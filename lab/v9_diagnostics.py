from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import numpy as np

from cap_flow_v9 import FULL_CONFIG
from v4_core import load_digits_subset, pick_distinctive_pixels
from v6_core import SignedEventCell, V6Config
from v9_core import V9Control, build_pool, evaluate, forward_pass, reset_states


@dataclass(frozen=True)
class DiagnosticMode:
    name: str
    event_scale: float = 1.0
    update_weights: bool = True
    update_events: bool = True
    all_pixels_init: bool = False
    correct_border_prob: float | None = None


@dataclass(frozen=True)
class RunResult:
    mode: str
    seed: int
    before_acc: float
    peak_acc: float
    peak_epoch: int
    final_acc: float
    final_gap: float
    peak_drop: float
    plastic_rate: float
    border_count: int
    border_before_acc: float
    border_after_acc: float
    border_before_gap: float
    border_after_gap: float
    border_flip_rate: float
    border_repair_rate: float
    border_retention_rate: float
    border_loss_rate: float
    mean_event: float
    event_std: float
    weight_count: int
    runtime_s: float
    false_positives: dict[int, int]
    false_negatives: dict[int, int]
    confusions: dict[tuple[int, int], int]
    mean_evidence: dict[int, float]
    wrong_pred_evidence: dict[int, float]
    mean_margin: dict[int, float]


MODES = {
    "full": DiagnosticMode("full"),
    "weights_only": DiagnosticMode("weights_only", update_events=False),
    "event_only": DiagnosticMode("event_only", update_weights=False),
    "event_x2": DiagnosticMode("event_x2", event_scale=2.0),
    "event_x4": DiagnosticMode("event_x4", event_scale=4.0),
    "event_only_x2": DiagnosticMode("event_only_x2", event_scale=2.0, update_weights=False),
    "event_only_x4": DiagnosticMode("event_only_x4", event_scale=4.0, update_weights=False),
    "all_pixels_init": DiagnosticMode("all_pixels_init", all_pixels_init=True),
    "random_sparse": DiagnosticMode("random_sparse", all_pixels_init=True),
    "correct_border": DiagnosticMode("correct_border", correct_border_prob=0.50),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return float(np.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def _scaled_config(config: V6Config, mode: DiagnosticMode) -> V6Config:
    return replace(
        config,
        event_bonus_scale=config.event_bonus_scale * mode.event_scale,
        event_weight_scale=config.event_weight_scale * mode.event_scale,
        event_credit_scale=config.event_credit_scale * mode.event_scale,
    )


def _build_class_pixels(x_train: np.ndarray, y_train: np.ndarray, config: V6Config, mode: DiagnosticMode) -> dict[int, list[int]]:
    if mode.all_pixels_init:
        return {cls: list(range(x_train.shape[1])) for cls in config.classes}
    return pick_distinctive_pixels(x_train, y_train, config.classes, top_k=12)


def _row_gap(row) -> float:
    _, _, label, _, class_signals, _, _ = row
    return class_signals[label] - max(value for cls, value in class_signals.items() if cls != label)


def _row_correct(row) -> bool:
    _, _, label, prediction, _, _, _ = row
    return label == prediction


def _border_metrics(before_rows, after_rows, threshold: float) -> tuple[int, float, float, float, float, float, float, float, float]:
    border_indices = [idx for idx, row in enumerate(before_rows) if abs(_row_gap(row)) < threshold]
    if not border_indices:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    before_correct = [_row_correct(before_rows[idx]) for idx in border_indices]
    after_correct = [_row_correct(after_rows[idx]) for idx in border_indices]
    before_gaps = [_row_gap(before_rows[idx]) for idx in border_indices]
    after_gaps = [_row_gap(after_rows[idx]) for idx in border_indices]
    flips = sum(int(before != after) for before, after in zip(before_correct, after_correct))
    before_wrong = sum(int(not value) for value in before_correct)
    before_right = sum(int(value) for value in before_correct)
    repairs = sum(int((not before) and after) for before, after in zip(before_correct, after_correct))
    retained = sum(int(before and after) for before, after in zip(before_correct, after_correct))
    losses = sum(int(before and (not after)) for before, after in zip(before_correct, after_correct))
    return (
        len(border_indices),
        sum(before_correct) / len(border_indices),
        sum(after_correct) / len(border_indices),
        _mean(before_gaps),
        _mean(after_gaps),
        flips / len(border_indices),
        repairs / before_wrong if before_wrong else 0.0,
        retained / before_right if before_right else 0.0,
        losses / before_right if before_right else 0.0,
    )


def _class_diagnostics(rows, classes: tuple[int, ...]):
    false_positives = Counter()
    false_negatives = Counter()
    confusions = Counter()
    evidence_values: dict[int, list[float]] = defaultdict(list)
    wrong_pred_evidence: dict[int, list[float]] = defaultdict(list)
    margin_values: dict[int, list[float]] = defaultdict(list)

    for _, _, label, prediction, class_signals, class_evidence, _ in rows:
        for cls in classes:
            evidence_values[cls].append(class_evidence[cls])
        margin_values[label].append(class_signals[label] - max(value for cls, value in class_signals.items() if cls != label))
        if prediction != label:
            false_negatives[label] += 1
            false_positives[prediction] += 1
            confusions[(label, prediction)] += 1
            wrong_pred_evidence[prediction].append(class_evidence[prediction])

    return (
        {cls: false_positives[cls] for cls in classes},
        {cls: false_negatives[cls] for cls in classes},
        dict(confusions),
        {cls: _mean(evidence_values[cls]) for cls in classes},
        {cls: _mean(wrong_pred_evidence[cls]) for cls in classes},
        {cls: _mean(margin_values[cls]) for cls in classes},
    )


def _cell_scale_for(control: V9Control, border_active: bool, cell: SignedEventCell, label: int, confuser: int, signature: str) -> float:
    cell_scale = 1.0
    if border_active:
        if control.push_mode == "all":
            cell_scale = control.border_scale
        elif control.push_mode == "target" and cell.preferred_label == label:
            cell_scale = control.border_scale
        elif control.push_mode == "confuser" and cell.preferred_label == confuser:
            cell_scale = control.border_scale
        elif control.push_mode == "target_confuser" and cell.preferred_label in {label, confuser}:
            cell_scale = control.border_scale
    if border_active and control.cross_event_scale > 1.0 and signature in {"cross_up", "cross_down"}:
        cell_scale *= control.cross_event_scale
    return cell_scale


def _update_cells_diagnostic(
    cells: list[SignedEventCell],
    sample: np.ndarray,
    rows,
    label: int,
    readout: dict[int, float],
    config: V6Config,
    control: V9Control,
    mode: DiagnosticMode,
    epoch: int,
) -> None:
    active_pixels = {idx for idx, value in enumerate(sample) if value > config.active_pixel_threshold}
    ranked = sorted(readout.items(), key=lambda item: item[1], reverse=True)
    confuser = next(cls for cls, _ in ranked if cls != label)
    sample_gap = readout[label] - readout[confuser]
    border_active = abs(sample_gap) < control.border_gap
    if control.early_epochs is not None:
        border_active = border_active and epoch <= control.early_epochs

    for cell_idx, out, weight, signature, delta in rows:
        cell = cells[cell_idx]
        cell_scale = _cell_scale_for(control, border_active, cell, label, confuser, signature)
        if cell.preferred_label == label:
            desired = config.target_output
            gate = 1.0 - readout[label]
            learn_scale = config.target_weight_scale * cell_scale
        elif cell.preferred_label == confuser:
            desired = -config.target_output
            gate = readout[confuser]
            learn_scale = config.confuser_weight_scale * cell_scale
        else:
            desired = 0.0
            gate = max(readout[confuser], readout[label])
            learn_scale = config.other_weight_scale * cell_scale

        credit = weight * (desired - out) * gate
        cell.trace = 0.9 * cell.trace + 0.1 * abs(delta)
        effective_credit = credit * max(0.05, cell.trace)
        cell.score = 0.9 * cell.score + 0.1 * effective_credit

        if mode.update_events:
            updated_event = cell.event_scores.get(signature, 0.0) + config.event_credit_scale * cell_scale * effective_credit
            cell.event_scores[signature] = float(np.clip(updated_event, -config.event_clip, config.event_clip))

        if mode.update_weights:
            for pixel_idx in active_pixels:
                weight_value = cell.weights.get(pixel_idx, 0.0)
                pixel_value = float(sample[pixel_idx])
                weight_value += learn_scale * effective_credit * pixel_value
                cell.weights[pixel_idx] = float(np.clip(weight_value, -config.weight_clip, config.weight_clip))


def train_epoch_diagnostic(
    cells: list[SignedEventCell],
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: V6Config,
    control: V9Control,
    mode: DiagnosticMode,
    epoch: int,
) -> float:
    order = list(range(len(y_train)))
    random.shuffle(order)
    plastic_updates = 0
    for idx in order:
        sample = x_train[idx]
        label = int(y_train[idx])
        reset_states(cells, config.state_leak)
        class_signals, _, cell_rows = forward_pass(cells, sample, config)
        prediction = max(class_signals, key=class_signals.get)
        confuser_signal = max(value for cls, value in class_signals.items() if cls != label)
        gap = class_signals[label] - confuser_signal
        if prediction != label:
            should_learn = True
        elif mode.correct_border_prob is not None and abs(gap) < control.border_gap:
            should_learn = random.random() < mode.correct_border_prob
        else:
            should_learn = random.random() < config.learn_correct_prob
        if not should_learn:
            continue
        plastic_updates += 1
        _update_cells_diagnostic(cells, sample, cell_rows, label, class_signals, config, control, mode, epoch)
    return plastic_updates / len(order)


def _event_stats(cells: list[SignedEventCell]) -> tuple[float, float]:
    values = [value for cell in cells for value in cell.event_scores.values()]
    return _mean(values), _std(values)


def _weight_count(cells: list[SignedEventCell]) -> int:
    return sum(len(cell.weights) for cell in cells)


def run_seed(mode: DiagnosticMode, seed: int, cell_count: int, epochs: int, control: V9Control, border_threshold: float) -> RunResult:
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    config = _scaled_config(FULL_CONFIG, mode)
    x_train, x_test, y_train, y_test = load_digits_subset(config.classes)
    class_pixels = _build_class_pixels(x_train, y_train, config, mode)
    cells = build_pool(cell_count=cell_count, class_pixels=class_pixels, config=config, seed=seed)

    before_acc, _, _, before_rows = evaluate(cells, x_test, y_test, config)
    peak_acc = before_acc
    peak_epoch = 0
    plastic_rates = []
    for epoch in range(1, epochs + 1):
        plastic_rate = train_epoch_diagnostic(cells, x_train, y_train, config, control, mode, epoch)
        plastic_rates.append(plastic_rate)
        acc, _, _, _ = evaluate(cells, x_test, y_test, config)
        if acc > peak_acc:
            peak_acc = acc
            peak_epoch = epoch

    final_acc, _, final_gap, after_rows = evaluate(cells, x_test, y_test, config)
    border = _border_metrics(before_rows, after_rows, border_threshold)
    false_pos, false_neg, confusions, mean_evidence, wrong_pred_evidence, mean_margin = _class_diagnostics(after_rows, config.classes)
    mean_event, event_std = _event_stats(cells)
    return RunResult(
        mode=mode.name,
        seed=seed,
        before_acc=before_acc,
        peak_acc=peak_acc,
        peak_epoch=peak_epoch,
        final_acc=final_acc,
        final_gap=final_gap,
        peak_drop=max(0.0, peak_acc - final_acc),
        plastic_rate=_mean(plastic_rates),
        border_count=border[0],
        border_before_acc=border[1],
        border_after_acc=border[2],
        border_before_gap=border[3],
        border_after_gap=border[4],
        border_flip_rate=border[5],
        border_repair_rate=border[6],
        border_retention_rate=border[7],
        border_loss_rate=border[8],
        mean_event=mean_event,
        event_std=event_std,
        weight_count=_weight_count(cells),
        runtime_s=time.perf_counter() - started,
        false_positives=false_pos,
        false_negatives=false_neg,
        confusions=confusions,
        mean_evidence=mean_evidence,
        wrong_pred_evidence=wrong_pred_evidence,
        mean_margin=mean_margin,
    )


def _format_class_counts(values: dict[int, int]) -> str:
    return ",".join(f"{cls}:{values.get(cls, 0)}" for cls in sorted(values))


def _format_top_confusions(results: list[RunResult], limit: int) -> str:
    counter = Counter()
    for result in results:
        counter.update(result.confusions)
    return ",".join(f"{label}->{pred}:{count}" for (label, pred), count in counter.most_common(limit))


def print_summary(mode: str, results: list[RunResult], top_confusions: int) -> None:
    print(
        f"summary mode={mode} seeds={len(results)} "
        f"before_acc={_mean([r.before_acc for r in results]):.4f} "
        f"peak_acc={_mean([r.peak_acc for r in results]):.4f} "
        f"final_acc={_mean([r.final_acc for r in results]):.4f} "
        f"final_std={_std([r.final_acc for r in results]):.4f} "
        f"final_gap={_mean([r.final_gap for r in results]):.4f} "
        f"peak_drop={_mean([r.peak_drop for r in results]):.4f} "
        f"border_count={_mean([float(r.border_count) for r in results]):.1f} "
        f"border_before_acc={_mean([r.border_before_acc for r in results]):.4f} "
        f"border_after_acc={_mean([r.border_after_acc for r in results]):.4f} "
        f"border_before_gap={_mean([r.border_before_gap for r in results]):.4f} "
        f"border_after_gap={_mean([r.border_after_gap for r in results]):.4f} "
        f"border_flip={_mean([r.border_flip_rate for r in results]):.4f} "
        f"border_repair={_mean([r.border_repair_rate for r in results]):.4f} "
        f"border_retention={_mean([r.border_retention_rate for r in results]):.4f} "
        f"border_loss={_mean([r.border_loss_rate for r in results]):.4f} "
        f"mean_event={_mean([r.mean_event for r in results]):.4f} "
        f"event_std={_mean([r.event_std for r in results]):.4f} "
        f"weights={_mean([float(r.weight_count) for r in results]):.1f} "
        f"runtime_s={_mean([r.runtime_s for r in results]):.2f}",
        flush=True,
    )
    fp = Counter()
    fn = Counter()
    for result in results:
        fp.update(result.false_positives)
        fn.update(result.false_negatives)
    print(f"class_false_positive mode={mode} {_format_class_counts(dict(fp))}", flush=True)
    print(f"class_false_negative mode={mode} {_format_class_counts(dict(fn))}", flush=True)
    print(f"top_confusions mode={mode} {_format_top_confusions(results, top_confusions)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v9 diagnostic ablations without modifying v9 core.")
    parser.add_argument("--modes", default="full,weights_only,event_only,event_x2,event_x4,correct_border")
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--border-gap", type=float, default=0.02)
    parser.add_argument("--border-scale", type=float, default=4.0)
    parser.add_argument("--border-threshold", type=float, default=0.02)
    parser.add_argument("--correct-border-prob", type=float, default=0.50)
    parser.add_argument("--top-confusions", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_modes = [name.strip() for name in args.modes.split(",") if name.strip()]
    control = V9Control(border_gap=args.border_gap, border_scale=args.border_scale)
    print(
        f"v9_diagnostics cells={args.cells} epochs={args.epochs} seeds={args.seeds} "
        f"seed_start={args.seed_start} "
        f"border_gap={args.border_gap:.4f} border_scale={args.border_scale:.2f} border_threshold={args.border_threshold:.4f}",
        f"correct_border_prob={args.correct_border_prob:.2f}",
        flush=True,
    )
    for name in requested_modes:
        if name not in MODES:
            raise SystemExit(f"unknown mode: {name}; available={','.join(sorted(MODES))}")
        mode = MODES[name]
        if name == "correct_border":
            mode = replace(mode, correct_border_prob=args.correct_border_prob)
        results = [
            run_seed(mode=mode, seed=seed, cell_count=args.cells, epochs=args.epochs, control=control, border_threshold=args.border_threshold)
            for seed in range(args.seed_start, args.seed_start + args.seeds)
        ]
        for result in results:
            print(
                f"run mode={result.mode} seed={result.seed} before={result.before_acc:.4f} peak={result.peak_acc:.4f} "
                f"peak_epoch={result.peak_epoch} final={result.final_acc:.4f} gap={result.final_gap:.4f} "
                f"border_n={result.border_count} border_before={result.border_before_acc:.4f} "
                f"border_after={result.border_after_acc:.4f} border_repair={result.border_repair_rate:.4f} "
                f"border_retention={result.border_retention_rate:.4f} border_loss={result.border_loss_rate:.4f} "
                f"mean_event={result.mean_event:.4f} "
                f"weights={result.weight_count} runtime_s={result.runtime_s:.2f}",
                flush=True,
            )
        print_summary(name, results, args.top_confusions)


if __name__ == "__main__":
    main()
