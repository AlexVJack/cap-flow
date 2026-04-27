from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class V6Config:
    classes: tuple[int, ...]
    drive_k: int
    foreign_k: int
    weight_low: float
    weight_high: float
    state_leak: float
    drive_scale: float
    self_scale: float
    event_bonus_scale: float
    score_scale: float
    event_weight_scale: float
    target_output: float
    learn_correct_prob: float
    target_weight_scale: float
    confuser_weight_scale: float
    other_weight_scale: float
    event_credit_scale: float
    event_clip: float
    active_pixel_threshold: float
    weight_clip: float


@dataclass
class SignedEventCell:
    preferred_label: int
    weights: dict[int, float]
    state: float = 0.0
    trace: float = 0.0
    score: float = 0.0
    event_scores: dict[str, float] = field(default_factory=dict)


def event_signature(old_state: float, new_state: float) -> str:
    delta = new_state - old_state
    if abs(delta) < 0.04:
        if new_state > 0.10:
            return "hold_pos"
        if new_state < -0.10:
            return "hold_neg"
        return "flat"
    if old_state <= 0.0 < new_state:
        return "cross_up"
    if old_state >= 0.0 > new_state:
        return "cross_down"
    if delta > 0.0:
        return "up_pos" if new_state >= 0.0 else "up_neg"
    return "down_pos" if new_state >= 0.0 else "down_neg"


def build_cell(preferred_label: int, class_pixels: dict[int, list[int]], config: V6Config, rng: random.Random) -> SignedEventCell:
    preferred = class_pixels[preferred_label]
    foreign = []
    for cls in config.classes:
        if cls != preferred_label:
            foreign.extend(class_pixels[cls])

    weights = {}
    for idx in rng.sample(preferred, k=min(config.drive_k, len(preferred))):
        weights[idx] = rng.uniform(config.weight_low, config.weight_high)
    for idx in rng.sample(foreign, k=min(config.foreign_k, len(foreign))):
        weights[idx] = -rng.uniform(config.weight_low, config.weight_high)
    return SignedEventCell(preferred_label=preferred_label, weights=weights)


def build_pool(cell_count: int, class_pixels: dict[int, list[int]], config: V6Config, seed: int):
    rng = random.Random(seed)
    cells = []
    classes = list(config.classes)
    for idx in range(cell_count):
        cells.append(build_cell(classes[idx % len(classes)], class_pixels, config, rng))
    return cells


def reset_states(cells: list[SignedEventCell], leak: float) -> None:
    for cell in cells:
        cell.state *= leak


def cell_step(cell: SignedEventCell, sample: np.ndarray, config: V6Config) -> tuple[float, str, float]:
    old_state = cell.state
    drive = sum(float(sample[idx]) * weight for idx, weight in cell.weights.items())
    drive = float(np.clip(drive, -1.5, 1.5))
    raw_state = config.self_scale * old_state + config.drive_scale * drive
    cell.state = float(np.clip(raw_state, -1.0, 1.0))
    signature = event_signature(old_state, cell.state)
    event_bonus = cell.event_scores.get(signature, 0.0)
    raw = cell.state + config.event_bonus_scale * event_bonus
    out = float(np.clip(raw, -1.0, 1.0))
    return out, signature, cell.state - old_state


def _softmax(values: dict[int, float]) -> dict[int, float]:
    peak = max(values.values())
    exps = {cls: math.exp(value - peak) for cls, value in values.items()}
    norm = sum(exps.values())
    return {cls: value / norm for cls, value in exps.items()}


def forward_pass(cells: list[SignedEventCell], sample: np.ndarray, config: V6Config):
    rows = []
    for idx, cell in enumerate(cells):
        out, signature, delta = cell_step(cell, sample, config)
        modulation = max(0.2, 1.0 + config.score_scale * cell.score + config.event_weight_scale * cell.event_scores.get(signature, 0.0))
        weight = max(0.05, abs(out)) * modulation
        rows.append((idx, out, weight, signature, delta))

    class_evidence = {}
    for cls in config.classes:
        cls_rows = [out for idx, out, _, _, _ in rows if cells[idx].preferred_label == cls]
        class_evidence[cls] = sum(cls_rows) / len(cls_rows)

    class_signals = _softmax(class_evidence)
    return class_signals, class_evidence, rows


def update_cells(
    cells: list[SignedEventCell],
    sample: np.ndarray,
    rows,
    label: int,
    readout: dict[int, float],
    class_evidence: dict[int, float],
    config: V6Config,
) -> None:
    active_pixels = {idx for idx, value in enumerate(sample) if value > config.active_pixel_threshold}
    ranked = sorted(readout.items(), key=lambda item: item[1], reverse=True)
    confuser = next(cls for cls, _ in ranked if cls != label)

    for cell_idx, out, weight, signature, delta in rows:
        cell = cells[cell_idx]
        if cell.preferred_label == label:
            desired = config.target_output
            gate = 1.0 - readout[label]
            learn_scale = config.target_weight_scale
        elif cell.preferred_label == confuser:
            desired = -config.target_output
            gate = readout[confuser]
            learn_scale = config.confuser_weight_scale
        else:
            desired = 0.0
            gate = max(readout[confuser], readout[label])
            learn_scale = config.other_weight_scale

        credit = weight * (desired - out) * gate
        cell.trace = 0.9 * cell.trace + 0.1 * abs(delta)
        effective_credit = credit * max(0.05, cell.trace)
        cell.score = 0.9 * cell.score + 0.1 * effective_credit
        updated_event = cell.event_scores.get(signature, 0.0) + config.event_credit_scale * effective_credit
        cell.event_scores[signature] = float(np.clip(updated_event, -config.event_clip, config.event_clip))

        for pixel_idx in active_pixels:
            weight_value = cell.weights.get(pixel_idx, 0.0)
            pixel_value = float(sample[pixel_idx])
            weight_value += learn_scale * effective_credit * pixel_value
            cell.weights[pixel_idx] = float(np.clip(weight_value, -config.weight_clip, config.weight_clip))


def evaluate(cells: list[SignedEventCell], x: np.ndarray, y: np.ndarray, config: V6Config):
    rows = []
    correct = 0
    gaps = []
    for sample, label in zip(x, y):
        reset_states(cells, config.state_leak)
        class_signals, class_evidence, cell_rows = forward_pass(cells, sample, config)
        prediction = max(class_signals, key=class_signals.get)
        correct += int(prediction == int(label))
        true_signal = class_signals[int(label)]
        confuser_signal = max(value for cls, value in class_signals.items() if cls != int(label))
        gap = true_signal - confuser_signal
        gaps.append(gap)
        err = 1.0 - max(0.0, gap + 0.5)
        rows.append((true_signal, err, int(label), prediction, class_signals, class_evidence, cell_rows))

    mae = sum(err for _, err, _, _, _, _, _ in rows) / len(rows)
    mean_gap = sum(gaps) / len(gaps)
    return correct / len(y), mae, mean_gap, rows


def prediction_hist(rows, classes: tuple[int, ...]) -> dict[int, int]:
    hist = {cls: 0 for cls in classes}
    for _, _, _, pred, _, _, _ in rows:
        hist[pred] += 1
    return hist


def label_metrics(rows, label: int) -> tuple[float, float, float]:
    label_rows = [row for row in rows if row[2] == label]
    label_acc = sum(int(row[3] == label) for row in label_rows) / len(label_rows)
    label_mae = sum(row[1] for row in label_rows) / len(label_rows)
    label_gap = sum(row[4][label] - max(v for cls, v in row[4].items() if cls != label) for row in label_rows) / len(label_rows)
    return label_acc, label_mae, label_gap


def train_epoch(cells: list[SignedEventCell], x_train: np.ndarray, y_train: np.ndarray, config: V6Config) -> float:
    order = list(range(len(y_train)))
    random.shuffle(order)
    plastic_updates = 0
    for idx in order:
        sample = x_train[idx]
        label = int(y_train[idx])
        reset_states(cells, config.state_leak)
        class_signals, class_evidence, cell_rows = forward_pass(cells, sample, config)
        prediction = max(class_signals, key=class_signals.get)
        should_learn = (prediction != label) or (random.random() < config.learn_correct_prob)
        if not should_learn:
            continue
        plastic_updates += 1
        update_cells(cells, sample, cell_rows, label, class_signals, class_evidence, config)
    return plastic_updates / len(order)
