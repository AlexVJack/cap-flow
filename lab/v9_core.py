from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from v6_core import (
    SignedEventCell,
    V6Config,
    build_pool,
    cell_step,
    evaluate,
    event_signature,
    forward_pass,
    label_metrics,
    prediction_hist,
    reset_states,
)


@dataclass(frozen=True)
class V9Control:
    border_gap: float = 0.02
    border_scale: float = 3.0
    early_epochs: int | None = None
    cross_event_scale: float = 1.0
    push_mode: str = "all"


DEFAULT_CONTROL = V9Control()


def update_cells(
    cells: list[SignedEventCell],
    sample: np.ndarray,
    rows,
    label: int,
    readout: dict[int, float],
    class_evidence: dict[int, float],
    config: V6Config,
    control: V9Control | None = None,
    epoch: int | None = None,
) -> None:
    del class_evidence
    control = control or DEFAULT_CONTROL
    active_pixels = {idx for idx, value in enumerate(sample) if value > config.active_pixel_threshold}
    ranked = sorted(readout.items(), key=lambda item: item[1], reverse=True)
    confuser = next(cls for cls, _ in ranked if cls != label)
    sample_gap = readout[label] - readout[confuser]
    border_active = abs(sample_gap) < control.border_gap
    if control.early_epochs is not None and epoch is not None:
        border_active = border_active and epoch <= control.early_epochs
    border_scale = control.border_scale if border_active else 1.0

    for cell_idx, out, weight, signature, delta in rows:
        cell = cells[cell_idx]
        cell_scale = 1.0
        if border_active:
            if control.push_mode == "all":
                cell_scale = border_scale
            elif control.push_mode == "target" and cell.preferred_label == label:
                cell_scale = border_scale
            elif control.push_mode == "confuser" and cell.preferred_label == confuser:
                cell_scale = border_scale
            elif control.push_mode == "target_confuser" and cell.preferred_label in {label, confuser}:
                cell_scale = border_scale
        if border_active and control.cross_event_scale > 1.0 and signature in {"cross_up", "cross_down"}:
            cell_scale *= control.cross_event_scale
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
        updated_event = cell.event_scores.get(signature, 0.0) + config.event_credit_scale * cell_scale * effective_credit
        cell.event_scores[signature] = float(np.clip(updated_event, -config.event_clip, config.event_clip))

        for pixel_idx in active_pixels:
            weight_value = cell.weights.get(pixel_idx, 0.0)
            pixel_value = float(sample[pixel_idx])
            weight_value += learn_scale * effective_credit * pixel_value
            cell.weights[pixel_idx] = float(np.clip(weight_value, -config.weight_clip, config.weight_clip))


def train_epoch(
    cells: list[SignedEventCell],
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: V6Config,
    control: V9Control | None = None,
    epoch: int | None = None,
) -> float:
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
        update_cells(cells, sample, cell_rows, label, class_signals, class_evidence, config, control=control, epoch=epoch)
    return plastic_updates / len(order)
