from __future__ import annotations

import random

import numpy as np

from v4_core import load_digits_subset, pick_distinctive_pixels
from v6_core import V6Config, SignedEventCell, event_signature, forward_pass, build_pool, update_cells


CONFIG = V6Config(
    classes=(1, 2),
    drive_k=6,
    foreign_k=2,
    weight_low=0.14,
    weight_high=0.30,
    state_leak=0.65,
    drive_scale=0.55,
    self_scale=0.35,
    event_bonus_scale=0.18,
    score_scale=0.30,
    event_weight_scale=0.20,
    target_output=0.82,
    learn_correct_prob=0.15,
    target_weight_scale=0.016,
    confuser_weight_scale=0.012,
    other_weight_scale=0.010,
    event_credit_scale=0.16,
    event_clip=1.0,
    active_pixel_threshold=0.15,
    weight_clip=0.60,
)


def test_event_signature_tracks_signed_regimes() -> None:
    assert event_signature(0.0, 0.20) == "cross_up"
    assert event_signature(0.30, -0.10) == "cross_down"
    assert event_signature(-0.40, -0.20) == "up_neg"
    assert event_signature(0.40, 0.20) == "down_pos"
    print("PASS event_signature_tracks_signed_regimes")


def test_forward_pass_allows_negative_evidence() -> None:
    sample = np.array([1.0, 1.0], dtype=float)
    cells = [
        SignedEventCell(preferred_label=1, weights={0: 0.5}),
        SignedEventCell(preferred_label=2, weights={1: -0.5}),
    ]
    class_signals, class_evidence, rows = forward_pass(cells, sample, CONFIG)
    assert sum(class_signals.values()) > 0.99999
    assert abs(sum(class_signals.values()) - 1.0) < 1e-5
    assert class_evidence[1] > 0.0, class_evidence
    assert class_evidence[2] < 0.0, class_evidence
    assert rows[1][1] < 0.0, rows
    print("PASS forward_pass_allows_negative_evidence")


def test_signed_credit_can_push_weights_negative() -> None:
    random.seed(42)
    np.random.seed(42)
    x_train, _, y_train, _ = load_digits_subset(CONFIG.classes)
    class_pixels = pick_distinctive_pixels(x_train, y_train, CONFIG.classes, top_k=12)
    cells = build_pool(cell_count=4, class_pixels=class_pixels, config=CONFIG, seed=42)
    sample = x_train[0]
    label = int(y_train[0])
    class_signals, class_evidence, rows = forward_pass(cells, sample, CONFIG)
    ranked = sorted(class_signals.items(), key=lambda item: item[1], reverse=True)
    confuser = next(cls for cls, _ in ranked if cls != label)
    confuser_idx = next(idx for idx, cell in enumerate(cells) if cell.preferred_label == confuser)
    before_event = cells[confuser_idx].event_scores.get(rows[confuser_idx][3], 0.0)
    before_weights = dict(cells[confuser_idx].weights)

    update_cells(cells, sample, rows, label, class_signals, class_evidence, CONFIG)

    after_event = cells[confuser_idx].event_scores.get(rows[confuser_idx][3], 0.0)
    negative_weights = sum(1 for value in cells[confuser_idx].weights.values() if value < 0.0)
    changed_weights = sum(1 for key, value in cells[confuser_idx].weights.items() if abs(value - before_weights.get(key, 0.0)) > 1e-9)
    assert after_event != before_event, (before_event, after_event)
    assert changed_weights > 0, changed_weights
    assert negative_weights > 0, cells[confuser_idx].weights
    print("PASS signed_credit_can_push_weights_negative")


if __name__ == "__main__":
    test_event_signature_tracks_signed_regimes()
    test_forward_pass_allows_negative_evidence()
    test_signed_credit_can_push_weights_negative()
