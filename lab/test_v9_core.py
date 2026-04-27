from __future__ import annotations

import random

import numpy as np

from v4_core import load_digits_subset, pick_distinctive_pixels
from v6_core import V6Config, build_pool, forward_pass
from v9_core import update_cells


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


def test_border_push_strengthens_near_zero_updates() -> None:
    random.seed(42)
    np.random.seed(42)
    x_train, _, y_train, _ = load_digits_subset(CONFIG.classes)
    class_pixels = pick_distinctive_pixels(x_train, y_train, CONFIG.classes, top_k=12)
    cells = build_pool(cell_count=4, class_pixels=class_pixels, config=CONFIG, seed=42)
    sample = x_train[0]
    label = int(y_train[0])
    class_signals, class_evidence, rows = forward_pass(cells, sample, CONFIG)

    # Force a near-zero border regime to verify v9 amplifies updates there.
    class_signals = {label: 0.501, next(cls for cls in CONFIG.classes if cls != label): 0.499}
    before = [dict(cell.weights) for cell in cells]
    update_cells(cells, sample, rows, label, class_signals, class_evidence, CONFIG)
    total_delta = sum(abs(cell.weights.get(k, 0.0) - before[idx].get(k, 0.0)) for idx, cell in enumerate(cells) for k in cell.weights)
    assert total_delta > 0.0001, total_delta
    print("PASS border_push_strengthens_near_zero_updates")


if __name__ == "__main__":
    test_border_push_strengthens_near_zero_updates()
