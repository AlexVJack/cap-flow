from __future__ import annotations

import random
import sys
import time

import numpy as np

from v4_core import load_digits_subset, pick_distinctive_pixels
from v6_core import V6Config
from v9_core import DEFAULT_CONTROL, build_pool, evaluate, label_metrics, prediction_hist, train_epoch


PAIR_CONFIG = V6Config(
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


MULTI_CONFIG = V6Config(
    classes=(0, 1, 2),
    drive_k=5,
    foreign_k=3,
    weight_low=0.12,
    weight_high=0.26,
    state_leak=0.65,
    drive_scale=0.52,
    self_scale=0.36,
    event_bonus_scale=0.14,
    score_scale=0.28,
    event_weight_scale=0.18,
    target_output=0.80,
    learn_correct_prob=0.10,
    target_weight_scale=0.014,
    confuser_weight_scale=0.010,
    other_weight_scale=0.008,
    event_credit_scale=0.14,
    event_clip=1.0,
    active_pixel_threshold=0.15,
    weight_clip=0.60,
)


FULL_CONFIG = V6Config(
    classes=tuple(range(10)),
    drive_k=5,
    foreign_k=3,
    weight_low=0.12,
    weight_high=0.26,
    state_leak=0.65,
    drive_scale=0.52,
    self_scale=0.36,
    event_bonus_scale=0.14,
    score_scale=0.28,
    event_weight_scale=0.18,
    target_output=0.80,
    learn_correct_prob=0.05,
    target_weight_scale=0.014,
    confuser_weight_scale=0.010,
    other_weight_scale=0.008,
    event_credit_scale=0.14,
    event_clip=1.0,
    active_pixel_threshold=0.15,
    weight_clip=0.60,
)


def run_probe(config: V6Config, cell_count: int) -> None:
    random.seed(42)
    np.random.seed(42)

    x_train, x_test, y_train, y_test = load_digits_subset(config.classes)
    class_pixels = pick_distinctive_pixels(x_train, y_train, config.classes, top_k=12)
    cells = build_pool(cell_count=cell_count, class_pixels=class_pixels, config=config, seed=42)

    before_acc, before_mae, before_gap, before_rows = evaluate(cells, x_test, y_test, config)
    print(f"before test_acc={before_acc:.4f} test_mae={before_mae:.4f} mean_gap={before_gap:.4f}", flush=True)

    for epoch in range(1, 7):
        t0 = time.perf_counter()
        plastic_rate = train_epoch(cells, x_train, y_train, config, control=DEFAULT_CONTROL, epoch=epoch)
        epoch_s = time.perf_counter() - t0
        train_acc, train_mae, train_gap, _ = evaluate(cells, x_train[: min(300, len(x_train))], y_train[: min(300, len(y_train))], config)
        test_acc, test_mae, test_gap, test_rows = evaluate(cells, x_test, y_test, config)
        mean_score = sum(cell.score for cell in cells) / len(cells)
        mean_event = sum(sum(cell.event_scores.values()) for cell in cells) / len(cells)
        mean_state = sum(cell.state for cell in cells) / len(cells)
        print(
            f"epoch={epoch} train_acc={train_acc:.4f} train_mae={train_mae:.4f} train_gap={train_gap:.4f} "
            f"test_acc={test_acc:.4f} test_mae={test_mae:.4f} test_gap={test_gap:.4f} "
            f"score={mean_score:.4f} event_score={mean_event:.4f} mean_state={mean_state:.4f} epoch_s={epoch_s:.2f} "
            f"plastic_rate={plastic_rate:.3f} pred_hist={prediction_hist(test_rows, config.classes)}",
            flush=True,
        )

    after_acc, after_mae, after_gap, after_rows = evaluate(cells, x_test, y_test, config)
    improvement_rate = sum(
        1 for (_, before_err, _, _, _, _, _), (_, after_err, _, _, _, _, _) in zip(before_rows, after_rows) if after_err < before_err
    ) / len(before_rows)
    print(f"after test_acc={after_acc:.4f} test_mae={after_mae:.4f} mean_gap={after_gap:.4f} improvement_rate={improvement_rate:.4f}", flush=True)

    for cls in config.classes:
        cls_acc, cls_mae, cls_gap = label_metrics(after_rows, cls)
        print(f"label={cls} acc={cls_acc:.4f} mae={cls_mae:.4f} mean_gap={cls_gap:.4f}", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pair"
    if mode == "pair":
        run_probe(PAIR_CONFIG, cell_count=8)
        return
    if mode == "multi":
        run_probe(MULTI_CONFIG, cell_count=12)
        return
    if mode == "full":
        run_probe(FULL_CONFIG, cell_count=40)
        return
    raise SystemExit("usage: python3 lab/cap_flow_v9.py [pair|multi|full]")


if __name__ == "__main__":
    main()
