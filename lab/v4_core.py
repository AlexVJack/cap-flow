from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TransitionConfig:
    classes: tuple[int, ...]
    excite_pref_k: int
    excite_foreign_k: int
    inhibit_foreign_k: int
    inhibit_pref_k: int
    excite_low: float
    excite_high: float
    inhibit_low: float
    inhibit_high: float
    settle_steps: int
    excite_self: float
    excite_drive: float
    excite_memory: float
    excite_inhibit: float
    inhibit_self: float
    inhibit_drive: float
    inhibit_excite: float
    memory_self: float
    memory_excite: float
    memory_inhibit: float
    event_bonus_scale: float
    readout_memory_weight: float
    readout_inhibit_weight: float
    tanh_scale: float
    score_scale: float
    event_weight_scale: float
    learn_correct_prob: float
    target_output: float
    confuser_mix: float
    target_excite_scale: float
    target_inhibit_scale: float
    confuser_excite_scale: float
    confuser_inhibit_scale: float
    other_inhibit_scale: float
    event_credit_scale: float
    event_clip: float
    active_pixel_threshold: float
    state_leak: float
    homeostatic_gate_target: float = 1.0
    homeostatic_gate_strength: float = 0.0
    homeostatic_min_plasticity: float = 1.0


@dataclass
class TransitionModule:
    preferred_label: int
    excite_weights: dict[int, float]
    inhibit_weights: dict[int, float]
    state: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    trace: float = 0.0
    score: float = 0.0
    inhibition: float = 0.0
    target_activity: float = 0.22
    event_scores: dict[str, float] = field(default_factory=dict)


def load_digits_subset(classes: tuple[int, ...], test_size: float = 0.2, random_state: int = 42):
    digits = load_digits()
    x = (digits.data.astype(np.float32) / 16.0).copy()
    y = digits.target.astype(np.int64)
    mask = np.isin(y, classes)
    x = x[mask]
    y = y[mask]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def pick_distinctive_pixels(x_train: np.ndarray, y_train: np.ndarray, classes: tuple[int, ...], top_k: int = 12) -> dict[int, list[int]]:
    class_pixels = {}
    for cls in classes:
        cls_mean = x_train[y_train == cls].mean(axis=0)
        other_mean = x_train[y_train != cls].mean(axis=0)
        diff = cls_mean - other_mean
        class_pixels[cls] = np.argsort(diff)[-top_k:].tolist()
    return class_pixels


def build_module(preferred_label: int, class_pixels: dict[int, list[int]], config: TransitionConfig, rng: random.Random) -> TransitionModule:
    preferred = class_pixels[preferred_label]
    foreign = []
    for cls in config.classes:
        if cls != preferred_label:
            foreign.extend(class_pixels[cls])

    excite_inputs = rng.sample(preferred, k=min(config.excite_pref_k, len(preferred)))
    if foreign:
        excite_inputs.extend(rng.sample(foreign, k=min(config.excite_foreign_k, len(foreign))))

    inhibit_inputs = []
    if foreign:
        inhibit_inputs.extend(rng.sample(foreign, k=min(config.inhibit_foreign_k, len(foreign))))
    inhibit_inputs.extend(rng.sample(preferred, k=min(config.inhibit_pref_k, len(preferred))))

    excite_weights = {idx: rng.uniform(config.excite_low, config.excite_high) for idx in excite_inputs}
    inhibit_weights = {idx: rng.uniform(config.inhibit_low, config.inhibit_high) for idx in inhibit_inputs}
    return TransitionModule(preferred_label=preferred_label, excite_weights=excite_weights, inhibit_weights=inhibit_weights)


def build_pool(module_count: int, class_pixels: dict[int, list[int]], config: TransitionConfig, seed: int):
    rng = random.Random(seed)
    modules = []
    class_groups = {cls: [] for cls in config.classes}
    classes = list(config.classes)
    for idx in range(module_count):
        cls = classes[idx % len(classes)]
        modules.append(build_module(cls, class_pixels, config, rng))
        class_groups[cls].append(idx)
    return modules, class_groups


def event_signature(delta: np.ndarray) -> str:
    labels = []
    for value in delta:
        if value > 0.04:
            labels.append("up")
        elif value < -0.04:
            labels.append("down")
        else:
            labels.append("flat")
    return "/".join(labels)


def reset_states(modules: list[TransitionModule], leak: float) -> None:
    for module in modules:
        module.state *= leak


def module_step(module: TransitionModule, sample: np.ndarray, config: TransitionConfig) -> tuple[float, str, np.ndarray]:
    old_state = module.state.copy()
    excite_drive = sum(float(sample[idx]) * weight for idx, weight in module.excite_weights.items())
    inhibit_drive = sum(float(sample[idx]) * weight for idx, weight in module.inhibit_weights.items())
    excite_drive = min(1.5, excite_drive)
    inhibit_drive = min(1.5, inhibit_drive)

    state = old_state.copy()
    for _ in range(config.settle_steps):
        excite = config.excite_self * state[0] + config.excite_drive * excite_drive + config.excite_memory * state[2] - config.excite_inhibit * state[1]
        inhibit = config.inhibit_self * state[1] + config.inhibit_drive * inhibit_drive + config.inhibit_excite * state[0]
        memory = config.memory_self * state[2] + config.memory_excite * state[0] - config.memory_inhibit * state[1]
        state = np.array([excite, inhibit, memory], dtype=np.float32)
        state = np.clip(state, 0.0, 1.0)

    module.state = 0.50 * module.state + 0.50 * state
    module.state = np.clip(module.state, 0.0, 1.0)
    delta = module.state - old_state
    signature = event_signature(delta)
    event_bonus = module.event_scores.get(signature, 0.0)
    raw = module.state[0] + config.readout_memory_weight * module.state[2] - config.readout_inhibit_weight * module.state[1] + config.event_bonus_scale * event_bonus
    adjusted = raw * max(0.2, 1.0 - module.inhibition)
    out = float(np.clip(0.5 + 0.5 * np.tanh(config.tanh_scale * adjusted), 0.0, 1.0))
    return out, signature, delta


def forward_pass(modules: list[TransitionModule], class_groups: dict[int, list[int]], sample: np.ndarray, config: TransitionConfig):
    rows = []
    for idx, module in enumerate(modules):
        out, signature, delta = module_step(module, sample, config)
        activity = max(0.0, out - module.target_activity)
        event_bonus = module.event_scores.get(signature, 0.0)
        weight = (0.05 + activity) * max(0.2, 1.0 + config.score_scale * module.score + config.event_weight_scale * event_bonus)
        rows.append((idx, out, weight, signature, delta))

    class_signals = {}
    for cls in config.classes:
        cls_rows = [out for idx, out, _, _, _ in rows if modules[idx].preferred_label == cls]
        class_signals[cls] = sum(cls_rows) / len(cls_rows)

    eps = 1e-6
    norm = sum(max(0.0, value) + eps for value in class_signals.values())
    class_signals = {cls: (max(0.0, value) + eps) / norm for cls, value in class_signals.items()}
    return class_signals, rows


def update_modules(modules: list[TransitionModule], sample: np.ndarray, rows, label: int, readout: dict[int, float], config: TransitionConfig) -> None:
    active_pixels = {idx for idx, value in enumerate(sample) if value > config.active_pixel_threshold}
    ranked = sorted(readout.items(), key=lambda item: item[1], reverse=True)
    confuser = next(cls for cls, _ in ranked if cls != label)

    for module_idx, out, weight, signature, delta in rows:
        module = modules[module_idx]
        module_cls = module.preferred_label
        if module_cls == label:
            credit = weight * (config.target_output - out) * (1.0 - readout[label])
        elif module_cls == confuser:
            credit = -weight * out * (config.confuser_mix * readout[confuser] + (1.0 - config.confuser_mix) * readout[label])
        else:
            credit = -config.other_inhibit_scale * weight * out * readout[label]

        eligibility = float(np.abs(delta).sum())
        module.trace = 0.90 * module.trace + 0.10 * eligibility
        plasticity_gate = 1.0
        if config.homeostatic_gate_strength > 0.0:
            overload = max(0.0, out - config.homeostatic_gate_target)
            plasticity_gate = max(config.homeostatic_min_plasticity, 1.0 - config.homeostatic_gate_strength * overload)

        effective_credit = credit * module.trace * plasticity_gate
        module.score = 0.9 * module.score + 0.1 * effective_credit
        updated_event = module.event_scores.get(signature, 0.0) + config.event_credit_scale * effective_credit
        module.event_scores[signature] = float(np.clip(updated_event, -config.event_clip, config.event_clip))

        for pixel_idx in active_pixels:
            excite = module.excite_weights.get(pixel_idx, 0.0)
            inhibit = module.inhibit_weights.get(pixel_idx, 0.0)
            pixel_value = float(sample[pixel_idx])

            if module_cls == label:
                excite += config.target_excite_scale * max(0.0, effective_credit) * pixel_value
                inhibit -= config.target_inhibit_scale * max(0.0, effective_credit) * pixel_value
            elif module_cls == confuser:
                excite += config.confuser_excite_scale * min(0.0, effective_credit) * pixel_value
                inhibit += config.confuser_inhibit_scale * abs(min(0.0, effective_credit)) * pixel_value
            else:
                inhibit += config.other_inhibit_scale * abs(min(0.0, effective_credit)) * pixel_value

            module.excite_weights[pixel_idx] = float(np.clip(excite, 0.0, 0.60))
            module.inhibit_weights[pixel_idx] = float(np.clip(inhibit, 0.0, 0.60))


def evaluate(modules: list[TransitionModule], class_groups: dict[int, list[int]], x: np.ndarray, y: np.ndarray, config: TransitionConfig):
    rows = []
    correct = 0
    gaps = []
    for sample, label in zip(x, y):
        reset_states(modules, config.state_leak)
        class_signals, module_rows = forward_pass(modules, class_groups, sample, config)
        prediction = max(class_signals, key=class_signals.get)
        correct += int(prediction == int(label))
        true_signal = class_signals[int(label)]
        confuser_signal = max(value for cls, value in class_signals.items() if cls != int(label))
        gap = true_signal - confuser_signal
        gaps.append(gap)
        err = 1.0 - max(0.0, gap + 0.5)
        rows.append((true_signal, err, int(label), prediction, class_signals, module_rows))

    mae = sum(err for _, err, _, _, _, _ in rows) / len(rows)
    mean_gap = sum(gaps) / len(gaps)
    return correct / len(y), mae, mean_gap, rows


def prediction_hist(rows, classes: tuple[int, ...]) -> dict[int, int]:
    hist = {cls: 0 for cls in classes}
    for _, _, _, pred, _, _ in rows:
        hist[pred] += 1
    return hist


def label_metrics(rows, label: int) -> tuple[float, float, float]:
    label_rows = [row for row in rows if row[2] == label]
    label_acc = sum(int(row[3] == label) for row in label_rows) / len(label_rows)
    label_mae = sum(row[1] for row in label_rows) / len(label_rows)
    label_gap = sum(row[4][label] - max(v for cls, v in row[4].items() if cls != label) for row in label_rows) / len(label_rows)
    return label_acc, label_mae, label_gap


def train_epoch(modules: list[TransitionModule], class_groups: dict[int, list[int]], x_train: np.ndarray, y_train: np.ndarray, config: TransitionConfig) -> float:
    order = list(range(len(y_train)))
    random.shuffle(order)
    plastic_updates = 0
    for idx in order:
        sample = x_train[idx]
        label = int(y_train[idx])
        reset_states(modules, config.state_leak)
        class_signals, module_rows = forward_pass(modules, class_groups, sample, config)
        prediction = max(class_signals, key=class_signals.get)
        should_learn = (prediction != label) or (random.random() < config.learn_correct_prob)
        if not should_learn:
            continue
        plastic_updates += 1
        update_modules(modules, sample, module_rows, label, class_signals, config)
    return plastic_updates / len(order)
