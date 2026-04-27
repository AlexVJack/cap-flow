# v9.1 Technical Note

## Summary

`cap-flow` is a small experimental learning system built around local signed state-transition events.

This note documents the current `v9 -> v9.1` finding on `sklearn.datasets.load_digits`:

```text
Learning from errors is not enough;
fragile correct decisions near the border also need local reinforcement.
```

`v9.1 correct_border prob=0.80` improves over the clean `v9 hard_scale` baseline on a `10` seed confirmation with `64` cells and `12` epochs.

This is not a claim that the system beats neural networks or that the mechanism scales broadly. The result is a reproducible local-learning finding with diagnostic decomposition.

## System Sketch

The system uses a pool of `SignedEventCell` objects.

Each cell has:

- a preferred class label;
- sparse signed pixel weights;
- a scalar internal state;
- a trace and score;
- event scores keyed by state-transition signatures.

For each sample, a cell computes a state update:

```text
old_state -> new_state -> signed transition event
```

The event vocabulary is fixed:

```text
flat
hold_pos
hold_neg
cross_up
cross_down
up_pos
up_neg
down_pos
down_neg
```

Class evidence is aggregated from cells assigned to each class, then converted to class probabilities with a softmax.

The architecture is intentionally small and local:

- no backpropagation;
- no dense hidden layer;
- no MLP readout;
- no learned router;
- fixed event vocabulary;
- local updates to weights, traces, scores, and event scores.

## Baselines

The stable historical baseline is `v6 signed event cells`.

For this note, the main comparison is:

```text
v9   = clean hard_scale cell-level baseline
v9.1 = v9 diagnostic control with correct_border learning
```

`v9` should remain a clean baseline. `v9.1` is not a retroactive patch to `v9`; it is a separate frontier candidate.

## What v9 Does

`v9 hard_scale` adds border-sensitive learning pressure.

If the target-vs-confuser gap is near zero, the update scale is increased.

Conceptually:

```text
wrong near-border case -> stronger local update
```

This acts mainly as repair pressure for near-miss errors.

## What v9.1 Changes

`v9.1 correct_border` keeps the `v9` structure but changes when learning is allowed.

The key diagnostic rule is:

```python
if prediction != label:
    should_learn = True
elif abs(gap) < border_gap:
    should_learn = random.random() < correct_border_prob
else:
    should_learn = random.random() < learn_correct_prob
```

This separates two pressures:

```text
repair    = learn from wrong near-border cases
stabilize = learn from correct but fragile near-border cases
```

The current frontier candidate uses:

```text
correct_border_prob = 0.80
border_gap = 0.02
border_scale = 4.0
```

## Main Result

Configuration:

```text
dataset: sklearn digits, classes 0..9
cells: 64
epochs: 12
seeds: 10
```

Approximate aggregate:

| mode | final_acc | peak_acc | final_gap | border_after_acc | peak_drop |
|---|---:|---:|---:|---:|---:|
| `v9 full` | `0.9145` | `0.9209` | `0.0151` | `0.9164` | `0.0064` |
| `v9.1 correct_border 0.80` | `0.9469` | `0.9497` | `0.0282` | `0.9473` | `0.0028` |

Read:

- `correct_border 0.80` holds after expanding from `6` to `10` seeds.
- It improves final accuracy, peak accuracy, mean gap, and border-set accuracy.
- It has lower peak-to-final drop than `v9 full` in this confirmation.

## Mechanism Decomposition

The diagnostic runner tracks a fixed pre-training border set.

For each sample whose initial target-vs-confuser gap is near zero, it records whether training repaired, retained, or lost the decision.

Metrics:

```text
border_repair    = initially wrong border cases that become correct
border_retention = initially correct border cases that stay correct
border_loss      = initially correct border cases that become wrong
```

Focused decomposition on seeds `8..9`:

| mode | border_repair | border_retention | border_loss |
|---|---:|---:|---:|
| `v9 full` | `0.8670` | `0.9629` | `0.0371` |
| `v9.1 correct_border 0.80` | `0.9180` | `0.9742` | `0.0258` |

This supports the working interpretation:

```text
v9   = repair pressure for near-miss errors
v9.1 = repair pressure + stabilization pressure for fragile correct decisions
```

## Ablations And Diagnostics

The diagnostic package includes modes for:

- `full`;
- `weights_only`;
- `event_only`;
- `event_x2`;
- `event_x4`;
- `event_only_x2`;
- `event_only_x4`;
- `random_sparse`;
- `correct_border`.

Current read from prior diagnostics:

- plastic weights carry most of the short-horizon accuracy lift;
- event scores are live but weaker than direct weight learning in the current setup;
- aggressive event scaling can degrade the regime;
- `correct_border` is the strongest diagnostic control so far;
- persistent class-level attractor/confuser dynamics are not yet strong enough to justify opening `v10` as an accuracy patch.

## How To Run

Install dependencies first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 0 --modes full,correct_border --correct-border-prob 0.80
```

Chunked confirmation pattern:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 0 --modes full,correct_border --correct-border-prob 0.80
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 2 --modes full,correct_border --correct-border-prob 0.80
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 4 --modes full,correct_border --correct-border-prob 0.80
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 6 --modes full,correct_border --correct-border-prob 0.80
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 8 --modes full,correct_border --correct-border-prob 0.80
```

Cheap smoke check:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 12 --epochs 1 --seeds 1 --modes full,weights_only,event_only,correct_border --correct-border-prob 0.80
```

Mechanistic tests:

```bash
PYTHONPATH=lab python lab/test_v6_core.py
PYTHONPATH=lab python lab/test_v9_core.py
```

## Limitations

- The result is on `sklearn digits`, which is a small testbed.
- This does not establish broad scaling behavior.
- The system is not competitive with modern neural baselines as a general ML method.
- Event scores currently appear weaker than the plastic weight path.
- The `10` seed aggregate is still small by benchmark standards.
- `v9.1` should not replace the historical `v6` baseline until second-dataset checks exist.
- `v10` should not be opened merely to raise accuracy.

## Why v10 Is Still Postponed

`v10` should only be opened if diagnostics show stable class-level dynamics that cell-level tissue does not resolve.

Signals that would justify `v10`:

- stable false-positive magnets across seeds;
- repeated confuser pairs across seeds;
- class-level attractor behavior not fixed by `v9.1`;
- evidence that a second local transition-event tissue over class states is needed.

Possible `v10` shape, if justified:

```text
pixel -> cell tissue -> class tissue -> decision
```

But at the current stage:

```text
v9   = clean baseline
v9.1 = confirmed frontier candidate
v10  = postponed
```

## Current Finding

The current finding is narrow but clear:

```text
Adding local reinforcement for fragile correct border decisions improves the signed-event cell-level system on digits.
```

The most compact formulation:

```text
Learning from errors is not enough;
fragile correct decisions near the border also need local reinforcement.
```
