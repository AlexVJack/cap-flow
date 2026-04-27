# v9.1 Technical Note

## Summary

`cap-flow` is a small experimental learning system built around local signed state-transition events.

This note documents the current `v9 -> v9.1` finding on `sklearn.datasets.load_digits`:

```text
Learning from errors is not enough;
fragile correct decisions near the border also need local reinforcement.
```

`v9.1 correct_border prob=0.80` improves over the clean `v9 hard_scale` baseline on a `10` seed confirmation with `64` cells and `12` epochs.

This is not a claim that the system beats neural networks or that the mechanism scales broadly. The result is a reproducible local-learning finding with a 10-seed diagnostic decomposition.

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

10-seed aggregate decomposition:

| mode | border_repair | border_retention | border_loss |
|---|---:|---:|---:|
| `v9 full` | `0.8762` | `0.9522` | `0.0478` |
| `v9.1 correct_border 0.80` | `0.9201` | `0.9714` | `0.0286` |

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
- `all_pixels_init`;
- `random_sparse` as a backwards-compatible alias for `all_pixels_init`;
- `correct_border`;
- `correct_border_weights_only`;
- `correct_border_event_only`.

Current read from prior diagnostics:

- plastic weights carry most of the short-horizon accuracy lift;
- event scores are live but weaker than direct weight learning in the current setup;
- aggressive event scaling can degrade the regime;
- `correct_border` is the strongest diagnostic control so far;
- persistent class-level attractor/confuser dynamics are not yet strong enough to justify opening `v10` as an accuracy patch.

## Open Questions And Next Validation

The current result is intentionally narrow. The next diagnostics should answer three questions before making a broader architecture claim:

```text
1. What does the event-score path contribute beyond plastic weight updates?
2. Does correct_border transfer to a second small dataset, or is it digits-specific?
3. If a second-dataset gain appears, does it come from near-border samples or from wider non-border decision-field changes?
```

The first point matters because, on digits, the short-horizon accuracy gain appears to be carried mostly by the plastic weight path. Event scores are still part of the signed state-transition dynamics, but their direct contribution remains under investigation.

The second and third points are meant to prevent overfitting the interpretation to `sklearn digits`. A useful next result would show whether `correct_border` is a transferable local learning gate, and whether event-score contribution is dataset-dependent.

## Exploratory Breast Cancer Follow-Up

This section records a small follow-up run after external feedback. It is public-reproducible, but it remains exploratory. The canonical reproduced result in this repository remains the `digits` 10-seed result above.

The follow-up tested a minimal tabular adaptation on `sklearn breast_cancer` and added a non-border decomposition to separate the fixed pre-training near-border subset from the rest of the test set.

Configuration:

```text
dataset: sklearn breast_cancer
cells: 32
epochs: 6
seeds: 10
correct_border_prob: 0.80
```

Key summary:

```text
mode                         final_acc  border_after  non_border_after  non_border_repair
full                         0.7807     0.9400        0.7289            0.1282
correct_border               0.8342     0.9400        0.7999            0.3875
correct_border_weights_only  0.7553     0.9224        0.7011            0.0500
correct_border_event_only    0.7772     0.9153        0.7326            0.1295
```

Current read:

```text
correct_border transfers to breast_cancer in this small diagnostic run.
The gain over full does not come from the fixed near-border subset.
It comes from wider non-border repair.
```

This also sharpens the event-path question. On `digits`, the short-horizon `correct_border` gain is mostly carried by plastic weights. On this `breast_cancer` follow-up, `correct_border_event_only` is stronger than `correct_border_weights_only`, while full `correct_border` remains strongest.

Careful interpretation:

```text
The event path is not uniformly decorative, but its contribution appears dataset-dependent.
This remains a small-scale diagnostic result, not a benchmark claim.
```

Reproduction command:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py \
  --dataset breast_cancer \
  --cells 32 \
  --epochs 6 \
  --seeds 10 \
  --seed-start 0 \
  --modes full,correct_border,correct_border_weights_only,correct_border_event_only \
  --correct-border-prob 0.80
```

Expected output is captured in [`RESULTS_breast_cancer_10seeds.txt`](RESULTS_breast_cancer_10seeds.txt).

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

Canonical 10-seed reproduction:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 10 --seed-start 0 --modes full,correct_border --correct-border-prob 0.80
```

This run can take a while on a laptop or small VM. On the capture machine it took roughly 13 minutes for both modes.

Expected output is captured in [`RESULTS_v9_1_digits_10seeds.txt`](RESULTS_v9_1_digits_10seeds.txt).

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
- On digits, event scores currently appear weaker than the plastic weight path.
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
