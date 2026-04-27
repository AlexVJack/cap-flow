# Signed Event Dynamics

Small experimental local learning system based on signed state-transition events.

This repository contains the public-facing `v9.1` proof-of-concept from `cap-flow`.
It learns on `sklearn` digits without backpropagation, using local credit assignment over sparse signed cells and state-transition events.

This is not a claim that the system replaces neural networks. The goal is narrower: document a reproducible local-learning mechanism and the diagnostic result that improved it.

## Current Finding

```text
Learning from errors is not enough;
fragile correct decisions near the border also need local reinforcement.
```

On `sklearn.datasets.load_digits`, `v9.1 correct_border prob=0.80` improves over the clean `v9 hard_scale` baseline across `10` seeds:

| mode | final_acc | peak_acc | final_gap | border_after_acc | peak_drop |
|---|---:|---:|---:|---:|---:|
| `v9 full` | `0.9145` | `0.9209` | `0.0151` | `0.9164` | `0.0064` |
| `v9.1 correct_border 0.80` | `0.9469` | `0.9497` | `0.0282` | `0.9473` | `0.0028` |

See [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md) for the mechanism, diagnostics, limitations, and reproduction commands.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Cheap smoke check:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 12 --epochs 1 --seeds 1 --modes full,weights_only,event_only,correct_border --correct-border-prob 0.80
```

Focused confirmation chunk:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 2 --seed-start 0 --modes full,correct_border --correct-border-prob 0.80
```

Canonical 10-seed reproduction:

```bash
PYTHONPATH=lab python lab/v9_diagnostics.py --cells 64 --epochs 12 --seeds 10 --seed-start 0 --modes full,correct_border --correct-border-prob 0.80
```

This run can take a while on a laptop or small VM. On the capture machine it took roughly 13 minutes for both modes.

Expected canonical output is captured in [`RESULTS_v9_1_digits_10seeds.txt`](RESULTS_v9_1_digits_10seeds.txt).

Exploratory second-dataset follow-up:

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

This follow-up is public-reproducible but remains exploratory, not a benchmark claim. Expected output is captured in [`RESULTS_breast_cancer_10seeds.txt`](RESULTS_breast_cancer_10seeds.txt).

Mechanistic checks:

```bash
PYTHONPATH=lab python lab/test_v6_core.py
PYTHONPATH=lab python lab/test_v9_core.py
```

## Repository Layout

```text
lab/v4_core.py          shared data/loading helpers from the earlier transition-event line
lab/v6_core.py          signed event cell baseline
lab/v9_core.py          clean v9 hard-scale control
lab/v9_diagnostics.py   diagnostic runner and v9.1 correct_border mode
lab/cap_flow_v9.py      v9 configs and simple probe entry point
TECHNICAL_NOTE.md       public technical note for the v9 -> v9.1 finding
RESULTS_*.txt           captured canonical reproduction output
```

## Limitations

- This is a small `sklearn digits` proof-of-concept.
- It does not establish broad scaling behavior.
- It is not intended as a general ML replacement.
- On digits, the short-horizon accuracy gain appears to be carried mostly by the plastic weight path; the event-score path is live but weaker under the current setup.
- `v10` is intentionally postponed until diagnostics justify a class-level tissue.

## Current Open Questions

- What does the event-score path contribute beyond plastic weight updates?
- Does `correct_border` transfer to a second small dataset, or is it digits-specific?
- If a second-dataset gain appears, does it come from near-border samples or from wider non-border decision-field changes?

A reproducible exploratory `breast_cancer` follow-up is included in [`RESULTS_breast_cancer_10seeds.txt`](RESULTS_breast_cancer_10seeds.txt). It is not a benchmark claim and does not replace the canonical digits reproduction; see [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).

## Status

```text
v9   = clean hard_scale baseline
v9.1 = confirmed frontier candidate
v10  = postponed
```
