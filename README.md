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
```

## Limitations

- This is a small `sklearn digits` proof-of-concept.
- It does not establish broad scaling behavior.
- It is not intended as a general ML replacement.
- The event-score path is currently weaker than the plastic weight path.
- `v10` is intentionally postponed until diagnostics justify a class-level tissue.

## Status

```text
v9   = clean hard_scale baseline
v9.1 = confirmed frontier candidate
v10  = postponed
```
