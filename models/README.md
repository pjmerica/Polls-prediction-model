# `models/` — the training code

Split by a single question: **does this model see race polling?**

| folder | models | shipped? |
|---|---|---|
| [`poll/`](poll/) | the **four production models** — win, margin, primary nominee, primary margin | yes, all four |
| [`fundamentals/`](fundamentals/) | no-polling variants of the general + primary models | **no** — reference only |

Each model writes its artifact into `../data/` (`*_xgb.json` + `*_features.json`) and its gain
table beside itself as `*_feature_importance.csv`.

## The workflow rule that governs every file here

> **Whenever the feature set changes, re-run the WHOLE thing end-to-end including the grid
> search.** Never reuse old hyperparameters.

Params tuned for one feature set can make a new set look worse than it is, which reads as "the
feature didn't help" and gets a good feature thrown away. Let regularization drop
non-predictive features rather than hand-curating them. This applies to all four models.

Two corollaries worth stating outright:

- **An unchanged result after a feature edit is a bug signal**, not a null result — it usually
  means the edit never reached the model (the classic cause is editing `features.py` but not
  `features_primary.py`, or vice versa).
- Notebooks must be run **one at a time**. `nbconvert` races and overwrites outputs on
  parallel runs.

## Validation is nested and never random

Hyperparameters are tuned by leave-one-cycle-out CV on **older cycles only** (1998–2016 for
the general models, ≤2020 for the primary ones). The headline numbers come from
**expanding-window evaluation** on modern cycles the tuner never saw — train strictly on
cycles *before* the test cycle.

Never report a tuning-fold number as a result, and never validate by random split: rows within
a race are not independent, so a random split leaks the answer across the fold boundary.
Exact windows are in [../docs/METHODOLOGY.md](../docs/METHODOLOGY.md).

## Never use these as features

`vote_pct`, `race_winning_pct`, `margin_actual` — they are the outcome. Also nothing that only
exists inside a defunct 538 file: production inputs are **raw polls + economic data only**,
because 2026+ cannot obtain anything else (see [../docs/AGENTS.md](../docs/AGENTS.md) rule 6).
