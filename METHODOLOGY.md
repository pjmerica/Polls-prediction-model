# Methodology — exact time windows & how every feature is built

The precise reference for **what time period each feature draws from**. Guiding rule: *every
feature must use only information available before that race's election.* Updated 2026-07-05
(raw-poll averages, 14 cycles, nested tuning).

## Cycles modeled
Even years **1998–2024** (14 cycles). 1998–2016 polls come from the frozen 538
pollster-ratings `raw_polls.csv` (top-two candidates per poll); 2018–2024 from the frozen
538 poll files. Odd-year races are excluded. General-election stage only (primaries filtered;
hypothetical general-matchup polls kept).

## Election-eve cutoff
**Nov 1 of the election year** is the cutoff for window math (conservative stand-in for
election day). `predict.py` uses the true election date for `days_to_elec`.

---

## A. Poll-based features — per candidate, within the cycle
**All aggregates are PLAIN averages — no weighting of any kind** (recency/sample/pollster-grade
weights were removed 2026-07-05; grades don't exist for future polls). Recency enters through
explicit features instead.

| feature | window |
|---|---|
| `poll_avg`, `poll_std`, `n_polls`, `poll_share`, `poll_lead`, `n_polls_over50`, `frac_polls_over50`, `race_total_polls`, `avg_sample` | all general-election polls in the cycle for that candidate/race |
| `poll_last` | the single most recent poll **with a known date** |
| `poll_last30` | polls within **30 days** of election (`days_to_elec ≤ 30`) |
| `poll_momentum` | slope of `pct` over polls within **60 days** (needs ≥3 polls) |
| `min_days` | days-to-election of the candidate's latest poll |
| `gap_x_recency` | `poll_lead` × recency factor `1/(1+min_days/30)` |
| `poll_adj` | plain mean of house-effect-adjusted pct (house effect from **train cycles only** per fold; from all history at predict time) |

## B. Lead-dynamics features
Running-mean margins over the race's poll dates (all in-cycle): `avg_margin_over_time`,
`min_margin`, `margin_trend`, `margin_volatility`, `n_lead_changes`, `lead_changed`.

## C. National environment
`natl_env_cand` = generic-ballot DEM−REP margin over the **30 days before the election**,
signed to the candidate's party. Source per cycle (see `cycles.py`): 1998–2016 computed from
the committed daily history file; 2018–2024 frozen constants; 2026+ passed to
`predict.py --natl-env` manually.

## D. Fundamentals
| feature | window |
|---|---|
| `prior_margin_cand` | most recent **prior** same-office election for that seat (2/4/6/8 yrs back; strictly before the cycle) |
| `is_incumbent`, `is_inc_party_race` | from frozen `races.csv` incumbent_party; **unknown = NaN**, never 0 |
| `is_president_party` | candidate's party == sitting president's party |

(The 538 partisan-lean file was removed entirely — single 2022 vintage = look-ahead leakage.)

## E. Macro features — per-cycle windows
Per metric, stats over **that cycle's own window** = prior even-year Nov 1 → this Nov 1
(e.g. 2024: 2022-11-01 → 2024-11-01). 7 full-window stats (`_eve/_mean/_max/_min/_std/_trend/
_last12_delta`) + 3/6/12-month recency cuts (`_avg/_max/_trend`). Metrics: unemployment,
inflation (CPI YoY computed on the full series, then windowed), cpi_core, gas, fed_funds,
unemp_u6, approval. Approval comes from `data/approval_monthly.csv` (Gallup via UCSB,
1993–2025-01). Pre-cycle months missing ⇒ NaN (XGBoost routes missing).

---

## Validation (nested — no selection bias)
- **Hyperparameter tuning:** leave-one-cycle-out CV over **1998–2016 only** (150 sampled
  configs, live grid search every run).
- **Honest evaluation:** leave-one-cycle-out over **2018–2024** — cycles the tuner never saw.
  Each fold trains on the other 13 cycles; the fold's `poll_adj` house effect is recomputed
  from its training cycles only.
- The single-split walkthrough (train = all but 2024, test = 2024) is also honest under this
  scheme.
- Never random splits.

## Production model (predict.py)
Trained on **all 14 cycles** with the tuned params; saved to `data/model_xgb.json` +
`data/model_features.json` by `model.ipynb`. `predict.py` builds identical features (same
`features.py` code) from the polling-agg raw poll feed and outputs per-candidate win
probabilities plus within-race normalized probabilities.

## Static-data principle
Every input is pulled once and **committed**: polls (all vintages), results, races.csv, macro,
approval, generic ballot. No model or predict run touches the network. Re-pull only to extend
to new months/cycles.
