# Methodology — exact time windows & how every feature is built

This is the precise reference for **what time period each feature draws from**. The guiding
rule: *every feature must use only information available before that race's election.* This
doc spells out the exact cutoffs so there's no ambiguity.

## Cycles modeled
`2018, 2020, 2022, 2024` (even-year general elections). Earlier cycles can't be modeled —
machine-readable polls only start in 2018.

## Election-eve cutoff
We treat **Nov 1 of the election year** as the cutoff ("election eve") for window math.
Actual election day is the first Tuesday after the first Monday in November; Nov 1 is a
clean, conservative stand-in that never includes post-election information.

---

## A. Poll-based features — per candidate, within the cycle
Polls carry an `end_date`. Every poll used has `end_date` on/before its race's election.
Features are built per candidate from **all of that candidate's polls in the cycle**:

| feature | window |
|---|---|
| `poll_avg`, `poll_wavg`, `poll_std`, `n_polls`, `poll_share`, `poll_lead`, `n_polls_over50`, `frac_polls_over50`, `race_total_polls`, `avg_grade`, `avg_pollscore`, `avg_sample` | all polls in the cycle for that candidate/race |
| `poll_last` | the single most recent poll |
| `poll_last30` | polls within **30 days** of the candidate's last poll (`days_to_elec ≤ 30`) |
| `poll_momentum` | linear slope of `pct` over polls within **60 days** (`days_to_elec ≤ 60`, needs ≥3 polls) |
| `min_days` | days-to-election of the candidate's latest poll |
| `gap_x_recency` | `poll_lead` × recency weight (closer to election = higher) |

`days_to_elec = election_date − poll end_date`. Weighting in `poll_wavg`: recency × √sample × pollster grade.

## B. Lead-dynamics features — the polling trajectory within the cycle
Computed by walking the race's polls in date order (running averages):

| feature | window |
|---|---|
| `avg_margin_over_time`, `min_margin`, `margin_trend`, `margin_volatility` | the candidate's lead-vs-best-opponent across **all poll dates in the cycle** |
| `n_lead_changes`, `lead_changed` | how often the front-runner flipped across the cycle's polls |

## C. National environment
| feature | window |
|---|---|
| `natl_env_cand` | generic-ballot DEM−REP averaged over the **30 days before election** (`0 ≤ days_to_elec ≤ 30`), per cycle, signed to the candidate's party |

## D. Fundamentals
| feature | window |
|---|---|
| `prior_margin_cand` | the **most recent *prior* election** for that exact seat (looks back 2/4/6/8 years; strictly before the current cycle) |
| `is_incumbent`, `is_inc_party_race` | from `races.csv` incumbent_party (known before the election) |
| `is_president_party` | candidate's party == sitting president's party (known before the election) |
| `lean_cand` | 538 partisan-lean file. **⚠️ KNOWN CAVEAT:** this file is a single **2022 vintage**, so for 2018/2020 races it carries information from *after* those elections (mild look-ahead leakage). It is barely used by the tuned model (heavy regularization), and `prior_margin_cand` covers the same "structural lean" signal cleanly. Kept for now; flagged. |

## E. Macro / economic features — **per-cycle windows (this is the key part)**
Each economic series is condensed over **that cycle's own window = the day after the prior
election eve, through this election eve**. So `max`/`min`/`std`/`trend` reflect *only that
cycle's* conditions, never the all-time history.

**Exact windows:**
| cycle | window (exclusive start → inclusive end) |
|---|---|
| **2018** | 2016-11-01 → 2018-11-01 |
| **2020** | 2018-11-01 → 2020-11-01 |
| **2022** | 2020-11-01 → 2022-11-01 |
| **2024** | 2022-11-01 → 2024-11-01 |

Per metric, **7 stats** over that window: `_eve` (last reading), `_mean`, `_max`, `_min`,
`_std`, `_trend` (slope), `_last12_delta` (value minus 12 months prior).

**Worked example — why per-cycle matters:** inflation peaked at ~9% in mid-2022.
- `inflation_max` for **2022** = **9.0** (the peak is inside 2020-11→2022-11)
- `inflation_max` for **2024** = **6.4** (2022-11→2024-11 only; the 9% peak is *before* this window, so excluded)

If we used an expanding-from-2016 window instead, 2024 would wrongly show 9.0. We don't.

**Note on YoY inflation:** computed as CPI / CPI-12-months-prior on the *full* series first,
then sliced to the window (so the first month of the window still has a valid YoY value).

**Metrics** (from `data/macro_monthly.csv`): `unemployment`, `inflation` (from CPI),
`cpi_core`, `gas`, `fed_funds`, `unemp_u6`, `approval` → 7 × 7 stats = 49 macro features/cycle.

---

## Validation windows
- **Leave-one-cycle-out CV:** train on 3 cycles, test the held-out 4th, rotate through all four.
  The pollster house-effect adjustment (`poll_wavg_adj`) is recomputed from **training cycles
  only** in each fold (no leakage).
- **Single split (for the walkthrough):** train ≤ 2022, test 2024.
- Never random splits — that would mix cycles and leak the future.

## Static-data principle
Past months/results never change, so the macro CSV (`data/macro_monthly.csv`, full history
back to 1947) and the result files are pulled **once and committed**. Changing a model
*feature* does not require re-downloading data — only re-running `model.ipynb` (with its
live grid search; see the workflow rule in README/AGENTS).
