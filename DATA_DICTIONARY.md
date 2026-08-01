# Data dictionary & missingness report

Two datasets are documented here:

1. **`polls_long_with_results.csv`** — the master long file from `build_dataset.ipynb`
   (one row per poll-candidate, with the race result joined on). **35,059 rows, 1998–2026**
   (updated 2026-07-21; was 22,546 rows / 2018–2026 before the 1998-2016 expansion —
   `raw_polls_538.csv` added 8,529 pre-2018 downballot polls, reshaped into this long format).
2. **The model feature table** — the collapsed candidate-level table built inside
   `model.ipynb` (one row per candidate per race, with engineered + macro features).
   **4,426 candidate-rows across 1,979 races, 1998–2024** (14 cycles; was 1,859 rows /
   2018–2024 before the expansion).

Missingness percentages were computed from the committed run. **Missingness here is almost
always structural and meaningful** (e.g. a poll with one entry has no std; a future race has
no result yet), not data corruption — and XGBoost handles NaN natively, so columns are not imputed.

**Pre-2018 rows have a different shape.** The 1998–2016 slice comes from 538's
`raw_polls.csv` (top-two candidates per poll only), which doesn't carry the API-era metadata
columns at all — `stage`, `pollster_rating_id`, `population`, `office_type`, `answer`,
`candidate_id`, etc. read as ~46% missing in the combined file, but that's **100% missing on
the pre-2018 slice and ~0% on 2018+**, not real per-row uncertainty. Pre-2018 rows are always
general-election by construction (the source only ever contained downballot generals), so a
missing `stage` there doesn't mean "unknown."

---

## 1. `polls_long_with_results.csv` — long poll file

### 1a. Identifiers & poll metadata
| column | type | missing | meaning |
|---|---|---|---|
| `poll_id` | str | 0.0% | Unique ID for the poll (a poll can hold several questions/candidates). |
| `pollster_id` | str | 0.0% | ID of the polling organization. |
| `pollster` | str | 0.0% | Pollster short name (e.g. "Marist"). |
| `display_name` | str | 0.0% | Pollster display name. |
| `sponsor_ids` | str | 53.0% | IDs of poll sponsor(s); null when self-sponsored. |
| `sponsors` | str | 53.0% | Sponsor name(s). |
| `question_id` | str | 0.0% | ID of the specific question (a poll may ask several matchups). |
| `created_at` | str | 0.0% | When the poll was added to the source DB. |
| `source` | str | 52.2% | Provenance tag (e.g. "538"); null for NYT-only rows. |
| `url`, `url_article`, `url_topline`, `url_crosstab` | str | 0.2–82% | Links to the poll / writeup / toplines / crosstabs. |
| `notes` | str | 95.0% | Free-text caveats (split sample, etc.). |
| `internal` | bool | 76.2% | True if released by a campaign/internal source. |
| `partisan` | str | 74.4% | Partisan lean of the sponsor (DEM/REP); null = nonpartisan. |
| `tracking` | bool | 98.6% | True if a tracking poll. |

### 1b. Pollster quality
| column | type | missing | meaning |
|---|---|---|---|
| `pollster_rating_id` | float | 2.0% | ID of the pollster's 538 rating. |
| `pollster_rating_name` | str | 2.2% | Name on the rating. |
| `numeric_grade` | float | 21.7% | 538 pollster quality grade (~0–3, higher = better). **Used in poll weighting.** Null = unrated. |
| `pollscore` | float | 18.9% | 538 pollscore (lower = better; newer metric). |
| `transparency_score` | float | 43.8% | 538 transparency score (0–10). |
| `methodology` | str | 14.1% | Mode (Live Phone, Online Panel, etc.). |

### 1c. Sample / population
| column | type | missing | meaning |
|---|---|---|---|
| `sample_size` | float | 0.9% | Number of respondents. **Used in poll weighting (√n).** |
| `population` | str | 0.1% | Sampled population: `lv` likely voters, `rv` registered, `v` voters, `a` adults. |
| `population_full` | str | 0.1% | Same, expanded. |
| `subpopulation` | str | 93.8% | Sub-group if the row is a crosstab cut. |

### 1d. Race identity
| column | type | missing | meaning |
|---|---|---|---|
| `state` | str | 0.0% | 2-letter state abbreviation (standardized; full names mapped). |
| `office` | str | 0.0% | `Senate` / `House` / `Governor` (our normalized office). |
| `office_type` | str | 0.0% | Raw office string from source ("U.S. Senate"). |
| `district` | float | 78.6% | House district number; **empty/NaN for Senate & Governor** (statewide) — that's why it reads 79% "missing". |
| `seat_name` | str | 37.4% | Raw seat label ("Class II", "Third Congressional District"). |
| `seat_number` | float | 10.7% | Numeric seat (House); null for many rows (filled via seat_name). |
| `race_id` | str | 0.0% | `year_state_office[-district]` — unique race key. |
| `year` | int | 0.0% | Election cycle. |
| `election_date` | str | 1.9% | Date of the election. |
| `stage` | str | 0.0% | `general` (we model these), also `primary`/`jungle primary`/`runoff`/`recall`. |

### 1e. The poll reading (candidate-level)
| column | type | missing | meaning |
|---|---|---|---|
| `candidate` | str | 0.0% | Candidate name as polled. |
| `candidate_id` | str | 0.0% | Candidate ID. |
| `answer` | str | 0.0% | Short answer label (usually last name). |
| `poll_party` | str | 0.0% | Candidate party as given by the poll. |
| `party_std` | str | 0.0% | **Normalized party: DEM / REP / OTH.** |
| `pct` | float | 0.0% | **The poll number — this candidate's support in this poll (the core predictor).** |
| `cand_key` | str | 0.0% | Normalized join key: `lastname firstinitial` (accent-stripped). **THE shared key** — polls ↔ results ↔ FEC ↔ bios ↔ candidate history, in both pipelines, so any change to `features.norm_name` re-keys every join and is retrain-triggering. **Fixed 2026-08-01**: intra-word punctuation is now deleted rather than turned into a space. Previously the two apostrophe characters took different paths (`"O’Rourke"` → `orourke b` via the NFKD/ASCII drop, `"O'Rourke"` → `rourke b` via `[^a-z\s]`→space), so one person became two rows that split their own polling; hyphens hit the same inconsistency and keyed 357 politicians off only the back half of their surname (`Ocasio-Cortez`→`cortez a`, `Hyde-Smith`→`smith c`). Hyphens additionally swallow surrounding whitespace, because sources type stray spaces (`"Debbie Mucarsel- Powell"` vs `"Mucarsel-Powell"` broke the nominee join and silently dropped `2024_FL_Senate_DEM` from primary training). Periods deliberately do **not** swallow space, or a middle initial glues onto the surname (`"Robert F. Kennedy"`→`fkennedy r`). **14 committed CSVs cache `cand_key` as a column** and therefore go stale whenever `norm_name` changes — re-run `scripts_rekey_cand_key.py` (lossless; the key is a pure function of the name column). |
| `start_date` | str | 0.0% | Poll field start. |
| `end_date` | str | 0.0% | Poll field end. **Used for recency weighting & days-to-election.** |
| `hypothetical` | float | 13.7% | 1 if a hypothetical matchup. |
| `ranked_choice_reallocated` / `ranked_choice_round` | float | 14% / 98% | RCV bookkeeping. |
| `nationwide_match` / `nationwide_batch` | float/bool | 100% / 14% | Internal source flags (nationwide_match is entirely empty). |
| `endorsed_candidate_id/name/party` | — | 100% | Endorsement fields — **entirely empty in this data; ignore.** |
| `sponsor_candidate_id/sponsor_candidate/_party` | — | ~90% | If a candidate sponsored the poll. |

### 1f. The result (joined; the label) — `has_result==1` rows only
| column | type | missing | meaning |
|---|---|---|---|
| `has_result` | int | 0.0% | **1 if this poll-candidate matched an election result, else 0.** Filter to 1 to model. |
| `won` | float | 24.4% | **The label: 1 if the candidate won the (general) race, else 0.** Null when `has_result==0` (e.g. 2026 future races, name-match misses). |
| `vote_pct` | float | 24.4% | Candidate's actual vote share. *Outcome — never a feature.* |
| `res_candidate` | str | 24.4% | Candidate name as it appears in the results file. |
| `res_party` | str | 24.4% | Normalized party from results (from `ballot_party`). |
| `race_winning_pct` | float | 24.4% | Winning candidate's vote share in that race. *Outcome — never a feature.* |

> The 24.4% missingness on the result columns = poll-candidates with no matched result: almost
> entirely **2026 (future, no result yet)** plus a small share of name-match misses. Among
> modeled cycles (2018–2024 general) the real district-poll match rate is 91–99%.

---

## 2. Model feature table (collapsed, in `model.ipynb`)

One row per candidate per race — **4,426 candidate-rows across 1,979 races, 1998–2024**
(updated 2026-07-21; was 1,859 rows / 2018–2024 before the 14-cycle expansion). Built from
the long file (filtered to `has_result==1`, general). All features below are **leak-free**
(no use of `vote_pct`/result). **No poll weighting of any kind** — every poll aggregate is a
plain mean; the old weighted-average features (`poll_wavg`, `avg_grade`, `avg_pollscore`,
`poll_wavg_adj`) were removed in the 2026-07-05 future-proofing overhaul (538 pollster
grades don't exist for future polls) and no longer appear in the model.

### 2a. Poll-derived features
| feature | missing | meaning |
|---|---|---|
| `poll_avg` | 0% | Simple mean of the candidate's polls. **Current #1 feature by gain** (39%). |
| `poll_last` | 0% | The candidate's most recent poll value. |
| `poll_last30` | 0% | Mean of polls in the final 30 days (falls back to all if none). |
| `poll_last7` | ~61% train / ~53% serve (PRIMARY model only) | Mean of polls in the final **7** days. Unlike `poll_last30` it does **NOT** fall back to the all-time mean on an empty window — that fallback is exactly the staleness bug it exists to counter — so it is NaN and XGBoost routes missing. **Built in both pipelines but a model feature only in `feature_list_primary`**: the general election is one fixed date, so on 2026-07-31 (95 days out) the window was populated for 39% of training rows and **0.0%** of live general rows (0 of 3370; min days_to_elec 99) — an always-missing-in-production feature, the same train/serve skew that killed `poll_adj`. Primaries are always imminent when predicted, so there it works (added 2026-07-31). |
| `n_polls_last7` | 0% | Count of polls in the final 7 days (0, never NaN). Primary model only. |
| `poll_std` | ~25% | Std of the candidate's polls. **NaN when only 1 poll exists** (std undefined). |
| `n_polls` | 0% | How many polls this candidate had. |
| `n_polls_over50` | 0% | Count of the candidate's polls above 50%. |
| `frac_polls_over50` | 0% | That as a fraction. |
| `race_total_polls` | 0% | Total polls in the race (all candidates; a per-CANDIDATE row count — sums to more than the distinct-survey count `n_surveys` used at predict time). |
| `avg_sample` | ~0% | Mean sample size. |
| `min_days` | 0% | Days-to-election of the candidate's latest poll. |

### 2b. Race-relative / gap features
| feature | missing | meaning |
|---|---|---|
| `poll_lead` | 0% | `poll_avg` minus the best OTHER candidate's `poll_avg` (per-candidate, not a race-wide constant — **fixed 2026-07-21**: previously used one constant subtracted from every candidate, so 2nd place always read exactly 0.0). Current #3 feature by gain. |
| `poll_lead_last7` | ~61% train (PRIMARY only) | `poll_last7` minus the best OTHER candidate's `poll_last7` — the final-week counterpart to `poll_lead`, which inherits `poll_avg`'s full-campaign staleness. NaN when either side has no final-week polls (never a silent 0, which would read as "tied"). Same primary-only scoping as `poll_last7`. |
| `poll_share` | 0% | Candidate's share of the summed polled support in the race. |
| `n_cands` | 0% | Number of candidates in the race. |
| `twoparty_margin_cand` | 0% | DEM−REP race margin, signed toward this candidate's party. Current #4 feature. |
| `abs_gap` | 0% | Absolute two-party gap (race closeness). |
| `tossup` | 0% | 1 if `abs_gap < 3`. |
| `undecided` | 0% | 100 − sum of polled support (undecided share). |
| `gap_x_recency` | 0% | `poll_lead` × closeness-to-election. Current #2 feature (was #1 pre-fix, at 58% of gain — largely an artifact of `poll_lead`'s old bug; dropped to 13% post-fix as `poll_avg` took over the signal it should have carried). |

### 2c. Lead dynamics over time
| feature | missing | meaning |
|---|---|---|
| `avg_margin_over_time` | 0% | Avg lead/deficit over the campaign (not just final). **Top-4 feature.** |
| `min_margin` | 0% | Worst point the candidate's lead hit. |
| `margin_trend` | 0% | Slope of the lead over time (widening/narrowing). |
| `margin_volatility` | 0% | Std of the lead over time (≈0 importance; kept for interpretability). |
| `n_lead_changes` | 0% | How many times the front-runner flipped in the polls. |
| `lead_changed` | 0% | 1 if the lead ever changed. |

### 2d. Fundamentals
| feature | missing | meaning |
|---|---|---|
| `prior_margin_cand` | **4.4%** (updated 2026-07-21; was ~59% before the 14-cycle expansion — most seats now have a matchable prior contest somewhere in 1998–2024) | Prior same-office election margin for the seat, signed to party. NaN when no prior contest anywhere in the 14-cycle window. |
| `is_incumbent` | 3.3% | 1 if this candidate's party holds the seat and they're running; NaN (not 0) when incumbency is unknown. |
| `is_inc_party_race` | 0% | 1 if the race has a known incumbent party. |

### 2e. Identity flags
| feature | missing | meaning |
|---|---|---|
| `is_dem` / `is_rep` | 0% | Party one-hots. |
| `is_senate` / `is_gov` | 0% | Office one-hots (House = both 0). |

### 2f. National environment & macro/climate (per-cycle; 0% missing — filled for every cycle)
`is_president_party` = 1 if the candidate's party holds the White House — the interaction key
that lets XGBoost learn the *direction* of each macro effect (e.g. high inflation hurts the
in-party candidate).

Each macro metric is condensed from **monthly** data (from `data/macro_monthly.csv`) over
**that cycle's own window = prior election eve → this election eve** (e.g. 2024 ←
2022-11-01→2024-09-30 — the window ends **Sep 30**, not eve, since October economic prints
publish after the election; fixed 2026-07-06). So `max` is *that cycle's* peak, not the
all-time max. Per metric there are **16 features, updated 2026-07-21** (was 7 before macro
recency cuts were added):

| naming pattern | meaning (layman) |
|---|---|
| `<metric>_eve` | the value right before the Sep-30 cutoff (the latest reading) |
| `<metric>_mean` | average level over the whole cycle window |
| `<metric>_max` / `_min` | the highest / lowest it got that cycle |
| `<metric>_std` | how much it bounced around (variance/spread) |
| `<metric>_trend` | slope — rising or falling into the election (NaN, not 0, if <2 observations — fixed 2026-07-14) |
| `<metric>_last12_delta` | change vs 12 months earlier (NaN, not 0, if <13 observations — fixed 2026-07-14) |
| `<metric>_avg_3mo` / `_6mo` / `_12mo` | average over the last 3/6/12 months of the window |
| `<metric>_max_3mo` / `_6mo` / `_12mo` | peak over the last 3/6/12 months |
| `<metric>_trend_3mo` / `_6mo` / `_12mo` | short-window slope over the last 3/6/12 months |

**Metrics** (`<metric>` ∈, **9 total, updated 2026-07-21** — was 7; `generic_ballot` and
`sentiment` added): `unemployment`, `inflation` (from CPI YoY), `cpi_core`, `gas`,
`fed_funds`, `unemp_u6`, `approval`, `generic_ballot`, `sentiment` → **144 macro
features/cycle** (9 metrics × 16 stats). Plus `natl_env_cand` (generic-ballot DEM−REP,
signed to the candidate's party; a single value, computed separately from the
`generic_ballot` macro metric above — see METHODOLOGY.md section C for the distinction).

Sources: economic metrics from **DBnomics** (BLS/EIA/Federal Reserve — see DATA_SOURCES.md §5
and `fetch_macro.py`); `approval` from Gallup/UCSB (1993–2025-01) + VoteHub API continuation
(2025-01+, verified current through 2026-06); `sentiment` from DBnomics UMich series (lags
~11-13 months, watchdog-monitored); `generic_ballot` from 538's historical file (1996-2016) +
VoteHub (2024+); `natl_env_cand` from the same generic-ballot lineage, last-30-days window.

> **Macro caveat:** these are national values *constant within a cycle*. n went from 4 cycles
> (2018-2024) to **14** (1998-2024) in the 2026-07-05 expansion — real progress, but 144 macro
> features on 14 national observations still invites memorization; regularization has zeroed
> most of them in the win model (heavy `colsample_bytree`/`reg_lambda`). They're kept for
> calibration and because the **margin model** is where they get a fairer test — check
> `margin_feature_importance.csv` after each retrain. Data is static (pulled once into
> `data/macro_monthly.csv`); it is **not** re-downloaded on every model run.

### 2g. Outcome (label / excluded-from-features)
| column | meaning |
|---|---|
| `won` | **The classification label (1/0).** |
| `vote_pct`, `race_winning_pct` | Actual results — **never used as features** (they're the answer). |
