# Data dictionary & missingness report

Two datasets are documented here:

1. **`polls_long_with_results.csv`** — the master long file from `build_dataset.ipynb`
   (one row per poll-candidate, with the race result joined on). 22,546 rows, 2018–2026.
2. **The model feature table** — the collapsed candidate-level table built inside
   `model.ipynb` (one row per candidate per race, with engineered + macro features).
   1,859 candidate-races, 2018–2024.

Missingness percentages were computed from the committed run. **Missingness here is almost
always structural and meaningful** (e.g. a poll with one entry has no std; a future race has
no result yet), not data corruption — and XGBoost handles NaN natively, so columns are not imputed.

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
| `cand_key` | str | 0.0% | Normalized join key: `lastname firstinitial` (accent-stripped). |
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

One row per candidate per race. Built from the long file (filtered to `has_result==1`,
general, 2018–2024). All features below are **leak-free** (no use of `vote_pct`/result).

### 2a. Poll-derived features
| feature | missing | meaning |
|---|---|---|
| `poll_avg` | 0% | Simple mean of the candidate's polls. |
| `poll_wavg` | 0% | **Weighted poll average** (recency × √sample × pollster grade). The single strongest feature. |
| `poll_last` | 0% | The candidate's most recent poll value. |
| `poll_last30` | 0% | Mean of polls in the final 30 days (falls back to all if none). |
| `poll_std` | **27.4%** | Std of the candidate's polls. **NaN when only 1 poll exists** (std undefined). |
| `n_polls` | 0% | How many polls this candidate had. |
| `n_polls_over50` | 0% | Count of the candidate's polls above 50%. |
| `frac_polls_over50` | 0% | That as a fraction. |
| `race_total_polls` | 0% | Total polls in the race (all candidates). |
| `avg_grade` | 5.3% | Mean `numeric_grade` of the candidate's polls. NaN if all unrated. |
| `avg_pollscore` | 4.6% | Mean pollscore. NaN if all unrated. |
| `avg_sample` | 0.2% | Mean sample size. |
| `min_days` | 0% | Days-to-election of the candidate's latest poll. |
| `poll_wavg_adj` | 0% | Weighted average **after pollster house-effect adjustment** (train-cycles-only). |

### 2b. Race-relative / gap features
| feature | missing | meaning |
|---|---|---|
| `poll_lead` | 0% | `poll_wavg` minus the best opponent's. **Top feature.** |
| `poll_share` | 0% | Candidate's share of the summed polled support in the race. |
| `n_cands` | 0% | Number of candidates in the race. |
| `twoparty_margin_cand` | 0% | DEM−REP race margin, signed toward this candidate's party. |
| `abs_gap` | 0% | Absolute two-party gap (race closeness). |
| `tossup` | 0% | 1 if `abs_gap < 3`. |
| `undecided` | 0% | 100 − sum of polled support (undecided share). |
| `gap_x_recency` | 0% | `poll_lead` × closeness-to-election. **Top-2 feature.** |

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
| `lean_cand` | ~59% | 538 partisan lean (state for Sen/Gov, district for House), signed to party. **NaN where the 2022-vintage lean file lacks the district** (post-redistricting). |
| `prior_margin_cand` | ~59% | Prior same-office election margin for the seat, signed to party. NaN when no prior contest. |
| `is_incumbent` | 0% | 1 if this candidate's party holds the seat and they're running. |
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
**that cycle's own window = prior election eve → this election eve** (2018 ← 2016-11→2018-11,
2020 ← 2018-11→2020-11, 2022 ← 2020-11→2022-11, 2024 ← 2022-11→2024-11). So `max` is *that
cycle's* peak, not the all-time max since 2016. Per metric there are **7 features**:

| naming pattern | meaning (layman) |
|---|---|
| `<metric>_eve` | the value right before the election (the latest reading) |
| `<metric>_mean` | average level over the whole 2016→eve window |
| `<metric>_max` | the highest it ever got (e.g. the inflation/gas peak) |
| `<metric>_min` | the lowest it got |
| `<metric>_std` | how much it bounced around (variance/spread) |
| `<metric>_trend` | slope — rising or falling into the election |
| `<metric>_last12_delta` | change vs 12 months earlier (the *change* voters feel) |

**Metrics** (`<metric>` ∈, **7 total**, as actually pulled): `unemployment`, `inflation`
(from CPI YoY), `cpi_core`, `gas`, `fed_funds`, `unemp_u6`, `approval` → **49 macro
features/cycle** (7 metrics × 7 stats). Plus `natl_env_cand` (538 generic-ballot DEM−REP,
signed to the candidate's party; a single value).

Exact per-cycle windows: 2018 ← 2016-11→2018-11 · 2020 ← 2018-11→2020-11 ·
2022 ← 2020-11→2022-11 · 2024 ← 2022-11→2024-11. (Full detail in [METHODOLOGY.md](METHODOLOGY.md).)

Sources: economic metrics from **DBnomics** (BLS/EIA/Federal Reserve — see DATA_SOURCES.md §5
and `fetch_macro.py`); `approval` from a documented monthly table; `natl_env_cand` from the
538 generic ballot. (`fetch_macro.py` lists more candidate series, e.g. GDP/sentiment/S&P, but
only the 7 above resolved cleanly via DBnomics; others skip gracefully.)

> **Macro caveat:** these are national values *constant within a cycle*, so with only 4 cycles
> (2018/2020/2022/2024) they carry little independent signal for **win/lose** and are heavily
> regularized away by the grid search (low `colsample_bytree` + high `reg_lambda`). They are kept
> for calibration and as the foundation for a future **margin** model. Data is static (pulled
> once into `data/macro_monthly.csv`); it is **not** re-downloaded on every model run.

### 2g. Outcome (label / excluded-from-features)
| column | meaning |
|---|---|
| `won` | **The classification label (1/0).** |
| `vote_pct`, `race_winning_pct` | Actual results — **never used as features** (they're the answer). |
