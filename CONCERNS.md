# Audit: risks, fixes, and what's still open

First written 2026-07-05 after a full repo audit; **updated the same day after the big
future-proofing overhaul.** Ground rule driving everything: **in production there is no
FiveThirtyEight.** Future inputs are (a) raw polls — candidate, party, state/district,
pollster, dates, pct, sample size — and (b) economic data. Anything that only exists inside
a 538 file is either leakage or a train/serve mismatch.

## Fixed (2026-07-05, second pass — the future-proofing overhaul)

### ✅ #1 Poll weighting removed — raw polls only
The old poll weight (recency × √sample × **538 pollster grade**) fed `poll_wavg` and through
it every top feature; grades don't exist for future polls → train/serve skew. Decision (user):
**no weighting at all** — every aggregate is a plain average. `avg_grade`/`avg_pollscore`
dropped from FEATURES. Recency still reaches the model via explicit features
(`poll_last`, `poll_last30`, `min_days`, `poll_momentum`, `gap_x_recency`). The 538
partisan-lean file is gone from the pipeline entirely.
Implemented in **`features.py`** — the ONE shared feature builder used by both `model.ipynb`
and `predict.py`, so train and predict features can never drift apart.

### ✅ #2 Prediction path exists: `predict.py`
Reads the **polling-agg** repo's raw polls (`data/raw/nyt_polls.csv` + `wikipedia_polls.csv`,
schema: pollster/candidate/party/stage/sample_size/end_date/implied_prob), dedups them,
builds features via `features.py`, loads the artifact `model.ipynb` saves
(`data/model_xgb.json` + `data/model_features.json`), and writes `predictions_<cycle>.csv`
with per-candidate and within-race-normalized win probabilities.
`--natl-env` (generic-ballot D−R average) must be supplied manually for now.

### ✅ Pre-2018 poll data found: training extended 4 → 14 cycles
538's pollster-ratings **`raw_polls.csv`** (frozen at `data/raw_polls_538.csv`) has 8,529
downballot general-election polls for **1998–2016**. Reshaped into the long dataset and merged
with the (already 1976+) results files: training went from 687 races / 4 cycles to
**~1,970 races / 14 cycles**. This also blunts the "cycle-constant features have n=4" worry
(#3 → n=14) — though 14 is still small; the macro features remain on probation.
Odd-year races (VA/NJ etc.) intentionally excluded.

### ✅ #4 Honest testing: nested tuning scheme
Hyperparameters are now selected by leave-one-cycle-out CV **on 1998–2016 only**; the
headline evaluation is leave-one-cycle-out on **2018–2024, which the tuner never saw**.
No more reporting the selection score as the performance estimate.

### ✅ #5–7 New non-538 sources
- **Approval** (`fetch_approval.py`): Gallup via UCSB American Presidency Project, scripted,
  1993 → 2025-01 (replaces the hand-typed table). Gap: UCSB has no Trump-2nd-term page yet;
  the script already tries those slugs — re-run it occasionally.
- **Generic ballot**: 538's daily historical file (frozen, `data/generic_ballot_hist_538.csv`)
  covers 1996–2016; 2018–2024 are frozen constants in `cycles.py`; **2026+ must be passed to
  `predict.py --natl-env`** (e.g. RealClearPolling average) until a scraper is written.
- **Future results** (for labeling 2026 once it happens): MIT Election Data + Science Lab
  (MEDSL) publishes official Senate/House/Governor returns — write a loader to the
  `res_*.csv` schema when 2026 certifications land.

### ✅ #8 Duplicates audited
- This repo: **clean** — NYT `*_current.csv` files turned out to be pure 2026-cycle (they
  contribute nothing to training), no overlap with the 2018–2024 historical files; the
  1998–2016 raw_polls slice is disjoint by construction. A cross-source dedup safety net now
  runs in `build_dataset.ipynb` anyway.
- **polling-agg repo: dirty** — `nyt_polls.csv` has ~1,353 internal duplicate rows and ~2,686
  rows duplicated between NYT and Wikipedia sources (80% with identical pct). `predict.py`
  dedups on (pollster, end_date, race, candidate), NYT preferred. **The polling-agg
  aggregator itself may be double-weighting these — worth fixing there too.**
- Also fixed while auditing: **primary-stage polls** were flowing into the dataset
  (a candidate's primary numbers contaminated their general averages); now filtered.
  Hypothetical-matchup general polls (~1k) are kept deliberately — they polled the real
  eventual matchup, just early.

### ✅ #9 Small fixes
- `poll_last` can no longer be a poll with an unknown date.
- Unknown incumbency is now **NaN** (XGBoost routes missing), not a silent "challenger".
- `undecided` clipped at 0.
- All cycle constants centralized in **`cycles.py`** (CYCLES, PRES_PARTY, eve windows,
  natl_env). Adding a cycle = edit one file.
- (Earlier same day) the House district `'1.0'`-vs-`'1'` key bug — fundamentals were dead for
  ALL House races; fixed with a normalizer (`features.dist_str`) + a hard assert.

### ✅ (2nd pass, same day) Economic data current + generic ballot automated + margin model
- **Econ fixed:** `fetch_macro.py` now overlays the **BLS public API** (no key) on the lagged
  DBnomics mirror — unemployment/CPI/core/U-6 current through **May–June 2026**.
- **Generic ballot automated:** RCP itself is Cloudflare-walled (403 to scripts), but
  `fetch_generic_ballot.py` reads Wikipedia's aggregator table (DDHQ, RCP, Silver Bulletin,
  VoteHub, …) and returns the mean D−R margin (2026 ≈ **D+5.8**). `predict.py` /
  `predict_margin.py` auto-fetch it; `--natl-env` overrides. This is the project's ONE live
  fetch, predict-time only (current-cycle info can't be frozen by definition).
- **Margin model built** — `margin_model.ipynb` + `predict_margin.py` +
  `data/margin_model_*.json`, **fully separate from the win/lose model** (user requirement).
  Target = actual vote margin vs best opponent; same features.py pipeline, same nested
  tune-old/eval-modern scheme; baselines = raw polled margin and a linear calibration of it.

## Still open, ranked

### 1. Approval series ends 2025-01
UCSB has no Trump-2nd-term page yet (`fetch_approval.py` auto-tries the slugs — re-run it
occasionally); Wikipedia's per-poll approval tables omit the year in date cells, so scraping
them is too fragile. Until fixed, 2026-cycle approval features only see 2024-11→2025-01.

### 2. Cycle-constant features still can't be strongly validated
n went 4 → 14, which is real progress, but 112 macro features on 14 national observations
still invites memorization; regularization has zeroed them in the win model so far. The
margin model is where they get their fair test — check `margin_feature_importance.csv` after
each run.

### 3. raw_polls (1998–2016) quirks — known and accepted
- Only the **top two** candidates per poll → third-party candidates invisible pre-2018
  (`n_cands`, `undecided`, `poll_share` behave slightly differently there).
- `polldate` is a single date (treated as end_date).
- Special elections share a race_id with the regular same-state race in rare cases
  (pre-existing limitation for 2018+ too, e.g. dual Senate seats; affects a handful of races).

### 4. polling-agg feed quality
The predict path is only as good as the scraper feed: the duplicate problem above, plus
`implied_prob` rounding (2 decimals = 1pp poll resolution), and primary/general stage tags
must stay accurate. Consider fixing dedup upstream in the polling-agg repo.

### 5. 2026 labels
After the 2026 elections, results must come from MEDSL/official returns (loader not yet
written), then: add 2026 to `cycles.py`, extend macro/approval, re-run the whole pipeline
(grid search included — the workflow rule).

## Improvement roadmap (2026-07-06 full review — ranked by expected value)
✅ Approval solved: VoteHub API continuation (all-pollster avg) after UCSB/Gallup ends 2025-01.
   VoteHub also has 532 generic-ballot polls (poll_type=generic-ballot) — see #3 below.

**Model architecture**
1. **Race-level two-party reframe** (one row per race, target = signed D−R margin, features =
   D-minus-R differences). Kills the "both candidates predicted to win" inconsistency AND the
   win/margin model split ambiguity; win prob becomes P(margin>0).
2. **Distributional margin** (quantile XGBoost 10/50/90 or similar): win prob from the margin
   distribution, and rung-by-rung pricing of Kalshi's margin-of-victory ladders (we already
   parse them) — the direct betting payoff.
3. **Snapshot training (fix the July-vs-eve mismatch)**: build features as-of T days out for
   multiple T per historical race, add days_out as a feature. Makes mid-campaign predictions
   honest — currently the single biggest train/serve gap.
4. **Own pollster reliability scores** from raw_polls 1998+ (per-pollster error/bias vs actual,
   prior cycles only) — future-proof replacement for the dead 538 grades.
5. **Calibration layer** (isotonic per office on out-of-fold preds) + ship the α≈0.5
   model/poll-softmax blend that already wins the sweep.

**Features (cheap → expensive)**
6. Partisan-poll handling: `partisan` col is ingested but UNUSED — frac_partisan, poll_avg
   excluding partisan/internal polls (directly fixes TX-28-style mirages).
7. Lead-significance: lead_se ≈ poll_std/√n_polls, lead_z = poll_lead/lead_se.
8. is_midterm + is_midterm×is_president_party (explicit midterm penalty).
9. RCV/runoff state flags (ME/AK; GA/LA) — tells the model where first-round margin ≠ winner.
10. Generic-ballot monthly recency cuts from VoteHub + House-G-US polls (1998-2022 committed,
    2024+2026 via VoteHub) — also automates natl_env, replacing the Wikipedia scrape.
11. State partisan lean from prior PRESIDENTIAL results (time-varying, self-computed,
    replaces the dropped 538 lean); district-level via MEDSL if wanted.
12. FEC fundraising ratio per race (free API, quarterly) — classic House fundamental.

**Ops**
13. Schema guards in predict loaders (pct in [0,100], races count sanity, required cols).
14. MEDSL results loader after Nov 2026; add 2026 to cycles.py; full retrain.
15. Market-snapshot history already accrues in polling-agg git history — build an extractor
    later to backtest model-vs-market edges against realized results.
