# Handoff: in-flight state, breakdown risks, next steps (2026-07-06)

For the next agent. Read AGENTS.md first (architecture + rules), CONCERNS.md second
(risk register + roadmap). This file: what's mid-flight RIGHT NOW, what's most likely to
break, and what to do next, in order.

## ⚠ IN-FLIGHT STATE (check this before touching anything)

1. **Retrain "run 2" (144 features: +sentiment, +generic_ballot) was executing in the
   background**: model.ipynb finished, margin_model.ipynb possibly still running.
   **The artifacts in data/ may be MIXED GENERATIONS** (win model from run 2, margin model
   from run 1) until the chain completes. Do NOT run refresh_dashboard.py or trust
   predictions until both notebooks show fresh outputs from the same feature set.
2. **FEC features are built but NOT wired in** (commit df15f97): data/fec_summary.csv +
   features.load_fec() + feature_list(fund=True) exist; the notebooks and predict scripts
   still run WITHOUT them. Wiring steps (= "run 3", the final planned retrain):
   - model.ipynb + margin_model.ipynb: `FEC = F.load_fec()` where FUNDS is loaded; pass
     `fec=FEC` to every `build_candidate_table` call (BASE build); change
     `FEATURES = F.feature_list(MACRO_FEATS)` → `F.feature_list(MACRO_FEATS, fund=True)`.
   - predict.py + predict_margin.py: same two changes where cand is built (predict_margin
     imports predict's loader but builds its own cand table — check both).
   - Run model.ipynb THEN margin_model.ipynb (never parallel), verify, then
     `py refresh_dashboard.py --no-feeds`, commit this repo, commit+push polling-agg.
3. Baselines to compare run 2/3 against — run 1 (vintage fixes, 112 features):
   WIN AUC .969 / Brier .069 / race-acc .859 (poll baseline .868);
   MARGIN MAE 6.47 vs calibrated-poll 7.52 / raw 7.90.

## Breakdowns I can see happening (ranked by likelihood × damage)

1. **Silent feature mismatch between artifact and predict path.** predict.py does
   `X = cand.reindex(columns=meta['features'])` — if the saved model expects fund_* but
   predict didn't pass `fec=`, reindex fills them ALL NaN and predicts garbage *without
   erroring*. After any wiring change, assert: every feature in model_features.json exists
   in the built cand table (add this assert — it doesn't exist yet).
2. **Editing a notebook while nbconvert is executing it** — nbconvert rewrites the file at
   completion and your edits vanish. Also: on a cell error, nbconvert can leave the
   PREVIOUS run's outputs in place (looks like "no change"). Always check the printed
   feature counts/dates in outputs match expectations.
3. **polling-agg feed schema drift.** predict assumes: implied_prob = pct/100, race_id like
   `2026-SEN-ME` / `2026-H-AL-01`, stage contains 'general', end_date parseable. A scraper
   change upstream silently shifts all of these. Schema guards are STILL NOT IMPLEMENTED
   (roadmap) — predictions would be confidently wrong.
4. **VoteHub is now a triple single-point-of-failure** (approval 2025+, generic-ballot
   monthly 2024-12+, candidate for natl_env). If the API dies or renames poll_type values,
   fetch_approval soft-skips (approval goes stale) and fetch_generic_ballot loses its
   current segment. No alerting exists — a stale feed looks like a working one.
5. **races.csv incumbency vintage vs redistricting.** The frozen races.csv (July 2025
   vintage from 538's unmaintained repo) assigns 2026 incumbents to districts that ELEVEN
   states then redrew (data/redistricted_2026.csv). In redrawn states, is_incumbent /
   prior_margin_cand can be wrong-district. The dashboard flags these rows (REDRAWN badge)
   but THE MODEL DOES NOT KNOW. 41% of predicted House races are affected.
6. **CI failures are swallowed.** Both polling-agg workflows run model_compare.py with
   `|| echo skipped` — if it starts crashing (e.g. schema drift), the dashboard quietly
   serves stale model_data.js forever. Consider making it fail loudly or adding a
   freshness check to the page (predictions_as_of is displayed — watch it).
7. **GitHub Pages transient deploy failures** — site serves a stale build while the repo
   looks fine. `gh run list --repo pjmerica/polling-agg-2026` → `gh run rerun <id>`.
8. **The 2024 generic-ballot hole.** generic_ballot_* features are NaN for the 2024 cycle
   (no surviving source). Do NOT "fix" this with hand-typed numbers — that provenance
   pattern is what we spent a day eliminating. Needs a real archived source or stays NaN.
9. **Windows papercuts**: use `py -X utf8` for scripts that print (cp1252 chokes on unicode);
   `python` alias doesn't exist (use `py`); rm -rf __pycache__ after editing helper modules;
   CRLF warnings are noise.
10. **polls_long_with_results.csv is gitignored** (12MB). Fresh clone → run
    build_dataset.ipynb FIRST (fully offline) or everything downstream fails.

## Next steps, in order

1. Finish the in-flight sequence above (verify run 2 → wire FEC → run 3 → publish).
2. **Add the missing asserts**: artifact-features ⊆ cand columns (predict), schema guards
   on the polling-agg loaders (pct∈[0,100], expected race count, required columns).
3. **Ask the user for two free API keys** (they were told; follow up):
   - api.data.gov key → FEC per-report as-of-Sep-30 totals (kills the Dec-31 caveat) +
     small-dollar share + true average-donation metric.
   - FollowTheMoney key → governor campaign finance (state-level; FEC is federal-only).
4. Redistricting-aware fundamentals: add `is_redistricted` feature and/or NaN the
   prior_margin/incumbency for redrawn districts (feature change ⇒ full retrain; consider
   bundling with the next feature batch instead of a standalone run).
5. After Nov 2026: MEDSL results loader → label 2026 → add cycle to cycles.py → retrain →
   **backtest the logged market edges** (polling-agg git history holds hourly market
   snapshots) before trusting edges for 2028 sizing.

## Model improvement possibilities (beyond CONCERNS.md's roadmap)

- **Race-level two-party reframe** and **distributional margin (quantile) model** — still
  the top two; they merge the split-model problem away and price Kalshi MOV ladders.
- **Snapshot training** (features as-of T days out) — biggest honesty gain for mid-campaign
  prediction, and it makes the FEC quarterly filings line up naturally.
- **Monotonic constraints** in XGBoost (win prob non-decreasing in poll_lead / fund_share)
  — cheap guard against weird extrapolation in thin-poll races.
- **Overperformance target** (actual margin MINUS polled margin) — removes the dominant
  poll signal from the target so fundamentals/money get full credit; often better-behaved
  than raw margin.
- **Isotonic calibration per office** on out-of-fold predictions; ship the α≈0.5
  model/poll-softmax blend that already wins the blend sweep.
- **Uncertainty-aware market comparison**: scale edge by prediction uncertainty (poll count,
  model disagreement) → a proper Kelly-ish sizing signal instead of raw edge sorting.

## Found in the 2026-07-06 late audit (numbers verified)
- **Dual-seat race collisions.** Training: 9 races (43 candidate rows) merge two same-state
  contests into one race_id (dual Senate seats: MN-2018, OK-2014/2022, NY-2010, MS/WY-2008,
  SC-2014, NE-2024; House special NY-19-2022) — two `won=1` rows per "race", race-relative
  features computed across candidates who never faced each other. Fix: add seat/special
  disambiguation to the race key in build_dataset.ipynb (poll files carry seat_name; results
  carry `special`), then retrain. Predict-time version FIXED same day (FL/OH 2026 specials
  now keyed `..._Senate-S`).
- **Cycle-correlated poll bias is large and unhedged.** Mean signed poll-margin error (D−R)
  by cycle swings from −3.9 (1998, 2012) to +6.7 (2020): within a cycle, errors share a
  common component of ±4-7 points. Per-race MAE (~6.5) understates PORTFOLIO risk for
  betting: if 2026 polls share a bias, every model edge moves together. Mitigations to
  build: cycle-bias prior feature (prior cycles' signed error by state/party, shrunken),
  and report edge-portfolio exposure by party on the dashboard.

## Retrain batch 4 (staged 2026-07-06, run AFTER run 3 publishes)
Feature-changing items implemented-or-designed but NOT yet in the training path:
1. **Dual-seat fix** — build_dataset.ipynb already patched (source-only): specials get
   district 'S' from results `special` flag + seat class (polls) + '-GS_' race marker
   (raw_polls); House special NY-19-2022 dropped. RE-RUN build_dataset first (regenerates
   polls_long_with_results.csv), THEN both model notebooks.
2. **Cycle-bias prior feature** (to build in features.py): mean signed poll-margin error of
   the state's (or region's) races in PRIOR cycles, shrunken toward the national prior-cycle
   mean. Strictly prior-cycle info = leak-free.
3. **Shrunken house effects**: multiply each pollster's dev by n/(n+K), K≈5 polls.
4. **is_redistricted feature** from data/redistricted_2026.csv (0 for all training cycles,
   1 for 2026 redrawn-state House rows) + consider NaNing prior_margin_cand there.
5. **Tune the poll-softmax baseline temperature** on TUNE_CYCLES before comparing (currently
   hardcoded 3.0 — potentially understates the baseline).
Then: run build_dataset → model.ipynb → margin_model.ipynb → refresh_dashboard → publish.

## Bias-robustness tooling (shipped 2026-07-06, active at predict time — no retrain needed)
- predict.py / predict_margin.py now emit win_prob_R3 / win_prob_D3 (pred_margin_R3/D3):
  re-predictions under a uniform ±3-point national poll shift; races whose PICK flips are
  marked `bias_fragile` and badged ≈ FRAGILE on the dashboard (treat as no-edge).
- Model tab meta line shows EDGE EXPOSURE (n D-lean vs R-lean edges): if your open edges
  are lopsided toward one party, you're making ONE correlated bet on 2026 poll bias.
- Tab shows a DATA STALE banner when model_data.js is >30h old (CI swallows compare
  failures; this makes them visible).
- refresh_dashboard.py runs a feed-freshness watchdog (approval/GB/unemployment ≤2mo lag,
  sentiment ≤13mo) — soft upstream deaths now print loud FEED STALE warnings.
- predict loader schema guards: required columns, pct∈[0,100], date parse rate, min races.

## Incident 2026-07-10: dashboard race collapse (41 -> 10) — resolved
Breakdown risk #6 ("CI failures are swallowed") materialized, chained with a new one:
data/raw/primaries.json is GITIGNORED, so CI market-refresh runs crashed the compare step
(FileNotFoundError, silenced by `|| echo`); Daily-refresh runs scraped a fresh Ballotpedia
calendar that only lists UPCOMING elections, so past primaries vanished and the
primaries-decided filter collapsed (23 states -> 8 -> 10 races, no governors). Local runs
looked fine (older, fuller primaries.json on disk) — classic works-on-my-machine.
FIX (polling-agg commit 47431f1): primary dates accumulate in committed
data/processed/primary_calendar_2026.json (max-date-per-state, merge-only, seeded with the
June calendar); compare tolerates a missing raw file. Plus model_predictions_as_of.txt
sidecar (CI checkouts reset mtimes — the staleness display was lying).
LESSON: any CI-consumed input must be committed or the consumer must tolerate its absence;
scraped "calendar" pages forget the past — accumulate, never replace.

## 2026-07-12: poll-data audit, population labels, party overrides
Full audit of the poll feeds + display + party handling. Details in the polling-agg repo's
**POLL_DATA_AUDIT.md**; the model-repo pieces:
- **Population labels (LV/RV/A/V)**: scrapers now capture surveyed population; the historical
  `polls_long_with_results.csv` already has a `population` column, so making it a MODEL
  feature later is a features.py change + retrain (not done yet — display-only for now).
- **`data/candidate_party_overrides.csv`** (NEW, committed): `model_party` + `display_party`
  columns. Osborn (NE-Sen) = model DEM / display IND — the "effective-party slot" pattern
  (see AGENTS.md rule 9). Loeffler = REP/REP (plain correction). Applied in predict.py; flows
  through features.py (`display_party` column) to predict/margin/explain output + the tab.
- **`data/dropped_out_2026.csv`** (NEW, committed): Duggan (MI-Gov withdrew) + NE-Sen fringe
  Dems (Burbank/Forbes) who diluted the Osborn two-way. predict.py drops their poll rows.
- **explain_2026.py** now reports `display_party`; the Model-vs-Markets tab marks the Dem-slot
  candidate's real affiliation, e.g. "Dan Osborn (I)".
MAINTENANCE: both CSVs are hand-maintained 2026 lists. When a candidate drops out or a party
label is wrong/independent-challenger, add a row and re-run refresh_dashboard.py. `cand_key`
= features.norm_name output ("lastname firstinitial", e.g. `osborn d`).
OPEN: population as a model feature (adult-poll downweight); the SHAP "leading Democrat"
pick can still land on a fringe candidate if the real challenger is an unfixed independent.
