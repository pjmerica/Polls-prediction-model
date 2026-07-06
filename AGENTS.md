# Instructions for a new agent (or developer) taking this over

Read this first, then CONCERNS.md (ranked risks + improvement roadmap), then METHODOLOGY.md
(exact windows for every feature). Last full update: **2026-07-06**.

## What this project is (one paragraph)
We predict U.S. downballot elections (Senate / House / Governor) from polls + fundamentals +
economy: a **win-probability model** and a **completely separate margin model** (user
requirement: keep them separate), both XGBoost, trained on 14 cycles (1998–2024, ~1,970
races). Predictions for the current cycle (2026) are published to the user's election
dashboard and compared against Kalshi/Polymarket prices. **Headline findings:** for
win/lose, polls are near the ceiling (model ties the poll baseline); for **margin**, the
model genuinely beats the polls (MAE ~6.5 vs ~7.5 for a calibrated poll baseline) — margin
is where the value is, and what markets price.

## THE TWO REPOS (coupled — know this first)
1. **This repo** (`Documents\Polling prediction model`) — data, models, predictions.
2. **polling-agg** (`Documents\Polling Agg\Polling agg and Prediction markets`, GitHub
   `pjmerica/polling-agg-2026`, GH Pages dashboard at pjmerica.github.io/polling-agg-2026) —
   poll/market scrapers (GitHub Actions refresh 2×/day + market refresh), the dashboard,
   and the **Model vs Markets tab**.

Coupling: `predict.py`/`predict_margin.py` READ polling-agg's `data/raw/*.csv`;
`refresh_dashboard.py` COPIES prediction CSVs into polling-agg's `data/processed/model_*.csv`;
polling-agg's Actions re-run `analysis/model_compare.py` (which regenerates
`docs/model_data.js` for the tab) every refresh. If either repo moves/renames, fix the
relative paths in `predict.py` (POLLING_AGG_RAW) and `refresh_dashboard.py` (AGG).

## The pipeline (run order)
```
1. build_dataset.ipynb   -> polls (1998-2016 raw_polls_538 + 2018-24 historical, ALL committed)
                            joined with results -> polls_long_with_results.csv
                            (12MB, GITIGNORED - regenerate on fresh clone; fully offline)
2. fetch_approval.py     -> Gallup/UCSB 1993-2025 + VoteHub API 2025+ -> data/approval_monthly.csv
   fetch_macro.py        -> DBnomics history + BLS-API overlay (current) -> data/macro_monthly.csv
                            (committed; re-run only to EXTEND)
3. model.ipynb           -> WIN model: tune on 1998-2016, honest eval on 2018-2024,
                            final fit on ALL cycles -> data/model_xgb.json + model_features.json
   margin_model.ipynb    -> MARGIN model (separate): same scheme, target = vote margin
                            -> data/margin_model_xgb.json + margin_model_features.json
4. predict.py            -> 2026 win probs   -> predictions_2026.csv
   predict_margin.py     -> 2026 margins     -> margin_predictions_2026.csv
   (both read polling-agg raw polls; auto-fetch natl_env; apply stale-candidate filter)
5. refresh_dashboard.py  -> ONE command: feeds -> predict both -> copy CSVs to polling-agg ->
                            model_compare.py. Then commit+push polling-agg to publish.
```
Shared modules: **cycles.py** (ALL cycle constants: CYCLES, PRES_PARTY, eve/prior_eve,
macro_cutoff, natl_env — extend a cycle by editing ONLY this), **features.py** (the ONE
feature builder used by training AND prediction — never fork feature logic outside it),
**macro_features.py** (per-cycle macro stats).

## Files
| file | what |
|---|---|
| `features.py` | Shared feature pipeline (train + predict). Plain raw-poll averages, NaN-not-zero missing, `norm_pollster`, `dist_str`. |
| `cycles.py` | Cycle constants + natl_env (computed 1998-2016, frozen 2018-24, live-fetched 2026+). |
| `model.ipynb` / `margin_model.ipynb` | The two models. SEPARATE by user requirement. |
| `predict.py` / `predict_margin.py` | Score 2026 from the polling-agg feed. Stale-candidate filter lives in predict.py (imported by predict_margin). |
| `refresh_dashboard.py` | One-command refresh (header documents every variable's feed). |
| `fetch_macro.py` / `fetch_approval.py` / `fetch_generic_ballot.py` | Data feeds (BLS API+DBnomics / UCSB+VoteHub / Wikipedia aggregators). |
| `build_dataset.ipynb` | Polls+results join. All inputs committed; offline. |
| `CONCERNS.md` | **Ranked risks + improvement roadmap. The living audit doc.** |
| `METHODOLOGY.md` | Exact time windows per feature. |
| `data/` | ALL inputs committed (polls, results, races.csv, macro, approval, generic ballot, model artifacts). |
| polling-agg: `analysis/model_compare.py` + `docs/model_data.js` | Model-vs-markets tab data (party markets + Kalshi margin-of-victory ladders + model-split flags, primaries-decided states only). |

## THE RULES (learned the hard way)
1. **Re-run the WHOLE model notebook (grid search included) whenever features change.**
   Stale hyperparameters once faked a regression. Applies to BOTH notebooks.
2. **Tune on 1998–2016 ONLY; 2018–2024 is the honest eval set.** Never report the tuning
   score as performance. Production artifacts then train on ALL cycles.
3. **Never use `vote_pct` / `race_winning_pct` / `margin_actual` as features** — they're
   outcomes (vote_pct rides through features.py as a label passenger for the margin model).
4. **Validate by cycle, never randomly.**
5. **All static data committed; no run touches the network** (exception: predict-time
   natl_env fetch, and the fetch_* scripts when extending).
6. **No 538-only inputs, no poll weighting.** Future = raw polls + econ only. Any new
   feature must be computable from a bare poll feed.
7. **Data available at election eve only**: macro windows end **Sep 30** (October prints
   publish after the election — vintage look-ahead), pct rounded to 1dp (feed precision),
   pollster house effects keyed by `norm_pollster`.
8. **Margin model stays separate from the win model** (user requirement). Where they
   disagree on a winner, the dashboard flags ⚠ SPLIT = treat as no-edge.

## TRAPS (don't repeat)
- **Run nbconvert executions ONE AT A TIME** — concurrent runs race and overwrite outputs.
- **Clear `__pycache__` when helper modules change.**
- CSV round-trips turn House districts into floats (`'1'`→`1.0`) — `features.dist_str`
  normalizes; a hard assert in model.ipynb guards House fundamentals coverage.
- The results files' `party` column is all-null — use `ballot_party`.
- The polling-agg feed has internal + cross-source duplicate polls — predict.py dedups.
- "General"-stage feed rows include **hypothetical primary-season matchups**; the
  stale-candidate filter (14d, with guards) removes primary losers (e.g. ME-Sen had 5
  candidates before it).
- 2026 races in RCV states (ME/AK) and runoff states: first-round leader can lose — ~3.5%
  of training labels disagree with margin sign; both models are weakest there.
- **GitHub Pages deploys occasionally fail transiently** ("try again later") — the site
  then serves a stale build. `gh run list --repo pjmerica/polling-agg-2026`, then
  `gh run rerun <id>`.
- RCP is Cloudflare-walled (403 to scripts) — generic ballot comes from Wikipedia's
  aggregator table instead; approval polls from VoteHub's open API (`api.votehub.com/polls`).
- The polls dataset (`polls_long_with_results.csv`) is gitignored (12MB) — regenerate via
  build_dataset.ipynb (offline) on a fresh clone BEFORE running models/predict.

## Current state (2026-07-06)
- A **full retrain of both models is/was running** (Sep-30 macro windows + pct rounding +
  pollster normalization = feature change ⇒ rule 1). Pre-fix honest-eval baselines to
  compare against: WIN AUC .969 / Brier .069 / race-acc .863 (poll baseline .868);
  MARGIN MAE 6.46 vs calibrated-poll 7.52 / raw-poll 7.90. Small shifts expected; margin
  MAE materially worse ⇒ investigate, don't auto-revert.
- After any retrain: `py refresh_dashboard.py --no-feeds`, commit this repo, commit+push
  polling-agg (`data/processed/model_*.csv`, `docs/model_data.js`).
- Coverage floor 1998. 2026: 108 general races predicted; dashboard compares 39
  (primaries-decided states with markets).

## Where to take it next
CONCERNS.md "Improvement roadmap" is the ranked list. Top of it: race-level two-party
reframe (kills split-model ambiguity), distributional margin (prices Kalshi MOV ladders
rung-by-rung), snapshot training (mid-campaign honesty), own pollster reliability scores,
partisan-poll features (`partisan` column is ingested but unused).
