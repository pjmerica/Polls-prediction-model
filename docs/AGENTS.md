# Instructions for a new agent (or developer) taking this over

Read this first, then HANDOFF.md (in-flight state + breakdown risks + next steps), then CONCERNS.md (ranked risks + improvement roadmap), then METHODOLOGY.md
(exact windows for every feature). Last full update: **2026-07-06**.

> ## ⚠ FILE LAYOUT CHANGED 2026-08-02 — read STRUCTURE.md
> Scripts moved out of the repo root into `models/poll/`, `models/fundamentals/`,
> `pipeline/fetch/`, `pipeline/build/` and `tools/`. **Any bare filename in this file or the
> other docs may now live in a subfolder** — most references below were written before the
> move and are kept as-is because they describe history. STRUCTURE.md has the current map and
> the run commands. Paths now resolve through `paths.py` (repo-root-relative), NOT from each
> script's own location. The predict/explain scripts and `refresh_dashboard.py` deliberately
> stayed at the root because the polling-agg CI workflow calls them there.

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
4. src/predict.py        -> 2026 win probs   -> outputs/predictions_2026.csv
   predict_margin.py     -> 2026 margins     -> margin_predictions_2026.csv
   (both read polling-agg raw polls; auto-fetch natl_env; apply stale-candidate filter)
5. src/refresh_dashboard.py -> ONE command: feeds -> predict both -> copy CSVs to polling-agg ->
                            model_compare.py. Then commit+push polling-agg to publish.
```
Shared modules: **cycles.py** (ALL cycle constants: CYCLES, PRES_PARTY, eve/prior_eve,
macro_cutoff, natl_env — extend a cycle by editing ONLY this), **features.py** (the ONE
feature builder used by training AND prediction — never fork feature logic outside it),
**macro_features.py** (per-cycle macro stats).

## Files
> Paths are as of the 2026-08-02 reorganisation — **STRUCTURE.md is the full map**, and it
> also lists what may live in the repo root. (This table listed bare root filenames until
> 2026-08-08, long after the models moved into `models/poll/`.)

| file | what |
|---|---|
| `src/features.py` | Shared feature pipeline (train + predict) for the GENERAL + margin models. Plain raw-poll averages, NaN-not-zero missing, `norm_pollster`, `dist_str`, `is_junk_answer`. |
| `src/features_primary.py` | The same job for the two PRIMARY models. **Grep both when changing a shared feature** — they are the classic fork risk. |
| `paths.py` (root) | THE path module. Every subfolder script resolves `data/` and the sibling polling-agg repo through it. Import before anything else. |
| `src/cycles.py` | Cycle constants + natl_env (computed 1998-2016, frozen 2018-24, live-fetched 2026+). |
| `models/poll/model.ipynb` / `margin_model.ipynb` | The two GENERAL models. SEPARATE by user requirement. |
| `models/poll/primary_model.py` / `primary_margin_model.py` | The two PRIMARY models (plain scripts, run in minutes). |
| `models/fundamentals/fundamentals_model.py` | No-polling reference models. NOT shipped to the dashboard. |
| `src/predict.py` / `predict_margin.py` | Score 2026 from the polling-agg feed. Stale-candidate filter + `drop_primary_losers` (dead-matchup removal) live in predict.py (imported by predict_margin). Also applies the two override files below + emits `n_surveys`, `display_party`, bias-sweep cols. |
| `src/predict_primary.py` / `predict_primary_margin.py` | The primary-side equivalents. |
| `src/explain_2026.py` | SHAP top-10 per race for BOTH general models -> `model_explanations_2026.json` (dashboard Explain modal). Friendly names + hover descriptions — **these strings are PUBLIC**, so they must match what the feature actually computes. |
| `src/explain_primary.py` | Same for the two primary models -> `primary_explanations_2026.json`. |
| `tools/build_missingness_report.py` | Regenerates `MISSINGNESS_REPORT.md` from current data (it is a GENERATED file — don't hand-edit). |
| `data/candidate_party_overrides.csv` | Hand-maintained party fixes. **Two columns**: `model_party` (what the model treats them as — fills the two-party slot) and `display_party` (real affiliation shown on the dashboard). E.g. Dan Osborn: model_party=DEM (de-facto challenger vs Ricketts), display_party=IND. |
| `data/dropped_out_2026.csv` | Candidates whose stale poll rows to remove (withdrew, or fringe also-rans diluting a race). E.g. Duggan (MI-Gov withdrew), NE-Sen fringe Dems. |
| `src/refresh_dashboard.py` | One-command refresh (header documents every variable's feed). Runs all four predicts + both explainers + copies to polling-agg. **The thing CI runs.** |
| `pipeline/fetch/fetch_macro.py` / `fetch_approval.py` / `fetch_generic_ballot.py` | Data feeds (BLS API+DBnomics / UCSB+VoteHub / Wikipedia aggregators). |
| `pipeline/build/build_dataset.ipynb` | Polls+results join. All inputs committed; offline. |
| `pipeline/build/build_office_level_table.py` | THE builder of `data/candidate_bios.csv` (leak-free, as-of-year office levels). Not `combine_candidate_bios.py` — that one is deprecated and now refuses to run. |
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
9. **Model party vs display party** (user decision 2026-07-12). An independent who is the
   de-facto two-party challenger (Dan Osborn vs Ricketts) is MODELED in the DEM slot
   (`party_std`=DEM, so poll_lead / two-party margin / normalization work) but DISPLAYED as
   IND. Do NOT "correct" `party_std` back to his real party — it would collapse the
   two-party framing and zero out his win prob. The split lives in
   `candidate_party_overrides.csv` (model_party vs display_party) + predict.py.

## TRAPS (don't repeat)
- **Run nbconvert executions ONE AT A TIME** — concurrent runs race and overwrite outputs.
- **Clear `__pycache__` when helper modules change.**
- CSV round-trips turn House districts into floats (`'1'`→`1.0`) — `features.dist_str`
  normalizes; a hard assert in model.ipynb guards House fundamentals coverage.
- The results files' `party` column is all-null — use `ballot_party`.
- The polling-agg feed has internal + cross-source duplicate polls — predict.py dedups
  **on the NORMALIZED pollster** (`F.norm_pollster`), not the raw string: the same survey is
  filed under two spellings ("Glengariff Group, Inc." vs "Glengariff Group") and the raw key
  let both through. The identical key lives in `predict.py`, `predict_primary.py` and
  `build_primary_dataset.py` — **keep all three in sync** (never-fork rule).
- **`features.norm_name` is THE shared join key** (polls ↔ results ↔ FEC ↔ bios ↔ history,
  both pipelines). Changing it re-keys every join ⇒ full retrain. Two live gotchas: hyphens
  must swallow adjacent whitespace (sources type "Mucarsel- Powell", and the mismatch
  silently DROPS a race from training), while periods must NOT (or "Robert F. Kennedy"
  becomes `fkennedy r`). **14 committed CSVs cache `cand_key` as a column** and go stale on
  any `norm_name` change — re-run `scripts_rekey_cand_key.py` (dry-run first).
- **Softmax temperature cannot be tuned by accuracy** — softmax is monotonic, so it never
  changes the argmax. Tune it on Brier. It was hardcoded at the worst value for weeks under
  a comment claiming otherwise; see METHODOLOGY.md.
- **Never re-run `classify()` on descriptor-less bio rows.** ~8000 rows are hand-coded
  (`src='manual'`, empty descriptor); classifying an empty string returns 0 and silently
  drops sitting members of Congress from office_level 4 to 0.
- **Dry-run every in-place data repair script** before `--apply`. Both repair scripts added
  2026-08-01 default to dry run for this reason; one of them would have wiped ~600 correct
  hand-coded rows on the first attempt.
- **NEVER regenerate `docs/*_data.js` locally without re-scraping markets first.**
  `polling-agg/data/raw/*.csv` (kalshi, polymarket) are GITIGNORED and scraped fresh by CI on
  every run — a local checkout has whatever vintage this machine last pulled. Regenerating the
  dashboard JS locally reads those stale CSVs and the push then OVERWRITES CI's correct prices.
  Found 2026-08-03: the page showed MI-07 Lawrence at 26% while Kalshi had ~91%, from a
  July-3 local CSV. Run `scrapers/kalshi.py` + `scrapers/polymarket.py` first. Model-side
  outputs (predictions, explanations) are safe to regenerate locally; market-side ones are not.
- **The three polling-agg workflows do NOT have the same checkout shape.** `model-refresh.yml`
  checks out BOTH repos; `market-refresh.yml` and `refresh.yml` check out polling-agg ONLY —
  and all three run the compare scripts. So anything in `analysis/model_compare*.py` that
  imports from the model repo must degrade gracefully when it is absent (import-else-mirror).
  A change exercised by only one workflow passes every local test and still breaks the other
  two on their next schedule.
- **Generated files must never be merge-resolved.** `docs/*.js` are regenerated output; a
  rebase conflict in them must be settled by taking either side to unstick the rebase and then
  RE-RUNNING the generator. A bare `git pull --rebase` in a push-retry loop leaves the tree
  dirty on conflict and every later attempt fails (killed both Daily refresh runs 2026-08-03).
- **`norm_name` has been re-implemented by hand THREE times** (build_dataset.ipynb,
  model_compare_primary.py, and features.py itself). Every copy drifted. If you find a fourth,
  import the real one — and when a copy is genuinely unavoidable (see the CI checkout asymmetry
  above), assert it matches whenever both are visible.
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
- The polls dataset (`polls_long_with_results.csv`) is **COMMITTED** (~14MB, force-added past
  the `*.csv` ignore rule so CI can see it) — a fresh clone does NOT need to rebuild it. Run
  build_dataset.ipynb only to extend/rebuild. (Three docs claimed it was gitignored, at three
  different sizes; corrected 2026-08-08.)
- **`is_incumbent` is PARTY-level, not personal** — it is `incumbent_party == candidate party`
  and nothing else, because `races.csv` has an incumbent PARTY column and no incumbent NAME.
  So several candidates in one race can all read 1 (16 of 114 general races in 2026; AK-Gov 3,
  SC-Sen 7), and a House member running for Governor reads 1. Don't "fix" the data — it is
  computing what it can. Don't describe it to users as personal incumbency either (that bug
  shipped to 128 live explanations). Renaming it ⇒ retrain.
- **Junk poll answers must be matched WHOLE-STRING, never by substring.** `Mike Rounds` and
  `Tony Knowles` are real officeholders that a `round`/`know` substring rule would delete. The
  slash-joined forms (`"Don't know/Someone else"`) are handled by splitting on `/` and
  requiring EVERY part to be a non-answer — 8 of these were being scored as real candidates
  until 2026-08-08.
- **`combine_candidate_bios.py` is deprecated and now refuses to run.** It writes the same
  `data/candidate_bios.csv` as `build_office_level_table.py` but with a frozen, leak-prone
  `office_level`. Use `pipeline/build/build_office_level_table.py`.
- **Scraper logs go in `logs/` (ignored as a directory), not the repo root.** Name-based
  ignore patterns silently committed 10 of 20 logs. docs/STRUCTURE.md has the closed list of
  what may live in the root - it is FIVE entries, and Python is not one of them.
- **Never build a data path from `__file__`** (`dirname(abspath(__file__))/"data"`). It is
  correct only while the file sits in the repo root; from `src/` it silently resolves to
  `src/data/`. Use `paths.data(...)` / `paths.out(...)`. The 2026-08-08 move surfaced four
  live instances, `features.DATA_DIR` among them.
- **Import `paths` as a module, never `from paths import out`.** `out` is a very common local
  variable name here, so a bare import gets shadowed and fails with
  `TypeError: 'DataFrame' object is not callable` at the END of a long run.
- **The two repos are checked out separately and can sit at different commits.** polling-agg
  looks for this repo's modules in `src/` then the root, and predictions in `outputs/` then
  the root; the root keeps a `refresh_dashboard.py` shim. Keep all three fallbacks working -
  a break here surfaces at 13:15 UTC, not in front of you. During the 2026-08-08 move,
  polling-agg's `from features import norm_name` silently fell back to a local mirror and
  disabled its own drift-assert, announcing it with one line of "note:".

## Current state
See HANDOFF.md for the dated log of every retrain and its numbers — this section is
intentionally NOT kept current move-by-move (it went stale after 2026-07-06 the first time);
HANDOFF.md's dated entries are the source of truth for "what changed when."
- After any retrain: `py refresh_dashboard.py --no-feeds`, commit this repo, commit+push
  polling-agg (`data/processed/model_*.csv`, `docs/model_data.js`).
- Coverage floor 1998, 14 cycles.

### Two staleness facts worth knowing (neither is a bug)
- **This repo's own `predictions_2026*.csv` / `margin_predictions_2026.csv` /
  `primary_predictions_2026.csv` are NOT kept live.** The daily GitHub Action refreshes
  polling-agg's copies (`data/processed/model_*.csv`) directly; the copies checked into
  *this* repo only update when someone runs `refresh_dashboard.py` locally (normally after
  a retrain). Seeing a multi-day-old `generated_at` in this repo's own CSV does not mean the
  live dashboard is stale — check polling-agg's `model_predictions_as_of.txt` for that.
- **`refresh_dashboard.py`'s `AGG` path** (`predict.py`'s `POLLING_AGG_RAW` too) is a
  hardcoded two-level-up relative path assuming the sibling directory layout
  `Documents/Polling prediction model` + `Documents/Polling Agg/Polling agg and Prediction
  markets`. If either directory is ever renamed or the repos are moved, both paths need a
  manual fix — there's no config layer, by design (see "THE TWO REPOS" above).

## Where to take it next
CONCERNS.md "Improvement roadmap" is the ranked list. Top of it: race-level two-party
reframe (kills split-model ambiguity), distributional margin (prices Kalshi MOV ladders
rung-by-rung), snapshot training (mid-campaign honesty), own pollster reliability scores,
partisan-poll features (`partisan` column is ingested but unused).
