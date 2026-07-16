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

## 2026-07-16: model predictions now AUTO-REFRESH daily (GitHub Action)

polling-agg .github/workflows/model-refresh.yml (daily 13:15 UTC + manual dispatch):
clones THIS repo (public - no secrets), mirrors the local sibling directory layout,
scrapes both market venues (market CSVs are gitignored - absent on fresh checkouts),
runs refresh_dashboard.py --no-feeds (predict win/margin/primary + SHAP + compares),
commits to polling-agg. Verified end-to-end (run 29529497733 -> commit c9b3652).
CONSEQUENCES:
- Local runs of refresh_dashboard.py are now only needed after RETRAINING (new
  artifacts must be committed+pushed to THIS repo before 13:15 UTC to reach the site,
  or run the refresh locally / dispatch the Action manually).
- polls_long_with_results.csv is now COMMITTED (14MB; CI needs it for house
  effects/bias priors) - build_dataset.ipynb is no longer a fresh-clone prerequisite
  for predicting, only for regenerating that file.
- The model tabs' "model as-of" should never trail the poll feed by more than ~1 day;
  if it does, check the Model refresh workflow runs first.

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

## 2026-07-12 (later): dropped poll_adj feature after ablation
User asked how the model looks with/without the pollster house-effect adjustment. Ran an
honest LOCO ablation (2018-2024). Findings (verified):
- **poll_adj** ranked #4/187 importance in BOTH models but removing it: win AUC .9683->.9679
  and race-acc .8623->.8620 (both noise), margin MAE 6.347->6.313 (slightly BETTER). It's
  ~redundant with poll_avg (poll_adj = poll_avg minus a per-pollster house-effect shift), so
  XGB re-routes the signal. Plus a train/serve hazard: house-effect table matches ~67% of
  2026-feed pollster names, so poll_adj is computed on a different basis for a third of future
  polls. **Decision (user): drop it.**
- Also measured for reference: Group A minus {poll_adj,bias_prior_cand,avg_sample} and the
  recency/dynamics block — see git history if you want fuller numbers. Only poll_adj was cut.
IMPLEMENTATION: removed from `feature_list()` in features.py (the column is still built in
build_candidate_table, harmless). Full re-tune of both models. compute_house_effect /
candidate_poll_adj machinery left in place (unused by features, cheap to keep).
LESSON: high XGB importance != marginal value when a feature is collinear with a kept one;
always ablate before trusting importance, and prefer cutting features that behave differently
on future data.

## 2026-07-14: full pipeline audit → Model-vs-Markets fixes + macro silent-zero retrain

Full audit of model + polling-agg Model-vs-Markets chain. Training/eval side came back
CLEAN (no label leakage: bias priors strictly past-cycle, Sep-30 macro windows, no
post-election polls, dual-seat specials keyed, win_prob_norm sums to 1). The page side had
two classes of FAKE edges, both fixed the same day:

1. **model_dem summed ALTERNATIVE primary candidates.** model_compare.py summed
   win_prob_norm over all Dem rows; races carrying leftover hypothetical-matchup polls
   (6 Dems in ME-Sen) published nonsense (ME-Sen "87% Dem" vs Kalshi 51.5 = fake 35-pt
   edge; NE-H-02 and two CA races showed 100%). FIX: model_dem = LEADING Dem's win_prob
   renormalized vs LEADING Rep's (matches the market's D/(D+R) vig normalization), plus an
   `unresolved_field` flag (page badge "? FIELD") when >1 same-party candidate survives the
   stale filter. ME-Sen honest number: Jackson 78.7 vs market ~52 — a real disagreement now,
   flagged. (User decision: leading matchup = Troy Jackson.)
2. **Independent-slot races compared different events.** NE-Sen: model P(Osborn)=.221 was
   compared to "Will the DEMOCRATIC party win" (.007) = fake 21-pt edge. Both venues list
   independent-win markets (Kalshi .32 / Poly .295); model_compare now matches those when
   dem_display != DEM (`slot_market='IND'`, page badge "I MKT"), normalized over the full
   D+R+I book. Honest edge: ~-9 pts (market likes Osborn MORE than the model).

Other fixes shipped with it:
- **macro_features.py silent zeros**: _trend (<2 obs) and _last12_delta (<13 obs) returned
  0.0, not NaN. 15 TRAINING values were wrong (2018/2020/2022 generic_ballot trends claimed
  "flat" from single-point windows) and 2026 sentiment_last12_delta silently asserted "no
  change" from year-old data. Feature-VALUE change ⇒ full re-tune+retrain of both models.
- **Expanding-window eval cells** added to both notebooks (train strictly < test cycle;
  the LOCO-minus-expanding gap measures LOCO's optimism from training on future cycles).
- **predict.py meta sidecar** (predictions_2026_meta.json): newest poll end_date consumed +
  natl_env used; dashboard meta line shows "newest poll used" so stale-poll edges are visible.
- **market-refresh.yml never committed docs/model_data.js** (regenerated every 2h, thrown
  away — model tab's market side actually updated only 2x/day). Fixed; both workflows also
  now fail LOUDLY on a compare crash (after the data commit) instead of `|| echo` swallowing.
- **Page bugs**: mv-meta line was rendering raw JS source (string concat inside the template
  literal); vig-normalize single-sided/ladder anchors; margin symmetrization uses each
  party's leading candidate.
- **natl_env train/serve mismatch documented** (538 model avg @ eve vs Wikipedia aggregator
  mean @ today) — METHODOLOGY.md section C.

UPDATE (same day): the measured LOCO-vs-expanding gap came back ~zero for the WIN model
(AUC +0.0015, AUC-PR +0.0003), so per user ("if it's negligible, let's do it the correct
way") **EXPANDING-WINDOW IS NOW THE PRIMARY HONEST EVAL** in both notebooks; LOCO is
demoted to a companion table that monitors the gap each retrain; the poll-baseline
benchmark + blend sweep train expanding-window too. Tuning stays LOCO-inside-1998-2016
(internal selection, no honesty issue, better use of the small old-cycle set — see
METHODOLOGY.md). Headline win-model numbers under the new scheme: AUC .966 / AUC-PR .947 /
KS .812 / Brier .072 / race-acc .864.
CAVEAT found when the margin run landed: the MARGIN model's gap is NOT negligible —
LOCO 6.23 vs expanding 7.24 MAE, entirely the 2018 fold (10.2 with only 10 train cycles;
2020/2022/2024 = 6.7/6.2/5.9 converge). The honest headline is now "MAE 7.24 over
2018-2024 as-they-happened", with the 2026-relevant fold (2024: 13 train cycles) at 5.87
and gap-free. Quote the margin model accordingly — do NOT keep citing 6.23/6.24.

## 2026-07-15: PRIMARY nominee model + "Primary vs Markets" page (built same-day)

User asked how fast a primary model could be built; answer: same day. Third separate model
(P(candidate becomes party nominee)); full design in METHODOLOGY.md "PRIMARY nominee model".

DATA STORY (the hard part):
- **NEGATIVE FINDING (do not re-attempt): 538's downloadable poll CSVs NEVER carried
  downballot regular-primary rows in any era.** Verified against in-season Wayback captures
  (Apr-2022 = generals+jungle only despite Oz/McCormick raging; May/Aug-2020 same) and
  post-season ones. The committed *_historical.csv were never filtered — there was nothing
  to filter.
- Recovery: **Wikipedia race pages' primary polling tables**, scraped by driving the
  polling-agg 2026 wikipedia scraper's OWN parser over 2018-2024 page titles
  (fetch_primary_polls_wikipedia.py) → 11,466 primary poll rows, 324 party-races.
  One parser for train AND serve.
- **Primary DATES = hand-entered static table** (fetch_primary_dates.py) after THREE failed
  iterations of Wikipedia prose-mining (filing deadlines/runoffs/news dates near 'primary'
  kept winning; poll-anchored windows got dragged by contamination). Dates are verifiable
  calendar facts — the hand-typed-data ban is about measurements. NY is per-office (2018/
  2022 split primaries). Every date cross-checked against the poll record; warnings printed.
- **Contamination guard**: some pages leak GENERAL polls into primary sections (NC-2022).
  build_primary_dataset drops rows dated after their primary (+2d): 3,062 rows dropped.
- Labels: nominee = the party's general-election candidate (res_*.csv join) — 96% of races
  matched. Kept: 214 contested labeled races / 6,666 poll rows / 1,076 candidate rows.

MODEL (primary_model.py — a SCRIPT, runs in minutes, not a notebook):
- Tune LOCO on ≤2020, honest eval EXPANDING-WINDOW on 2022+2024 vs poll-leader baseline.
- **Headline: AUC .971 / AUC-PR .918 / KS .865 / Brier .046 / race-acc .895 vs poll-leader
  .723** (2022: .924/66 races; 2024: .865/37). The +17pt pick edge over the naive baseline
  comes from recency features (poll-leader uses stale means; primaries break late).
- **FUND FEATURES EXCLUDED from the artifact (leakage)**: FEC receipts are cycle-END totals
  and nominees raise most money AFTER winning the primary → fund_share partially encodes
  the training label, while a mid-2026 candidate has no such money (train/serve skew).
  Measured: identical picks without fund (.895 = .895); only Brier differs (.035 vs .046 =
  the leak-suspect juice). Revisit only with as-of-primary-date FEC reports.
- Macro: available per-primary-date via build_macro_asof() (also the machinery snapshot
  training needs) but ~noise on 214 races — artifact ships NO-macro; both ablations
  re-measured and printed on every training run.

SERVE: predict_primary.py (feed stage='primary', per-race dates from *_current.csv →
primaries.json → calendar; same dedup/stale/junk filters; jungle states CA/WA/LA/AK
excluded) → 183 races, 62 upcoming. polling-agg analysis/model_compare_primary.py prices
vs POLYMARKET candidate primary markets (Kalshi has none downballot with usable ids;
its 'nominee' markets are 2028-VP trivia) → docs/primary_model_data.js → new
"Primary vs Markets" tab (upcoming primaries only — inverse of the general tab's filter).
Thin-poll races (<3 polls) sort LAST regardless of edge size: a 90-pt "edge" off one stale
survey (CT-Gov REP: 1 poll vs a 97% market favorite = Stewart presumably out) is the market
knowing something the polls don't. Both CI workflows run + commit the primary compare.

SAME-DAY UPGRADES (2026-07-15, after user review):
- **Explain modal** on the Primary tab (explain_primary.py -> SHAP top-10 per predicted
  nominee; reuses explain_2026's label machinery).
- **RESULTS-BASED LABELS** (user: "the primary winner might not be the one in the general"
  — Platner): fetch_primary_results_2026.py scrapes actual primary-results tables (2026 +
  --hist 2018-24). THREE parser bugs caught by known-winner validation before they poisoned
  anything: (1) Lt-Gov primary tables nested in gubernatorial primary sections crowned
  running mates (Husted over DeWine, "Fetterman" as 2018-PA-gov); (2) joint-ticket cells
  ("Richard Cordray and Betty Sutton" -> 'sutton r') — take the first wikilink; (3)
  '(withdrawn)' annotations leaking into name keys. Also: NEVER trust `cmd | tail` exit
  codes — a masked crash shipped an empty hist file once.
- **Nickname-alias merging** (features_primary.merge_nickname_aliases, train+serve): same
  person split across name variants ('Bobby'/'Robert Charles') diluted their own polls —
  36 merges in the 2026 feed alone. Same-last-name + nickname-equivalence, within-race only
  (Mayra vs Eric Flores stay distinct). The GENERAL model's feed likely has the same issue
  — queued as roadmap item #21.
- **Population splits** (user request): poll_avg/last/last30/std/n_polls/lead per LV/RV/A.
  Ablation: identical picks, Brier slightly better -> kept.
- **New headline** (corrected labels): race-acc .910 / AUC-PR .923 / Brier .046 vs
  poll-leader .723. **2026 backtest vs actual winners: 84 contested decided primaries,
  picks 81.0% vs poll-leader 67.9%, AUC .962.** (The pre-correction v1 artifact scores
  85.7% on the same set; the 4-race delta is binomial noise at n=84 and was isolated to
  the label/tune changes, NOT the population features — v2 stands because its labels are
  verifiably correct. Both numbers recorded here deliberately.) The value-add stat: in
  the 19 races where model and poll-leader disagreed, model right 14 / polls 3 / both
  wrong 2. Full reproducible scorecard: **analysis/primary_backtest_2026.ipynb**
  (executed outputs committed; bump AS_OF + re-run the results scraper to extend after
  the August primaries). Corrected labels also cleared two fake
  high-confidence "misses" (NM-Gov "Haaland lost" was an LG-table artifact — she won;
  ME-Gov was the Bobby/Robert duplicate). Remaining high-confidence misses are ALL House
  races — the office the training set lacks; treat House primary probabilities as softer.

AUDIT ITEMS DEFERRED (user: "address after"): snapshot training (top priority — training
races' freshest poll is median 6d pre-election vs ~112d for 2026 predictions, so honest-eval
numbers do NOT describe July forecasts), race-level two-party reframe, shipping the α≈0.5
blend, has_result selection bias, is_incumbent still wrong-district in redrawn states,
grid-search selects on AUC not Brier, edge/uncertainty weighting, backtest logging schema.
