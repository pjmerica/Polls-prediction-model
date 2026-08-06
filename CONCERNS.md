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
  **Dedup key fixed 2026-07-31: it now uses the NORMALIZED pollster (`F.norm_pollster`), not
  the raw string.** Keying on the raw string missed exactly the cross-source case the dedup
  exists for — the same survey filed under two spellings survived as two independent polls
  and was double-counted in every aggregate: `"Glengariff Group, Inc."` 41.4 **and**
  `"Glengariff Group"` 41.0 for the same 2026-07-11 field date, likewise Mitchell Research,
  Susquehanna, Rosetta Stone. MI-Sen-DEM carried **99 poll rows for 36 real surveys**.
  Removed 2254 duplicate rows from the 2026 primary feed and 1040 from the general feed.
  `norm_pollster` also gained a trailing-descriptor strip (`Communications`, `& Associates`,
  `LLC`, …) — anchored to the END and never allowed to empty the string, and verified not to
  merge distinct firms (`Lester & Associates` vs `Ron Lester and Associates` stay separate;
  all 19 collapsed groups are real spelling variants of one pollster).
  **The same key must stay in sync in three places** — `predict.py`, `predict_primary.py`,
  `build_primary_dataset.py` (never-fork rule).
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

### 1. ~~Approval series ends 2025-01~~ RESOLVED (verified 2026-07-21)
This was fixed by the VoteHub API continuation (line "✅ Approval solved" above) but this
entry was never removed from the open list, contradicting that line. Verified 2026-07-21:
`data/approval_monthly.csv` is current through **2026-06** — the UCSB→VoteHub handoff is
live and working, not stalled at 2025-01. No action needed; entry kept (struck through) so
the next reader doesn't re-investigate a non-problem.

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

## New worries found in the 2026-07-06 post-change sweep

### ⚠ Mid-decade redistricting breaks House fundamentals for 2026 (NEW, potentially big)
Several states redrew congressional maps between 2024 and 2026 (e.g. the Texas/North
Carolina-style mid-decade redraws). Our `prior_margin_cand` and incumbency joins key on
**district numbers** — for a redrawn district, "TX-28's 2024 margin" describes an electorate
that partially no longer exists, and the model can't tell. Polls of the new district are
fine; the fundamentals are quietly wrong. **Mitigation ideas:** a `redistricted_2026` flag
per state/district (small manual table, or scrape Ballotpedia's redistricting tracker) so
the model can discount priors; longer-term, population-overlap-weighted prior margins.
Until then: treat House edges in redrawn states with extra suspicion.

### Model-vs-market tab staleness was masked (fixed same day)
`generated_at` refreshed on every Action run even when the model predictions were weeks old.
The payload now carries `predictions_as_of` (prediction file mtime) and the tab shows it.

### Residual seams (accepted, documented)
Approval Gallup→VoteHub methodology seam at 2025-01; natl_env 538-average→aggregator-mean
seam at 2026; ~33% of 2026 feed poll rows are from pollsters with no house-effect history
(they get zero adjustment — correct but weaker); Wikipedia/Kalshi-title regex fragility in
fetch_generic_ballot.py and MOV_RX (soft-fail to missing values, never crash).

## More data worth pulling in (beyond the roadmap)
- **Special-election overperformance index**: 2025–26 special-election margins vs district
  lean are a famously strong national-environment signal, fully public (Wikipedia/DDHQ),
  and we have none of it.
- **Expert race ratings** (Cook/Sabato/IE): Wikipedia's per-cycle election pages carry the
  ratings tables — parseable like the generic-ballot table; a consensus-rating feature is
  cheap and strong, especially for unpolled House races.
- **FEC fundraising** (free API, quarterly): receipts ratio per race — the classic
  candidate-quality proxy for thin-polled districts.
- **MEDSL district-level presidential returns**: time-varying district partisan lean
  (replaces the dead 538 lean properly, helps House + redistricting mitigation).
- **UMich consumer sentiment** (DBnomics `UMICH/SOC`): one more macro series, arguably more
  election-relevant than CPI level; trivial to add to fetch_macro.py SERIES.
- **VoteHub generic-ballot polls** (532 already available via the same API we use for
  approval): per-poll natl_env + monthly recency cuts, replacing the Wikipedia scrape.

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

## Added by the 2026-07-14 audit (deferred by user — "address after")
Fixed same day: page fake edges (multi-Dem sum → leading matchup; IND-slot vs Dem-party
market → independent-win markets), macro silent zeros (⇒ retrain), market-refresh not
committing model_data.js, workflow crash-swallowing, expanding-window eval cells (now
permanent in both notebooks). See HANDOFF.md 2026-07-14. Still open, NOT yet on the list
above:
16. **has_result selection bias**: training keeps only races whose results name-matched —
    likely over-represents clean two-party races vs what the 2026 feed serves.
17. **Tune/select on Brier (or logloss), not AUC** — the page's currency is probabilities.
18. **Uncertainty-weighted edges** on the page (poll count, bias-sweep spread, win/margin
    disagreement) so thin-poll mega-edges rank below well-supported small ones.
19. **is_incumbent still wrong-district in redrawn states** (PVI patch fixes prior_margin
    only); consider is_redistricted feature or NaN-ing incumbency there (⇒ retrain).
20. Backtest logging schema NOW (pre-Nov-2026) so the edge backtest is a join, not
    archaeology (extends #15).
21. **Nickname-duplicate candidates in the GENERAL model's feed** (found 2026-07-15 via the
    primary model): the same person can appear under name variants ('Bobby' vs 'Robert
    Charles'), splitting their polls across two cand_keys. The primary pipeline merges
    them (features_primary.merge_nickname_aliases, 36 merges in the feed); predict.py's
    general path does NOT yet — leading candidates' poll_avg can be diluted. Port the
    merge into load_agg_polls (feature-affecting ⇒ verify + full retrain).
    **Partly addressed 2026-08-01** — a *different* cause of the same symptom was fixed in
    `norm_name` itself (punctuation splitting one person into two keys; see the
    `cand_key` entry in DATA_DICTIONARY.md). The NICKNAME half of this item is still open
    for the general path.

## Added by the 2026-08-01 name-key / bio-leakage pass

22. **`bio_office_level`: the two models disagree, and it has never been settled.**
    On the general WIN model it is a top-5 feature by mean |SHAP| (and #9 by gain) at
    **99.9% training coverage**. On the PRIMARY model the with/without test shows it adding
    **nothing** — race_acc .910 both ways, Brier .0231 vs .0233 — at **56.1%** coverage.
    Serve-time coverage differs too: general 87.3% vs primary 47.2%, so both carry some
    train/serve gap (mild, unlike `poll_last7`'s 39%→0% cliff). Worth one honest
    with/without run per model rather than keeping it by inertia on one side and ignoring
    the null on the other. Deferred by user ("we will fix all of this later").
23. **`poll_last7` is primary-only and should be revisited for the general model in late
    October.** It is 0.0% populated for 2026 general races today (95 days out) vs 39% in
    training — an always-missing-in-production feature. Two ways to ship it: gate training
    rows on days-to-election so the model only sees it at the horizon being served, or
    simply add it once the live window fills near election day. Either is feature-affecting
    ⇒ full re-tune + retrain.
24. **14 committed CSVs cache `cand_key` as a column.** They silently go stale whenever
    `features.norm_name` changes, and a stale key breaks joins rather than erroring.
    `scripts_rekey_cand_key.py` repairs them, but the real fix is to stop caching a derived
    key — derive it at load time, or add a CI check that recomputes and diffs it.
25. **Wikipedia race pages sometimes omit a candidate's prior office** (Bean's 2022 page
    reads 0 despite 2016 reading 2). Pre-existing source-completeness gap, separate from the
    future-tense leak fixed 2026-08-01; not overridden, still worth a cleanup pass.

## Added 2026-08-03 — full-repo audit findings

29. ~~**`poll_momentum` is a PRODUCTION general-model feature that is 100% NaN at serve time.**~~ **FIXED 2026-08-03** — see the resolution note at the end of this item.
    Same structural problem as `poll_last7` (item 23), except this one actually shipped. It
    needs >=3 dated polls within 60 days of the election; the general election is a fixed date,
    so on 2026-08-03 the live feed's minimum `days_to_elec` is 95 and **0 of 1,667 rows**
    qualify. Training coverage is 54.1%, serve coverage 0.0%.
    It is not harmless: it ranks **12th of 187 by gain**, and it appears in **82 of the
    dashboard's top-10 SHAP blocks with a null value in every one** — so the Explain modal
    shows users a driver that contributed nothing.
    Measured with/without on the held-out cycles:
        WITH (production)   AUC .9685  Brier .0691  race_acc .8570
        WITHOUT             AUC .9678  Brier .0697  race_acc .8630
    Dropping it IMPROVES race-accuracy by 0.6pt and costs ~0.0007 Brier. On that evidence it
    should probably come out of `feature_list`, but that is a feature change -> full re-tune +
    retrain of BOTH general models, so it is logged rather than done unilaterally.
    Note it self-heals in late October, when the 60-day window finally contains polls — which
    is exactly why it was never noticed: the model is only wrong about it for most of the cycle.

    **RESOLUTION (2026-08-03, user call): `poll_momentum` now uses ALL of a candidate's dated
    polls instead of a final-60-day window.** Serve coverage 0% → **39.4%**; train coverage
    54% → 60%. The two definitions measure the same thing (r = 0.917 on the 2,396 training rows
    where both exist), so nothing is lost.
    **Justified by serve-time availability, NOT by accuracy** — and that distinction matters,
    because the single-seed numbers above are misleading. Re-run across 5 seeds:
        A window (old)  race_acc .8613 ± .0024      MARGIN MAE 7.348 ± .038
        B all-poll (new) race_acc .8631 ± .0011     MARGIN MAE 7.390 ± .064
        D BOTH           race_acc .8602 ± .0018     MARGIN MAE 7.348 ± .044
        C neither        race_acc .8620 ± .0010     MARGIN MAE 7.372 ± .046
    The between-variant gaps are the same size as the seed spread, and the two models disagree
    on which they prefer — so on accuracy this feature simply does not matter much. What DOES
    matter is that the old version was permanently NaN in production and polluting 82 Explain
    modals with a null driver.
    **Carrying BOTH was tested at the user's suggestion and is the WORST option for the win
    model** (.8602, below every alternative): at r = 0.92 they are near-duplicates, so the
    spare column mainly gives the trees something sparser to overfit.
    **A FORKED SECOND COPY nearly made this a silent non-change for the primaries.**
    `features_primary.py` carried its own 60-day implementation, so the first "all four models
    retrained" pass actually retrained BOTH primary models on the OLD definition. The tell was
    the primary margin model returning byte-identical results (MAE 17.03, identical
    hyperparameters) — a retrain that changes nothing means the input changed nothing.
    There is now ONE definition, `F.poll_momentum_slope()`, called by both feature modules.
    Lesson: verifying a shared-feature change through one module is not verification. Grep for
    other call sites first.
    The primary-side numbers are NOT the general model's, and the near-duplicate argument does
    NOT transfer: primary coverage 29.1% → **53.1%**, but the old/new correlation is only
    **r = 0.353** (vs 0.917 general), so for primaries this is a genuinely different feature.
    Post-fix, honestly retrained: primary nominee race_acc .915 → **.907**, primary margin MAE
    17.03 → **16.67**. Both moves are inside noise on 102 races / 320 rows — but the primary
    margin model's weak 2022 fold flipped from a loss to a win vs the calibrated baseline
    (15.97 → 14.69 vs 15.39), so both folds now beat it.
    All four models retrained on the new definition (the two primaries twice — see above).
30. **`sentiment_last12_delta` is also 100% NaN at serve time**, for a different reason: the
    consumer-sentiment series in `data/macro_monthly.csv` ends **2025-08**, 12 months stale,
    while every other macro series is current to 2026-06. The `_last12_delta` window cannot be
    filled, so the feature is correctly NaN (this is the 2026-07-14 silent-zero fix working).
    `refresh_dashboard.py`'s freshness guard already tolerates it with a 13-month threshold and
    a "DBnomics mirror lags ~1yr" note, so it is KNOWN — but it ranks 144/187, so the practical
    cost is low and the honest fix is a better sentiment source, not a code change.
31. **Audit clean elsewhere.** Checked and found NO problems in: cross-repo duplicated helper
    functions (only `norm_name` had copies, now fixed); unscoped CSS selectors beyond the
    `.mv-etab` collision fixed today; `inf` values or all-NaN columns in either primary model's
    live matrix; unreferenced/dead scripts; model artifacts or importance CSVs being gitignored
    (all 8 tracked). Redundant recomputation across the refresh chain (house effects + bias
    priors rebuilt by 4 scripts) measures ~2s total — not worth caching.

## Added 2026-08-02 — thin polling + the fundamentals model

26. **The primary model is overconfident below 3 distinct surveys** — ~9-12 points, held out
    on 2022+2024, permutation p=0.024 (`analysis/poll_volume_breakpoint.ipynb`). ~33% of its
    features are NaN there: `poll_momentum` is 100% missing by construction (needs 3+ dated
    polls), `poll_std` needs 2. The GENERAL model shows no such break at any volume —
    fundamentals carry it when polls are thin. **Shipped so far: a DISPLAY fix only** (the
    dashboard's "reliable only (4+ polls)" gate). The model's numbers are unchanged.
    Options measured, none yet chosen by the user:
      a. piecewise temperature (T=0.7 below 4 surveys, 0.4 above) — best on metrics
         (Brier .0218 -> .0206) and it TRANSFERS across cycles, but a cliff at exactly 3
         surveys is an artifact of these 102 races, not a fact about primaries;
      b. continuous evidence-scaled temperature — smoother and more defensible in principle,
         measured neutral-to-worse because it also flattens the well-polled races;
      c. fix `low_confidence_field` instead — it currently keys only on the leader's margin
         over a uniform 1/n split, which one poll clears easily; requiring a minimum survey
         count would LABEL thin races honestly regardless of the number shown.
    Note that NONE of these rescues 2026_CT_Governor_REP, where the single poll (Stewart 42,
    Fazio 13) is simply wrong — calibration widens error bars, it cannot reverse a bad input.
27. **Measure poll volume in DISTINCT SURVEYS, not poll rows.** One survey of an N-candidate
    primary emits N rows, so `n_polls`/`race_total_polls` conflate "well-polled race" with
    "crowded field". An early row-based pass put the breakpoint in the wrong place AND
    reported the 1-survey bucket as the worst; on surveys it is actually UNDER-confident
    (10/10 called correctly) — a lone poll usually appears in a race so lopsided nobody polled
    it twice. "Fewer polls = worse" is NOT monotone.
28. **The fundamentals (no-polling) model — work list CLOSED 2026-08-02.** All items done or
    resolved; the model stays a reference floor, NOT wired into predict.py or the dashboard.
      - #5 fundraising leakage: **PROVEN**, not assumed. FEC coverage runs to the year AFTER
        the cycle (2022 → 2023-01-31), so ~18 months of post-primary money is in the total.
        The eventual nominee is the top fundraiser **92.4%** of the time vs the poll leader
        winning 69.6%; nominees hold a median **6.8×** the runner-up's share. Reassuringly,
        the PRIMARY model has **zero** FEC features, so no primary prediction was affected.
      - #1 hyperparameters re-tuned (LOCO over old cycles only): general Brier .111 → .105,
        race_acc .791 → .811; primary Brier .153 → .132.
      - #2 softmax temperature fit after the params: **T=1.0** for the fundamentals primary
        (vs 0.4 for the poll model — a flat temperature is right precisely because this model
        has little signal to sharpen).
      - #6 **answered: do NOT blend.** `analysis/fundamentals_vs_polls_thin.py` runs both
        models head to head on the same held-out races. The fundamentals model loses on
        accuracy in EVERY survey bucket of BOTH models (primary 2-3 surveys: .471 vs .706)
        and its Brier is worse everywhere. It IS better calibrated on thin primary races
        (gap −0.004 vs +0.206) but that is worth nothing while it picks the wrong winner more
        than half the time. **The thin-poll fix therefore belongs in the poll model's own
        calibration (item 26), not in a second model.**
      - #7 "nondeterminism" was **not** nondeterminism. XGBoost with a fixed random_state is
        bit-for-bit identical here (5 back-to-back fits → identical probabilities to 8dp).
        The .803-vs-.791 difference straddled commit 457dc1d, which regenerated
        `polls_long_with_results.csv`. Different DATA, not a different seed. **General lesson:
        that file is gitignored, so a metric change between runs can come from a silent input
        change git will not show you — compare model numbers only within one commit.**
      - #3 feature selection: deliberately SKIPPED per user; the 161-feature general variant
        stays as is.

## Added 2026-08-05/06 — the roster-integrity pass

The Aug 4 primaries gave the first real scoreboard (135/149 = 90.6% season-to-date,
calibrated: 94% in the 90%+ band, 80% at 70–90%, 78% below). Reading the 14 misses turned
up a single dominant failure mode that had nothing to do with the model.

32. ~~**Withdrawn candidates score as live front-runners.**~~ **FIXED** — but read this as a
    recurring maintenance burden, not a closed bug.
    **CT-Gov-REP is the worst case found: Erin Stewart at 99% on a 357-day-old poll.** She
    suspended her campaign on the eve of the May GOP convention; the actual nominee (Ryan
    Fazio, 92% of delegates) sat at **0.4%** — six days before the primary.
    The same pattern explains several season misses: Hogan 100% in MD-Gov-REP, Gowdy 99% in
    SC-Sen-REP (Graham won), Crenshaw 98% in TX-2. In every case a well-known name in one old
    survey beat the actual candidate.
    Also removed: Mandela Barnes and Sara Rodriguez (WI-Gov, withdrew 7/30 and 7/17), Missy
    Hughes (WI-Gov, 6/22 — still ON THE BALLOT, the deadline had passed), and three CT
    candidates who never cleared the 15% convention threshold.
    **The list is hand-maintained, so this WILL recur.** `data/dropped_out_2026.csv` is the
    only thing standing between the model and a stale field.

33. **`poll_age_days` / `stale_polling` added** to make item 32 visible instead of inferred.
    `n_polls` cannot distinguish "one poll last week" from "one poll last September", and only
    the second lets the field change underneath the model. Flags races with no poll in 90 days.
    **10 of 54 upcoming races are flagged**, including HI-1 (296 days, voted 8/8) and
    CT-Gov-REP (357 days). Surfaced on the primary dashboard as an orange `STALE {n}d` chip.

34. ~~**Junk poll placeholders score as candidates.**~~ **FIXED** — `"A Progressive Challenger"`
    was the model's **94% front-runner in FL-23-DEM**, beating a real named candidate.
    `is_junk_answer` only matched `generic X`; it now also catches article-led hypotheticals.
    Validated against all 1,523 feed names and 1,820 corpus names: 13 flagged, all genuine
    junk, zero real people.

35. ~~**Feed typos split one candidate into two half-strength entries.**~~ **FIXED.**
    `norm_name` deliberately does not fuzzy-match, so a one-character misspelling is invisible —
    both halves just look weakly polled. Found in SIX races: `"Josh Elliot"`/`"Josh Elliott"`
    (7 polls each, six days before CT's primary), Raffensperger split across TWO races,
    `"Esther Kim-Varet"`/`"Esther Kim Varet"`, `"Joe Strada"`/`"Joe Stranda"`.
    Fixed via `data/name_aliases.csv` (exact-match only — auto-merging two real people is worse
    than a split) plus `warn_near_duplicate_names()`, which now runs in both feeds every time.
    That detector suppresses pairs `norm_name` already merges ("Mark R. Warner"/"Mark Warner",
    the two O'Rourke apostrophes) — 10 such pairs exist, and warning about them would train the
    reader to ignore the check.

36. **OPEN: the general model has no equivalent of the primary staleness flag.** `poll_age_days`
    is emitted only by `predict_primary.py`. The general side has the same exposure — its feed
    carries hypothetical matchups for months — and `drop_stale_candidates` (14-day relative
    rule) does not catch a race where EVERY poll is old.

37. **OPEN: results scraped for races that have not happened.** The convention-vote guard
    (item 31) fires on statewide races under 25k votes, and caught a fifth race live
    (ME-Sen-DEM, 571 votes). But the underlying scraper still takes "the last table on the
    page" with no check that the election date has passed. A date guard would be the real fix.
