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
explicit features instead. Poll `pct` is rounded to 1 decimal in BOTH training and predict
paths (the live feed's resolution — instrument harmonization, 2026-07-06). Pollster names are
normalized (`features.norm_pollster`) before house-effect lookup so 2026-feed names match the
538-era history (row match 63% → 67%).

| feature | window |
|---|---|
| `poll_avg`, `poll_std`, `n_polls`, `poll_share`, `poll_lead`, `n_polls_over50`, `frac_polls_over50`, `race_total_polls`, `avg_sample` | all general-election polls in the cycle for that candidate/race |
| `poll_last` | the single most recent poll **with a known date** |
| `poll_last30` | polls within **30 days** of election (`days_to_elec ≤ 30`) |
| `poll_momentum` | slope of `pct` over polls within **60 days** (needs ≥3 polls) |
| `min_days` | days-to-election of the candidate's latest poll |
| `gap_x_recency` | `poll_lead` × recency factor `1/(1+min_days/30)` |
| ~~`poll_adj`~~ | **DROPPED as a feature 2026-07-12.** Was the plain mean of house-effect-adjusted pct. An ablation (honest LOCO eval) showed it added no out-of-sample value — win AUC/accuracy unchanged, margin MAE slightly *better* without it — because it was largely redundant with `poll_avg`. It also had a train/serve risk (the pollster house-effect table matches only ~67% of 2026-feed pollster names). The column is still computed but is no longer fed to either model. |

## B. Lead-dynamics features
Running-mean margins over the race's poll dates (all in-cycle): `avg_margin_over_time`,
`min_margin`, `margin_trend`, `margin_volatility`, `n_lead_changes`, `lead_changed`.

## C. National environment
`natl_env_cand` = generic-ballot DEM−REP margin over the **30 days before the election**,
signed to the candidate's party. Source per cycle (see `cycles.py`): 1998–2016 computed from
the committed daily history file; 2018–2024 frozen constants; 2026+ passed to
`predict.py --natl-env` manually (default: fetched live from the Wikipedia aggregator table).

**Known train/serve mismatch (2026-07-14 audit):** training values are 538's *model-based*
generic-ballot average over the last 30 pre-election days; the 2026 value is a *different
instrument* (Wikipedia aggregator mean) at a *different time anchor* (today, mid-campaign,
not election eve). Both measure the same quantity but with different house-effect handling
and smoothing — treat 2026 `natl_env_cand` as approximate. The value actually used is
recorded in `predictions_2026_meta.json` per run.

## D. Fundamentals
| feature | window |
|---|---|
| `prior_margin_cand` | most recent **prior** same-office election for that seat (2/4/6/8 yrs back; strictly before the cycle) |
| `is_incumbent`, `is_inc_party_race` | from frozen `races.csv` incumbent_party; **unknown = NaN**, never 0 |
| `is_president_party` | candidate's party == sitting president's party |

(The 538 partisan-lean file was removed entirely — single 2022 vintage = look-ahead leakage.)

## E. Macro features — per-cycle windows
Per metric, stats over **that cycle's own window** = prior even-year Nov 1 → **this Sep 30**
(e.g. 2024: 2022-11-01 → 2024-09-30). The window ends Sep 30 — not eve — because October
economic prints (CPI mid-Nov, jobs report ±election day) are not reliably published before
the election; using them would be vintage look-ahead (fixed 2026-07-06). 7 full-window stats (`_eve/_mean/_max/_min/_std/_trend/
_last12_delta`) + 3/6/12-month recency cuts (`_avg/_max/_trend`). Metrics: unemployment,
inflation (CPI YoY computed on the full series, then windowed), cpi_core, gas, fed_funds,
unemp_u6, approval. Approval comes from `data/approval_monthly.csv` (Gallup via UCSB,
1993–2025-01). Pre-cycle months missing ⇒ NaN (XGBoost routes missing).

**Silent-zero fix (2026-07-14):** `_trend` with <2 observations and `_last12_delta` with <13
used to return 0.0 ("flat"), violating the missing=NaN rule — 15 training values were wrong
(2018/2022 generic_ballot trends claimed flat from single-point windows) and at predict time
a lagging series (sentiment, ~1yr behind) got a silently stale "no change". Now NaN.
Both models retrained (feature-value change ⇒ full re-tune per the standing rule).

---

## Validation (nested — no selection bias)
- **Hyperparameter tuning:** leave-one-cycle-out CV over **1998–2016 only** (150 sampled
  configs, live grid search every run). LOCO is fine *inside* the tune block: it's internal
  selection on cycles the evaluation never touches (no honesty issue), and it uses the
  small old-cycle set more efficiently than expanding folds would.
- **Honest evaluation (PRIMARY, 2026-07-14): EXPANDING-WINDOW over 2018–2024** — each fold
  trains strictly on cycles BEFORE the test cycle (2018 ← 1998–2016, …, 2024 ← 1998–2022),
  exactly what a real forecaster could have done. The tuner never saw these cycles.
- **Companion: LOCO over 2018–2024** (each fold trains on the other 13 cycles, future ones
  included) is still printed to monitor the optimism gap each retrain. Measured 2026-07-14:
  - **Win model: gap ≈ 0** (AUC +0.0015, AUC-PR +0.0003, KS +0.0034, race-acc −0.0016) —
    the switch cost nothing.
  - **Margin model: gap = 1.0 MAE pt** (LOCO 6.23 → expanding 7.24), **entirely the 2018
    fold** (MAE 10.2 with only 10 training cycles; 2020/2022/2024 = 6.7/6.2/5.9 converge to
    LOCO). Read the headline accordingly: the eval-mean is dragged by small-training-set
    folds, while the 2026-relevant fold (2024, trained on 13 cycles) shows no gap at all.
- The model-vs-poll-baseline benchmark and blend sweep also run expanding-window.
- The single-split walkthrough (train = all but 2024, test = 2024) is expanding-window by
  construction (2024 is the last cycle).
- Never random splits.

**Win-model metrics reported (candidate level):** ROC-AUC, **AUC-PR** (average precision —
sensitive to the ~37% win base rate; the honest positive-class number), **KS** (max
separation between winners' and losers' predicted-prob CDFs = max(TPR−FPR)), Brier, LogLoss,
plus race-winner accuracy vs the tuned poll-softmax baseline. AUC-PR + KS added 2026-07-12.
The **margin model** reports MAE / R² only — AUC-PR and KS are classification metrics and
don't apply to a regression target.

## Production model (predict.py)
Trained on **all 14 cycles** with the tuned params; saved to `data/model_xgb.json` +
`data/model_features.json` by `model.ipynb`. `predict.py` builds identical features (same
`features.py` code) from the polling-agg raw poll feed and outputs per-candidate win
probabilities plus within-race normalized probabilities.

## Static-data principle
Every input is pulled once and **committed**: polls (all vintages), results, races.csv, macro,
approval, generic ballot. No model or predict run touches the network. Re-pull only to extend
to new months/cycles.

---

# PRIMARY nominee model (added 2026-07-15)

A third, separate model: P(candidate wins their party's nomination), for the dashboard's
"Primary vs Markets" tab (Polymarket candidate primary markets; upcoming primaries only —
the inverse of the general tab's decided-primaries filter).

**Scope:** regular DEM/REP partisan primaries for Senate/Governor/House. Jungle/top-two/RCV
states (CA, WA, LA, AK) excluded — "advance to the general" is a different target. Runoffs
excluded from MVP (each round is its own contest; label = eventual nominee is future work).

**Data:** historical (2018–2024) primary polls scraped from Wikipedia race pages'
"Democratic/Republican primary" polling tables (`fetch_primary_polls_wikipedia.py`, driving
the SAME parser the polling-agg 2026 scraper uses — one parsing implementation for train
and serve). Negative finding, verified so nobody re-tries it: **538's downloadable poll
CSVs never carried downballot regular-primary rows in any era** — in-season Wayback
captures (Apr-2022, May/Aug-2020, Nov-2020/2024) all show general+jungle only.
Primary DATES are extracted per race page (`fetch_primary_dates.py`, prose regex + mode,
validated against hand-checked dates; last-poll+4d fallback flagged `approx`) — primaries
move between cycles and states, so dates are per-(cycle,state,office), never assumed.

**Labels (upgraded 2026-07-15, same day):** won = the ACTUAL primary winner, scraped from
the same Wikipedia race pages' results tables (fetch_primary_results_2026.py --hist; last
table per party-race so runoffs supersede round 1). The original nominee-join (candidate
appears among the party's general-election candidates) remains the fallback for races
without parsed results — it mislabels primary winners who later withdrew (the Platner
scenario) and misses nickname variants ('Bob' vs 'Robert Casey'). Results-scraper
hardening, each caught by known-winner validation: down-ticket guard (Lt-Gov primary
tables inside gubernatorial primary sections once crowned running mates), joint-ticket
cells (take the first wikilink, not the merged cell text), parenthetical annotations
stripped. Nickname-alias merging unifies same-person variants within a race at BOTH train
and predict time ('Bobby'/'Robert Charles' split one ME-Gov-26 candidate's polls across
two keys; 36 merges in the 2026 feed).

**Population splits (2026-07-15, user request):** poll_avg/last/last30/std/n_polls/lead per
LV / RV / A surveyed-population class ('v' folds into RV; absent class = NaN). Ablation:
identical picks, Brier slightly better. Per-class momentum/dynamics: too sparse at ~200
races, documented not forgotten.

**Candidate features (2026-07-17, user request; OVERFIT-TRIMMED 2026-07-18).**
The build produced ten candidate-history features across two fact-checked sources —
(a) electoral track record from the committed results archives (candidate_history.py,
strictly-prior-cycle, fusion-lines deduped) and (b) officeholder bios scraped from the
race-page candidate bullets (fetch_candidate_bios.py: office_level + prior_candidacy).
Both passed heavy fact-checking (check_candidate_history.py: 13 exact known-truth asserts
+ 0.09% collision audit; check_officeholder.py: FOUR parser bugs each caught by
known-truth asserts — candidacy-as-office, US-title variants, US-vs-state house,
ENDORSEMENT-lists-as-candidates which were 3/4 of the first scrape; final 7/7 levels
exact, 98% cross-source consistency).

**But an overfitting review (asked: "are we scared this is overfit?") showed the
aggregate .935 race-acc was one-cycle luck** — the ten-feature set HURT 2022 (.955→.924)
while helping 2024 (.865→.946). A per-cycle, 6-seed feature-subset sweep found that a
SINGLE feature — `bio_office_level` (highest office held: 4 fed/3 statewide/2 state-leg/
1 local/0; 35% coverage; a durable mechanism, not a quirk) — does ALL the honest work:
best Brier and AUC-PR in BOTH cycles, no 2022 regression, and it RAISED the 2026
out-of-sample backtest from 83.3% to **86.9%**. Dropping the other nine improved
generalization. So the artifact keeps `bio_office_level` only. The hist_*/results
machinery + candidate_history.py stay in the repo (fact-checked, may return with more
training data) but are no longer model features. Lesson: 9 thin-coverage features on
~200 races overfit; fact-checking data accuracy does NOT prevent feature-count overfit —
a per-cycle seed sweep is the check that catches it.
Known caveat on bios: race pages are read as they exist TODAY; post-election edits can
tint descriptors — mitigated by using office LEVEL (rarely edited) and dropping in_office.

**Two more bio-scraper parser bugs found 2026-07-23** (user asked to bring `bio_office_level`
to the GENERAL model too — see HANDOFF.md's 2026-07-23 entry for the full in-flight state;
this note is the permanent methodology record). Both affect the ALREADY-LIVE primary model's
`candidate_bios.csv`, not just the new general-model extension work:
  a. **Citation-link name mangling**: `parse_page()`'s `li.find("a")` could grab a footnote
     link (`<a href="#cite_note-...">[8]</a>`) instead of the candidate's actual (unlinked)
     name text, corrupting **640 of 4,441 rows (14.4%)** of the then-committed bio data
     (e.g. "Matthew W. Morgan" → "ew W. Morgan"). Fixed by skipping `#cite_note`-href links
     when hunting for the name link.
  b. **Incumbent-context ambiguity**: bare "incumbent senator" / "incumbent Representative
     [from X] since Y" (Wikipedia omits "U.S." when the page itself makes the office
     obvious) matched none of `classify()`'s office_level regexes, misclassifying **107 of
     2,848 "incumbent"-descriptor rows** as level 0/unknown; also missed leadership-title
     phrasing ("Majority Leader of the United States House of Representatives" — Eric
     Cantor). Fixed by giving `classify()` the page's office as context.
Both fixed in `fetch_candidate_bios.py`; the already-committed Senate/Governor/2026 data was
re-scraped with both fixes (old version archived per the project's archive-don't-delete
convention: `archive/candidate_bios_<timestamp>_pre-<fix>.csv`). **Re-run
`check_officeholder.py` after any future change to this scraper and read the printed
DISAGREEMENT list, not just the pass/fail line — that's how both of these bugs were found
even though the battery numerically "passed" (96%/consistency ≥85%) both times.**

**General-model extension (CONCLUDED 2026-07-24 — not shipped; full arc in HANDOFF.md):**
the existing bio scrape only covered Senate/Governor 2018-2024 (+ House 2026 only), and —
found after the user challenged a premature "redundant with polls" ablation conclusion —
its target list was SYSTEMATICALLY biased: derived from primary-POLL pages, it excluded
uncontested/safe-seat incumbent races entirely (36% of 2024 Senate races had no target;
Whitehouse/Cantwell/Klobuchar were unreachable). After rebuilding targets from the results
files + House extension (`fetch_house_candidate_bios_hist.py`) + two more subsection-
heading parser fixes ("Nominee", "Advanced to general"), coverage reached **58.1% of the
general model's 14-cycle candidate table (67.7% among winners)**. The re-ablation on fixed
coverage was MIXED, and the feature was NOT shipped: win-model calibration flipped positive
(AUC/AUC-PR/KS/Brier) but race-acc stayed −0.005 with the 2020/2024 folds regressing (a
matched-races-only split ruled out NaN dilution); the margin model was uniformly worse in
all four eval cycles. `feature_list(candidate_bios=True)` + `load_candidate_bios()` remain
in features.py as opt-in machinery. Same conclusion shape as poll_adj and
primary_margin/primary_uncontested (also built + ablated out, 2026-07-22/23): for the
GENERAL election, poll-based features already carry nearly all of this signal — in the
PRIMARY model, where polling is weak, the same feature earns its keep (see the overfit-trim
section above). METHODOLOGICAL NOTE worth keeping: the first (null) ablation was run on
biased coverage and nearly closed the question wrongly — measure a feature's COVERAGE
STRUCTURE (who's missing and why), not just its coverage rate, before trusting an ablation.

**Leak-free as-of-year office-level table (2026-07-25/26, user request).**
IMPORTANT FOR ANY FUTURE AGENT — READ THIS BEFORE "FIXING" A CANDIDATE'S office_level:
**`bio_office_level` is INTENTIONALLY year-varying. The same candidate SHOULD read
different levels in different election years. That is not a bug — it is the whole design.**
The rule is: `office_level` = the highest office the candidate held **STRICTLY BEFORE that
election year** (no look-ahead). So a first-time 2018 candidate reads 0 in 2018 even if they
later became a Senator, and reads 4 only from the cycle after they took federal office.
Verified spot-checks (these are the CORRECT, expected values — do not "correct" them):
  - Abigail Spanberger: 2018 → 0 (newcomer), 2020/2022/2024 → 4 (was a US Rep by then)
  - James Lankford: 2010 House → 0 (first run), 2012+ House/Senate/Gov → 4
  - Aaron Bean: 2016 → 2 (state senator then); 2024 → 4 (US Rep by then)

This replaced the old `combine_candidate_bios.py` merge, which stamped each person's FROZEN
peak office_level onto ALL their rows — a genuine look-ahead leak (a 2010 first-timer who
later reached the Senate would have wrongly read 4 in 2010). `combine_candidate_bios.py` is
DEPRECATED (docstring says so; kept, not deleted, per the archive rule). The authoritative
builder is now **`build_office_level_table.py`** — run it after ANY bio-source change:
    py -X utf8 build_office_level_table.py     # -> data/candidate_bios.csv (rebuilt fresh)
Two source kinds, both contributing an as-of-year level:
  - WIKIPEDIA (`candidate_bios_{senate,governor,house}.csv`): already contemporaneous — each
    row was scraped from THAT YEAR's own race page, so its office_level is used as-is and is
    PREFERRED on any key overlap. (Caveat: a race page occasionally omits a candidate's prior
    office — e.g. Bean's 2022 page reads 0 despite 2016 reading 2. That's a Wikipedia
    source-completeness gap, not a builder bug; flagged for later cleanup, not overridden.)
  - BALLOTPEDIA (`candidate_bios_ballotpedia.csv`, gap-filler only): person-level but carries
    per-office TENURE DATES in an `offices_json` column (e.g.
    `[["U.S. House VA 7",2019,2025],["Governor of Virginia",2026,null]]`). The builder computes
    the as-of-year level = max office-level among offices whose tenure STARTED before `year`.
    This is what makes even Ballotpedia rows time-varying and leak-free instead of a frozen peak.
Coverage after this pass: **68.8% of winner-rows** (1,357/1,971), up from 67.7%. It is
very uneven by era — **2012–2024 ≈ 88–100%**, **1998–2010 ≈ 32–48%** (older Wikipedia race
pages are far sparser). The Ballotpedia scrape captured 21 of 91 targeted 2012+ uncovered
winners before hitting a persistent multi-hour IP block; **59 winners remain uncovered** and
the scrape is resumable (`py -X utf8 fetch_candidate_bios_ballotpedia.py --winners-only`
skips done rows; progress is saved and not poisoned). Two crash/schema fixes shipped with
this: (1) `build_office_level_table.py` now normalizes the uncovered-list's `"S"` statewide
district placeholder to `""` for Senate/Governor (matches Wikipedia + production key);
(2) `features.load_candidate_bios` now uses the crash-safe `dist_str()` helper instead of a
raw `int(district)` that blew up on any non-numeric district value. **Status unchanged:
bio_office_level is still opt-in machinery, NOT shipped to the general model** (the mixed
ablation above still stands); this pass improved the data's correctness (leak-free) and
coverage so a future re-ablation runs on clean inputs. Re-ablate on this table BEFORE
shipping — do not assume the earlier verdict transfers to the corrected data.

**Headline (expanding-window, results labels, +bio_office_level).** Report the STABLE
metrics first — they hold across both eval cycles: **AUC-PR .966, Brier .024** (vs
polls-only .924/.045). Race-winner accuracy is .929 mean but is cycle-dependent by a few
points on ~50 races/cycle (2022 .939, 2024 .919), so it's the secondary number, not the
headline. **2026 out-of-sample backtest** (84 contested decided primaries vs scraped
actual winners): **picks 86.9%, AUC .964, Brier .048** (poll-leader baseline 67.9%).
High-confidence misses concentrate in HOUSE races — the office the training set lacks.

**Features (features_primary.py):** within-FIELD poll structure (plain means, no weighting):
poll_avg/last/last30/std, poll_share, poll_lead, momentum, undecided, n_cands, field
dynamics (lead changes, margin trajectory), days-to-primary recency cuts; fund_receipts_ln +
fund_share recomputed WITHIN the party field (FEC); is_defending_party (races.csv
incumbent_party — true candidate incumbency is not derivable from committed data),
is_pres_party, office/party dummies. NO natl_env / bias priors / house effects (no partisan
channel within a party). Macro is available via `build_macro_asof(primary_date)` (windows
end at the last month published before THAT primary — the generalized Sep-30 rule) but the
artifact ships WITHOUT macro: the training set is a few hundred races and the with-macro
ablation is re-measured on every training run (primary_model.py prints it).

**Validation:** tune = small LOCO grid over cycles ≤2020; honest eval = EXPANDING-WINDOW on
cycles ≥2022 (train strictly before the test cycle), vs the poll-leader baseline. Same
scheme as the general model. Expect wider error bars than the general model: primary
polling is structurally worse (late deciders, name recognition) and n is small — the page
explainer says so.

**Artifacts:** data/primary_model_xgb.json + primary_model_features.json (trained on all
labeled cycles). primary_model.py is a SCRIPT (runs in minutes), not a notebook.
