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

**Headline (expanding-window, results-based labels):** AUC .971 / AUC-PR .923 / Brier .046 /
race-acc .910 vs poll-leader .723. **2026 out-of-sample backtest** (84 contested decided
primaries vs scraped actual winners): picks 85.7% vs poll-leader 67.9%, AUC .962.
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
