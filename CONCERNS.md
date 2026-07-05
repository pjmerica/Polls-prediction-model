# Audit: what worries me, and the improvement plan

Written 2026-07-05 after a full repo audit. Ordered by how much each item worries me.
Ground rule for everything below: **in production there is no FiveThirtyEight.** Future inputs
are (a) raw polls — candidate, party, state/district, pollster, dates, pct, sample size — and
(b) economic data. Anything that only exists inside a 538 file is either leakage or a
train/serve mismatch waiting to happen.

## Fixed in this pass (2026-07-05)

### ✅ BUG: House fundamentals were silently dead (district `'1.0'` vs `'1'` keys)
`polls_long_with_results.csv` round-trips the district column as float, so `model.ipynb` saw
`'1.0'` while every lookup table built from `races.csv`/results (`inc_map`, `dist_lean`,
`margin_map`) used `'1'`. Result: **all 470 House races** had `is_incumbent=0`,
`is_inc_party_race=0`, `prior_margin_cand=NaN`, `lean_cand=NaN` — verified 0% lookup hit rate
before the fix, 100% after. House is exactly where polls are thin and fundamentals matter most.
Fixed with a `_dist_str` normalizer + a hard `assert` in the feature cell so it can't regress
silently. Full notebook re-run (with grid search, per the workflow rule) required and done.

### ✅ Live web fetches on every model run
`model.ipynb` downloaded 3 GitHub CSVs (538 partisan-lean ×2, `races.csv`) on **every** run.
Now cached in `data/` and committed. Also committed: the NYT `*_current.csv` poll files (the
2024 cycle is over — a re-fetch would silently pull 2026 polls into the training set) and the
`res_*.csv` results files. **Every model input is now in git; no run touches the network.**
The bonus: the frozen `races.csv` vintage already contains 2026/2028 incumbency records.

## Open worries, ranked

### 1. 538 pollster ratings are baked into the poll weighting (train/serve skew)
`numeric_grade` enters the poll weight `w`, which feeds `poll_wavg` — and through it
`poll_lead`, `gap_x_recency`, `twoparty_margin_cand`, `poll_wavg_adj`, `poll_share`… i.e.
**every top feature in the importance table**. Future polls carry no 538 grades, so `w`
degrades to recency×sample only, and future features are computed on a systematically
different scale than the training features. `avg_grade`/`avg_pollscore` are also FEATURES
(importance 0.0, harmless but dead).
**Plan:** remove `numeric_grade` from `w`; drop `avg_grade`/`avg_pollscore`/`transparency_score`
from the pipeline entirely; re-tune + full re-run (feature change ⇒ grid search). If pollster
quality is worth recovering, compute our **own** pollster reliability score from historical
poll-vs-result error per pollster (computable from our own committed data, per-cycle, leak-free
— pollster *names* will still exist in future polls).

### 2. There is no prediction path — the whole repo is retrospective
`model.ipynb` only does CV on 2018–2024. Nothing can score a 2026 race. The future-facing
pipeline needs a `predict.py` that builds features from **only** future-available inputs:
raw polls (schema-agnostic loader), `data/macro_monthly.csv` extended to the new cycle,
incumbency (frozen `races.csv` already covers 2026), prior margins (2024 results, committed).
**Plan:** build `predict.py` + a `FUTURE_SCHEMA` contract listing exactly which raw poll
columns are required; refuse to run if a feature would need a 538-only column.

### 3. Cycle-constant features with n=4 cycles
The 112 macro features, `natl_env_cand`, and `PRES_PARTY` are constant within a cycle — the
model sees each of them at exactly **4 distinct values**. Today heavy regularization zeroes
essentially all of them (see `feature_importance.csv`: every macro feature = 0.0), so they are
dead weight rather than damage. But they are a standing memorization hazard: any future tuning
run that lands on weaker regularization can use them to fingerprint the cycle and re-learn
"2018 was a blue wave" instead of economics.
**Plan:** cut to a small theory-driven set (~5–10: approval eve/trend, inflation 6–12mo,
unemployment trend, gas 6mo), always interacted with `is_president_party`. Their honest test is
the future **margin** model, not win/lose. Accept that with 4 cycles these can never be
strongly validated.

### 4. The reported CV numbers are also the tuning target (optimism bias)
The grid search picks the config with the best leave-one-cycle-out AUC over 150 candidates,
and then the same folds (and the 2024 "test" split, which was inside the tuning folds) are
reported as honest performance. The selection maximum of 150 noisy CV scores is biased upward.
**Plan:** nested CV (tune within the 3 training cycles of each fold) for the headline numbers;
or at minimum label the current numbers "model selection score, not unbiased estimate" in the
docs. The poll-softmax baseline is untuned, so the "polls are the ceiling" conclusion survives
this bias — the model's true gap to the baseline is, if anything, *larger* than reported.

### 5. `natl_env` is 4 hardcoded numbers of 538 provenance
Fine as frozen history, but 2026 needs a generic-ballot value from a non-538 source, and a
1-value-per-cycle feature has the same n=4 problem as #3. **Plan:** the AGENTS.md TODO stands —
`fetch_generic_ballot.py` (RealClearPolling / Wikipedia per-poll tables) → monthly CSV →
commit → recency cuts like the macro features. In production, compute it from the raw
generic-ballot polls in whatever poll feed we use.

### 6. The presidential-approval series is hand-typed
`fetch_macro.py`'s `APPROVAL` dict is a hardcoded monthly table ("538/Gallup avg") ending
2024-12, with no scripted provenance. It feeds 16 of the macro features. **Plan:** re-source
from Gallup's published monthly averages (documented URL + parse script), extend past 2024,
commit the CSV like everything else.

### 7. Election results for 2026+ need a new source
`prior_margin_cand`, the `won` label, and incumbency all come from the 538 `election-results`
repo, which is unmaintained. For the 2026 cycle we can still *predict* (priors = committed 2024
results), but to *label* 2026 for retraining we need official returns.
**Plan:** MIT Election Data + Science Lab (medsl) publishes Senate/House/Governor returns; add
a loader once 2026 certifications land. Convert to the existing `res_*.csv` schema.

### 8. Possible double-counting where the NYT and 538 poll files overlap
Dedup keys are `poll_id`/`question_id`, which are only comparable within one source. If the
same 2024 poll appears in both `*_current.csv` (NYT) and `*_historical.csv` (538 snapshot)
with different IDs, it is counted twice — inflating `n_polls` and over-weighting duplicated
polls. **Plan (cheap check):** dedup on `(pollster, end_date, race, candidate, round(pct))`
across sources and diff the row count; if overlap exists, add that as a fallback dedup key in
`build_dataset.ipynb`.

### 9. Smaller code-level items (grab bag)
- `poll_last = gc['pct'].iloc[-1]` — NaT `end_date` rows sort last, so a dateless poll can be
  "the most recent poll". Drop NaT before taking `iloc[-1]`.
- `is_incumbent=0` conflates "open seat / challenger" with "we couldn't match the race" —
  before today's district fix that silently mislabeled the entire House. The new assert guards
  the House aggregate, but a per-race NaN (letting XGBoost route missing) would be cleaner
  than a hard 0.
- `undecided = 100 − Σ poll_wavg` can go negative in crowded races (e.g. CA top-two with two
  Dems polling 50+ each); harmless to XGBoost but nonsense as a quantity.
- Cycle constants are hardcoded in ≥4 places (`CYCLES`, `PRES_PARTY`, `PRIOR_EVE`, `natl_env`,
  `APPROVAL`). Extending to 2026 means touching all of them — centralize in one module when
  adding the first new cycle.

## Improvement queue (agreed direction)
1. **De-538 the feature pipeline** (worry #1): drop grade/pollscore from `w` and FEATURES,
   re-tune, full re-run. This is the prerequisite for any production use.
2. **`predict.py` + input schema contract** (worry #2).
3. **Generic-ballot fetcher** (worry #5, unblocks recency-cut version of `natl_env`).
4. **Margin / overperformance target** — the long-standing "where a model can actually beat
   polls" direction; also where macro features get a fair test.
5. **Nested CV** for honest headline numbers (worry #4).
6. Overlap-dedup check (#8) and the grab-bag fixes (#9) opportunistically.
