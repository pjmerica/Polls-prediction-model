# Instructions for a new agent (or developer) working on this repo

Read this first. It explains what the project is, how the pieces fit, the hard-won
rules, and the known traps. If you only read one file, read this one.

## What this project is (one paragraph)
We predict whether a U.S. candidate wins their election (Senate / House / Governor),
from polling plus context. There is **no single dataset** of "polls + who won," so we
assemble one: download polls and results from public sources, join them, engineer
features, and train an XGBoost classifier — validated honestly with leave-one-cycle-out
cross-validation. **The headline finding: a one-variable poll model is the ceiling** —
the fancy model matches but does not beat "whoever leads the polls wins" on win/lose.
We keep building features anyway because (a) calibration/probabilities improve and (b)
the infrastructure is the basis for a future *margin* model where features can help.

## The pipeline (run order)
```
1. build_dataset.ipynb   -> joins polls (1998-2016 raw_polls + 2018-24 historical, all committed)
                            with results -> polls_long_with_results.csv (one row per poll-candidate)
2. fetch_approval.py     -> ONE-TIME Gallup approval pull (UCSB) -> data/approval_monthly.csv
   fetch_macro.py        -> ONE-TIME econ pull (DBnomics) -> data/macro_monthly.csv
                            (both committed; re-run only to EXTEND to new months)
3. model.ipynb           -> features via features.py, tunes on 1998-2016, evaluates on
                            2018-2024, saves data/model_xgb.json + data/model_features.json
4. predict.py            -> win probabilities for FUTURE races from the polling-agg repo's
                            raw poll feed (..\Polling Agg\...\data\raw\)
5. margin_model.ipynb    -> SEPARATE margin model (predicts victory margin in pct points);
   predict_margin.py        own artifact data/margin_model_*.json. Keep it fully separate
                            from the win/lose model (user requirement).
```
Shared modules: **cycles.py** (all cycle constants — extend a cycle by editing ONE file),
**features.py** (the feature builder used by BOTH model.ipynb and predict.py — never fork it),
**macro_features.py** (per-cycle macro stats).

## Files
| file | what it does |
|---|---|
| `build_dataset.ipynb` | Downloads & joins polls + election results -> `polls_long_with_results.csv`. |
| `fetch_macro.py` | One-time DBnomics pull -> `data/macro_monthly.csv` (monthly, back to 1947). Run once; static. |
| `macro_features.py` | Reads `data/macro_monthly.csv`, builds per-cycle macro features (no network). |
| `model.ipynb` | The model: feature engineering, tuning, CV, benchmark. |
| `data/` | Cached downloads (gitignored) **except** `macro_monthly.csv` (committed). |
| `data_samples/` | Tiny committed samples of the result files (so schema is visible without downloading). |
| `CONCERNS.md` | **Ranked audit of risks + the improvement queue** (2026-07-05). Read before extending the model — includes the "no 538 in production" constraint. |
| `METHODOLOGY.md` | **Exact time windows** for every feature (per-cycle macro windows, poll recency, etc.). |
| `DATA_SOURCES.md` | Every data source, exact URL, and how it was found. |
| `DATA_DICTIONARY.md` | Every column/feature explained (layman + technical). |
| `MISSINGNESS_REPORT.md` | Per-column missingness for both datasets. |

## THE RULES (learned the hard way — follow them)

1. **Re-run the WHOLE `model.ipynb` (including the grid search) whenever you change features.**
   Hyperparameters tuned for one feature set can make a *new* feature set look worse than
   it is. We saw a macro-feature "regression" that was purely stale hyperparameters; re-tuning
   fixed it. The notebook searches params live in section 5 — let it.

2. **Let regularization drop useless features; don't hand-curate.** The grid search picks
   heavy regularization (low `colsample_bytree`, high `reg_lambda`) which zeroes out
   non-predictive columns. So *add* candidate features and tune, rather than pre-excluding.

3. **Never use `vote_pct` / `race_winning_pct` as features.** They're the outcome. The label
   is `won`.

4. **Validate by year, never randomly.** Train on whole cycles, test on a held-out cycle
   (leave-one-cycle-out). Random splits leak the future and inflate scores. Hyperparameters
   are tuned on the 1998-2016 cycles ONLY; 2018-2024 is the untouched honest eval set.

5. **Static data is pulled once and committed.** Past months/results never change. ALL model
   inputs are committed; no run touches the network. Re-pull only to *extend* to new months.

6. **No 538-only inputs, no poll weighting.** Production future = raw polls + econ data only
   (538 is dead). Pollster grades/pollscore/partisan-lean are banned; all poll aggregates are
   plain averages (see features.py header + CONCERNS.md). Feature changes go in features.py so
   model.ipynb and predict.py can never drift apart.

## TRAPS that have bitten us (don't repeat)
- **Run nbconvert on `model.ipynb` ONE AT A TIME.** Launching several concurrent
  `nbconvert --execute --inplace` runs makes them race and overwrite each other's outputs,
  so you read stale results. Wait for one to finish before starting another.
- **Clear `__pycache__` when a helper module changes** (`rm -rf __pycache__` or run with
  `PYTHONDONTWRITEBYTECODE=1`). A stale `.pyc` silently ran old `macro_features` code once.
- **If a notebook cell references a feature column by name, update it when feature names change.**
  A leftover `approval_eve` reference (renamed to `approval_yr_eve`) made the whole notebook
  error out, and nbconvert left the *previous* run's outputs in place — looked like the change
  "did nothing." Always confirm the printed feature count matches `macro_features.py`'s direct output.
- **The 'party' column in the results files is 100% null** — the real party is in `ballot_party`.
- **`district` is empty for Senate/Governor** (statewide) — that's correct, not missing data.
- **FRED's `fredgraph.csv` host was unreachable from both the sandbox AND the user's machine**, so
  `fetch_macro.py` now uses **DBnomics** (free, no key) pulling the upstream agencies (BLS/EIA/Fed).
  DBnomics does *not* mirror FRED itself — use the agency series codes (already wired in the script).

## Environment
```
pip install pandas numpy requests xgboost scikit-learn jupyter matplotlib openpyxl shap
```
Polls/results download on first `build_dataset.ipynb` run. `fetch_macro.py` pulls from DBnomics (no key). The macro CSV is already committed, so you usually don't need to re-run it.

## Current honest performance (LOCO on 2018–2024; tuned on 1998–2016 only)
- XGBoost: AUC 0.969, Brier 0.069, race-winner accuracy 0.863
- Poll-only baseline (softmax of poll_avg): AUC 0.966, Brier 0.071, race-acc 0.868
- Model now edges the baseline on AUC/Brier (calibration); baseline still a hair ahead on
  picking winners. Coverage floor is **1998** (frozen 538 raw_polls file).

## If you're extending this
- **Most promising direction:** switch the target from win/lose to **margin** or
  **overperformance vs polls** — polls don't already contain that answer, so features have room.
- **House is the weak spot** (district partisan-lean only ~41% covered); a time-varying
  district PVI would help there.

### TODO: generic-ballot recency cuts (planned, not yet built)
We want monthly generic-ballot D−R over time so we can add 3/6/12-month avg/max/trend cuts
(like the macro features). Current state: `natl_env` in `model.ipynb` is just 4 hardcoded
per-cycle eve values (2018 +7.8, 2020 +7.5, 2022 +0.7, 2024 +0.1) — the daily 2018–2024
series lived only on the Internet Archive, which is now unreachable.
**Plan / sources for a `fetch_generic_ballot.py` (pull once → `data/generic_ballot_monthly.csv`
→ commit; only fetch live in production):**
- **RealClearPolling** `realclearpolling.com/polls/state-of-the-union/generic-congressional-vote`
  (and per-cycle equivalents) — most consistent structured source.
- **Wikipedia** "Generic ballot"/"Opinion polling for the YYYY U.S. House elections" — has
  per-poll tables but the exact article title/table index varies by cycle (the 2022 *main*
  House article does NOT contain the poll list; the dedicated polling article title still needs
  to be pinned down). `pd.read_html(io.StringIO(requests.get(url).text))` is the parse path.
- **GitHub `fivethirtyeight/data/congress-generic-ballot/generic_topline_historical.csv`**
  exists but only covers **1995–2016** (too old for our 2018+ cycles).
Once the monthly CSV exists, mirror the macro recency-cut logic in `macro_features.py`.
