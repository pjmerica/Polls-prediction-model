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
1. build_dataset.ipynb   -> downloads polls + results, joins them
                            -> writes polls_long_with_results.csv  (one row per poll-candidate)
2. fetch_macro.py        -> ONE-TIME pull of monthly economic data (DBnomics)
                            -> writes data/macro_monthly.csv  (static, committed, never re-pull)
3. model.ipynb           -> reads both files, builds features, tunes + trains XGBoost,
                            cross-validates, benchmarks vs polls-only
```

## Files
| file | what it does |
|---|---|
| `build_dataset.ipynb` | Downloads & joins polls + election results -> `polls_long_with_results.csv`. |
| `fetch_macro.py` | One-time DBnomics pull -> `data/macro_monthly.csv` (monthly, back to 1947). Run once; static. |
| `macro_features.py` | Reads `data/macro_monthly.csv`, builds per-cycle macro features (no network). |
| `model.ipynb` | The model: feature engineering, tuning, CV, benchmark. |
| `data/` | Cached downloads (gitignored) **except** `macro_monthly.csv` (committed). |
| `data_samples/` | Tiny committed samples of the result files (so schema is visible without downloading). |
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
   (leave-one-cycle-out). Random splits leak the future and inflate scores.

5. **Static data is pulled once and committed.** Past months/results never change. `fetch_macro.py`
   and the committed CSVs exist so we don't re-pull on every change. Only re-pull to *extend* to
   a new cycle.

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
pip install pandas numpy requests xgboost scikit-learn jupyter matplotlib openpyxl
```
Polls/results download on first `build_dataset.ipynb` run. `fetch_macro.py` pulls from DBnomics (no key). The macro CSV is already committed, so you usually don't need to re-run it.

## Current honest performance (leave-one-cycle-out CV, 2018–2024)
- XGBoost: AUC ~0.96, Brier ~0.08, race-winner accuracy ~0.86
- Poll-only baseline: AUC ~0.965, Brier ~0.071, race-acc ~0.87  ← still ahead
- Coverage floor is **2018** (no machine-readable downballot polls before then).

## If you're extending this
- **Most promising direction:** switch the target from win/lose to **margin** or
  **overperformance vs polls** — polls don't already contain that answer, so features have room.
- **House is the weak spot** (district partisan-lean only ~41% covered); a time-varying
  district PVI would help there.
