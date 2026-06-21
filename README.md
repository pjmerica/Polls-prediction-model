# Polls prediction model

Predicts whether a U.S. candidate **wins their election** — Senate, House, and Governor
races, 2018–present — using polling plus political and economic context.

---

## For a non-technical reader (what is this?)

Every election, pollsters ask voters who they'll support. This project asks: **how well
can you predict the actual winner from those polls, and does adding "context" (the economy,
the president's approval, who's the incumbent) help?**

We gathered every poll we could find since 2018, matched each one to who actually won,
and trained a machine-learning model. The honest headline result:

> **Just betting on whoever is ahead in the polls is already about as good as it gets.**
> Our fancier model ties it but doesn't beat it for calling winners. Where the model *does*
> help is giving a trustworthy **probability** (e.g. "this candidate has a 72% chance"),
> which a simple "who's ahead" rule can't.

That's a real, well-known fact about U.S. elections — the polls already contain most of
the signal. We keep adding features (economy, incumbency, etc.) because they sharpen the
probabilities and set up a future model that predicts *margin of victory*, where there's
more room to add value.

---

## For a data scientist (how it works)

**Task:** binary classification, `won ∈ {0,1}`, one row per candidate per race.
**Model:** XGBoost, hyperparameters grid-searched live.
**Validation:** leave-one-cycle-out CV (train on 3 election cycles, test the held-out 4th,
rotate). Never random splits — that would leak the future.

**Why no single dataset exists:** there's no public file of "polls + who won," so we join two:

| layer | source | notes |
|---|---|---|
| **Polls** | NYT poll CSVs (current) + Internet Archive snapshot of FiveThirtyEight (historical) | 538 was dissolved 3/2025; NYT publishes the same schema. Polls start **2018**. |
| **Results** (who won) | [FiveThirtyEight `election-results` repo](https://github.com/fivethirtyeight/election-results) | per-candidate `winner` flag, all races 1976–2024 |
| **Partisan lean** | 538 `partisan-lean` (state + district) | CPVI-style; district file is 2022 vintage (~41% House coverage) |
| **National environment** | 538 generic ballot (Internet Archive) | per-cycle DEM−REP |
| **Macro / economy** | DBnomics monthly series (one-time pull, see below) | unemployment, inflation (CPI), core CPI, gas, fed funds, U-6, approval — condensed per cycle |

**Headline finding (cross-validated):** a one-variable "poll softmax" baseline
(AUC ≈ 0.965, Brier ≈ 0.071) matches or beats the full 88-feature XGBoost
(AUC ≈ 0.96, Brier ≈ 0.08) on win/lose. Polls are the ceiling. The features improve
calibration and would matter more for a margin model.

📄 **Deep docs:** [AGENTS.md](AGENTS.md) (start here if you're contributing) ·
[METHODOLOGY.md](METHODOLOGY.md) (**exact time windows for every feature**) ·
[DATA_SOURCES.md](DATA_SOURCES.md) (every URL + how found) ·
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) (every variable) ·
[MISSINGNESS_REPORT.md](MISSINGNESS_REPORT.md).

---

## Pipeline (run order)

```
1. build_dataset.ipynb   → downloads polls + results, joins them
                           → polls_long_with_results.csv  (one row per poll-candidate)
2. fetch_macro.py        → ONE-TIME pull of monthly economic data (DBnomics)
                           → data/macro_monthly.csv  (static; committed; never re-pull)
3. model.ipynb           → features + tuning + cross-validation + poll-only benchmark
```

### Macro data is pulled once and committed
Economic history doesn't change retroactively (2018's inflation is fixed forever), so we
pull it **once** and save `data/macro_monthly.csv` — monthly readings back to 1947 for the
core series (CPI, unemployment), kept in full in case older polls surface. The
model then uses, for each election, **that cycle's own window (prior election eve → this
election eve)** — 2018 ← Nov 2016–Nov 2018, 2020 ← Nov 2018–Nov 2020, etc. — and condenses
each indicator into trajectory stats (eve level, mean, max, min, std, trend, 12-month change),
so e.g. `inflation_max` is *that cycle's* peak, not the all-time peak. XGBoost decides which matter.

## Run

```bash
pip install pandas numpy requests xgboost scikit-learn jupyter matplotlib openpyxl shap
```

1. **`build_dataset.ipynb`** — run top to bottom (downloads polls + results, caches to `data/`).
2. **`python fetch_macro.py`** — run **once** on a machine with internet (pulls from DBnomics; creates
   `data/macro_monthly.csv`; commit it). Skip if the CSV is already committed.
3. **`model.ipynb`** — run top to bottom.

> ⚠️ **Workflow rule:** whenever you change the model's feature set, **re-run the entire
> `model.ipynb` end-to-end including the grid search** — never reuse old hyperparameters.
> Params tuned for one feature set can make a new set look worse than it is. Let regularization
> drop non-predictive features rather than hand-curating. (More rules + traps in [AGENTS.md](AGENTS.md).)

## Next steps
- Predict **margin / overperformance vs polls** instead of win/lose — where features can
  actually beat the polls.
- A time-varying **district PVI** to strengthen House (the weakest office).
- Probability **calibration** (isotonic/Platt) for even tighter Brier.
