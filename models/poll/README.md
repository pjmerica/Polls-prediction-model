# `models/poll/` — the four production models

All four use race polling and all four are shipped to the dashboard. Numbers below are read
straight out of the committed `../../data/*_features.json` artifacts (2026-08-08), so they
match the model currently in production.

| file | model | artifact | headline |
|---|---|---|---|
| `model.ipynb` | **Win probability** | `data/model_xgb.json` | AUC .966 / Brier .072 / race-acc .864 — **ties** the poll-leader baseline |
| `margin_model.ipynb` | **Margin of victory** | `data/margin_model_xgb.json` | MAE ~6.5–7.2 vs ~7.5 calibrated-poll baseline — **beats** polls |
| `primary_model.py` | **Primary nominee** | `data/primary_model_xgb.json` | race-acc .923 (2022) / .892 (2024) vs poll-leader .769 / .676 |
| `primary_margin_model.py` | **Primary margin** | `data/primary_margin_model_xgb.json` | MAE 14.69 (2022) / 18.65 (2024) vs poll 16.33 / 23.18 |

Notebooks are the two general models; the two primary models are plain scripts that run in
minutes. Each writes its gain table beside itself as `*_feature_importance.csv`.

```bash
py -X utf8 models/poll/primary_model.py
py -X utf8 models/poll/primary_margin_model.py
py -X utf8 -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=5400 models/poll/model.ipynb
```

## Two things about these models that surprise people

**The win model does not beat the polls, and that is the honest finding.** For calling a
winner, "whoever leads the polls" is already near the ceiling. What the model adds is a
*calibrated probability* rather than a binary read. The margin models are where the extra
features actually earn their keep. Do not "fix" the win model to beat the baseline — that
result is real and well-known about U.S. elections.

**The primary ranker's scores are not probabilities.** `primary_model.py` is an
`xgbranker` (`rank:pairwise`); raw scores are unbounded and comparable **only within a race**.
They become probabilities through a within-race softmax at `softmax_temp = 0.5`, and that one
number feeds *both* the dashboard and the Explain modal — if those two disagree, the
temperature is out of sync.

> Tune the temperature by **Brier, not accuracy**. Softmax is monotonic, so it cannot change
> the ranking and accuracy is flat across every value — it was once hardcoded at the single
> worst setting for months because accuracy looked fine. The full grid is stored in
> `data/primary_model_features.json` under `softmax_temp_grid`.

## Honest reading of the primary margin model

It beats the calibrated-poll baseline on both eval folds now, but on **611 labelled rows over
two eval cycles**. Primary margins are ~2x harder than general ones (target std 40.8). Treat
single-fold movement as noise.

Also primary-specific: `best_other` is computed over the **results** field, not the polled
subset, because the polled subset often omits the actual runner-up — taking the max over
polled candidates would compare the front-runner to the wrong person and inflate accuracy.
Races with one polled candidate (`poll_lead = 0` by construction) are **flagged, not dropped**;
the dashboard greys them with a leading `~`.

## Before you retrain

Read [../README.md](../README.md) — the re-tune-everything rule, the nested validation scheme,
and the never-use-these-as-features list all apply here.
