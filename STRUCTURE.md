# Repo layout

Reorganised 2026-08-02. Before, 36 scripts sat in the repo root; the split separates
**poll-based models** from **fundamentals (no-polling) models**, and separates both from the
data pipeline and one-off tools.

```
.
├── paths.py                  THE path module - every script resolves data/ and the sibling
│                             polling-agg repo from the REPO ROOT via this, not from its own
│                             location. Import it before anything else in a subfolder script.
│
├── features.py               Shared feature builders. features.py = general/margin,
├── features_primary.py       features_primary.py = primary. NEVER-FORK: training and predict
├── cycles.py                 both import these, so they cannot drift apart.
├── macro_features.py
├── candidate_history.py
│
├── predict.py                ── ENTRYPOINTS (stay at the root; CI calls them by name) ──
├── predict_margin.py         Live 2026 predictions from the raw poll feed.
├── predict_primary.py
├── predict_primary_margin.py
├── explain_2026.py           SHAP explanations powering the dashboard's Explain modal.
├── explain_primary.py
├── refresh_dashboard.py      Orchestrator: feeds -> predict -> explain -> copy to polling-agg.
│                             THE thing CI runs (`python -u refresh_dashboard.py --no-feeds`).
│
├── models/
│   ├── poll/                 Models that USE polling (production).
│   │   ├── model.ipynb           general win model      -> data/model_xgb.json
│   │   ├── margin_model.ipynb    general margin model   -> data/margin_model_xgb.json
│   │   ├── primary_model.py      primary nominee ranker -> data/primary_model_xgb.json
│   │   ├── primary_margin_model.py  primary margin      -> data/primary_margin_model_xgb.json
│   │   └── *_feature_importance.csv   gain tables, written beside the model that made them
│   └── fundamentals/         Models that use NO race polling (reference priors, not shipped).
│       └── fundamentals_model.py -> data/fundamentals_model_{general,primary}*.json
│
├── pipeline/
│   ├── fetch/                Scrapers / API pulls (network). fetch_*.py
│   └── build/                Dataset assembly from fetched data.
│       ├── build_dataset.ipynb          -> polls_long_with_results.csv (gitignored, 15MB)
│       ├── build_primary_dataset.py     -> data/primary_polls_long.csv
│       ├── build_office_level_table.py  -> data/candidate_bios.csv
│       ├── combine_candidate_bios.py    (DEPRECATED - superseded by build_office_level_table)
│       └── freeze_2026_dataset.py
│
├── tools/                    One-off repairs and audits, not part of any pipeline.
│   ├── scripts_rekey_cand_key.py          re-derive cached cand_key after a norm_name change
│   ├── scripts_fix_future_office_level.py strip future-tense office leakage from bios
│   ├── check_candidate_history.py
│   ├── check_officeholder.py
│   └── measure_office_coverage.py
│
├── analysis/                 Investigations. Notebook + a .py module holding its logic.
│   ├── poll_volume_breakpoint.{ipynb,py}  where thin polling breaks each model
│   ├── fundamentals_vs_polls_thin.py      head-to-head: do fundamentals help thin races? (no)
│   ├── fundamentals_on_unpolled.py        does the no-poll model work where polls don't exist?
│   └── primary_backtest_2026.ipynb
│
└── data/                     All committed inputs + model artifacts.
```

## Running things after the move

Everything is runnable from the **repo root**:

```bash
py -X utf8 refresh_dashboard.py --no-feeds      # full predict + explain + dashboard copy
py -X utf8 models/poll/primary_model.py         # retrain the primary nominee model
py -X utf8 models/poll/primary_margin_model.py  # retrain the primary MARGIN model
py -X utf8 models/fundamentals/fundamentals_model.py
py -X utf8 pipeline/build/build_primary_dataset.py
py -X utf8 tools/scripts_rekey_cand_key.py      # dry run; --apply to write
py -X utf8 -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=5400 models/poll/model.ipynb
```

## How paths work now

The old idiom — `HERE = os.path.dirname(os.path.abspath(__file__))`, then `data/` relative to
it — was correct only while every file sat in the root. It is now:

```python
import os as _os, sys as _sys
_sys.path.insert(0, <dirname x N to reach the repo root>)
from paths import ROOT, AGG
```

`paths.py` also appends `pipeline/fetch`, `pipeline/build`, `models/*` and `tools` to
`sys.path`, so cross-script imports (`import fetch_candidate_bios`, `from
build_primary_dataset import ...`) keep working by bare module name exactly as before —
without turning every folder into a package.

**Notebooks** carry a prelude cell that walks up to the directory containing `paths.py` and
`chdir`s there, so their bare relative paths (`data/...`) resolve whether Jupyter was started
in the notebook's folder or at the root. It is idempotent — nbconvert rewrites the notebook on
execution and must not double-apply it.

## What deliberately did NOT move

- **The five predict/explain scripts and `refresh_dashboard.py`** stay at the root. The
  polling-agg CI workflow (`.github/workflows/model-refresh.yml`) runs
  `python -u refresh_dashboard.py` with `working-directory: 'Polling prediction model'` and
  copies `predictions_2026.csv` / `primary_predictions_2026.csv` from the root by literal
  path. Moving them would mean editing a workflow in the *other* repo that only runs on a
  schedule — a failure you would not see until 13:15 UTC.
- **Prediction outputs** (`predictions_2026.csv`, `primary_predictions_2026.csv`,
  `margin_predictions_2026.csv`, and the `*_meta.json` sidecars) still land in the root, for
  the same reason.
- **The shared feature modules** stay at the root so `import features` works everywhere
  without ceremony.
