# Repo layout

Reorganised **2026-08-08** (second pass). The 2026-08-02 pass moved 36 scripts out of the
root into `models/`, `pipeline/`, `tools/` and `analysis/`; this pass finished the job by
moving the last loose files out too, leaving **4 files in the repo root**.

| root file | why it is allowed to stay |
|---|---|
| `README.md` | entry point |
| `paths.py` | THE path module — must sit at the root, since everything resolves *from* it |
| `requirements.txt` | dependency pin; CI's pip cache keys on this literal path |
| `refresh_dashboard.py` | **compatibility shim only** (3 lines) → `src/refresh_dashboard.py` |
| `polls_long_with_results.csv` | the committed 14MB training file (see note below) |

Everything else now lives in a folder. Run commands are unchanged in spirit — still from the
repo root — but the entrypoints are spelled `src/…`:

## Every folder has a README

Added 2026-08-08. **The folder's README is the authoritative description of what is in it** —
this file is the map, those are the detail. Start with the one nearest the code you're
touching.

| folder | what's in it |
|---|---|
| [`../src/`](../src/README.md) | all first-party Python — entrypoints + shared feature modules |
| [`../data/`](../data/README.md) | every committed input + trained model artifact |
| [`../outputs/`](../outputs/README.md) | generated predictions (gitignored, regenerated every run) |
| [`../models/`](../models/README.md) | training code — [`poll/`](../models/poll/README.md) (4 shipped) · [`fundamentals/`](../models/fundamentals/README.md) (not shipped) |
| [`../pipeline/`](../pipeline/README.md) | [`fetch/`](../pipeline/fetch/README.md) (the only network code) · [`build/`](../pipeline/build/README.md) (offline assembly) |
| [`../tools/`](../tools/README.md) | audits, fact-checks, one-off repairs |
| [`../analysis/`](../analysis/README.md) | investigations + [`worklists/`](../analysis/worklists/README.md) |
| [`../archive/`](../archive/README.md) | superseded data snapshots (archive-don't-delete) |
| [`../data_samples/`](../data_samples/README.md) | tiny committed extracts, for documentation |
| [`../logs/`](../logs/README.md) | scraper run-logs (gitignored as a directory) |
| `docs/` | this folder — see [README.md](README.md) for the reading order |

**A README is only useful if it survives a fresh clone.** `data/`, `outputs/`, `logs/` and
`archive/` ignore their contents wholesale, so each needed an explicit un-ignore. Note the
idiom this forces: ignore `folder/*`, never `folder/`, because **git never looks inside an
excluded directory** — a `!folder/README.md` under a bare `folder/` rule silently does
nothing. `logs/scrape/` needed its parent re-included first for the same reason.

```
.
├── paths.py                  THE path module - every script resolves data/, outputs/ and the
│                             sibling polling-agg repo from the REPO ROOT via this, not from
│                             its own location. STAYS AT THE ROOT (everything resolves FROM it).
│                             Import it before anything else in a subfolder script.
│
├── refresh_dashboard.py      3-line COMPATIBILITY SHIM -> src/refresh_dashboard.py. Exists so
│                             an old polling-agg checkout calling the root path still works
│                             instead of failing at 13:15 UTC. Contains no logic.
│
├── README.md                 The only doc at the root; the rest are in docs/.
├── requirements.txt          CI's pip cache keys on this literal path.
├── polls_long_with_results.csv   Committed 14MB training file (force-added past `*.csv`).
│
├── src/                      ALL first-party Python. Run from the repo root: `py src/predict.py`.
│   ├── features.py           Shared feature builders. features.py = general/margin,
│   ├── features_primary.py   features_primary.py = primary. NEVER-FORK: training and predict
│   ├── cycles.py             both import these, so they cannot drift apart.
│   ├── macro_features.py
│   ├── candidate_history.py
│   │
│   ├── predict.py            ── ENTRYPOINTS ── live 2026 predictions from the raw poll feed.
│   ├── predict_margin.py
│   ├── predict_primary.py
│   ├── predict_primary_margin.py
│   ├── explain_2026.py       SHAP explanations powering the dashboard's Explain modal.
│   ├── explain_primary.py
│   └── refresh_dashboard.py  Orchestrator: feeds -> predict -> explain -> copy to polling-agg.
│                             THE thing CI runs (`python -u src/refresh_dashboard.py --no-feeds`).
│
├── outputs/                  GENERATED predictions + meta sidecars + SHAP JSON. Gitignored in
│                             full; recreated by any refresh run. Written via `paths.out()`,
│                             never by a hand-spelled path. polling-agg reads these (with a
│                             legacy root fallback) and commits its OWN copy under
│                             data/processed/ - that copy is what the dashboard serves.
│
├── docs/                     Every deep doc except README.md (AGENTS, CONCERNS, HANDOFF,
│                             METHODOLOGY, DATA_SOURCES, DATA_DICTIONARY, MISSINGNESS,
│                             STRUCTURE - this file).
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
│       ├── build_dataset.ipynb          -> polls_long_with_results.csv (~14MB, COMMITTED)
│       ├── build_primary_dataset.py     -> data/primary_polls_long.csv
│       ├── build_office_level_table.py  -> data/candidate_bios.csv
│       ├── combine_candidate_bios.py    DEPRECATED 2026-07-25 - REFUSES TO RUN. It writes the
│       │                                same data/candidate_bios.csv with a frozen (leaky)
│       │                                office_level. Use build_office_level_table.py.
│       └── freeze_2026_dataset.py
│
├── tools/                    One-off repairs and audits, not part of any pipeline.
│   ├── scripts_rekey_cand_key.py          re-derive cached cand_key after a norm_name change
│   ├── scripts_fix_future_office_level.py strip future-tense office leakage from bios
│   ├── build_missingness_report.py        regenerates MISSINGNESS_REPORT.md from current data
│   ├── check_candidate_history.py
│   ├── check_officeholder.py
│   └── measure_office_coverage.py
│
├── analysis/                 Investigations. Notebook + a .py module holding its logic.
│   ├── poll_volume_breakpoint.{ipynb,py}  where thin polling breaks each model
│   ├── fundamentals_vs_polls_thin.py      head-to-head: do fundamentals help thin races? (no)
│   ├── fundamentals_on_unpolled.py        does the no-poll model work where polls don't exist?
│   ├── primary_backtest_2026.ipynb
│   └── worklists/            Point-in-time audit outputs. NOTHING reads these (see its README).
│
├── logs/                     Scraper run-logs. Gitignored except its README. Was 20 loose
│                             .txt files in the repo root until 2026-08-08.
│
└── data/                     All committed inputs + model artifacts.
```

## Running things after the move

Everything is runnable from the **repo root**:

```bash
py -X utf8 src/refresh_dashboard.py --no-feeds   # full predict + explain + dashboard copy
py -X utf8 src/predict.py                        # one model at a time, if you prefer
py -X utf8 refresh_dashboard.py --no-feeds       # the root shim - identical behaviour
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

## The cross-repo coupling (read before moving anything else)

The entrypoints and prediction outputs *were* pinned to the repo root by
polling-agg's `.github/workflows/model-refresh.yml`, which runs on a schedule. That is why
the 2026-08-02 pass left them there. The 2026-08-08 pass moved them anyway — but only after
making **both sides tolerate either layout**, because a mistake here does not fail in front
of you, it fails at 13:15 UTC and then serves stale predictions until someone notices.

Three things make that safe, and all three must stay true:

1. **`refresh_dashboard.py` shim at the root.** The workflow now calls
   `src/refresh_dashboard.py`, but an older checkout calling the bare root path still works.
2. **polling-agg searches both layouts.** `analysis/model_compare{,_primary}.py` look for the
   model repo's modules in `src/` then the root, and its predictions in `outputs/` then the
   root. This one bit us during the move: `from features import norm_name` silently fell back
   to a local mirror and *disabled its own drift-assert* — a failure that printed one line of
   "note:" and nothing else.
3. **The workflow's push-retry `cp` tries `outputs/` then the root.**

If you ever move these again, update all three together — and grep the *other* repo, not just
this one.

- **The shared feature modules** live in `src/`. `import features` still works everywhere
  because `paths.py` puts `src/` on `sys.path`; scripts inside `src/` carry a 2-line prelude
  that puts the repo ROOT on `sys.path` first so they can `import paths` at all.

## What may live in the repo root (added 2026-08-08)

The 2026-08-02 move cleared the root of scripts, but loose *files* kept accumulating there —
by 2026-08-08 there were 20 scraper logs and 3 orphaned CSVs, and the logs were half-committed
purely by accident of naming. The root is now a closed list. **If a new file does not fit one
of these five categories, it belongs in a subfolder:**

| allowed in root | why |
|---|---|
| `README.md` | entry point; every other doc lives in `docs/` |
| `paths.py` | everything resolves *from* it, so it cannot itself be resolved via itself |
| `requirements.txt` | CI's pip cache keys on this literal path |
| `refresh_dashboard.py` | the 3-line shim, and **only** because a scheduled cross-repo workflow may call the old path |
| `polls_long_with_results.csv` | the committed 14MB training file |

That is the whole list — five entries. Python goes in `src/`, generated predictions in
`outputs/`, docs in `docs/`.

Everything else has a home: scraper logs → `logs/`, one-off audit outputs → `analysis/worklists/`,
data → `data/`, repairs/audits → `tools/`. Two rules learned the hard way:

1. **Prefer a directory-level ignore (`logs/*`) over name patterns.** The three name-specific
   log patterns matched 10 of 20 files and silently committed the rest.
2. **A file with no consumer is a documentation problem, not a file problem.** Before moving
   or deleting one, grep the repo for its name; if nothing reads it, say so in a README next
   to it (see `analysis/worklists/README.md`) rather than leaving the next reader to guess.
3. **Never build a data path from `__file__`.** `os.path.dirname(os.path.abspath(__file__))`
   + `"data"` is correct only while the file sits in the repo root — move it one level and it
   silently points at `src/data/`, which either crashes (if reading) or writes to the wrong
   place (worse). Use `paths.data(...)` / `paths.out(...)`. Moving five files on 2026-08-08
   surfaced four live instances of this, including `features.DATA_DIR`, which every model
   reads through.
4. **Do not `from paths import out`.** `out` is one of the most common local variable names
   in this repo (`out = pd.DataFrame(...)` appears in a dozen functions) and a bare import
   gets shadowed — `TypeError: 'DataFrame' object is not callable`, at the very end of a long
   run. Import the module: `import paths as _paths`, then `_paths.out(...)`.
