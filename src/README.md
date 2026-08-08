# `src/` — all first-party Python

Everything here is **run from the repo root**, never from inside this folder:

```bash
py -X utf8 src/refresh_dashboard.py --no-feeds   # the whole pipeline, one command
py -X utf8 src/predict.py                        # or one model at a time
```

Moved here from the repo root on 2026-08-08. Before that these 12 files sat loose at the top
level; see [../docs/STRUCTURE.md](../docs/STRUCTURE.md) for the full map.

## Two groups

### Shared feature modules — imported by training AND prediction

| file | what |
|---|---|
| `features.py` | Feature builders for the **general + margin** models. Poll averages, `norm_name` (THE join key), `norm_pollster`, `dist_str`, `is_junk_answer`. |
| `features_primary.py` | The same job for the **two primary** models. |
| `cycles.py` | Cycle constants + `natl_env` (generic-ballot margin per cycle). |
| `macro_features.py` | Economic/approval windows joined onto a cycle. |
| `candidate_history.py` | Prior-candidacy lookups built from the results archives. |

> **NEVER-FORK rule.** Training and `predict*.py` both import these, which is the only reason
> they cannot compute a feature differently. `features.py` and `features_primary.py` are
> *already* two implementations of overlapping ideas — **grep both** when changing a shared
> feature. An unchanged retrain result after a feature edit is a bug signal, not a null result.

### Entrypoints — the scripts that actually get run

| file | writes |
|---|---|
| `predict.py` | `outputs/predictions_2026.csv` (win probabilities) |
| `predict_margin.py` | `outputs/margin_predictions_2026.csv` |
| `predict_primary.py` | `outputs/primary_predictions_2026.csv` |
| `predict_primary_margin.py` | `outputs/primary_margin_predictions_2026.csv` |
| `explain_2026.py` | `outputs/model_explanations_2026.json` (+ a polling-agg copy) |
| `explain_primary.py` | `outputs/primary_explanations_2026.json` (+ a polling-agg copy) |
| `refresh_dashboard.py` | orchestrates all six, then copies into polling-agg and regenerates its compare pages. **This is what CI runs.** |

`predict.py` also owns two filters the others import: the **stale-candidate** filter and
`drop_primary_losers()` (dead-matchup removal).

## Three rules specific to this folder

1. **Never build a data path from `__file__`.** `dirname(abspath(__file__)) + "data"` was
   correct only while these files sat in the repo root — from `src/` it silently resolves to
   `src/data/`. Use `paths.data(...)` and `paths.out(...)`. Moving these files surfaced four
   live instances of this, including `features.DATA_DIR`.
2. **Import `paths` as a module; never `from paths import out`.** `out` is one of the most
   common local variable names in this repo (`out = pd.DataFrame(...)`), so a bare import
   gets shadowed and dies with `TypeError: 'DataFrame' object is not callable` — at the very
   *end* of a long run.
3. **Every file here needs the 2-line `sys.path` prelude** before `import paths`, because
   `paths.py` lives one level up in the repo root.

## Explainer strings are public

The `FRIENDLY` / `DESC` dicts in `explain_2026.py` are rendered in the dashboard's Explain
modal, so they must describe what the code *actually computes*. A wrong description shipped to
128 live race explanations once (`is_incumbent` — see [../docs/CONCERNS.md](../docs/CONCERNS.md) #49).
