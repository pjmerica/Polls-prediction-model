# `pipeline/build/` — assembling the training tables (offline)

No network access. Every input is already committed in `data/`, so these are reproducible and
safe to re-run; each rebuilds its output from scratch rather than appending to itself.

| script | writes | notes |
|---|---|---|
| `build_dataset.ipynb` | `polls_long_with_results.csv` (**repo root**, ~14MB) | joins 1998–2016 `raw_polls_538` + 2018–24 historical polls with results. The file is **committed** (force-added past `*.csv`), so a fresh clone does not need this step. |
| `build_primary_dataset.py` | `data/primary_polls_long.csv` | the primary-side equivalent |
| `build_office_level_table.py` | `data/candidate_bios.csv` | **THE** builder of the bio table every consumer reads |
| `freeze_2026_dataset.py` | `data/candidate_table_2026.csv` + `dataset_2026_meta.json` | point-in-time snapshot of the live cycle |
| `combine_candidate_bios.py` | — | **DEPRECATED, refuses to run** |

## `build_office_level_table.py` vs `combine_candidate_bios.py`

Both write `data/candidate_bios.csv`. Only the first is correct.

`combine_candidate_bios.py` stamped each person's **frozen** office level onto all of their
rows — so a first-time candidate who later became a Senator would read level 4 in their early
races. That is a look-ahead leak into a feature used by every model.

`build_office_level_table.py` replaced it (2026-07-25) with a **leak-free, as-of-year**
computation: the highest office held *strictly before* each election year, derived from
Ballotpedia tenure dates.

The deprecated script stayed runnable for eight months and four `fetch_*` docstrings still
pointed at it, so running it would have silently reverted the good table with no error and no
obvious diff. It now hard-exits unless `ALLOW_DEPRECATED_BIO_COMBINE=1`.

> **Corollary:** `bio_office_level` is *intentionally* year-varying. The same person reads a
> lower level in an earlier cycle, and a first-time candidate is 0 even if they later won
> office. Do **not** "correct" these.

## The frozen-snapshot rule

`freeze_2026_dataset.py` records values that were live at the moment it ran — e.g.
`natl_env_used: 5.57` in `dataset_2026_meta.json`, while a later run uses 6.75. That drift is
the point of a snapshot. Leave it alone.

## Notebook mechanics

`build_dataset.ipynb` carries a prelude cell that walks up to the folder containing `paths.py`
and `chdir`s there, so its relative paths resolve whether Jupyter started here or at the root.
It is **idempotent** — nbconvert rewrites the notebook on execution and must not double-apply
it. Run notebooks **one at a time**; parallel nbconvert runs race and overwrite outputs.
