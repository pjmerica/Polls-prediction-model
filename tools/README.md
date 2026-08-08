# `tools/` — audits, fact-checks, and one-off repairs

Not part of any pipeline. Nothing here runs automatically; each is invoked by hand when the
thing it checks or repairs is in question. Run from the repo root.

## Fact-check batteries (run these after touching what they cover)

| script | checks |
|---|---|
| `check_candidate_history.py` | `src/candidate_history.py` — run after any change to the history logic |
| `check_officeholder.py` | candidate BIO features against known-truth cases |
| `measure_office_coverage.py` | `bio_office_level` coverage of the general-model candidate table; regenerates the gap worklist |

These exist because bio features are scraped from prose and fail *quietly* — a regex that
stops matching produces level 0, which is indistinguishable from "this person held no office."

## Report generators

| script | writes |
|---|---|
| `build_missingness_report.py` | `docs/MISSINGNESS_REPORT.md` — **generated; do not hand-edit** |

Run it after any change to the training data. The report was hand-maintained once and went
stale by ~12,500 rows while still being the document people consulted to decide whether a
feature was safe to use.

## One-shot repairs

| script | does |
|---|---|
| `scripts_rekey_cand_key.py` | recomputes every **precomputed** `cand_key` column after a `norm_name` change |
| `scripts_fix_future_office_level.py` | strips future-tense office leakage out of scraped bios |

> **Always dry-run an in-place repair first.** These scripts rewrite committed CSVs. One of
> them nearly wiped ~600 hand-coded bio rows (sitting members of Congress went 4 → 0 in the
> dry-run output, which is the only reason it was caught). `scripts_rekey_cand_key.py`
> defaults to a dry run and needs `--apply` to write. Keep that pattern for anything new here.

### Why re-keying is needed at all

`features.norm_name` is **the** join key across polls ↔ results ↔ FEC ↔ bios ↔ history, in
both pipelines. 17 committed CSVs cache `cand_key` as a column, so changing `norm_name` makes
every one of them silently wrong — rows stop matching and simply vanish from training. A
punctuation change once split 357 people and dropped a whole race.

Any `norm_name` change therefore means: re-key, then **full retrain**.
