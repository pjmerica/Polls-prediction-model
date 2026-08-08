# Audit worklists (records, not inputs)

Moved here from the repo root on 2026-08-08. **Nothing in the pipeline reads these files** —
verified by grepping the whole repo for each filename before moving them. They are the
output of past coverage audits, kept as a record of what was checked and hand-coded.

They are gitignored (`*.csv`), so they exist only in a working copy.

| file | what it is | produced by |
|---|---|---|
| `missing_2026_bio.csv` | 2026 candidates with no `bio_office_level` at the time of the audit — the gap list behind the ~13.7% serve-time bio-coverage figure in DATA_DICTIONARY.md | ad-hoc audit query |
| `missing_2026_real.csv` | the subset of the above judged to be *real* candidates (junk poll answers removed), i.e. the genuine hand-code target list | ad-hoc audit query |
| `office_level_handcode_worklist.csv` | historical candidates needing a manual `office_level`; `unc` lists the cycles they appear in | office-coverage audit (`tools/measure_office_coverage.py` territory) |

**If you are looking for the live version of any of this**, do not read these files — they are
point-in-time snapshots and will drift. Regenerate from the current data instead:

- current bio coverage → `py -X utf8 tools/measure_office_coverage.py`
- the authoritative bio table → `data/candidate_bios.csv`, rebuilt by
  `py -X utf8 pipeline/build/build_office_level_table.py`
- hand-codes that actually feed the model → `data/candidate_bios_manual.csv`
