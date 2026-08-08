# `archive/` — superseded data snapshots (kept, never deleted)

Timestamped copies of `candidate_bios*` files taken **before** a change that rewrote them.
Nothing reads this folder. It exists so a bad transformation can be diffed against what came
before, rather than reconstructed from a scraper re-run that may no longer reproduce the same
pages.

Naming: `<file>_<YYYYMMDD>_<HHMMSS>_<what-changed-next>.csv`

| snapshot | taken before |
|---|---|
| `..._pre-namefix.csv` | the `norm_name` punctuation fix |
| `..._pre-incumbent-context-fix.csv` | making `classify()` page-context aware |
| `..._pre-nominee-subsection-fix.csv` | the nominee-subsection parser fix |
| `..._pre-unbiased-targetlist-fix.csv` | replacing the biased (primary-poll-derived) target list |
| `..._last-shared-write-before-split.csv` | splitting one shared bios file into per-office files |
| `..._pre-tenure-schema.csv` | the Ballotpedia tenure-date schema change |
| `..._pre-table-backfill.csv` | the Senate/Governor table backfill |

## The two poisoned files are the useful ones

`candidate_bios_ballotpedia_20260724_220526_POISONED_ratelimited_0hits.csv` and
`..._234942_partial_softblocked.csv` are kept deliberately.

Ballotpedia does not return an error when it rate-limits — it serves **valid-looking pages
with no data**. A blocked scrape therefore produces a complete, well-formed CSV in which every
office level is 0, which is indistinguishable from "none of these people held office" unless
you compare hit counts against a previous run. These two files are what that failure looks
like on disk. Check any new bio scrape against them before trusting it.

## The rule

**Archive, don't delete.** Anything superseded gets a timestamped copy here with a suffix
saying what changed next. The whole folder is gitignored (`*.csv`), so these live only in a
working copy — if you need one preserved permanently, say so explicitly.
