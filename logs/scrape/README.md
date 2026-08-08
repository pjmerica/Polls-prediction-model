# `logs/scrape/` — scraper progress traces

Gitignored. See [../README.md](../README.md) for why this folder exists as a directory-level
ignore rather than a set of filename patterns.

These are `... | tee logs/scrape/<name>.txt` traces from long scrapes. They are kept only so a
half-failed run can be diffed against the next attempt — **no tool reads them**, and nothing
here is source data. The CSVs the scrapers write into `data/` are.

## What they are good for

Exactly one thing: **spotting a scrape that "succeeded" but collected nothing.** Ballotpedia
soft-blocks by serving valid-looking pages with no data, so a rate-limited run exits cleanly
with a complete CSV full of zeros. Comparing the hit counts in a new log against the previous
one is how that gets caught (see `../../archive/README.md` for the two poisoned files kept as
reference).

## Convention

`<scraper-name>_<YYYYMMDD>.txt`.

Anything genuinely worth keeping — a coverage number, a decision, a bug found — belongs in
`docs/HANDOFF.md` or `docs/CONCERNS.md`, not in an untracked log file. Assume everything here
will be deleted without warning.
