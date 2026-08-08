# `pipeline/` — getting data in and assembling it

Two stages, deliberately separated:

| folder | touches the network? | what |
|---|---|---|
| [`fetch/`](fetch/) | **yes** — this is the only place that does | 14 scrapers / API pulls → `data/*.csv` |
| [`build/`](build/) | no — offline | joins fetched data into the training tables |

## Pull once, commit, never re-download

Nothing in this pipeline re-downloads on a normal run. Historical polls, results, macro,
approval, generic ballot, FEC, candidate bios — **all committed once** and re-pulled only to
*extend* to a new month or cycle. The single exception is the live generic-ballot fetch at
predict time, since current-cycle information cannot be frozen by definition.

That is why a full retrain is reproducible offline, and why a scraper breaking upstream does
not silently change last year's training data.

## Run order

```
1. pipeline/fetch/*.py            (only to extend/refresh a source)
2. pipeline/build/build_dataset.ipynb        -> polls_long_with_results.csv (repo root)
   pipeline/build/build_primary_dataset.py   -> data/primary_polls_long.csv
   pipeline/build/build_office_level_table.py-> data/candidate_bios.csv
3. models/  (retrain)
4. src/refresh_dashboard.py  (predict + publish)
```

## Failure mode to watch for

**A dead upstream fails softly.** The fetch scripts are written to degrade rather than crash,
so a stale feed looks exactly like a working one. `src/refresh_dashboard.py` runs an explicit
`check_feed_freshness()` for this reason — if it warns that a series is months behind, treat
that as a broken fetch, not as a quiet month in the data.

Second: **one output file, one writer.** Three bio scrapers once shared
`data/candidate_bios.csv`, each with its own "resume from what's there" logic; running them
back to back silently discarded ~9,500 scraped rows. Each scraper now writes its **own** file
and a single build step merges them.
