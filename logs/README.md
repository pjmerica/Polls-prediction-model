# Scraper run-logs

Consolidated here from the repo root on 2026-08-08 (20 files were loose in the root).

`logs/` is **gitignored in full**. These are transient progress traces — the kind written with
`... | tee scrape_log.txt` while a long scrape runs — kept only so a scrape that half-failed
can be diffed against the next attempt. **They are not source data**; the CSVs the scrapers
write into `data/` are.

## Why the directory rule matters

Before the move, `.gitignore` listed three *name-specific* patterns
(`ballotpedia_scrape*.txt`, `backfill_*_scrape_log.txt`, `backfill_*_log.txt`). Those matched
10 of the 20 logs. The other 10 — `candidate_bios_rescrape_log{,2,3,4}.txt`,
`house_candidate_bios_scrape_log{,2..5}.txt`, `house_primary_scrape_log.txt` — did not match
any pattern and were **committed to git**, purely because of how they happened to be named.
A single `logs/` rule now covers every scraper log, including ones future scrapers invent.

## Convention

Write new run-logs to `logs/scrape/<scraper-name>_<YYYYMMDD>.txt`. Anything genuinely worth
keeping (a coverage number, a decision, a bug found) belongs in `HANDOFF.md` or `CONCERNS.md`,
not in a log file that no tool reads and git does not track.
