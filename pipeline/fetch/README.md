# `pipeline/fetch/` — the scrapers (the only network code in the repo)

Run from the repo root, e.g. `py -X utf8 pipeline/fetch/fetch_macro.py`. Each script is safe
to re-run: they resume from their own output and rebuild deterministically.

**You normally do not need to run any of these.** Everything they produce is already committed
in `data/`. Run one only to *extend* a source to a new month or cycle.

| script | writes | source |
|---|---|---|
| `fetch_macro.py` | `macro_monthly.csv`, `umcsent_fred.csv` | DBnomics history + BLS API overlay + **FRED `UMCSENT`** for consumer sentiment |
| `fetch_approval.py` | `approval_monthly.csv` | Gallup/UCSB 1993–2025 + VoteHub API 2025+ |
| `fetch_generic_ballot.py` | `generic_ballot_monthly.csv` | Wikipedia aggregator table (RCP is Cloudflare-walled — 403 to scripts) |
| `fetch_fec.py` / `fetch_fec_detail.py` | `fec_summary.csv`, `fec_detail.csv` | FEC API (needs `FEC_API_KEY` in `.env`) |
| `fetch_governor_finance.py` | `governor_finance.csv` | FollowTheMoney (`FTM_API_KEY`) |
| `fetch_district_pvi.py` | `district_pvi_current.csv` | Cook PVI |
| `fetch_primary_dates.py` | `primary_dates_hist.csv` | Wikipedia |
| `fetch_primary_polls_wikipedia.py` | `primary_polls_wikipedia.csv` | Wikipedia primary-poll tables |
| `fetch_primary_results_2026.py` | `primary_results_2026.csv`, `primary_results_hist.csv` | Wikipedia |
| `fetch_house_primary_results_hist.py` | `house_primary_results_hist.csv` | Wikipedia |
| `fetch_candidate_bios.py` | `candidate_bios_senate.csv`, `_governor.csv` | Wikipedia race pages |
| `fetch_house_candidate_bios_hist.py` | `candidate_bios_house.csv` | Wikipedia race pages |
| `fetch_candidate_bios_ballotpedia.py` | `candidate_bios_ballotpedia.csv`, `uncovered_candidates.csv` | Ballotpedia |

Secrets live in `.env` at the repo root (gitignored). Long scrapes should `tee` their progress
into `logs/scrape/`.

## The bio scrapers: one file per office, one merger

The three bio scrapers write **separate** files on purpose. They used to share
`data/candidate_bios.csv`, each with its own resume-from-what's-there logic that assumed it was
the only writer — running them back to back silently dropped ~9,500 already-scraped House rows.

After running any of them, rebuild the merged table with the **leak-free** builder:

```bash
py -X utf8 pipeline/build/build_office_level_table.py
```

**Not** `combine_candidate_bios.py` — it writes the same path with a frozen `office_level`
(a look-ahead leak) and now refuses to run without an explicit override.

## Rate limits and poisoned output

Ballotpedia soft-blocks aggressive scraping and returns **valid-looking pages with no data**.
A rate-limited run therefore produces a complete CSV full of zeros rather than an error — one
such file is preserved in `../../archive/` with `POISONED_ratelimited_0hits` in its name.
Always check hit counts against the previous run before trusting a bio scrape.

## Historical pages are read as they exist *today*

Wikipedia bios are edited after elections, so a date range can close (`2019-present` →
`2019-2023`) and leak small amounts of later information into training-era rows. Office
*level* — the actual feature — rarely changes from such edits. Documented in
[../../docs/METHODOLOGY.md](../../docs/METHODOLOGY.md); do not attempt to "correct" as-of-year
office levels by hand.

## `fetch_macro.py` never deletes data (2026-08-08)

It used to. A skipped series was simply absent from the frames it concatenated, and the file
was rewritten anyway — so one read timeout **deleted that metric's entire history**. On
2026-08-08 that wiped `sentiment` and its 16 model features; nothing failed loudly, and only
`predict.py`'s artifact-feature assert stopped a run from serving 16 empty features.

Two guards now run before the file is written:

- **Carry-forward** — a metric in the previous file that this run could not fetch keeps its
  existing rows, with a loud `!! CARRIED FORWARD ... STALE, not refreshed` notice.
- **Shrink warning** — a metric that returns FEWER rows than last time prints
  `!! <metric> SHRANK <old> -> <new>`. A partial or truncated response is otherwise
  indistinguishable from a successful one.

**Read the output before committing `data/macro_monthly.csv`.** The shrink warning is a
warning, not a block. It immediately caught a real regression: with the BLS API returning
**503**, four series (`cpi`, `cpi_core`, `unemp_u6`, `unemployment`) silently truncated from
2026-06/07 back to 2025-01. If you see that, wait for BLS and re-run.

## Three sources for one series, on purpose

Consumer sentiment has the most fragile supply chain in the file:

1. **DBnomics** (`SCSMICH/MICS/ICS`) — the original source. Its mirror **stalled at 2025-08**,
   which is what made `sentiment_last12_delta` 100% NaN at serve time (CONCERNS #30).
2. **FRED `UMCSENT`** — the same University of Michigan index, current, **no API key**.
   Verified identical where the two overlap: 664 months, max absolute difference **0.0**.
   Applied as an **overlay**, never a replacement — our file has 302 pre-1978 rows that FRED's
   monthly series lacks (UMich surveyed quarterly before 1978).
3. **`data/umcsent_fred.csv`** — a committed cache, refreshed on every successful fetch and
   read when FRED is unreachable. FRED is genuinely flaky from some networks: measured inside
   one hour, 3 read timeouts → 10/10 successes averaging 0.7s → more timeouts. A series that
   updates monthly should not depend on that.
