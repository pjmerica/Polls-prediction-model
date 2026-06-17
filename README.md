# Polls prediction model

Building a model to predict whether a candidate will win an election (U.S. Senate, House, and Governor races, **2018–present**).

## Data

There is no single downloadable file with "every poll + who won," so this project assembles one from free public sources. FiveThirtyEight was dissolved in March 2025 and its poll endpoints now return HTML, so the notebook uses live replacements that share the **same 538 schema**:

- **Current polls** — [New York Times poll CSVs](https://www.nytimes.com/newsgraphics/polls/senate.csv) (`senate.csv`, `house.csv`, `governor.csv`) — same columns as old 538, updated continuously, CC-BY.
- **Historical polls** — Internet Archive snapshot of FiveThirtyEight's `*_polls_historical.csv` (frozen but complete).
- **Results / who won** — [FiveThirtyEight `election-results` repo](https://github.com/fivethirtyeight/election-results), which includes a per-candidate `winner` flag.

### Time range

Election *results* go back to 1976, but machine-readable *polls* effectively begin in **2018** (when 538's poll database starts). You can't model races that have no polls, so 2018–present is the working range — ~26k poll rows, ~17k with a known winner.

## `build_dataset.ipynb`

Downloads all sources, standardizes state codes and candidate names, and joins each poll to its race's eventual result.

Output is **long format**: one row per individual poll (per candidate), with the race outcome (`won`, `vote_pct`, `race_winning_pct`) attached, plus a `race_id` for grouping. All original poll columns (pollster, sample size, dates, rating, methodology) are preserved so nothing is lost before modeling. A `has_result` flag marks which poll rows matched a result.

Saved to `polls_long_with_results.csv` (gitignored — regenerate by running the notebook).

Approximate match rate (poll → result): Senate ~76%, Governor ~58%, House ~53% (House is lower because district polling is sparse and names are harder to match).

## Run

```bash
pip install pandas numpy requests jupyter
```

Open `build_dataset.ipynb` and run top to bottom. The first run downloads source data into a local `data/` folder and caches it. (The Internet Archive can rate-limit with HTTP 429 — just re-run the download cell; cached files are skipped.)

## Next steps

- Collapse the long table to one row per race for modeling.
- Split train/test **by year** (e.g. train ≤ 2022, test 2024) to avoid leaking the future.
- Don't use `vote_pct` / `race_winning_pct` as model inputs — they're outcomes.
