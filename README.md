# Polls prediction model

Building a model to predict whether a candidate will win an election (U.S. Senate, House, and Governor races, 2000–present).

## Data

There is no single downloadable file with "every poll + who won," so this project assembles one from two free public sources:

- **Polls** — [FiveThirtyEight poll files](https://projects.fivethirtyeight.com/polls-page/data/) (`senate_polls.csv`, `house_polls.csv`, `governor_polls.csv`, plus their `*_historical.csv` versions)
- **Results / who won** — [FiveThirtyEight `election-results` repo](https://github.com/fivethirtyeight/election-results), which includes a per-candidate `winner` flag (data back to 1998)

## `build_dataset.ipynb`

Downloads both sources, normalizes candidate names, and joins each poll to its race's eventual result.

Output is **long format**: one row per individual poll (per candidate), with the race outcome (`won`, `vote_pct`, `race_winning_pct`) attached, plus a `race_id` for grouping. All original poll columns (pollster, sample size, dates, rating, methodology) are preserved so nothing is lost before modeling.

Saved to `polls_long_with_results.csv` (gitignored — regenerate by running the notebook).

## Run

```bash
pip install pandas numpy requests
```

Open `build_dataset.ipynb` and run top to bottom. The first run downloads source data into a local `data/` folder and caches it.

## Next steps

- Collapse the long table to one row per race for modeling.
- Split train/test **by year** (e.g. train ≤ 2020, test 2022/2024) to avoid leaking the future.
- Don't use `vote_pct` / `race_winning_pct` as model inputs — they're outcomes.
