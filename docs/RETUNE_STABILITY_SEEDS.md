# Re-tune stability of the 2026 forecast

Built by `analysis/retune_stability.py` (audit M5). Each run repeats the trainer's whole tuning procedure with different randomness (another sample of the random search, other stage-2 and model seeds, the temperature and offset refitted for the win and no-polls models) and scores the same served 2026 table. The audit's bar: no race moves more than 5 pts. "Move" is against the served model; "spread" is between the re-tunes themselves. Louisiana House is left out of the polled comparison (its odds come from the margin model and the runoff simulation).

| Model | Run | Settings shared with the served ensemble | Notes |
|---|---|---|---|
| win | 1 | 4 of 5 | temperature 0.8849, minor offset -8.0 |
| win | 2 | 4 of 5 | temperature 0.8849, minor offset -8.0 |
| win | 3 | 4 of 5 | temperature 0.8849, minor offset -8.0 |

## Win model (win-chance pts)

- Races: 175. Against the served model: largest move 3.9, median 0.70, races over 5: 0, over 2: 23.
- Between re-tunes: largest spread 5.4, median 0.70, races over 5: 2.
- Favourite changes in the worst run: 1.

| Race | Largest move | Spread between re-tunes |
|---|---|---|
| 2026_CO_House-5 | 3.9 | 5.2 |
| 2026_FL_House-25 | 3.4 | 5.4 |
| 2026_MO_House-2 | 3.1 | 1.5 |
| 2026_AR_House-2 | 3.0 | 2.2 |
| 2026_PA_House-8 | 3.0 | 3.7 |
| 2026_TX_House-34 | 2.9 | 4.7 |
| 2026_GA_Governor | 2.9 | 2.6 |
| 2026_WA_House-3 | 2.7 | 3.8 |
| 2026_NY_House-17 | 2.7 | 3.3 |
| 2026_OR_Governor | 2.7 | 3.6 |
| 2026_NC_House-1 | 2.7 | 4.8 |
| 2026_NE_Senate | 2.5 | 2.5 |
