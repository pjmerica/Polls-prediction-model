# Re-tune stability of the 2026 forecast

Built by `analysis/retune_stability.py` (audit M5). Each run repeats the trainer's whole tuning procedure with different randomness (another sample of the random search, other stage-2 and model seeds, the temperature and offset refitted for the win and no-polls models) and scores the same served 2026 table. The audit's bar: no race moves more than 5 pts. "Move" is against the served model; "spread" is between the re-tunes themselves. Louisiana House is left out of the polled comparison (its odds come from the margin model and the runoff simulation).

| Model | Run | Settings shared with the served ensemble | Notes |
|---|---|---|---|
| win | 1 | 0 of 5 | temperature 0.8849, minor offset -8.0 |
| win | 2 | 0 of 5 | temperature 0.8849, minor offset -8.0 |
| win | 3 | 0 of 5 | temperature 0.8849, minor offset -8.0 |

## Win model (win-chance pts)

- Races: 175. Against the served model: largest move 10.9, median 1.14, races over 5: 15, over 2: 59.
- Between re-tunes: largest spread 12.7, median 1.11, races over 5: 18.
- Favourite changes in the worst run: 3.

| Race | Largest move | Spread between re-tunes |
|---|---|---|
| 2026_CO_House-5 | 10.9 | 10.7 |
| 2026_PA_House-8 | 9.8 | 12.7 |
| 2026_WA_House-3 | 8.2 | 8.1 |
| 2026_NC_House-1 | 8.0 | 9.8 |
| 2026_MN_House-1 | 7.5 | 7.0 |
| 2026_AR_House-2 | 7.2 | 6.5 |
| 2026_MO_House-2 | 7.1 | 6.1 |
| 2026_TX_House-34 | 6.7 | 7.3 |
| 2026_IA_House-1 | 6.5 | 5.9 |
| 2026_GA_Governor | 5.9 | 3.9 |
| 2026_FL_House-9 | 5.8 | 2.9 |
| 2026_VA_House-1 | 5.7 | 4.6 |
