# Re-tune stability of the 2026 forecast

Built by `analysis/retune_stability.py` (audit M5). Each run repeats the trainer's whole tuning procedure with different randomness (another sample of the random search, other stage-2 and model seeds, the temperature and offset refitted for the win and no-polls models) and scores the same served 2026 table. The audit's bar: no race moves more than 5 pts. "Move" is against the served model; "spread" is between the re-tunes themselves. Louisiana House is left out of the polled comparison (its odds come from the margin model and the runoff simulation).

| Model | Run | Settings shared with the served ensemble | Notes |
|---|---|---|---|
| fund | 1 | 0 of 5 | temperature 0.811, minor offset -2.0 |
| fund | 2 | 0 of 5 | temperature 0.811, minor offset -2.0 |
| fund | 3 | 1 of 5 | temperature 0.811, minor offset -2.0 |

## No-polls model (win-chance pts)

- Races: 329. Against the served model: largest move 3.5, median 0.09, races over 5: 0, over 2: 6.
- Between re-tunes: largest spread 4.0, median 0.09, races over 5: 0.
- Favourite changes in the worst run: 0.

| Race | Largest move | Spread between re-tunes |
|---|---|---|
| 2026_CA_House-4 | 3.5 | 4.0 |
| 2026_ME_Governor | 3.1 | 0.9 |
| 2026_CA_House-14 | 2.9 | 2.9 |
| 2026_LA_House-4 | 2.2 | 1.4 |
| 2026_MI_House-1 | 2.1 | 2.8 |
| 2026_IN_House-1 | 2.1 | 2.2 |
| 2026_LA_House-2 | 1.7 | 1.7 |
| 2026_WY_Governor | 1.7 | 1.8 |
| 2026_LA_House-1 | 1.6 | 0.8 |
| 2026_CA_House-1 | 1.3 | 1.5 |
| 2026_NV_House-1 | 1.3 | 1.3 |
| 2026_NC_House-8 | 1.2 | 1.2 |
