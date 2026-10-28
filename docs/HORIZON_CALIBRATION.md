# Calibration of the published win odds by horizon

Built by `analysis/race_horizon_eval.py` (audit item 8). Held-out 2018-24: each cycle is predicted with only the polls that ended at least H days before the election, by the race-level model trained on earlier cycles, with the temperature and minor-candidate offset fitted on earlier cycles. A forecast H days out is calibrated when, in each band, the predicted and actual win rates match.

## Summary by horizon (mean over 2018, 2020, 2022, 2024)

| Days out | Races | Brier | Log loss | Favourite won | Favourite's average chance |
|---|---|---|---|---|---|
| 0 | 686 | 0.059 | 0.192 | 88.7% | 88.1% |
| 10 | 662 | 0.065 | 0.210 | 88.0% | 87.4% |
| 35 | 599 | 0.067 | 0.216 | 87.9% | 86.3% |
| 60 | 515 | 0.082 | 0.266 | 85.3% | 84.6% |

## By cycle, 10 days out (the horizon closest to the last week of the campaign)

| Cycle | Races | Brier | Favourite won | Favourite's average chance | Upsets expected | Upsets |
|---|---|---|---|---|---|---|
| 2018 | 218 | 0.066 | 88.1% | 85.9% | 30.7 | 26 |
| 2020 | 146 | 0.081 | 82.9% | 86.8% | 19.3 | 25 |
| 2022 | 169 | 0.055 | 91.1% | 87.5% | 21.1 | 15 |
| 2024 | 129 | 0.058 | 89.9% | 89.3% | 13.8 | 13 |

## Confident favourites: expected vs actual upsets

Each race's favourite, by its published chance. "Chance of this many upsets or more" is exact (Poisson-binomial over the favourites' own odds): a small number means the favourites lost more often than their odds said.

| Days out | Favourite's chance | Favourites | Upsets expected | Upsets | Chance of this many upsets or more |
|---|---|---|---|---|---|
| 0 | 95%+ | 336 | 6.2 | 4 | 0.87 |
| 0 | 90%-95% | 90 | 6.3 | 4 | 0.89 |
| 0 | 75%-90% | 126 | 20.7 | 18 | 0.78 |
| 10 | 95%+ | 302 | 5.4 | 4 | 0.79 |
| 10 | 90%-95% | 98 | 7.1 | 5 | 0.84 |
| 10 | 75%-90% | 119 | 19.4 | 22 | 0.29 |
| 35 | 95%+ | 257 | 4.8 | 2 | 0.95 |
| 35 | 90%-95% | 85 | 6.3 | 8 | 0.30 |
| 35 | 75%-90% | 113 | 19.1 | 14 | 0.93 |
| 60 | 95%+ | 192 | 4.3 | 3 | 0.80 |
| 60 | 90%-95% | 73 | 5.4 | 7 | 0.30 |
| 60 | 75%-90% | 105 | 18.0 | 13 | 0.93 |

## Calibration by confidence band (every candidate)

The 95% interval is the Wilson interval on the band's actual win rate; a band is off only when its predicted rate falls outside it.

| Days out | Band | Candidates | Predicted | Won | 95% interval |
|---|---|---|---|---|---|
| 0 | 0%-2% | 579 | 0.5% | 0.0% | 0.0%-0.7% |
| 0 | 2%-10% | 355 | 4.5% | 2.3% | 1.1%-4.4% (outside) |
| 0 | 10%-30% | 150 | 18.3% | 20.0% | 14.4%-27.1% |
| 0 | 30%-50% | 100 | 39.7% | 42.0% | 32.8%-51.8% |
| 0 | 50%-70% | 97 | 60.3% | 61.9% | 51.9%-70.9% |
| 0 | 70%-90% | 157 | 81.4% | 81.5% | 74.7%-86.8% |
| 0 | 90%-98% | 222 | 95.2% | 96.4% | 93.1%-98.2% |
| 0 | 98%-100% | 204 | 99.1% | 100.0% | 98.2%-100.0% |
| 10 | 0%-2% | 548 | 0.5% | 0.2% | 0.0%-1.0% |
| 10 | 2%-10% | 323 | 4.9% | 2.5% | 1.3%-4.8% (outside) |
| 10 | 10%-30% | 157 | 18.5% | 22.9% | 17.0%-30.1% |
| 10 | 30%-50% | 100 | 39.5% | 37.0% | 28.2%-46.8% |
| 10 | 50%-70% | 102 | 60.4% | 66.7% | 57.1%-75.1% |
| 10 | 70%-90% | 155 | 81.2% | 78.1% | 70.9%-83.9% |
| 10 | 90%-98% | 209 | 94.9% | 96.2% | 92.6%-98.0% |
| 10 | 98%-100% | 191 | 99.0% | 99.5% | 97.1%-99.9% |
| 35 | 0%-2% | 422 | 0.5% | 0.0% | 0.0%-0.9% |
| 35 | 2%-10% | 287 | 5.0% | 3.5% | 1.9%-6.3% |
| 35 | 10%-30% | 156 | 19.1% | 17.9% | 12.7%-24.7% |
| 35 | 30%-50% | 103 | 38.7% | 36.9% | 28.2%-46.5% |
| 35 | 50%-70% | 106 | 60.7% | 67.0% | 57.6%-75.2% |
| 35 | 70%-90% | 146 | 80.6% | 82.2% | 75.2%-87.5% |
| 35 | 90%-98% | 187 | 94.9% | 94.7% | 90.4%-97.1% |
| 35 | 98%-100% | 155 | 99.0% | 100.0% | 97.6%-100.0% |
| 60 | 0%-2% | 246 | 0.6% | 0.8% | 0.2%-2.9% |
| 60 | 2%-10% | 248 | 5.0% | 3.6% | 1.9%-6.8% |
| 60 | 10%-30% | 159 | 19.4% | 15.1% | 10.4%-21.5% |
| 60 | 30%-50% | 98 | 39.3% | 45.9% | 36.4%-55.8% |
| 60 | 50%-70% | 102 | 60.7% | 57.8% | 48.1%-67.0% |
| 60 | 70%-90% | 143 | 80.1% | 84.6% | 77.8%-89.6% |
| 60 | 90%-98% | 169 | 94.9% | 95.3% | 90.9%-97.6% |
| 60 | 98%-100% | 96 | 98.9% | 97.9% | 92.7%-99.4% |
