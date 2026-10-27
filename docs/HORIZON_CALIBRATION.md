# Calibration of the published win odds by horizon

Built by `analysis/race_horizon_eval.py` (audit item 8). Held-out 2018-24: each cycle is predicted with only the polls that ended at least H days before the election, by the race-level model trained on earlier cycles, with the temperature and minor-candidate offset fitted on earlier cycles. A forecast H days out is calibrated when, in each band, the predicted and actual win rates match.

## Summary by horizon (mean over 2018, 2020, 2022, 2024)

| Days out | Races | Brier | Log loss | Favourite won | Favourite's average chance |
|---|---|---|---|---|---|
| 0 | 686 | 0.059 | 0.193 | 88.4% | 88.3% |
| 10 | 662 | 0.066 | 0.213 | 87.7% | 87.6% |
| 35 | 599 | 0.068 | 0.219 | 87.9% | 86.5% |
| 60 | 515 | 0.083 | 0.270 | 85.3% | 84.7% |

## Calibration by confidence band (every candidate)

| Days out | Band | Candidates | Predicted | Won |
|---|---|---|---|---|
| 0 | 0%-2% | 579 | 0.5% | 0.2% |
| 0 | 2%-10% | 359 | 4.5% | 1.9% |
| 0 | 10%-30% | 148 | 18.3% | 22.3% |
| 0 | 30%-50% | 97 | 39.6% | 40.2% |
| 0 | 50%-70% | 97 | 60.3% | 62.9% |
| 0 | 70%-90% | 149 | 81.4% | 79.2% |
| 0 | 90%-98% | 233 | 95.2% | 97.0% |
| 0 | 98%-100% | 202 | 99.1% | 99.5% |
| 10 | 0%-2% | 543 | 0.5% | 0.2% |
| 10 | 2%-10% | 327 | 4.7% | 2.4% |
| 10 | 10%-30% | 157 | 17.9% | 24.8% |
| 10 | 30%-50% | 102 | 39.6% | 36.3% |
| 10 | 50%-70% | 104 | 60.6% | 67.3% |
| 10 | 70%-90% | 150 | 81.9% | 76.0% |
| 10 | 90%-98% | 218 | 95.2% | 96.3% |
| 10 | 98%-100% | 184 | 99.1% | 99.5% |
| 35 | 0%-2% | 422 | 0.5% | 0.0% |
| 35 | 2%-10% | 285 | 4.9% | 2.8% |
| 35 | 10%-30% | 157 | 18.5% | 19.1% |
| 35 | 30%-50% | 104 | 38.6% | 37.5% |
| 35 | 50%-70% | 107 | 60.8% | 65.4% |
| 35 | 70%-90% | 145 | 81.1% | 81.4% |
| 35 | 90%-98% | 190 | 95.1% | 95.8% |
| 35 | 98%-100% | 152 | 99.0% | 100.0% |
| 60 | 0%-2% | 246 | 0.5% | 0.8% |
| 60 | 2%-10% | 245 | 5.0% | 3.3% |
| 60 | 10%-30% | 158 | 18.9% | 16.5% |
| 60 | 30%-50% | 102 | 38.8% | 43.1% |
| 60 | 50%-70% | 108 | 61.2% | 60.2% |
| 60 | 70%-90% | 140 | 81.0% | 83.6% |
| 60 | 90%-98% | 168 | 95.0% | 95.8% |
| 60 | 98%-100% | 94 | 99.0% | 97.9% |
