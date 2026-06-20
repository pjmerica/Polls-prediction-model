# Missingness report

_Generated from the committed data. Long file: 22,546 rows._

## `polls_long_with_results.csv` (long poll file)

| column | dtype | % missing | # missing |
|---|---|---|---|
| `poll_id` | str | 0.0% | 0 |
| `pollster_id` | str | 0.0% | 0 |
| `pollster` | str | 0.0% | 0 |
| `sponsor_ids` | str | 53.0% | 11,945 |
| `sponsors` | str | 53.0% | 11,945 |
| `display_name` | str | 0.0% | 0 |
| `pollster_rating_id` | float64 | 2.0% | 446 |
| `pollster_rating_name` | str | 2.2% | 489 |
| `numeric_grade` | float64 | 21.7% | 4,903 |
| `pollscore` | float64 | 18.9% | 4,259 |
| `methodology` | str | 14.1% | 3,168 |
| `transparency_score` | float64 | 43.8% | 9,881 |
| `state` | str | 0.0% | 0 |
| `start_date` | str | 0.0% | 0 |
| `end_date` | str | 0.0% | 0 |
| `sponsor_candidate_id` | float64 | 91.7% | 20,678 |
| `sponsor_candidate` | str | 89.3% | 20,143 |
| `sponsor_candidate_party` | str | 89.3% | 20,143 |
| `question_id` | str | 0.0% | 0 |
| `sample_size` | float64 | 0.9% | 213 |
| `population` | str | 0.1% | 33 |
| `subpopulation` | str | 93.8% | 21,157 |
| `population_full` | str | 0.1% | 33 |
| `tracking` | object | 98.6% | 22,234 |
| `created_at` | str | 0.0% | 0 |
| `notes` | str | 95.0% | 21,425 |
| `url` | str | 0.2% | 49 |
| `url_article` | str | 70.4% | 15,864 |
| `url_topline` | str | 81.6% | 18,396 |
| `url_crosstab` | str | 78.7% | 17,750 |
| `source` | str | 52.2% | 11,772 |
| `internal` | object | 76.2% | 17,176 |
| `partisan` | str | 74.4% | 16,769 |
| `year` | int64 | 0.0% | 0 |
| `office_type` | str | 0.0% | 0 |
| `seat_name` | str | 37.4% | 8,425 |
| `seat_number` | float64 | 10.7% | 2,419 |
| `election_date` | str | 1.9% | 424 |
| `stage` | str | 0.0% | 0 |
| `poll_party` | str | 0.0% | 0 |
| `pct` | float64 | 0.0% | 0 |
| `answer` | str | 0.0% | 0 |
| `candidate` | str | 0.0% | 0 |
| `candidate_id` | str | 0.0% | 0 |
| `race_id` | str | 0.0% | 0 |
| `nationwide_match` | float64 | 100.0% | 22,546 |
| `ranked_choice_reallocated` | float64 | 13.7% | 3,099 |
| `hypothetical` | float64 | 13.7% | 3,099 |
| `office` | str | 0.0% | 0 |
| `district` | float64 | 78.6% | 17,729 |
| `cand_key` | str | 0.0% | 0 |
| `party_std` | str | 0.0% | 0 |
| `endorsed_candidate_id` | float64 | 100.0% | 22,542 |
| `endorsed_candidate_name` | str | 100.0% | 22,542 |
| `endorsed_candidate_party` | str | 100.0% | 22,542 |
| `nationwide_batch` | object | 13.7% | 3,099 |
| `ranked_choice_round` | float64 | 97.6% | 21,998 |
| `won` | float64 | 24.4% | 5,509 |
| `vote_pct` | float64 | 24.4% | 5,509 |
| `res_party` | str | 24.4% | 5,509 |
| `res_candidate` | str | 24.4% | 5,509 |
| `race_winning_pct` | float64 | 24.4% | 5,509 |
| `has_result` | int64 | 0.0% | 0 |

## Model feature table (collapsed candidate-level, 2018–2024)

_1,859 candidate-races. Only columns with notable missingness or that are engineered are shown; macro features are 0% (filled per cycle)._

| feature | % missing | # missing |
|---|---|---|
| `poll_avg` | 0.0% | 0 |
| `poll_wavg` | 0.0% | 0 |
| `poll_std` | 27.4% | 509 |
| `avg_grade` | 5.3% | 99 |
| `avg_pollscore` | 4.6% | 85 |
| `avg_sample` | 0.2% | 4 |
| `lean_cand` | 59.4% | 1,105 |
| `prior_margin_cand` | 59.4% | 1,105 |
| `approval_eve` | 0.0% | 0 |
| `approval_max` | 0.0% | 0 |
| `approval_mean` | 0.0% | 0 |
| `approval_std` | 0.0% | 0 |
| `approval_trend` | 0.0% | 0 |
| `gas_eve` | 0.0% | 0 |
| `gas_max` | 0.0% | 0 |
| `gas_mean` | 0.0% | 0 |
| `gas_std` | 0.0% | 0 |
| `gas_trend` | 0.0% | 0 |
| `gdp_eve` | 0.0% | 0 |
| `gdp_max` | 0.0% | 0 |
| `gdp_mean` | 0.0% | 0 |
| `gdp_std` | 0.0% | 0 |
| `gdp_trend` | 0.0% | 0 |
| `inflation_eve` | 0.0% | 0 |
| `inflation_max` | 0.0% | 0 |
| `inflation_mean` | 0.0% | 0 |
| `inflation_std` | 0.0% | 0 |
| `inflation_trend` | 0.0% | 0 |
| `unemployment_eve` | 0.0% | 0 |
| `unemployment_max` | 0.0% | 0 |
| `unemployment_mean` | 0.0% | 0 |
| `unemployment_std` | 0.0% | 0 |
| `unemployment_trend` | 0.0% | 0 |
