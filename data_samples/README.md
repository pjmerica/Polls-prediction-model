# `data_samples/` — tiny committed extracts of the results files

Three small illustrative samples of the election-results files:

| file | sample of |
|---|---|
| `res_senate_sample.csv` | `data/res_senate.csv` |
| `res_house_sample.csv` | `data/res_house.csv` |
| `res_governor_sample.csv` | `data/res_governor.csv` |

They exist so a reader can see the **shape** of the results data — column names, how a
candidate row looks, how parties and vote counts are recorded — without opening a large file
or having the full `data/` directory. Referenced from
[../docs/DATA_SOURCES.md](../docs/DATA_SOURCES.md).

**Nothing in the pipeline reads these.** They are documentation. The models read the full files
in `data/`.

Uniquely in this repo, `data_samples/` is explicitly un-ignored in `.gitignore`
(`!data_samples/`, `!data_samples/*.csv`) so these survive a fresh clone despite the blanket
`*.csv` rule.

## One gotcha visible in the samples

The results files' `party` column is **all null** — use `ballot_party` instead. That is not a
sampling artifact; it is true of the full files, and it has caught people out.
