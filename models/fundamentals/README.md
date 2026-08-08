# `models/fundamentals/` — the no-polling models (NOT shipped)

```bash
py -X utf8 models/fundamentals/fundamentals_model.py
```

Trains polling-free variants of the general and primary models — incumbency, partisan lean,
seat history, fundraising, candidate office level, and the macro block, but **no race polls**.
Writes `data/fundamentals_model_{general,primary}{,_features}.json` and the two gain tables
here.

**Nothing here is blended into production.** It exists to answer "how much of the result is
structural, before anyone is polled?" and as a potential floor for races with thin polling.

## What it found (and why it stayed unshipped)

| | race-acc | vs |
|---|---|---|
| General, no polls | **.811** | .868 with polls |
| Primary, no polls | **.435** | its own naive baseline of .451 |

The general result is the interesting one: most general elections really are structurally
determined, with `is_incumbent` alone carrying **43% of the gain**.

The primary result is the instructive one — it lands **below its own naive baseline** ("pick
the highest office-level candidate"). Within a single party there is no incumbency edge, no
partisan lean, and no seat history to lean on, so the features that carry the general model
have nothing to say. A model that cannot beat its own naive baseline does not get shipped.

Two follow-up investigations in [`../../analysis/`](../../analysis/) tested whether these
models help where polling is thin — `fundamentals_vs_polls_thin.py` (answer: no) and
`fundamentals_on_unpolled.py`. See [../../docs/CONCERNS.md](../../docs/CONCERNS.md) #28.

## Note on `is_incumbent` here

Its 43% share of the gain is a **party-level** signal, not personal incumbency:
`is_incumbent` is `incumbent_party == candidate party`, because `races.csv` has an incumbent
*party* column and no incumbent name. Read the number as "the party holding this seat usually
holds it again", which is exactly the structural claim this model is testing.
