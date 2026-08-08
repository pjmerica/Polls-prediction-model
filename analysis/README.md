# `analysis/` — investigations, not production

Nothing here runs in the pipeline or ships to the dashboard. These are the "we wondered
whether…" scripts, kept so their answers stay reproducible instead of living in someone's
memory. **Several of them found "no" and are kept precisely for that.**

Notebooks pair with a `.py` module holding the logic, so the analysis is importable and
testable rather than trapped in cells.

| file | question | answer |
|---|---|---|
| `fundamentals_vs_polls_thin.py` | Do the no-polling models beat the poll models on thin-poll races? | **No.** |
| `fundamentals_on_unpolled.py` | Does the fundamentals model work where polls don't exist at all? | see script |
| `poll_volume_breakpoint.{ipynb,py}` | At what poll count does each model break down? | see notebook |
| `primary_backtest_2026.ipynb` | Backtest of the primary model on 2026 | — |
| `score_candidate_history.py` | Out-of-fold win + margin predictions per candidate → CSV | — |
| [`worklists/`](worklists/) | point-in-time audit outputs — **nothing reads them** | — |

## Negative results are results

`fundamentals_vs_polls_thin.py` is the reason the fundamentals models are not blended into
production. Deleting it would make that decision look arbitrary, and someone would redo the
work. If you run an ablation and it comes back null, **leave the script here** and record the
verdict in [../docs/CONCERNS.md](../docs/CONCERNS.md).

## Two cautions when reading old results

**A verdict expires when the data changes.** The own-primary-margin ablation returned null in
July 2026 and the feature was dropped — but the training data has since changed (the
dead-matchup filter, 2026-08-06), so that verdict does **not** automatically carry over. Re-run
before citing an old null.

**An unchanged result can mean the change never landed.** `features.py` and
`features_primary.py` are separate implementations; editing one and testing the other looks
exactly like "the feature didn't help." Grep both.
