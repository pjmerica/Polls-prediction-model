# `docs/` — the deep documentation

`README.md` stays at the repo root as the entry point; everything else lives here (moved
2026-08-08).

## Read in this order

| doc | what it is | read it when |
|---|---|---|
| [AGENTS.md](AGENTS.md) | architecture + **the rules learned the hard way** + traps | **first**, before changing anything |
| [STRUCTURE.md](STRUCTURE.md) | the folder map, run commands, and what may live in the repo root | you need to find or place a file |
| [HANDOFF.md](HANDOFF.md) | in-flight state + a dated log of every retrain and its numbers | you want "what changed when" |
| [CONCERNS.md](CONCERNS.md) | the living risk register + ranked improvement roadmap | you're deciding what to work on next |
| [METHODOLOGY.md](METHODOLOGY.md) | **exact time windows for every feature** | you're touching a feature or judging leakage |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | every variable, its meaning, its missingness | you don't know what a column means |
| [DATA_SOURCES.md](DATA_SOURCES.md) | every source URL and how it was found | you need to re-pull or extend a source |
| [MISSINGNESS_REPORT.md](MISSINGNESS_REPORT.md) | per-column missingness, train **and** serve | you're about to trust a sparse feature |

## Which of these are authoritative

- **HANDOFF.md is the source of truth for "what changed when."** AGENTS.md's "current state"
  section is deliberately *not* kept current move-by-move — it went stale once and is not
  trusted for dates.
- **MISSINGNESS_REPORT.md is GENERATED.** Do not hand-edit it; rebuild with
  `py -X utf8 tools/build_missingness_report.py`. It was hand-maintained once and drifted
  ~12,500 rows out of date while remaining the file people consulted before trusting a feature.
- **CONCERNS.md is append-only in spirit.** Items are marked RESOLVED or
  RESOLVED-AS-NOT-A-BUG rather than deleted, so a question that was already investigated
  doesn't get re-litigated from scratch.

## Keeping docs honest

Most of the errors found in the 2026-08-08 audit were **documentation that disagreed with the
code**, not broken code:

- `is_incumbent` was described as personal incumbency in DATA_DICTIONARY.md and in the public
  Explain modal, while the code computes a party-level flag. That wrong text reached 128 live
  race explanations.
- Three docs called `polls_long_with_results.csv` gitignored, at three different sizes. It is
  committed.
- Four `fetch_*` docstrings pointed at a deprecated script that would have silently corrupted
  the bio table.

When you change behaviour, grep the docs for the old claim. A wrong doc is worse than a
missing one, because it gets believed.
