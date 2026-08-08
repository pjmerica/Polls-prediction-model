# -*- coding: utf-8 -*-
"""Regenerate MISSINGNESS_REPORT.md from the CURRENT data.

Added 2026-08-08. The report was previously hand-written and went stale without anyone
noticing: it described `polls_long_with_results.csv` as "22,546 rows" long after the 4->14
cycle expansion took it to 35,052, and its feature-table section still said "2018-2024" when
the model trains on 1998-2024. Every percentage in it was computed against the smaller file.

A stale missingness report is worse than none - it is exactly the document someone consults
before deciding whether a feature is safe to use.

Run it after any change that alters the training data:

    py -X utf8 tools/build_missingness_report.py

It rebuilds the file in place and stamps it with the row counts it actually measured, so the
next reader can tell at a glance whether it matches the data in front of them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, data  # noqa: E402

import pandas as pd  # noqa: E402

POLLS = os.path.join(ROOT, "polls_long_with_results.csv")
FEATS = data("candidate_table_2026.csv")
OUT = os.path.join(ROOT, "docs", "MISSINGNESS_REPORT.md")   # docs/ since the 2026-08-08 move


def table(df):
    """Markdown rows: column | dtype | % missing | # missing."""
    out = []
    n = len(df)
    for c in df.columns:
        miss = int(df[c].isna().sum())
        pct = (100.0 * miss / n) if n else 0.0
        out.append(f"| `{c}` | {df[c].dtype} | {pct:.1f}% | {miss:,} |")
    return "\n".join(out)


def main():
    if not os.path.exists(POLLS):
        sys.exit(f"missing {POLLS} - run pipeline/build/build_dataset.ipynb first")

    polls = pd.read_csv(POLLS, low_memory=False)
    years = sorted(polls["year"].dropna().unique().astype(int)) if "year" in polls else []
    trained = polls[polls["has_result"] == 1] if "has_result" in polls else polls

    parts = [
        "# Missingness report",
        "",
        "> **Generated file — do not hand-edit.** Rebuild with",
        "> `py -X utf8 tools/build_missingness_report.py` after any change to the training data.",
        "",
        f"_Source: `polls_long_with_results.csv`, **{len(polls):,} rows**, "
        f"{len(polls.columns)} columns. Cycles: {years[0]}–{years[-1]} "
        f"({len(years)} present). Rows with a joined result (`has_result=1`, what actually "
        f"trains): **{len(trained):,}**._",
        "",
        "## `polls_long_with_results.csv` (long poll file)",
        "",
        "One row per candidate per poll. High missingness in the 538-era metadata columns",
        "(`pollster_id`, `sponsors`, `numeric_grade`, …) is EXPECTED and not a defect: the",
        "1998–2016 archive carries far less metadata than the 2018+ feed, and none of those",
        "columns are model features. Check `METHODOLOGY.md` before treating any number here",
        "as a problem.",
        "",
        "| column | dtype | % missing | # missing |",
        "|---|---|---|---|",
        table(polls),
        "",
    ]

    if os.path.exists(FEATS):
        f = pd.read_csv(FEATS, low_memory=False)
        parts += [
            "## Live 2026 feature table (`data/candidate_table_2026.csv`)",
            "",
            f"_{len(f):,} candidate-rows — the SERVE-time table, which is where train/serve",
            "skew shows up. A feature that is well-populated in training but mostly NaN here",
            "is the failure mode that got `poll_adj` dropped (CONCERNS.md) and that keeps",
            "`poll_last7` out of the general model._",
            "",
            "| column | dtype | % missing | # missing |",
            "|---|---|---|---|",
            table(f),
            "",
        ]

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {OUT}: {len(polls):,} poll rows"
          + (f", {len(pd.read_csv(FEATS, low_memory=False)):,} feature rows"
             if os.path.exists(FEATS) else ""))


if __name__ == "__main__":
    main()
