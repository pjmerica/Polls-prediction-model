# -*- coding: utf-8 -*-
"""Fact-check battery for candidate_history.py. Run after any change to the history
builder or the results files. Hard-fails on known-truth violations.

Iteration 1: known-truth assertions (hand-verified public facts).
Iteration 2: name-collision audit - same (state, norm_name) appearing in TWO offices or
             districts in the SAME cycle can't be one person on most ballots; measures
             how much identity-conflation the (name,state) join carries.
Iteration 3 lives in check_officeholder.py (cross-source vs Wikipedia descriptors).

    py -X utf8 check_candidate_history.py
"""
import pandas as pd

import features as F
from candidate_history import CandidateHistory, _general_rows

def main():
    h = CandidateHistory()
    n = F.norm_name

    # ---- iteration 1: known truths (facts checkable against public record) ----
    checks = [
        # (label, cycle, state, name, field, op, value)
        ("Stevens 4 House wins by 2026", 2026, "MI", "Haley Stevens",
         "hist_prior_wins", "==", 4),
        ("Rogers won House 7x, lost 2024 Sen", 2026, "MI", "Mike Rogers",
         "hist_prior_wins", ">=", 6),
        ("Rogers 2024 Sen loss = last run, close race", 2026, "MI", "Mike Rogers",
         "hist_years_since_last_run", "==", 2.0),
        ("El-Sayed never ran a tracked general", 2026, "MI", "Abdul El-Sayed",
         "hist_prior_runs", "==", 0),
        ("Espaillat EXACTLY 5 House wins (fusion lines deduped)", 2026, "NY",
         "Adriano Espaillat", "hist_prior_wins", "==", 5),
        ("Espaillat pre-2016 = no tracked runs", 2016, "NY", "Adriano Espaillat",
         "hist_prior_runs", "==", 0),
        ("John James: 2 Sen losses + House wins", 2026, "MI", "John James",
         "hist_prior_runs", ">=", 4),
        ("John James ever won", 2026, "MI", "John James", "hist_ever_won", "==", 1),
        ("Wasserman Schultz long House career", 2026, "FL", "Debbie Wasserman Schultz",
         "hist_prior_wins", ">=", 9),
        ("Roy Cooper won Gov 2016+2020", 2026, "NC", "Roy Cooper",
         "hist_prior_wins", ">=", 2),
        ("Paxton: AG only, no tracked federal/gov runs", 2026, "TX", "Ken Paxton",
         "hist_prior_runs", "==", 0),
        ("David Jolly won a House race", 2026, "FL", "David Jolly",
         "hist_ever_won", "==", 1),
    ]
    fails = 0
    for label, cyc, st, name, field, op, want in checks:
        got = h.history(cyc, st, n(name))[field]
        ok = (got == want) if op == "==" else (got >= want)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {field}={got} (want {op}{want})")
        fails += (not ok)

    # leak-safety: as-of semantics (a 2018 lookup must not see 2018+ results)
    s24 = h.history(2024, "MI", n("Haley Stevens"))["hist_prior_wins"]
    s19 = h.history(2018, "MI", n("Haley Stevens"))["hist_prior_wins"]
    ok = (s24 == 3 and s19 == 0)
    print(f"  {'PASS' if ok else 'FAIL'}  as-of leak check: Stevens wins @2024={s24} "
          f"(want 3) @2018={s19} (want 0)")
    fails += (not ok)

    # ---- iteration 2: same-cycle identity collisions ----
    g = _general_rows()
    dup = (g.groupby(["cycle", "state", "cand_key"])["office"].nunique() > 1)
    multi_office = int(dup.sum())
    # same cycle+office+state+name should be ONE candidacy row-group; House multi-district
    hh = g[g["office"] == "House"].copy()
    print(f"\n  collision audit: same (cycle,state,name) in MULTIPLE offices: "
          f"{multi_office} of {g.groupby(['cycle','state','cand_key']).ngroups} "
          f"({multi_office / g.groupby(['cycle','state','cand_key']).ngroups:.2%})")
    if multi_office:
        ex = g[g.set_index(["cycle", "state", "cand_key"]).index.isin(
            dup[dup].index)].sort_values(["cycle", "state"])
        print(ex.head(8).to_string(index=False))

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECKS FAILED'}")
    return fails

if __name__ == "__main__":
    raise SystemExit(main())
