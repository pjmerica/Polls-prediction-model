# -*- coding: utf-8 -*-
"""Fact-check iteration 3: candidate BIO features (fetch_candidate_bios.py output).

Pass 1: known-truth office levels (hand-verified public facts).
Pass 2: cross-source consistency - candidates our RESULTS archives show winning a
        federal/statewide general should classify office_level >= 3 wherever a bio
        exists. Reports the agreement rate and every disagreement.

    py -X utf8 check_officeholder.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import pandas as pd

import features as F
import features_primary as FP
from candidate_history import CandidateHistory



def main():
    bios = pd.read_csv("data/candidate_bios.csv", low_memory=False)
    n = F.norm_name
    fails = 0

    def lvl(year, office, st, name, party):
        m = bios[(bios["year"] == year) & (bios["office"] == office)
                 & (bios["state"] == st) & (bios["cand_key"] == n(name))
                 & (bios["party"].map(F.npar) == party)]
        return int(m["office_level"].iloc[0]) if len(m) else None

    # ---- pass 1: known truths ----
    checks = [
        ("McMorrow = state senator (2)", 2026, "Senate", "MI", "Mallory McMorrow", "DEM", 2),
        ("Stevens = US rep (4)", 2026, "Senate", "MI", "Haley Stevens", "DEM", 4),
        ("El-Sayed = county health director (1)", 2026, "Senate", "MI", "Abdul El-Sayed", "DEM", 1),
        ("Buttigieg = fmr US cabinet (4)", 2026, "Senate", "MI", "Pete Buttigieg", "DEM", 4),
        ("Paxton = TX AG (3)", 2026, "Senate", "TX", "Ken Paxton", "REP", 3),
        ("Talarico = TX state rep (2)", 2026, "Senate", "TX", "James Talarico", "DEM", 2),
        ("Barr = US rep (4)", 2026, "Senate", "KY", "Andy Barr", "REP", 4),
    ]
    for label, y, of, st, name, pty, want in checks:
        got = lvl(y, of, st, name, pty)
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got}")
        fails += (not ok)

    # ---- pass 2: cross-source consistency ----
    hist = CandidateHistory()
    b = bios.copy()
    b["party_std"] = b["party"].map(F.npar)
    scored = agree = 0
    disagreements = []
    for r in b.itertuples():
        if pd.isna(r.cand_key):
            continue
        h = hist.history(int(r.year), r.state, r.cand_key)
        if h["hist_prior_wins"] >= 1:                 # won a tracked general before
            scored += 1
            if r.office_level >= 3:
                agree += 1
            else:
                disagreements.append((r.year, r.state, r.name,
                                      r.office_level, str(r.descriptor)[:70]))
    rate = agree / scored if scored else float("nan")
    print(f"\n  cross-source: {scored} bio'd candidates with prior tracked general wins; "
          f"{rate:.0%} classify office_level>=3")
    for x in disagreements[:12]:
        print("   DISAGREE:", x)
    # threshold: below 85% means the classifier or the join is broken
    ok = rate >= 0.85
    print(f"  {'PASS' if ok else 'FAIL'}  consistency >= 85%")
    fails += (not ok)

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECKS FAILED'}")
    return fails

if __name__ == "__main__":
    raise SystemExit(main())
