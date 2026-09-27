# -*- coding: utf-8 -*-
"""Re-classify Wikipedia bio rows with the current classify() (2026-09-25).

    py -X utf8 tools/repair_bio_relatives.py            # dry run: list every change
    py -X utf8 tools/repair_bio_relatives.py --apply    # rewrite the three per-office files
    py -X utf8 pipeline/build/build_office_level_table.py   # then rebuild candidate_bios.csv

classify() credited candidates with OTHER people's offices: "grandson of former U.S. Senator
Paul Laxalt" (Adam Laxalt -> 4, really 3), "father-in-law of U.S. representative Max Miller"
(Bernie Moreno -> 4, really 0), "Running mate: Shane Hernandez, former state representative"
(Tudor Dixon -> 2, really 0). classify() now strips those clauses; this re-runs it ONLY on
rows whose descriptor contains one, so hand-coded rows (empty descriptor, src='manual' - see
the AGENTS.md trap) are never touched.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths  # noqa: E402

import argparse
import os
import re

import pandas as pd

from fetch_candidate_bios import classify  # noqa: E402

FILES = ["candidate_bios_senate.csv", "candidate_bios_governor.csv", "candidate_bios_house.csv"]
RX = re.compile(r"(?i)running\s*mate|\b(?:son|daughter|grandson|granddaughter|grandfather|"
                r"grandmother|father|mother|husband|wife|spouse|brother|sister|nephew|niece|"
                r"cousin|uncle|aunt|widow|widower|child|grandchild)(?:\s*-\s*in\s*-\s*law)?\s+of\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    total = 0
    for fn in FILES:
        path = os.path.join(paths.DATA, fn)
        b = pd.read_csv(path, low_memory=False)
        desc = b["descriptor"].fillna("").astype(str)
        manual = (b["src"].astype(str).eq("manual") if "src" in b.columns
                  else pd.Series(False, index=b.index))
        # every row that HAS a descriptor (so the current classify() - relatives, running
        # mates, and the statewide offices added 2026-09-25 - applies uniformly); rows with no
        # descriptor are hand-coded and must never be re-classified (AGENTS.md trap)
        hit = desc.str.strip().ne("") & ~manual
        new = [classify(d, o) for d, o in zip(desc[hit], b.loc[hit, "office"])]
        chg = b.loc[hit].assign(new_level=new)
        chg = chg[chg["new_level"] != chg["office_level"]]
        total += len(chg)
        print(f"== {fn}: {int(hit.sum())} descriptor rows, {len(chg)} change")
        for r in chg.itertuples():
            print(f"   {r.year} {r.state} {r.office:8} {str(r.name)[:24]:24} "
                  f"{r.office_level} -> {r.new_level}   | {str(r.descriptor)[:110]}")
        if args.apply and len(chg):
            b.loc[chg.index, "office_level"] = chg["new_level"].astype(int)
            b.to_csv(path, index=False)
    print(f"\n{total} rows change" + ("" if args.apply else "  (dry run - pass --apply)"))


if __name__ == "__main__":
    main()
