# -*- coding: utf-8 -*-
"""One-shot repair: re-classify office_level for bios whose descriptor names a FUTURE office.

    py -X utf8 scripts_fix_future_office_level.py [--apply]

Why (2026-08-01): Wikipedia bios are written after the fact, so a candidate's blurb can
describe an office they only won LATER. classify() read those as offices already held:
    Raphael Warnock, 2016 GA Senate: "pastor ... and future U.S. Senator for this seat" -> 4
    Ron DeSantis,    2012 FL House:  "Iraq War veteran, former prosecutor and future
                                      Florida governor"                                 -> 3
Both held NO office at the time. That is label leakage - bio_office_level would be telling the
model "this person later became a senator" as if it were a pre-election credential.

classify() now strips future-office phrases (see fetch_candidate_bios.classify). This script
re-runs it over the committed bio CSVs so the stored office_level catches up WITHOUT
re-scraping (no network).

CRITICAL: only rows with real descriptor prose are re-classified. ~8000 rows are hand-coded
(src='manual', descriptor empty) - running classify() on an empty descriptor returns 0 and
would silently wipe correct hand-coded levels for sitting members of Congress (caught in
dry-run: Peltola/Risch/Schiff 4 -> 0). Those rows are left exactly as they are.
"""
import argparse
import os

import pandas as pd

from fetch_candidate_bios import classify

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["data/candidate_bios.csv", "data/candidate_bios_house.csv",
         "data/candidate_bios_senate.csv", "data/candidate_bios_governor.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total = 0
    for rel in FILES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, low_memory=False)
        if not {"descriptor", "office_level", "office"} <= set(d.columns):
            print(f"{rel}: missing columns, skipped")
            continue

        # ONLY rows with actual descriptor prose (see docstring).
        has_desc = d["descriptor"].notna() & (d["descriptor"].astype(str).str.strip() != "")
        fresh = d["office_level"].copy()
        fresh.loc[has_desc] = [classify(r.descriptor, office=r.office)
                               for r in d[has_desc].itertuples()]
        changed = fresh != d["office_level"]
        n = int(changed.sum())
        total += n
        print(f"{os.path.basename(rel):32s} {n:4d} / {len(d):6d} re-classified "
              f"({int(has_desc.sum())} rows have prose, {int((~has_desc).sum())} hand-coded "
              f"left alone)")
        for r in d[changed].head(8).itertuples():
            new = classify(r.descriptor, office=r.office)
            print(f"     {r.year} {r.name}: {r.office_level} -> {new}")
        if args.apply and n:
            d["office_level"] = fresh
            d.to_csv(path, index=False)

    print()
    print(f"TOTAL re-classified: {total}")
    print("WROTE the files." if args.apply else "DRY RUN - pass --apply to write.")


if __name__ == "__main__":
    main()
