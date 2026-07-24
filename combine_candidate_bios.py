# -*- coding: utf-8 -*-
"""Combine the three per-office candidate-bio files into data/candidate_bios.csv, the file
every consumer (features.py, features_primary.py, check_officeholder.py) actually reads.

Added 2026-07-24 (user instruction) after a real incident: fetch_candidate_bios.py and
fetch_house_candidate_bios_hist.py used to both write to the SAME candidate_bios.csv, each
with its own "resume from what's already there" logic that assumed it was the only writer.
Running them back-to-back silently discarded ~9,500 already-scraped House rows when the
Senate/Governor script's resume-read saw a version of the file that predated the House
script's additions. Fix: each office scrapes to its OWN file
(candidate_bios_senate.csv / _governor.csv / _house.csv) and this script is the ONLY thing
that ever writes the combined candidate_bios.csv - a single, explicit, reviewable step
instead of an implicit merge buried inside a scraper's resume logic.

    py -X utf8 combine_candidate_bios.py

Run this after ANY of the three per-office scrapers, before trusting data/candidate_bios.csv
for a fact-check run or a features.py rebuild. Safe to re-run - always rebuilds the combined
file from scratch off the three source files, never appends to itself.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["candidate_bios_senate.csv", "candidate_bios_governor.csv",
          "candidate_bios_house.csv"]
OUT = os.path.join(HERE, "data", "candidate_bios.csv")

def main():
    frames = []
    for fn in SOURCES:
        p = os.path.join(HERE, "data", fn)
        if not os.path.exists(p):
            print(f"WARNING: {fn} not found - skipping (combined file will be missing "
                  f"this office's rows)")
            continue
        df = pd.read_csv(p, low_memory=False)
        frames.append(df)
        print(f"{fn}: {len(df)} rows")
    if not frames:
        raise SystemExit("no source files found - nothing to combine")

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["year", "office", "state", "district", "party", "cand_key"], keep="first")
    dupes = before - len(combined)
    if dupes:
        print(f"dropped {dupes} cross-file duplicate rows "
              f"(same year/office/state/district/party/cand_key in >1 source file)")

    combined.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(combined)} total bios")
    print("by office:", combined["office"].value_counts().to_dict())
    print("office_level distribution:", combined["office_level"].value_counts().to_dict())

if __name__ == "__main__":
    main()
