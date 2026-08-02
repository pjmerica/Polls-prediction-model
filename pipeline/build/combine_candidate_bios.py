# -*- coding: utf-8 -*-
"""DEPRECATED 2026-07-25 - use build_office_level_table.py instead.

This did a plain concat of the per-office files + a naive Ballotpedia merge that stamped
each Ballotpedia person's FROZEN office_level onto all their rows (a look-ahead risk: a
first-time candidate who later became a Senator would read level 4 in their early races).
build_office_level_table.py supersedes it with a LEAK-FREE as-of-year computation from
Ballotpedia's per-office tenure dates (highest office held strictly BEFORE each election
year). Kept (not deleted, per the archive-don't-delete rule) only for reference.

Original purpose:
Combine the three per-office candidate-bio files into data/candidate_bios.csv, the file
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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import os

import pandas as pd

HERE = ROOT   # repo root (paths.py) - this file lives in a subfolder
# WIKIPEDIA sources share the full (year,office,state,district,party,cand_key) schema and
# concat directly. BALLOTPEDIA (candidate_bios_ballotpedia.csv, added 2026-07-24) is
# PERSON-level (candidate,state,office_level - one profile covers all a person's races), so
# it's expanded back to per-race rows below and only fills gaps Wikipedia missed.
SOURCES = ["candidate_bios_senate.csv", "candidate_bios_governor.csv",
          "candidate_bios_house.csv"]
BALLOTPEDIA = "candidate_bios_ballotpedia.csv"
UNCOVERED = "uncovered_candidates.csv"
OUT = os.path.join(HERE, "data", "candidate_bios.csv")

def _expand_ballotpedia():
    """candidate_bios_ballotpedia.csv (person-level) -> per-race rows matching the Wikipedia
    schema. Each BP hit is a person's office_level; map it onto their (year,office,state,
    district,party,cand_key) rows from uncovered_candidates.csv (the exact rows BP was asked
    to fill). Returns a DataFrame or None if either file is absent."""
    bp_path = os.path.join(HERE, "data", BALLOTPEDIA)
    unc_path = os.path.join(HERE, "data", UNCOVERED)
    if not (os.path.exists(bp_path) and os.path.exists(unc_path)):
        return None
    import features as F


    bp = pd.read_csv(bp_path, low_memory=False)
    bp = bp[bp["office_level"].notna()]              # only real hits
    lvl_map = {(r.candidate, r.state): r.office_level for r in bp.itertuples()}
    prior_map = {(r.candidate, r.state): getattr(r, "bio_prior_candidacy", 0)
                 for r in bp.itertuples()}
    unc = pd.read_csv(unc_path, low_memory=False)
    rows = []
    for r in unc.itertuples():
        key = (r.candidate, r.state)
        if key not in lvl_map:
            continue
        rows.append(dict(
            year=r.year, office=r.office, state=r.state,
            district=("" if pd.isna(r.district) else r.district),
            party=r.party, name=r.candidate, cand_key=r.cand_key,
            office_level=int(lvl_map[key]),
            bio_in_office=0,                          # BP infobox doesn't give this cleanly
            bio_prior_candidacy=int(prior_map.get(key, 0)),
            src="ballotpedia"))
    return pd.DataFrame(rows) if rows else None

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

    wiki = pd.concat(frames, ignore_index=True)
    before = len(wiki)
    wiki = wiki.drop_duplicates(
        subset=["year", "office", "state", "district", "party", "cand_key"], keep="first")
    if before - len(wiki):
        print(f"dropped {before - len(wiki)} cross-file duplicate rows (Wikipedia sources)")

    # Ballotpedia fills ONLY rows Wikipedia doesn't already cover (Wikipedia preferred).
    combined = wiki
    bp = _expand_ballotpedia()
    if bp is not None:
        wiki_keys = set(zip(wiki["year"], wiki["office"], wiki["state"],
                            wiki["district"].fillna("").astype(str), wiki["party"], wiki["cand_key"]))
        bp["_k"] = list(zip(bp["year"], bp["office"], bp["state"],
                            bp["district"].astype(str), bp["party"], bp["cand_key"]))
        bp_new = bp[~bp["_k"].isin(wiki_keys)].drop(columns="_k")
        combined = pd.concat([wiki, bp_new], ignore_index=True)
        print(f"{BALLOTPEDIA}: {len(bp)} expanded rows, {len(bp_new)} NEW (gap-filling; "
              f"Wikipedia preferred on overlap)")

    combined.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(combined)} total bios")
    print("by office:", combined["office"].value_counts().to_dict())
    print("office_level distribution:", combined["office_level"].value_counts().to_dict())

if __name__ == "__main__":
    main()
