# -*- coding: utf-8 -*-
"""Build the ONE authoritative, leak-free office-level table -> data/candidate_bios.csv.

Replaces the ad-hoc combine_candidate_bios.py merge with a single tidy master keyed by
(year, office, state, district, party, cand_key), one office_level each, computed as the
highest office the candidate held STRICTLY BEFORE that election year (user requirement
2026-07-25 - no look-ahead: a first-time 2018 candidate reads 0 even if they later became
a Senator).

Two source kinds, both contributing an as-of-year level:
  WIKIPEDIA (data/candidate_bios_{senate,governor,house}.csv): already contemporaneous -
    each row was scraped from that YEAR's own race page, which describes the candidate as
    they were then (verified: Abigail Spanberger reads 0 on her 2018 page, 4 on 2020+). So
    its office_level is used as-is. Preferred source on any overlap.
  BALLOTPEDIA (data/candidate_bios_ballotpedia.csv): person-level, but carries per-office
    TENURE DATES (offices_json, e.g. [["U.S. House ...",2019,2025],["Governor ...",2026,
    null]]). We compute the as-of-year level = max office-level among offices that STARTED
    before `year`. Gap-filler only (Wikipedia wins overlaps). This is what makes Ballotpedia
    rows leak-free and time-varying instead of a frozen peak.

    py -X utf8 build_office_level_table.py
Writes data/candidate_bios.csv (every consumer reads this). Rebuilds from scratch each run.
"""
import json
import os

import pandas as pd

import features as F
from fetch_candidate_bios_ballotpedia import classify_ballotpedia

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WIKI_SOURCES = ["candidate_bios_senate.csv", "candidate_bios_governor.csv",
               "candidate_bios_house.csv"]
OUT = os.path.join(DATA, "candidate_bios.csv")

def _bp_asof_level(offices, year):
    """Highest office-level among offices whose tenure STARTED strictly before `year`.
    offices = [(phrase, start, end_or_None)]. Returns 0 if none qualify (held no office
    before that race), or None if offices is empty/unknown."""
    if not offices:
        return None
    levels = [classify_ballotpedia(phrase) for phrase, start, _ in offices
              if start is not None and start < year]
    return max(levels) if levels else 0

def main():
    # ---- Wikipedia: already per-year contemporaneous, used as-is ----
    wiki_frames = []
    for fn in WIKI_SOURCES:
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            df = pd.read_csv(p, low_memory=False)
            df["src"] = "wikipedia"
            wiki_frames.append(df)
            print(f"{fn}: {len(df)} rows")
    if not wiki_frames:
        raise SystemExit("no Wikipedia source files - nothing to build")
    wiki = pd.concat(wiki_frames, ignore_index=True)
    wiki = wiki.drop_duplicates(
        subset=["year", "office", "state", "district", "party", "cand_key"], keep="first")
    KEY = ["year", "office", "state", "district", "party", "cand_key"]
    wiki["district"] = wiki["district"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    wiki_keys = set(map(tuple, wiki[KEY].astype(str).values))

    # ---- Ballotpedia: expand person-rows to the uncovered per-race rows, as-of-year level ----
    bp_path = os.path.join(DATA, "candidate_bios_ballotpedia.csv")
    unc_path = os.path.join(DATA, "uncovered_candidates.csv")
    bp_rows = []
    if os.path.exists(bp_path) and os.path.exists(unc_path):
        bp = pd.read_csv(bp_path, low_memory=False)
        bp = bp[bp["office_level"].notna()]
        off_map = {}
        for r in bp.itertuples():
            try:
                off_map[(r.candidate, r.state)] = json.loads(getattr(r, "offices_json", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                off_map[(r.candidate, r.state)] = []
        prior_map = {(r.candidate, r.state): getattr(r, "bio_prior_candidacy", 0) for r in bp.itertuples()}
        unc = pd.read_csv(unc_path, low_memory=False)
        unc["district"] = unc["district"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        # uncovered_candidates.csv encodes statewide Senate/Governor districts as "S"; the
        # Wikipedia bio schema (and the production candidate-table key) use "" there. Normalize
        # so the gap-fill rows match on key AND stay numeric-district-safe downstream
        # (features.load_candidate_bios does int(district) on House rows only when non-empty).
        unc.loc[unc["office"].isin(["Senate", "Governor"]), "district"] = ""
        for r in unc.itertuples():
            key = (r.candidate, r.state)
            if key not in off_map:
                continue
            rowkey = tuple(str(x) for x in (r.year, r.office, r.state, r.district, r.party, r.cand_key))
            if rowkey in wiki_keys:            # Wikipedia already covers it - it wins
                continue
            lvl = _bp_asof_level(off_map[key], int(r.year))
            if lvl is None:
                continue
            bp_rows.append(dict(
                year=r.year, office=r.office, state=r.state, district=r.district,
                party=r.party, name=r.candidate, cand_key=r.cand_key,
                office_level=int(lvl), bio_in_office=0,
                bio_prior_candidacy=int(prior_map.get(key, 0)), src="ballotpedia"))
        print(f"candidate_bios_ballotpedia.csv: {len(bp)} profiles -> {len(bp_rows)} "
              f"leak-free as-of-year rows (gap-filling; Wikipedia preferred)")

    combined = pd.concat([wiki, pd.DataFrame(bp_rows)], ignore_index=True) if bp_rows else wiki
    # final dedup guard (Wikipedia first, so it wins any residual collision)
    combined = combined.drop_duplicates(subset=KEY, keep="first")
    combined.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(combined)} rows "
          f"({(combined['src']=='wikipedia').sum()} wiki, {(combined['src']=='ballotpedia').sum()} ballotpedia)")
    print("office_level distribution:", combined["office_level"].value_counts().sort_index().to_dict())

if __name__ == "__main__":
    main()
