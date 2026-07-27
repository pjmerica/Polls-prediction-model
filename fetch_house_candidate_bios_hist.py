# -*- coding: utf-8 -*-
"""Scrape HISTORICAL (1998-2024) House candidate BIO descriptors from the same Wikipedia
per-state "House elections in X" pages fetch_house_primary_results_hist.py already reads
-> data/candidate_bios_house.csv (its OWN file - see the note below on why).

WHY THIS WAS MISSING: fetch_candidate_bios.py's historical target list derives from
primary_polls_wikipedia.csv (Senate/Governor only by design - see that file's own
docstring), so it never had House pages to scrape for 1998-2024 despite its own parser
(parse_page, house=True path) already knowing how to read House pages structurally
(district-heading detection is already there). Only the 2026 predict-time run ever saw
House pages, via a different, separately-built target list. This script drives the SAME
parse_page()/classify() functions (imported, not reimplemented) over an independently-built
House page-target list - same page set as fetch_house_primary_results_hist.py, and shares
that script's at-large-state title pattern + fix.

SEPARATE OUTPUT FILE (changed 2026-07-24, user instruction after a real incident): this
script used to append to the SAME data/candidate_bios.csv that fetch_candidate_bios.py
writes. Running the two in sequence, with fetch_candidate_bios.py's own resume logic
reading whatever candidate_bios.csv happened to contain at that moment, silently discarded
~9,500 already-scraped House rows when fetch_candidate_bios.py ran AFTER this script but
started its "existing data" read from a version of the file that predated this script's
House rows. Fix: Senate, Governor, and House each get their OWN output file
(candidate_bios_senate.csv / _governor.csv / _house.csv), written ONLY by the script(s)
that scrape that office. combine_candidate_bios.py concatenates all three into
candidate_bios.csv (what every consumer actually reads) as an explicit, separate, manual
step - no script ever overwrites another's output again.

COVERAGE: within Senate/Governor candidates specifically, the existing bios already matched
39.1% of general-model candidate rows (2018-2024) - real signal, not the ~15% that looked
thin when House's zero coverage was averaged in. This should bring House up to a similar
order of magnitude, not fix pre-2010 sparsity (Wikipedia editing-depth pattern is the same
one found scraping House primary RESULTS - see that script's docstring; expect it here too).

    py -X utf8 fetch_house_candidate_bios_hist.py
Writes data/candidate_bios_house.csv (safe to re-run - resumes from THIS file only, deduped
by (year,office,state,district,party,cand_key)). Run combine_candidate_bios.py afterward to
rebuild the merged candidate_bios.csv every consumer reads.
"""
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets"))
from scrapers.wikipedia_polls import fetch_page, STATES  # noqa: E402

from fetch_candidate_bios import (parse_page, parse_results_tables, classify,  # noqa: E402
                                  PRIOR_CAND_RX)
import features as F  # noqa: E402
from cycles import CYCLES  # noqa: E402

OUT = os.path.join(HERE, "data", "candidate_bios_house.csv")

URL_HOUSE_MULTI = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of"
                   "_Representatives_elections_in_{state}")
URL_HOUSE_ATLARGE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of"
                     "_Representatives_election_in_{state}")   # singular "election"
TERRITORIES = {"GU", "PR", "VI", "AS", "MP", "DC"}

# INVESTIGATED 2026-07-24 (why is WA missing from House bios 2020/2022/2024?): NOT a URL
# bug - both "Washington" plain and "Washington_(state)" fetch the House-elections page
# fine (verified directly with fetch_page after an initial wrong assumption here). The
# real cause is still open - likely a transient fetch failure or rate-limit during the
# original ~700-page scrape run (California's 2024 page separately confirmed to fetch fine
# now too, suggesting the same transient-failure explanation, not a systematic URL issue).
# No code fix here; the actual remedy is re-running the scrape for the specific missing
# (year, state) pairs, which naturally retries the fetch.
HOUSE_TITLE_OVERRIDE = {}

def _at_large_states_by_cycle():
    """Same ground-truth logic as fetch_house_primary_results_hist.py - not duplicated
    reasoning, just not importable across the two scripts without an awkward shared module
    for one small function."""
    r = pd.read_csv(os.path.join(HERE, "data", "res_house.csv"), low_memory=False)
    r = r[r["cycle"].isin(CYCLES)]
    def extract_dist(s):
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else 0
    r["d"] = r["office_seat_name"].map(extract_dist)
    mx = r.groupby(["cycle", "state_abbrev"])["d"].max()
    out = {}
    for (cyc, st), d in mx.items():
        if st in TERRITORIES:
            continue
        if d <= 1:
            out.setdefault(int(cyc), set()).add(st)
    return out

def main():
    at_large_by_cycle = _at_large_states_by_cycle()
    pages = [(cyc, st) for cyc in CYCLES for st in STATES if st not in TERRITORIES]
    print(f"{len(pages)} (cycle, state) House pages to scrape")

    existing = pd.read_csv(OUT, low_memory=False) if os.path.exists(OUT) else pd.DataFrame()
    already = set()
    if len(existing):
        h_existing = existing[existing["office"] == "House"]
        already = set(zip(h_existing["year"], h_existing["state"]))
        print(f"{len(already)} (year, state) House pages already in "
              f"{os.path.basename(OUT)} - will skip those, dedup happens at the end")

    frames = [existing] if len(existing) else []
    n_with_bios = 0
    for i, (year, st) in enumerate(pages):
        # skip pages already in OUR OWN output file (2026-07-24: `already` was computed
        # but never consulted before - every re-run re-fetched all ~700 pages)
        if (year, st) in already:
            continue
        state = HOUSE_TITLE_OVERRIDE.get(st, STATES[st])
        s = state.replace(" ", "_")
        at_large = st in at_large_by_cycle.get(year, set())
        url = (URL_HOUSE_ATLARGE if at_large else URL_HOUSE_MULTI).format(year=year, state=s)
        html = fetch_page(url)
        if html is None:
            continue
        rows = parse_page(html, house=True)
        # SUPPLEMENT with results-table rows (2026-07-27): most pre-2012 multi-district House
        # pages have NO candidate bullets, only "Party|Candidate|Votes|%" results tables
        # (verified: 2008 WA, 2010 FL, 2004 TX). Bullets carry real descriptors so they WIN;
        # only add a table candidate the bullets didn't find (matched per district on
        # party + normalized name). Incumbents come through with a synthesized descriptor.
        have = {(d, F.npar(p), F.norm_name(n)) for d, p, n, _ in rows}
        for tr in parse_results_tables(html, house=True, office="House"):
            if (tr[0], F.npar(tr[1]), F.norm_name(tr[2])) not in have:
                rows.append(tr)
        if at_large:
            # at-large pages have no "District N" heading for parse_page's own detector to
            # find, so district stays None there - same gap fixed in the results scraper's
            # parser (at_large=True), applied here at the row level since parse_page's
            # signature is shared with the LIVE 2026 scrape and shouldn't change behavior.
            rows = [(1, party, name, desc) for (_d, party, name, desc) in rows]
        df = pd.DataFrame(rows, columns=["district", "party", "name", "descriptor"])
        df["year"], df["office"], df["state"] = year, "House", st
        df["cand_key"] = df["name"].map(F.norm_name)
        df["office_level"] = df["descriptor"].map(lambda d: classify(d, office="House"))
        df["bio_in_office"] = df["descriptor"].astype(str).str.contains(
            "present", case=False).astype(int)
        df["bio_prior_candidacy"] = df["descriptor"].astype(str).str.contains(
            PRIOR_CAND_RX).astype(int)
        if len(df):
            n_with_bios += 1
            print(f"  {year} {st}{' (at-large)' if at_large else ''}: {len(df)} candidate bios")
        frames.append(df)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(pages)} pages checked, {n_with_bios} had bios")
        time.sleep(0.8)

    allb = pd.concat(frames, ignore_index=True)
    allb = allb.drop_duplicates(subset=["year", "office", "state", "district",
                                        "party", "cand_key"], keep="first")
    allb.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(allb)} total bios ({len(allb) - len(existing)} net new)")
    print("office_level distribution (House only, this run's new rows):")
    new_house = allb[(allb["office"] == "House") & (allb["year"].isin(CYCLES))]
    print(new_house.groupby("year").size().to_string())

if __name__ == "__main__":
    main()
