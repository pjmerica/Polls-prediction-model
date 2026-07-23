# -*- coding: utf-8 -*-
"""Scrape HISTORICAL (1998-2024) House candidate BIO descriptors from the same Wikipedia
per-state "House elections in X" pages fetch_house_primary_results_hist.py already reads
-> appends to data/candidate_bios.csv.

WHY THIS WAS MISSING: fetch_candidate_bios.py's historical target list derives from
primary_polls_wikipedia.csv (Senate/Governor only by design - see that file's own
docstring), so it never had House pages to scrape for 1998-2024 despite its own parser
(parse_page, house=True path) already knowing how to read House pages structurally
(district-heading detection is already there). Only the 2026 predict-time run ever saw
House pages, via a different, separately-built target list. This script drives the SAME
parse_page()/classify() functions (imported, not reimplemented) over an independently-built
House page-target list - same page set as fetch_house_primary_results_hist.py, and shares
that script's at-large-state title pattern + fix.

COVERAGE: within Senate/Governor candidates specifically, the existing bios already matched
39.1% of general-model candidate rows (2018-2024) - real signal, not the ~15% that looked
thin when House's zero coverage was averaged in. This should bring House up to a similar
order of magnitude, not fix pre-2010 sparsity (Wikipedia editing-depth pattern is the same
one found scraping House primary RESULTS - see that script's docstring; expect it here too).

    py -X utf8 fetch_house_candidate_bios_hist.py
Appends to data/candidate_bios.csv (existing 2018-2026 Senate/Governor/2026-House rows are
preserved; this only adds 1998-2024 House rows, deduped by (year,office,state,district,
party,cand_key) same as the existing script does).
"""
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets"))
from scrapers.wikipedia_polls import fetch_page, STATES  # noqa: E402

from fetch_candidate_bios import parse_page, classify, PRIOR_CAND_RX, OUT  # noqa: E402
import features as F  # noqa: E402
from cycles import CYCLES  # noqa: E402

URL_HOUSE_MULTI = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of"
                   "_Representatives_elections_in_{state}")
URL_HOUSE_ATLARGE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of"
                     "_Representatives_election_in_{state}")   # singular "election"
TERRITORIES = {"GU", "PR", "VI", "AS", "MP", "DC"}

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
        print(f"{len(already)} (year, state) House pages already in candidate_bios.csv "
              f"(2026 predict-time scrape) - will still re-check, dedup happens at the end")

    frames = [existing] if len(existing) else []
    n_with_bios = 0
    for i, (year, st) in enumerate(pages):
        state = STATES[st]
        s = state.replace(" ", "_")
        at_large = st in at_large_by_cycle.get(year, set())
        url = (URL_HOUSE_ATLARGE if at_large else URL_HOUSE_MULTI).format(year=year, state=s)
        html = fetch_page(url)
        if html is None:
            continue
        rows = parse_page(html, house=True)
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
