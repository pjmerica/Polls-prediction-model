# -*- coding: utf-8 -*-
"""Scrape HISTORICAL (1998-2024) House primary results from Wikipedia's per-state
"X United States House of Representatives elections in Y" summary pages ->
data/house_primary_results_hist.csv.

BACKGROUND: fetch_primary_results_2026.py already has House-page parsing machinery
(URL_HOUSE, house=True in parse_results_tables) but its --hist mode only ever scraped
Senate/Governor, because it derives its historical page list from primary_polls_wikipedia.csv,
which was deliberately scoped Senate+Governor-only (fetch_primary_polls_wikipedia.py:
"House historical primary polling is too sparse for pages to exist reliably"). That was a
call about POLL tables specifically; RESULTS tables turn out to behave differently - this
script drives the SAME parser (parse_results_tables, imported) over independently-built House
page targets, one page per (state, cycle) rather than per-district (these pages summarize an
entire state's House delegation).

COVERAGE (actual, from running this script 2026-07-22 - supersedes an earlier, WRONG
docstring claim of a hard ~2010 cutoff based on spot-checking only Michigan/Texas): coverage
GROWS GRADUALLY, not at a clean cutoff - distinct states with any House primary data climbs
from 6 (1998) to 11 (2000-02) to ~13-22 (2004-08) to 31 (2010) to 48+ (2012 onward, roughly
full). This is Wikipedia editing depth fading with distance from the present, not a format
change at one date - some smaller/lower-profile states (Idaho, Nebraska) had detailed 1998
pages while larger states (Michigan, Texas) didn't get primary detail until later.

FINAL VALIDATED NUMBERS (data/house_primary_results_hist.csv, this run): 11,218 candidate
rows, 4,900 House party-races, across all 14 cycles (1998-2024). Per-cycle race counts: 1998
76, 2000 92, 2002 124, 2004 116, 2006 185, 2008 177, 2010 242, 2012 563, 2014 508, 2016 526,
2018 591, 2020 580, 2022 570, 2024 550 (also printed at the end of every run).

FACT-CHECK (2026-07-22, all PASS on the final file): exactly one is_winner=True per race,
0/4900 exceptions. Five hard historical spot-checks: 2014 VA-7 REP (Eric Cantor's primary
loss to Brat), 2018 MA-7 DEM (Pressley beat incumbent Capuano), 2020 NY-16 DEM (Bowman beat
incumbent Engel), 2022 MI-3 REP (Meijer LOST to Gibbs), 2018 NY-14 DEM (Ocasio-Cortez beat
incumbent Crowley, 56.7%) - all correct, including the two where the "obvious" pick would
be wrong (Cantor and Meijer were sitting incumbents who LOST).

MERGED-ROUND BUG (found + fixed 2026-07-22): some pages (11 Texas-2012 races) have no
separate "Runoff" heading, so first-round AND runoff results land in one table_seq -
without a guard, pct columns summed to 150-200% and a margin-of-victory feature would
silently compare the winner against a stale first-round also-ran instead of the real
runner-up. Two-pass fix in main(): (1) collapse a candidate appearing twice within one
table_seq to their higher-vote (runoff) row; (2) when a race's WHOLE pct sum is still >130%
after pass 1, keep only the top-2-by-votes if their pct values sum to ~100 (the decided
runoff pair) and drop the rest as first-round-only debris. Pass 2's >130% gate is load-
bearing: an earlier, narrower version that triggered on "do the top two sum to ~100" alone
would have silently deleted real 3rd-place-and-lower candidates from 69 ordinary (non-
runoff) 3+ candidate races, where the top two legitimately dominate the field on their own.
Verified against those 69 races post-fix: row counts unchanged.

Five remaining races have an out-of-[50,150]% pct sum after all fixes (2006 OH-4 DEM, 2006
OR-3 REP, 2008 OR-4 DEM, 2018/2020 AK-1 DEM) - inspected individually, not a parsing bug:
single-candidate pages where the source table's pct column uses a different denominator
(e.g. total ballots cast, not just the party's own field) and the two Alaska rows are a
genuinely crowded field where this scraper likely isn't seeing every minor candidate. None
of these threaten is_winner (unambiguous in all five) or leave a materially wrong margin.

AT-LARGE STATES (single House seat, no per-district sections): AK, DE, MT, ND, SD, VT, WY -
cycle-dependent (SD/MT gained/lost a second seat across redistricting; see
_at_large_states_by_cycle). These use a DIFFERENT Wikipedia title pattern (singular
"election in", not plural "elections in" - verified: the plural title 404s for SD/2018,
the singular title is correct) and parse_results_tables needs at_large=True since there's no
"District N" heading to detect (added there 2026-07-22).

    py -X utf8 fetch_house_primary_results_hist.py
Writes data/house_primary_results_hist.csv (committed; fetch once - static-data principle).
Takes a while: 14 cycles x ~50 states = ~700 page fetches at the polite 0.8s/page pace.
"""
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets"))
from scrapers.wikipedia_polls import fetch_page, STATES  # noqa: E402

from fetch_primary_results_2026 import parse_results_tables  # noqa: E402
import features as F  # noqa: E402
from cycles import CYCLES  # noqa: E402

URL_HOUSE_MULTI = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of"
                   "_Representatives_elections_in_{state}")
URL_HOUSE_ATLARGE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of"
                     "_Representatives_election_in_{state}")   # singular "election"

# territories: not part of the general model's scope (no electoral votes / voting House
# seat in the sense the results files track for the model) - excluded outright.
TERRITORIES = {"GU", "PR", "VI", "AS", "MP", "DC"}

def _at_large_states_by_cycle():
    """{cycle: set(state_abbrev)} - states with exactly one House district that cycle,
    from the committed results archive (ground truth, not a hardcoded guess: SD lost its
    2nd seat after 1980 reapportionment, MT regained one in 2020 - both show up correctly
    here because they're read from res_house.csv, not asserted)."""
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
    print(f"{len(pages)} (cycle, state) pages to scrape across {len(CYCLES)} cycles")

    allrows = []
    n_with_primaries = 0
    for i, (year, st) in enumerate(pages):
        state = STATES[st]
        s = state.replace(" ", "_")
        at_large = st in at_large_by_cycle.get(year, set())
        url = (URL_HOUSE_ATLARGE if at_large else URL_HOUSE_MULTI).format(year=year, state=s)
        html = fetch_page(url)
        if html is None:
            continue
        rows = parse_results_tables(html, f"{year}_{st}_House", house=True,
                                    at_large=at_large)
        if rows:
            n_with_primaries += 1
            n_races = len(set(r["race_id"] for r in rows))
            print(f"  {year} {st}{' (at-large)' if at_large else ''}: "
                  f"{len(rows)} result rows, {n_races} party-races")
        allrows.extend(rows)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(pages)} pages checked, "
                  f"{len(allrows)} rows so far, {n_with_primaries} pages had primary data")
        time.sleep(0.8)

    df = pd.DataFrame(allrows)
    if not len(df):
        raise SystemExit("no results parsed at all - page structure or URL pattern changed?")

    # winner = top votes in the LAST table per (race_id): runoff supersedes round 1
    last = df.groupby("race_id")["table_seq"].transform("max")
    fin = df[df["table_seq"] == last].copy()

    # MERGED-ROUND GUARD (found 2026-07-22, fact-check pass on real data: Texas-2012 races
    # had first-round AND runoff results inside the SAME table_seq - no separate "Runoff"
    # heading on the page for the parser's stage tracker to key off, so both rounds got
    # treated as one table. Two symptoms, two passes:
    # (1) a candidate who made the runoff appears TWICE (round-1 tally + runoff tally) -
    #     collapse to their higher-vote (runoff) row.
    # (2) eliminated first-round-only candidates still linger as single rows below the
    #     runoff pair, inflating the field and corrupting a "margin vs runner-up" feature
    #     (e.g. TX-25 2012: Williams 58.0% / Riddle 42.0% are the real runoff, but 10 more
    #     stale round-1 rows remained). Detected by: once the two current vote-leaders'
    #     pct values sum to ~100 (the runoff is fully decided between just them), every
    #     lower-vote row in that race is round-1-only debris - drop it.
    # Without both passes, TX-2012 pct columns summed to ~150-200% and margin-of-victory
    # would silently compare a runoff winner against a stale round-1 also-ran.
    dup = fin.duplicated(subset=["race_id", "candidate"], keep=False)
    n_dup_races = fin.loc[dup, "race_id"].nunique() if dup.any() else 0
    if dup.any():
        fin = (fin.sort_values("votes", ascending=False)
                  .drop_duplicates(subset=["race_id", "candidate"], keep="first"))

    def _trim_stale_runoff_rows(g):
        # only trim when the WHOLE race's pct sum is anomalously high (>130%) - a normal
        # 3+ candidate race where the top two happen to dominate (common, NOT a runoff
        # signature) still sums close to 100% across its full field and must NOT be
        # touched (verified: 69 such legitimate races exist in this dataset - an earlier,
        # narrower version of this check that only looked at the top two would have
        # silently deleted their real 3rd-place-and-below candidates).
        if g["pct"].sum() <= 130:
            return g
        g = g.sort_values("votes", ascending=False)
        if len(g) >= 2 and abs(g["pct"].iloc[0] + g["pct"].iloc[1] - 100) < 1.0:
            return g.iloc[:2]
        return g
    before_n = len(fin)
    fin = fin.groupby("race_id", group_keys=False)[fin.columns].apply(_trim_stale_runoff_rows)
    n_trimmed_rows = before_n - len(fin)
    if n_dup_races or n_trimmed_rows:
        print(f"merged-round guard: collapsed duplicates in {n_dup_races} races, "
              f"trimmed {n_trimmed_rows} stale round-1-only rows from detected runoffs")

    fin["is_winner"] = fin.groupby("race_id")["votes"].transform("max") == fin["votes"]
    fin["cand_key"] = fin["candidate"].map(F.norm_name)

    out_path = os.path.join(HERE, "data", "house_primary_results_hist.csv")
    fin.to_csv(out_path, index=False)

    fin["year"] = fin["race_id"].str.split("_").str[0].astype(int)
    print(f"\nsaved -> {out_path}")
    print(f"{fin['race_id'].nunique()} House party-races with primary results, "
          f"{len(pages)} pages checked, {n_with_primaries} pages had ANY primary data")
    print("\nraces per cycle (coverage GROWS GRADUALLY 1998->2012, not a cutoff - see the "
          "docstring's COVERAGE note; treat pre-2010 rows as real but sparse):")
    print(fin.groupby("year")["race_id"].nunique().to_string())

if __name__ == "__main__":
    main()
