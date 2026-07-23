# -*- coding: utf-8 -*-
"""Scrape 2026 PRIMARY RESULTS from Wikipedia race pages -> data/primary_results_2026.csv.

Ground truth for the primary model: the actual primary winner per (race, party) —
directly, not via the post-primary-general-polling proxy (which is blind exactly where
the primary IS the contest, e.g. deep-blue NYC House seats).

Parsing: results wikitables (headers contain Candidate + Votes/%) under primary-stage
section context. Per (race_id, party) the LAST results table in document order wins —
that's the runoff where one exists (verified: TX-Sen-REP round 1 = Cornyn 42.0/Paxton
40.5, runoff = Paxton 63.8 -> nominee Paxton). Tables under non-primary context (e.g.
prior-cycle results in Background sections) are excluded by the stage tracker; mixed-
party tables are skipped as a second guard.

Doubles as the LABELING machinery for adding cycle 2026 to training after November.

    py -X utf8 fetch_primary_results_2026.py            # 2026 (pages from predictions)
    py -X utf8 fetch_primary_results_2026.py --hist     # 2018-2024 (pages from the
                                                        # historical polls scrape) ->
                                                        # data/primary_results_hist.csv
HIST NOTE: general-election candidate lists are a flawed label source - a primary winner
who later drops out (Platner, ME-Sen 2026) appears NOWHERE in them and the replacement
gets mislabeled as the winner. Actual results are the truth; build_primary_dataset
prefers them and reports every disagreement with the nominee-join.
"""
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets"))
from bs4 import BeautifulSoup  # noqa: E402
from scrapers.wikipedia_polls import fetch_page, infer_section_context, STATES  # noqa: E402

import features as F  # noqa: E402

URL_SEN = "https://en.wikipedia.org/wiki/{year}_United_States_Senate_election_in_{state}"
URL_GOV = "https://en.wikipedia.org/wiki/{year}_{state}_gubernatorial_election"
URL_HOUSE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of_Representatives"
             "_elections_in_{state}")

PARTY_ROW = {"republican": "REP", "democratic": "DEM"}

def parse_results_tables(html, base_race_id, house=False, at_large=False):
    """[{race_id(+party), party, candidate, votes, pct, table_seq}] for every
    primary-context results table. race_id gets '-<district>' inserted for House.

    at_large=True (added 2026-07-22, historical House scrape): the page has ONE district
    for the whole state (AK/DE/MT/ND/SD/VT/WY, cycle-dependent - some regained/lost a seat
    across redistricting, e.g. SD 2-seat pre-1982, MT 2-seat 2020+) and carries no
    "District N" heading to detect, so seed district=1 up front instead of requiring one to
    appear (matches res_house.csv's own at-large convention, verified: office_seat_name=
    'District 1'). Without this, at-large-year House pages parsed zero rows (the
    `house and district is None` guard below dropped everything)."""
    soup = BeautifulSoup(html, "html.parser")
    rows, seq = [], 0
    stage, district, other_office = "general", (1 if at_large else None), False
    OTHER_OFFICE_RX = re.compile(
        r"lieutenant|attorney general|secretary of state|treasurer|auditor|"
        r"comptroller|agriculture|superintendent|land commissioner|railroad",
        re.I)
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name != "table":
            text = el.get_text(" ", strip=True)
            if house:
                m = re.search(r"District\s+(\d+)", text)
                if m:
                    district = int(m.group(1)); stage = "general"
            s, _p = infer_section_context(text)
            if s:
                stage = s
                other_office = False    # a new primary/general section resets the guard
            # DOWN-TICKET GUARD: gubernatorial pages nest Lieutenant-Governor (etc.)
            # primary results INSIDE the party-primary sections - without this the
            # last-table heuristic crowned running mates (Husted over DeWine, Driscoll
            # over Healey, Fetterman as 2018 PA "governor" winner...). A subsection
            # naming another office poisons tables until the next stage-bearing heading.
            if OTHER_OFFICE_RX.search(text):
                other_office = True
            continue
        if other_office or stage not in ("primary", "primary runoff"):
            continue
        if "wikitable" not in (el.get("class", []) or []):
            continue
        heads = " ".join(th.get_text(" ", strip=True) for th in el.find_all("th")).lower()
        if "candidate" not in heads or ("votes" not in heads and "%" not in heads):
            continue
        seq += 1
        parties, trows = set(), []
        for tr in el.find_all("tr"):
            els = tr.find_all(["td", "th"])
            cells = [c.get_text(" ", strip=True) for c in els]
            if len(cells) < 4:
                continue
            party = PARTY_ROW.get(cells[1].strip().lower())
            if party is None:
                continue
            # TICKET cells: states electing Governor+LG jointly list 'Richard Cordray and
            # Betty Sutton' in one cell -> naive parsing crowned 'sutton r'. Names are
            # individually wikilinked, so the FIRST link is the head of the ticket;
            # fallback splits on ticket separators.
            a = els[2].find("a")
            cand = a.get_text(" ", strip=True) if a else re.split(
                r"\s+and\s+|/", cells[2])[0]
            # strip ALL parenthetical annotations: (incumbent), (write-in), (withdrawn)...
            cand = re.sub(r"\s*\([^)]*\)\s*", " ", cand).strip()
            votes = re.sub(r"[^\d]", "", cells[3])
            pct = re.sub(r"[^\d.]", "", cells[4]) if len(cells) > 4 else ""
            if not cand or not votes:
                continue
            parties.add(party)
            trows.append((party, cand, int(votes), float(pct) if pct else None))
        if not trows or len(parties) != 1:
            continue          # mixed-party table = not a primary results table
        party = parties.pop()
        rid = base_race_id + (f"-{district}" if house and district else "") + "_" + party
        if house and district is None:
            continue
        for p_, cand, votes, pct in trows:
            rows.append(dict(race_id=rid, party=party, candidate=cand,
                             votes=votes, pct=pct, table_seq=seq))
    return rows

def page_targets(hist):
    """[(year, st, office)] - 2026 from the predictions CSV, hist from the polls scrape."""
    if not hist:
        preds = pd.read_csv(os.path.join(HERE, "primary_predictions_2026.csv"))
        need = set()
        for rid in preds["race_id"].unique():
            _y, st, of_di, _pty = rid.split("_")
            need.add((2026, st, of_di.split("-")[0]))
        return sorted(need)
    polls = pd.read_csv(os.path.join(HERE, "data", "primary_polls_wikipedia.csv"),
                        low_memory=False)
    out = set()
    for page in polls.loc[polls["stage"] == "primary", "src_page"].unique():
        y, off, st = page.split("-")
        out.add((int(y), st, {"SEN": "Senate", "GOV": "Governor"}[off]))
    return sorted(out)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", action="store_true",
                    help="scrape 2018-2024 results for the historical training pages")
    args = ap.parse_args()
    out_path = os.path.join(HERE, "data",
                            "primary_results_hist.csv" if args.hist
                            else "primary_results_2026.csv")

    pages = page_targets(args.hist)
    print(f"{len(pages)} pages to scrape")

    allrows = []
    for year, st, office in pages:
        state = STATES.get(st)
        if not state:
            continue
        s = state.replace(" ", "_")
        if office == "Senate":
            url, base, house = (URL_SEN.format(year=year, state=s),
                                f"{year}_{st}_Senate", False)
        elif office == "Governor":
            url, base, house = (URL_GOV.format(year=year, state=s),
                                f"{year}_{st}_Governor", False)
        else:
            url, base, house = (URL_HOUSE.format(year=year, state=s),
                                f"{year}_{st}_House", True)
        html = fetch_page(url)
        if html is None:
            print(f"  {st} {office}: page fetch failed")
            continue
        rows = parse_results_tables(html, base, house=house)
        if rows:
            print(f"  {st} {office}: {len(rows)} result rows, "
                  f"{len(set(r['race_id'] for r in rows))} party-races")
        allrows.extend(rows)
        time.sleep(0.8)

    df = pd.DataFrame(allrows)
    if not len(df):
        raise SystemExit("no results parsed - page structure changed?")
    # winner = top votes in the LAST table per (race_id): runoff supersedes round 1
    last = df.groupby("race_id")["table_seq"].transform("max")
    fin = df[df["table_seq"] == last].copy()
    fin["is_winner"] = fin.groupby("race_id")["votes"].transform("max") == fin["votes"]
    fin["cand_key"] = fin["candidate"].map(F.norm_name)

    # hand-checked ground truth: hard-fail on regression (these are the exact races the
    # Lieutenant-Governor table bug corrupted, plus a runoff case)
    KNOWN = {"2018_OH_Governor_DEM": "cordray r", "2018_OH_Governor_REP": "dewine m",
             "2018_HI_Governor_DEM": "ige d", "2022_MA_Governor_DEM": "healey m",
             "2026_TX_Senate_REP": "paxton k"}
    wmap = fin[fin["is_winner"]].groupby("race_id")["cand_key"].agg(set).to_dict()
    for rid, want in KNOWN.items():
        if rid in wmap:
            assert want in wmap[rid], f"{rid}: parsed winner {wmap[rid]} != known {want}"

    fin.to_csv(out_path, index=False)
    w = fin[fin["is_winner"]]
    print(f"\nsaved -> {out_path}: {fin['race_id'].nunique()} party-races with results, "
          f"{len(w)} winners (known-winner validation passed)")

if __name__ == "__main__":
    main()
