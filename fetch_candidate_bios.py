# -*- coding: utf-8 -*-
"""Scrape candidate BIO descriptors from Wikipedia race pages' primary-section candidate
lists -> office-experience features ("smaller offices" the results archives can't see:
state legislators, mayors, AGs, county officials).

Bullet format on race pages:  "Mallory McMorrow , state senator from the 8th district
(2019-present)" / "Abdul El-Sayed , former Wayne County health director (2023-2025) and
candidate for governor in 2018".

office_level: 4 federal (US sen/rep/cabinet) > 3 statewide (gov/LG/AG/SoS/treasurer) >
2 state legislature > 1 local (mayor/county/city/sheriff/judge) > 0 none-detected.
Plus: bio_in_office (descriptor says 'present'), bio_prior_candidacy ('candidate for' /
'nominee for' - catches runs our results files never tracked, e.g. El-Sayed's 2018
gubernatorial primary bid).

Output: data/candidate_bios.csv (committed). Known honesty caveat (documented in
METHODOLOGY): historical pages are read as they exist TODAY; post-election edits can
close date ranges ('2019-2023'), leaking small amounts of later info into training-era
bios. Office LEVEL - the feature - rarely changes from such edits.

    py -X utf8 fetch_candidate_bios.py
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

OUT = os.path.join(HERE, "data", "candidate_bios.csv")
URL_SEN = "https://en.wikipedia.org/wiki/{year}_United_States_Senate_election_in_{state}"
URL_GOV = "https://en.wikipedia.org/wiki/{year}_{state}_gubernatorial_election"
URL_HOUSE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of_Representatives"
             "_elections_in_{state}")

LEVELS = [
    # US-form variants: 'U.S.', 'US', 'U.S' (missing periods happen: 'U.S representative')
    (4, r"(u\.?s\.?|united states) (senator|representative|secretary)|"
        r"member of (the )?(u\.?s\.?|united states) house|member of congress|"
        r"white house|congress(wo)?man"),
    (3, r"\bgovernor\b|lieutenant governor|attorney general|secretary of state|"
        r"state treasurer|state auditor|state comptroller|commissioner of|"
        r"superintendent of public"),
    (2, r"state senator|state representative|state assembly|state house|"
        r"speaker of the .{0,30}house|state senate|general assembly|state delegate|"
        r"house of delegates"),
    (1, r"\bmayor\b|county (executive|commissioner|supervisor|clerk|judge|attorney|"
        r"treasurer|health director)|city council|county council|sheriff|"
        r"school board|city commissioner|selectman|alderman|district attorney|"
        r"state's attorney|\bjudge\b"),
]
PRIOR_CAND_RX = re.compile(r"candidate for|nominee for", re.I)

def classify(desc):
    d = str(desc).lower()
    # candidacy mentions are NOT offices held: 'candidate for governor in 2018' must not
    # classify as governor (caught by the El-Sayed known-truth check)
    d = re.sub(r"(candidate|nominee) for [^,;]+", " ", d)
    for lvl, rx in LEVELS:
        if re.search(rx, d):
            return lvl
    return 0

def parse_page(html, house=False):
    """[(district_or_None, party, name, descriptor)] from primary-section <li> bullets."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    stage, party, district = "general", "", None
    in_candidates = False
    SKIP_SUB_RX = re.compile(r"endorse|polling|results|debate|see also|references|"
                             r"fundrais|predictions|notes", re.I)
    CAND_SUB_RX = re.compile(r"candidate|declared|potential|withdr|declined|filed", re.I)
    for el in soup.find_all(["h2", "h3", "h4", "h5", "ul"]):
        if el.name != "ul":
            text = el.get_text(" ", strip=True)
            if house:
                m = re.search(r"District\s+(\d+)", text)
                if m:
                    district = int(m.group(1)); stage, party = "general", ""
            s, p = infer_section_context(text)
            if s:
                stage = s
                in_candidates = False
            if p:
                party = p
                in_candidates = False
            # subsection tracking: ENDORSEMENT lists live INSIDE primary sections and
            # once scraped Russ Feingold as a Wisconsin "candidate" - only collect
            # bullets under candidate-ish subheadings
            if SKIP_SUB_RX.search(text):
                in_candidates = False
            elif CAND_SUB_RX.search(text):
                in_candidates = True
            continue
        if stage != "primary" or not party or not in_candidates:
            continue
        for li in el.find_all("li", recursive=False):
            t = li.get_text(" ", strip=True)
            t = re.sub(r"\[\s*\d+\s*\]", "", t)          # strip cite brackets
            if not (8 < len(t) < 400) or "," not in t:
                continue
            a = li.find("a")
            name = (a.get_text(" ", strip=True) if a else t.split(",")[0]).strip()
            if not name or len(name.split()) > 5:
                continue
            desc = t[len(name):].lstrip(" ,").strip()
            if not desc:
                continue
            out.append((district if house else None, party, name, desc))
    return out

def main():
    # target pages = every page the primary polls came from + the 2026 predict set
    polls = pd.read_csv(os.path.join(HERE, "data", "primary_polls_wikipedia.csv"),
                        low_memory=False)
    targets = set()
    for page in polls.loc[polls["stage"] == "primary", "src_page"].unique():
        y, off, st = page.split("-")
        targets.add((int(y), {"SEN": "Senate", "GOV": "Governor"}[off], st))
    preds = pd.read_csv(os.path.join(HERE, "primary_predictions_2026.csv"))
    for r in preds.drop_duplicates(["state", "office"]).itertuples():
        targets.add((2026, r.office, r.state))
    targets = sorted(targets)
    print(f"{len(targets)} pages to scrape")

    done = set()
    frames = []
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, low_memory=False)
        frames.append(old)
        done = set(zip(old["year"], old["office"], old["state"]))
        print(f"resuming: {len(done)} pages already fetched")

    for i, (year, office, st) in enumerate(targets):
        if (year, office, st) in done:
            continue
        state = STATES.get(st)
        if not state:
            continue
        s = state.replace(" ", "_")
        house = office == "House"
        url = (URL_SEN if office == "Senate" else
               URL_GOV if office == "Governor" else URL_HOUSE).format(year=year, state=s)
        html = fetch_page(url)
        rows = parse_page(html, house=house) if html else []
        df = pd.DataFrame(rows, columns=["district", "party", "name", "descriptor"])
        df["year"], df["office"], df["state"] = year, office, st
        df["cand_key"] = df["name"].map(F.norm_name)
        df["office_level"] = df["descriptor"].map(classify)
        df["bio_in_office"] = df["descriptor"].astype(str).str.contains(
            "present", case=False).astype(int)
        df["bio_prior_candidacy"] = df["descriptor"].astype(str).str.contains(
            PRIOR_CAND_RX).astype(int)
        frames.append(df)
        if rows:
            print(f"  {year} {office} {st}: {len(rows)} candidate bios")
        if (i + 1) % 15 == 0:
            pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
        time.sleep(0.8)

    allb = pd.concat(frames, ignore_index=True)
    allb = allb.drop_duplicates(subset=["year", "office", "state", "district",
                                        "party", "cand_key"], keep="first")
    allb.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(allb)} bios")
    print("office_level distribution:", allb["office_level"].value_counts().to_dict())

if __name__ == "__main__":
    main()
