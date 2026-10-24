"""Every 2026 November race's candidates -> data/general_roster_2026.csv. 2026-10-23.

Source: Wikipedia's three 2026 overview pages (House, Senate, governors). Each has one table row
per district or state with its November candidates ("Name (Party)"), the seat's incumbent and
status, and (House and Senate) the 2026 Cook PVI on the map in force, so the 2025-26 redrawn
districts come with their NEW lean. models/fundamentals/predict_unpolled.py scores every race
from this roster, so races no pollster covers still get a model probability.

Rules:
  - candidates marked withdrawn / disqualified / deceased are dropped; write-ins are kept but
    flagged (write_in=1) and left out of scoring;
  - the House "Special elections" table (already decided) is skipped;
  - party comes from the candidate's own label: Democratic (incl. DFL / Democratic-NPL) -> DEM,
    Republican -> REP, Independent / no party -> IND, anything else -> OTH.

Re-run after any nominee change (replacement nominees, court-ordered maps):

    python pipeline/fetch/fetch_general_roster_2026.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "src"))
from paths import ROOT  # noqa: E402

import datetime as dt  # noqa: E402
import re  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

import features as F  # noqa: E402  (norm_name: the shared candidate join key)

H = {"User-Agent": "Mozilla/5.0 (research; polling model; github.com/pjmerica)"}
API = "https://en.wikipedia.org/api/rest_v1/page/html/{}"
PAGES = {"House": "2026_United_States_House_of_Representatives_elections",
         "Senate": "2026_United_States_Senate_elections",
         "Governor": "2026_United_States_gubernatorial_elections"}
OUT = _os.path.join(ROOT, "data", "general_roster_2026.csv")

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"}
_DROP = re.compile(r"withdr|disqualif|deceased|died|suspended|removed from (the )?ballot", re.I)
_REF = re.compile(r"\[\s*[^\]]{1,8}\s*\]")
_PVI = re.compile(r"^(R\+\d+|D\+\d+|EVEN)$")


def party_code(p):
    s = p.lower()
    if "democratic" in s or "dfl" in s or "farmer" in s:
        return "DEM"
    if "republican" in s:
        return "REP"
    if "independent" in s or "no party" in s or "nonpartisan" in s or "unaffiliated" in s:
        return "IND"
    return "OTH"


def pvi_value(s):
    """'R+17' -> -17, 'D+5' -> 5, 'EVEN' -> 0 (Dem-positive, the convention of district_pvi_current.csv)."""
    s = s.strip()
    if s == "EVEN":
        return 0.0
    m = re.match(r"([RD])\+(\d+)", s)
    return (float(m.group(2)) * (1 if m.group(1) == "D" else -1)) if m else None


def candidates(td):
    out = []
    for li in td.find_all("li"):
        txt = _REF.sub("", li.get_text(" ", strip=True)).replace("▌", " ").strip()
        txt = re.sub(r"\s+", " ", txt)
        m = re.match(r"^(?:Y\s+)?(.*?)\s*\(([^()]*)\)", txt)
        if not m or not m.group(1).strip():
            continue
        name, party = m.group(1).strip(), m.group(2).strip()
        rest = txt[m.end():]
        out.append(dict(candidate=name, party_raw=party, party=party_code(party),
                        write_in=int("write-in" in (party + rest).lower()),
                        dropped=int(bool(_DROP.search(party + " " + rest)))))
    return out


def cand_cell(cells):
    """The candidates cell: the last cell that holds a list of 'Name (Party)' items."""
    for c in reversed(cells):
        if c.find("li") and "(" in c.get_text():
            return c
    return None


def parse(office, html):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for t in soup.find_all("table"):
        heads = " ".join(th.get_text(" ", strip=True) for th in t.find_all("th")[:16])
        if "Candidates" not in heads:
            continue
        h = t.find_previous(["h2", "h3"])
        special_table = bool(h and "special" in h.get_text(" ", strip=True).lower())
        if special_table and office == "House":
            continue
        for tr in t.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if len(cells) < 3 or cells[0].name != "th":
                continue
            first = cells[0].get_text(" | ", strip=True)
            if office == "House":
                m = re.match(r"^([A-Za-z .]+?)\s*\|\s*(\d+|at-large|at large)", first, re.I)
                if not m or m.group(1).strip() not in STATES:
                    continue
                st = STATES[m.group(1).strip()]
                dist = "1" if m.group(2).lower().startswith("at") else str(int(m.group(2)))
            else:
                name = re.sub(r"\(.*?\)|\[.*?\]|\|", " ", first).strip()
                name = re.sub(r"\s+", " ", name)
                if name not in STATES:
                    continue
                st = STATES[name]
                # the Senate page lists FL and OH (Class 3 seats held by appointees) in its own
                # "Special elections" table; the model keys a special 'S' (2026_OH_Senate-S)
                dist = "S" if (special_table or "special" in first.lower()) and office == "Senate" else ""
            pvi_txt = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
            pvi = pvi_value(pvi_txt) if _PVI.match(pvi_txt) else None
            cc = cand_cell(cells[1:])
            if cc is None:
                continue
            inc = cells[2].get_text(" ", strip=True) if office != "Governor" else cells[1].get_text(" ", strip=True)
            status = " ".join(c.get_text(" ", strip=True) for c in cells[2:] if c is not cc)[:200]
            race_id = f"2026_{st}_{office}" + (f"-{dist}" if dist else "")
            for c in candidates(cc):
                rows.append(dict(race_id=race_id, office=office, state=st, district=dist, pvi=pvi,
                                 incumbent=_REF.sub("", inc)[:80], status=_REF.sub("", status), **c))
    return rows


def main():
    rows = []
    for office, title in PAGES.items():
        r = requests.get(API.format(title), headers=H, timeout=120)
        r.raise_for_status()
        got = parse(office, r.text)
        print(f"{office}: {len({x['race_id'] for x in got})} races, {len(got)} candidate rows")
        rows += got
    d = pd.DataFrame(rows)
    d["cand_key"] = d["candidate"].map(F.norm_name)
    d["fetched_at"] = dt.date.today().isoformat()
    d = d.drop_duplicates(["race_id", "cand_key"])
    d.to_csv(OUT, index=False)
    live = d[(d["dropped"] == 0) & (d["write_in"] == 0)]
    two = live.groupby("race_id")["party"].apply(lambda s: {"DEM", "REP"} <= set(s)).sum()
    print(f"saved {OUT}: {d['race_id'].nunique()} races, {len(live)} live candidates; "
          f"{two} races with both a Democrat and a Republican")


if __name__ == "__main__":
    main()
