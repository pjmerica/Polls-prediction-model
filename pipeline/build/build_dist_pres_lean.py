"""District presidential margins for House races -> data/district_pres_margin.csv (2026-10-27).

For every House race year 2002-2026, the DEM-minus-REP presidential margin (% of the total vote)
in each district at the most recent presidential election STRICTLY BEFORE that year, tabulated on
the district lines IN FORCE that year. features.load_dist_pres_lean turns it into a lean (minus the
national margin) for the `dist_pres_lean_cand` feature, next to the state lean `pres_lean_cand`
(user 2026-10-27: keep both). A House race had only the state lean before, so a D+12 state read the
same for every district in it (CO-5: tied polls, R+14 seat, model D 79%).

Source: The Downballot (formerly Daily Kos Elections), presidential results by congressional
district, https://www.the-downballot.com/p/data - free with credit and a link; they ask that the
spreadsheets not be reproduced whole, so the downloads stay in gitignored data/raw/pres_by_cd/ and
only this derived table is committed. Re-download with --fetch.

Sheet -> race years (the presidential column used; lines = that sheet's map):
  districts used 2006-10: 2002/2004 <- 2000 (only where the 2002 map lasted to 2006: not GA, and
                          not TX's five 2006 court-redrawn districts in 2004; TX 2002 had other
                          lines), 2006/2008 <- 2004, 2010 <- 2008
  2012-14: 2012 <- 2008, 2014 <- 2012 | 2016: 2016 <- 2012 | 2018: 2018 <- 2016
  2020: 2020 <- 2016 | 2022: 2022 <- 2020 | 2024: 2024 <- 2020
  2026 lines (incl. the 2025-26 redraws; MO and VA on their old maps): 2026 <- 2024, exact votes
1998/2000 races (1996 presidential results by district) are not covered, nor Florida's 2016 races
(2012 results on the 2016 court map): NaN, which the models treat as missing.

    py -X utf8 pipeline/build/build_dist_pres_lean.py [--fetch]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT  # noqa: E402

import os
import re
import sys

import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "pres_by_cd")
OUT = os.path.join(ROOT, "data", "district_pres_margin.csv")
SHEETS = {   # file: (sheet id, gid)
    "s2006map.csv": ("1l7W130tPRF6dQ4JoqlQJSJexnEg4rZct-7kwDXqgoLE", "429358610"),
    "s2012map.csv": ("1xn6nCNM97oFDZ4M-HQgoUT3X4paOiSDsRMSuxbaOBdg", "0"),
    "s2016map.csv": ("1VfkHtzBTP5gf4jAu8tcVQgsBJ1IDvXEHjuMqYlOgYbA", "0"),
    "s2018map.csv": ("1zLNAuRqPauss00HDz4XbTH2HqsCzMe0pR8QmD1K8jk8", "0"),
    "s2020map.csv": ("1XbUXnI9OyfAuhP5P3vWtMuGc5UJlrhXbzZo3AwMuHtk", "0"),
    "s2022map.csv": ("1CKngqOp8fzU22JOlypoxNsxL6KSAH920Whc-rd7ebuM", "1871835782"),
    "s2024map.csv": ("1ng1i_Dm_RMDnEvauH44pgE6JCUsapcuu8F2pCfeLWFo", "620838163"),
    "s2026votes.csv": ("1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0", "1491069057"),
    # 2008 on the 2012 lines, Jan 2012 (some states' maps were not final yet): used ONLY for
    # California, which the 2012-14 sheet leaves blank for 2008 (CA's map was final Aug 2011)
    "s2012pred_g0.csv": ("1JEl9zKk-6RYKxcBVTutQzwdy1tgTc9AOrKEJOs_T1zY", "0"),
}
# (file, race year, presidential cycle, D column, R column) - column positions in the sheet
PLAN = [
    ("s2006map.csv", 2002, 2000, 7, 8), ("s2006map.csv", 2004, 2000, 7, 8),
    ("s2006map.csv", 2006, 2004, 5, 6), ("s2006map.csv", 2008, 2004, 5, 6),
    ("s2006map.csv", 2010, 2008, 3, 4),
    ("s2012map.csv", 2012, 2008, 5, 6), ("s2012map.csv", 2014, 2012, 3, 4),
    ("s2016map.csv", 2016, 2012, 5, 6),
    ("s2018map.csv", 2018, 2016, 3, 4),
    ("s2020map.csv", 2020, 2016, 5, 6),
    ("s2022map.csv", 2022, 2020, 3, 4),
    ("s2024map.csv", 2024, 2020, 6, 7),
]
# 2002/2004 read the 2006 lines: drop states whose map changed in between
TX_2006_REDRAWN = {"15", "21", "23", "25", "28"}   # LULAC v. Perry (2006)
CD = re.compile(r"^([A-Z]{2})-(AL|\d{1,2})$")


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for f, (sid, gid) in SHEETS.items():
        r = requests.get(f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}",
                         timeout=60)
        r.raise_for_status()
        open(os.path.join(RAW, f), "wb").write(r.content)
        print("fetched", f)


def rows(f):
    t = pd.read_csv(os.path.join(RAW, f), header=None, dtype=str, keep_default_na=False)
    t = t[t[0].str.strip().str.match(CD)].copy()
    t["state"] = t[0].str.strip().str.extract(CD)[0]
    t["district"] = t[0].str.strip().str.extract(CD)[1].replace("AL", "1").astype(int).astype(str)
    return t


def num(s):
    return pd.to_numeric(s.str.replace(",", "").str.replace("%", "").str.strip(), errors="coerce")


def main():
    if "--fetch" in sys.argv:
        fetch()
    out = []
    for f, year, pres, dc, rc in PLAN:
        t = rows(f)
        t["margin"] = num(t[dc]) - num(t[rc])
        if year in (2002, 2004):
            t = t[t["state"] != "GA"]
            t = t[~((t["state"] == "TX") & ((year == 2002) | t["district"].isin(TX_2006_REDRAWN)))]
        if f == "s2012map.csv" and year == 2012:
            ca = rows("s2012pred_g0.csv")
            ca = ca[ca["state"] == "CA"].set_index("district")
            fill = (num(ca[3]) - num(ca[4]))
            m = (t["state"] == "CA") & t["margin"].isna()
            t.loc[m, "margin"] = t.loc[m, "district"].map(fill)
        out.append(t.assign(year=year, pres_cycle=pres)[["year", "state", "district", "pres_cycle", "margin"]])
    t = rows("s2026votes.csv")       # exact 2024 votes on the 2026 lines
    d, r, tot = num(t[4]), num(t[5]), num(t[6])
    t["margin"] = 100 * (d - r) / tot
    out.append(t.assign(year=2026, pres_cycle=2024)[["year", "state", "district", "pres_cycle", "margin"]])
    o = pd.concat(out).dropna(subset=["margin"])
    o["margin"] = o["margin"].round(2)
    o = o.sort_values(["year", "state", "district"], key=lambda s: s.astype(int) if s.name == "district" else s)
    dup = o.duplicated(["year", "state", "district"])
    assert not dup.any(), o[dup].head()
    o.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(o)} rows")
    print(o.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
