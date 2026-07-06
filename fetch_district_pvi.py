"""Current-map Cook PVI per district + state -> data/district_pvi_current.csv.

Source: Wikipedia's Cook PVI tables (435 districts + 51 states), which track Cook's
published updates for the CURRENT maps — the only free machine-readable PVI. Historical
PVI back to 1998 is paywalled (Cook PDFs), so this is a PREDICT-TIME dataset only:
training keeps prior_margin_cand (time-varying by construction). Its predict-time job is
the 2025-26 mid-decade redistricting patch: in redrawn districts our prior-margin joins
describe OLD boundaries, and 2*PVI is a far better estimate of the new district's
structural margin than a wrong-district number.

Committed snapshot; re-run occasionally (Cook updates PVIs after each redraw — the very
latest June-2026 redraws may lag on Wikipedia for a while; check `fetched_at`).

    python fetch_district_pvi.py
"""
import io
import re

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (research; polling model)"}
URL = "https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index"

STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
}

def pvi_num(s):
    """'R+27' -> -27, 'D+5' -> +5, 'EVEN' -> 0 (DEM-positive)."""
    s = str(s).strip().upper()
    if s.startswith("EVEN"):
        return 0.0
    m = re.match(r"([DR])\+(\d+)", s)
    if not m:
        return None
    return float(m.group(2)) * (1 if m.group(1) == "D" else -1)

def main():
    r = requests.get(URL, timeout=60, headers=H)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    rows = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        if "District" in cols and "PVI" in cols and len(t) > 400:
            for _, x in t.iterrows():
                m = re.match(r"([A-Za-z .]+?)\s+(\d+|at-large)$", str(x["District"]).strip(),
                             re.I)
                if not m:
                    continue
                st = STATE_ABBR.get(m.group(1).strip())
                di = "1" if m.group(2).lower() == "at-large" else str(int(m.group(2)))
                v = pvi_num(x["PVI"])
                if st and v is not None:
                    rows.append(dict(state=st, district=di, pvi=v))
        elif "State" in cols and "PVI" in cols and 45 <= len(t) <= 55:
            for _, x in t.iterrows():
                st = STATE_ABBR.get(str(x["State"]).strip())
                v = pvi_num(x["PVI"])
                if st and v is not None:
                    rows.append(dict(state=st, district="", pvi=v))
    df = pd.DataFrame(rows).drop_duplicates(["state", "district"])
    df["fetched_at"] = pd.Timestamp.now().date().isoformat()
    df.to_csv("data/district_pvi_current.csv", index=False)
    n_d = (df["district"] != "").sum()
    print(f"saved data/district_pvi_current.csv: {n_d} districts + {len(df)-n_d} states")

if __name__ == "__main__":
    main()
