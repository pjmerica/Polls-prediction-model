"""Current-cycle generic-ballot (national environment) number for predict.py.

RCP itself is behind Cloudflare (403 for scripts), but Wikipedia's per-cycle House-elections
article carries a table of the major *aggregators* (Decision Desk HQ, RealClearPolitics,
FiftyPlusOne, ...) with their current Democratic/Republican generic-ballot averages. We take
the MEAN of the aggregators' D-R margins — an average of averages, robust to any one outlet.

This is the ONLY live fetch in the project, and it runs at predict time only (current-cycle
information cannot be frozen, by definition). Historical cycles come from committed files via
cycles.py. predict.py calls get_natl_env(); --natl-env <number> overrides it; if the fetch
fails, the feature is left missing (NaN) with a warning.

    python fetch_generic_ballot.py [cycle]     # prints the number + the table
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import io
import re
import sys

import pandas as pd
import requests



H = {"User-Agent": "Mozilla/5.0 (research; polling-prediction-model)"}
URL = "https://en.wikipedia.org/wiki/{cycle}_United_States_House_of_Representatives_elections"

def _margin_from_text(s):
    """'Democrats +5.3%' -> +5.3 ; 'Republicans +2.1%' -> -2.1 ; else NaN."""
    m = re.search(r"(Democrat|Republican)\w*\s*\+\s*([\d.]+)", str(s))
    if not m:
        return float("nan")
    v = float(m.group(2))
    return v if m.group(1).startswith("Democrat") else -v

def get_natl_env(cycle=2026, verbose=False):
    """Mean aggregator generic-ballot D-R margin for `cycle`, or None if unavailable."""
    try:
        r = requests.get(URL.format(cycle=cycle), timeout=60, headers=H)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
    except Exception as e:
        if verbose:
            print(f"fetch failed: {type(e).__name__}: {e}")
        return None

    for t in tables:
        cols = [str(c) for c in (t.columns.get_level_values(-1)
                                 if hasattr(t.columns, "get_level_values") else t.columns)]
        has = lambda w: any(w in c for c in cols)
        if has("Democrat") and has("Republican") and has("Margin") and len(t) <= 25:
            mcol = [c for c in t.columns if "Margin" in str(c)][-1]
            t = t.copy()
            t["_src"] = t[t.columns[0]].astype(str).str.replace(r"\[\d+\]", "", regex=True)
            t["_m"] = t[mcol].map(_margin_from_text)
            # drop the page's own 'Average' row so it isn't double-counted in our mean
            t = t[~t["_src"].str.lower().str.contains("average")]
            margins = t["_m"].dropna()
            if len(margins) == 0:
                continue
            if verbose:
                for s, m in zip(t["_src"], t["_m"]):
                    if pd.notna(m):
                        print(f"  {s[:40]:40} {m:+.1f}")
            return round(float(margins.mean()), 2)
    if verbose:
        print("no aggregator table found on the page")
    return None

# ---------------------------------------------------------------------------
# Monthly generic-ballot series -> data/generic_ballot_monthly.csv
#   1998-2022 : per-poll House-G-US rows in the frozen data/raw_polls_538.csv
#   2024-12+  : VoteHub open API (poll_type=generic-ballot)
#   2023 .. 2024-11 : NO machine-readable per-poll source survives (538's daily series died
#   with the Internet Archive) -> those months are simply absent (NaN features downstream).
# Value = monthly mean of per-poll (DEM% - REP%).
# ---------------------------------------------------------------------------
def build_monthly(out="data/generic_ballot_monthly.csv"):
    # base layer: 538's DAILY estimates 1995-2016 (dense + smooth), monthly means
    daily = pd.read_csv("data/generic_ballot_hist_538.csv", low_memory=False)
    daily = daily[daily["subgroup"].astype(str).str.lower() == "all polls"].copy()
    daily["date"] = pd.to_datetime(daily["modeldate"], errors="coerce")
    daily["margin"] = (pd.to_numeric(daily["dem_estimate"], errors="coerce")
                       - pd.to_numeric(daily["rep_estimate"], errors="coerce"))
    base = (daily.dropna(subset=["date", "margin"])
                 .set_index("date")["margin"].resample("MS").mean())

    rp = pd.read_csv("data/raw_polls_538.csv", low_memory=False)
    g = rp[rp["type_simple"] == "House-G-US"].copy()
    dem = pd.to_numeric(g["cand1_pct"], errors="coerce").where(
        g["cand1_party"].astype(str).str.upper().str.startswith("D"),
        pd.to_numeric(g["cand2_pct"], errors="coerce"))
    rep = pd.to_numeric(g["cand2_pct"], errors="coerce").where(
        g["cand1_party"].astype(str).str.upper().str.startswith("D"),
        pd.to_numeric(g["cand1_pct"], errors="coerce"))
    hist = pd.DataFrame({"end": pd.to_datetime(g["polldate"], errors="coerce"),
                         "margin": dem - rep}).dropna()

    vh_rows = []
    try:
        r = requests.get("https://api.votehub.com/polls",
                         params={"poll_type": "generic-ballot"}, timeout=60, headers=H)
        r.raise_for_status()
        for p in r.json():
            ans = {str(a.get("choice", "")).lower()[:3]: a.get("pct") for a in p.get("answers", [])}
            d_, r_ = ans.get("dem"), ans.get("rep")
            end = pd.to_datetime(p.get("end_date"), errors="coerce")
            if d_ is not None and r_ is not None and pd.notna(end):
                vh_rows.append((end, float(d_) - float(r_)))
        print(f"votehub generic-ballot polls: {len(vh_rows)}")
    except Exception as e:
        print(f"votehub fetch failed ({type(e).__name__}) - historical part only")
    vh = pd.DataFrame(vh_rows, columns=["end", "margin"])
    if len(vh):
        vh = vh[vh["end"] > hist["end"].max()]     # no overlap with the frozen history

    allp = pd.concat([hist, vh], ignore_index=True)
    polled = allp.set_index("end")["margin"].resample("MS").mean()
    # daily-estimate base (dense, 1995-2016) wins where present; per-poll months fill the rest
    monthly = base.combine_first(polled).dropna().round(2)
    outdf = monthly.reset_index()
    outdf.columns = ["date", "value"]
    outdf["metric"] = "generic_ballot"
    outdf.to_csv(out, index=False)
    print(f"saved -> {out}  ({len(outdf)} months, "
          f"{outdf['date'].min().date()} .. {outdf['date'].max().date()})")

if __name__ == "__main__":
    if "--monthly" in sys.argv:
        build_monthly()
    else:
        cyc = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2026
        v = get_natl_env(cyc, verbose=True)
        print(f"\nnatl_env({cyc}) = {v}  (DEM-REP, mean of aggregators)")
