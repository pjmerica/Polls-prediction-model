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

if __name__ == "__main__":
    cyc = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    v = get_natl_env(cyc, verbose=True)
    print(f"\nnatl_env({cyc}) = {v}  (DEM-REP, mean of aggregators)")
