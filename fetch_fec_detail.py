"""FEC API detailed candidate totals (itemized individual money) -> data/fec_detail.csv.

Complements the bulk fec_summary.csv: the API's /candidates/totals adds
individual_ITEMIZED_contributions per candidate, which combined with the bulk file's
TOTAL individual money gives the small-dollar (unitemized, <$200) share — the grassroots
metric. True average donation is impossible from any FEC aggregate (no donor counts);
small-dollar share is the standard proxy.

Requires FEC_API_KEY in .env (free: https://api.data.gov/signup/). ~450 paged requests
for all 15 cycles at 1000/hr limit; run once, commit, re-run to refresh the current cycle.

    python fetch_fec_detail.py            # missing cycles + always refresh current
    python fetch_fec_detail.py --all
"""
import argparse
import os
import time

import pandas as pd
import requests

CYCLES = list(range(1998, 2027, 2))
OUT = "data/fec_detail.csv"
BASE = "https://api.open.fec.gov/v1/candidates/totals/"

def api_key():
    env = dict(l.strip().split("=", 1) for l in open(".env", encoding="utf-8-sig")
               if "=" in l and not l.lstrip().startswith("#"))
    k = env.get("FEC_API_KEY", "").strip()
    if not k:
        raise SystemExit("FEC_API_KEY missing from .env")
    return k

def fetch_cycle(cycle, key):
    rows = []
    for office in ("S", "H"):
        page = 1
        while True:
            r = requests.get(BASE, params={
                "api_key": key, "cycle": cycle, "office": office,
                "election_full": "true", "per_page": 100, "page": page}, timeout=60)
            if r.status_code == 429:
                time.sleep(30); continue
            r.raise_for_status()
            j = r.json()
            for x in j.get("results", []):
                rows.append(dict(
                    cycle=cycle,
                    cand_id=x.get("candidate_id"),
                    cand_name=x.get("name"),
                    office=("Senate" if office == "S" else "House"),
                    state=x.get("state"),
                    district=x.get("district"),
                    party=x.get("party"),
                    receipts=x.get("receipts"),
                    indiv_itemized=x.get("individual_itemized_contributions"),
                    coverage_end=x.get("coverage_end_date"),
                ))
            pages = j.get("pagination", {}).get("pages", 1)
            if page >= pages:
                break
            page += 1
            time.sleep(0.25)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    key = api_key()

    old = pd.read_csv(OUT, dtype={"district": str}) if os.path.exists(OUT) else pd.DataFrame()
    have = set(old["cycle"].unique()) if len(old) else set()
    todo = [c for c in CYCLES if args.all or c not in have or c == CYCLES[-1]]

    frames = [old[~old["cycle"].isin(todo)]] if len(old) else []
    for cyc in todo:
        rows = fetch_cycle(cyc, key)
        frames.append(pd.DataFrame(rows))
        print(f"  {cyc}: {len(rows)} candidates")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"saved -> {OUT}  ({len(df)} rows)")

if __name__ == "__main__":
    main()
