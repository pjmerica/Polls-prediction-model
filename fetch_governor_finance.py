"""Governor campaign finance (state-level) via FollowTheMoney -> data/governor_finance.csv.

The FEC covers federal races only; governors file with 50 state agencies. FollowTheMoney
(NIMSP) aggregates them all. Requires FTM_API_KEY in .env (free registered key).

Query grammar (discovered 2026-07-06): office-type filter `c-r-ot=G` (Governor) + `y=` year,
grouped by candidate (`gro=c-t-id`), paged with `p=N`. Fields per record: candidate, state,
general party, incumbency, election status, Total_$ (total raised).

Coverage: solid back to ~2000 (1998 partial in some states). Committed once; re-run to
extend/refresh the current cycle.

    python fetch_governor_finance.py
"""
import json
import os
import time

import pandas as pd
import requests

YEARS = list(range(1998, 2027, 2))
OUT = "data/governor_finance.csv"

def api_key():
    env = dict(l.strip().split("=", 1) for l in open(".env", encoding="utf-8-sig")
               if "=" in l and not l.lstrip().startswith("#"))
    k = env.get("FTM_API_KEY", "").strip()
    if not k:
        raise SystemExit("FTM_API_KEY missing from .env")
    return k

def fetch_year(year, key):
    rows, page = [], 0
    empty_retries = 0
    while True:
        r = requests.get("https://api.followthemoney.org/",
                         params={"y": year, "c-r-ot": "G", "gro": "c-t-id",
                                 "p": page, "mode": "json", "APIKey": key},
                         timeout=90)
        j = json.loads(r.text)
        recs = j.get("records", [])
        if not recs or recs[0] == "No Records":
            # FTM soft-blocks bursts: empty result may be a throttle, not real absence
            if page == 0 and empty_retries < 3:
                empty_retries += 1
                print(f"    {year}: empty response, backing off 90s (retry {empty_retries}/3)")
                time.sleep(90)
                continue
            break
        for x in recs:
            if x.get("Office_Sought", {}).get("Office_Sought") != "GOVERNOR":
                continue          # c-r-ot=G can include a few Lt-Gov style offices
            rows.append(dict(
                cycle=year,
                state=x.get("Election_Jurisdiction", {}).get("Election_Jurisdiction"),
                cand_name=x.get("Candidate", {}).get("Candidate"),
                party=x.get("General_Party", {}).get("General_Party"),
                incumbency=x.get("Incumbency_Status", {}).get("Incumbency_Status"),
                status=x.get("Election_Status", {}).get("Election_Status"),
                receipts=pd.to_numeric(x.get("Total_$", {}).get("Total_$"), errors="coerce"),
            ))
        paging = j.get("metaInfo", {}).get("paging", {})
        if page >= int(paging.get("maxPage", 0)):
            break
        page += 1
        time.sleep(3)
    return rows

def main():
    key = api_key()
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set(old["cycle"].unique()) if len(old) else set()
    allrows = old.to_dict("records") if len(old) else []
    for y in YEARS:
        if y in have and y != YEARS[-1]:
            print(f"  {y}: already fetched ({sum(1 for r in allrows if r['cycle']==y)} rows)")
            continue
        rows = fetch_year(y, key)
        allrows += rows
        print(f"  {y}: {len(rows)} governor candidates")
        time.sleep(5)
    df = pd.DataFrame(allrows).dropna(subset=["state", "cand_name"])
    df.to_csv(OUT, index=False)
    print(f"saved -> {OUT}  ({len(df)} rows, {df['cycle'].min()}-{df['cycle'].max()})")

if __name__ == "__main__":
    main()
