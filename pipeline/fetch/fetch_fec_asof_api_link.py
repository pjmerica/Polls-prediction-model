"""As-of-Sep-30 money for campaigns the bulk linkage file does not link (2026-10-28).

    py -X utf8 pipeline/fetch/fetch_fec_asof_api_link.py CYCLE:CAND_ID [CYCLE:CAND_ID ...]

fetch_fec_asof.py maps committees to candidates through the FEC's bulk candidate-committee
linkage file (ccl) for each cycle. A candidate who ran again later can be missing from an earlier
cycle's linkage (Lucas Kunce MO-Sen 2024, Laura Gillen NY-4 2024, Val Demings FL-10 2016), so their
as-of money was never fetched and the model saw none. This asks the FEC API for the candidate's
committees in that cycle instead, then applies fetch_fec_asof's own rules (election-year Q3 /
October monthly, else the off-year year-end; cycle-to-date columns; principal + authorized
committees; transfers between them netted) and appends to data/fec_asof.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402,F401

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import fetch_fec_asof as A  # noqa: E402


def committees(key, cand, cycle):
    r = requests.get(f"https://api.open.fec.gov/v1/candidate/{cand}/committees/",
                     params={"api_key": key, "cycle": cycle, "per_page": 50}, headers=A.H, timeout=90)
    r.raise_for_status()
    return [c["committee_id"] for c in r.json().get("results", []) if c.get("designation") in ("P", "A")]


def main():
    key = A.api_key()
    old = pd.read_csv(A.OUT)
    have = set(zip(old["cycle"], old["cand_id"]))
    rows = []
    for arg in _sys.argv[1:]:
        cyc, cand = arg.split(":")
        cyc = int(cyc)
        if (cyc, cand) in have:
            print(f"{cyc} {cand}: already in {A.OUT}")
            continue
        acc = None
        for cm in committees(key, cand, cyc):
            allrep = A._committee_reports(key, cm, cyc, None)
            rep = A._best([x for x in allrep if x.get("report_type") in ("Q3", "M10")], cyc)
            if rep is None:
                rep = A._best([x for x in allrep if x.get("report_type") == "YE"], cyc - 1)
            if rep is None:
                continue
            if acc is None:
                acc = dict(cycle=cyc, cand_id=cand, n_committees=0, **{k: 0.0 for k in A.YTD})
            acc["n_committees"] += 1
            for k, v in A._ytd_or_nan(rep).items():
                acc[k] = acc[k] + v
            acc["asof"] = max(acc.get("asof", ""), str(rep.get("coverage_end_date", ""))[:10])
        if acc is None:
            print(f"{cyc} {cand}: no Q3 / year-end report found")
            continue
        t = acc.get("transfers_out_auth", 0.0)
        if acc["n_committees"] > 1 and t == t and t > 0:
            acc["receipts"] = max(acc["receipts"] - t, 0.0)
            acc["disbursements"] = max(acc["disbursements"] - t, 0.0)
        acc["self_fund"] = acc["cand_contrib"] + acc["cand_loans"]
        rows.append(acc)
        print(f"{cyc} {cand}: receipts {acc['receipts']:,.0f} through {acc['asof']} ({acc['n_committees']} committee(s))")
    if rows:
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(A.OUT, index=False)
        print(f"added {len(rows)} -> {A.OUT}")


if __name__ == "__main__":
    main()
