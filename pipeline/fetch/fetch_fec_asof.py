"""FEC money AS OF SEPTEMBER 30 of each election year -> data/fec_asof.csv (2026-10-01).

Why: fec_summary.csv (bulk webl) is END-OF-CYCLE - it includes money raised AFTER the election.
Ossoff 2020 raised ~$128M after Sep 30 (pre-general, pre-runoff and year-end reports) on top of
~$27M before it, so training fund_* features carried look-ahead the live model never has. This
file is the cumulative cycle-to-date total a forecaster could have seen on Sep 30:

    the Q3 report of the election year (monthly filers: the October monthly, covering
    September) - its "_ytd" columns are CYCLE-TO-DATE on Form 3, so it alone is the total
    through Sep 30. A committee with no election-year report falls back to its off-year
    year-end (also cycle-to-date). Never add the two: that double-counts the off year.

Live cycle (2026) before Oct 15: Q3 is not filed yet, so the election-year part is the latest
quarterly (Q2, through Jun 30) - `asof` records which. Re-run after Oct 15.

Committees -> candidates via FEC bulk candidate-committee linkage files (ccl, no key needed).
Only principal + authorized committees (designation P/A) count.

    py -X utf8 pipeline/fetch/fetch_fec_asof.py            # all cycles missing + current
    py -X utf8 pipeline/fetch/fetch_fec_asof.py --all
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402,F401

import argparse
import io
import os
import time
import zipfile

import pandas as pd
import requests

CYCLES = list(range(1998, 2027, 2))
OUT = "data/fec_asof.csv"
API = "https://api.open.fec.gov/v1/reports/house-senate/"
H = {"User-Agent": "Mozilla/5.0 (research; polling model)"}
YTD = {
    "receipts": "total_receipts_ytd",
    "indiv_contrib": "total_individual_contributions_ytd",
    "pac_contrib": "other_political_committee_contributions_ytd",
    "party_contrib": "political_party_committee_contributions_ytd",
    "cand_contrib": "candidate_contribution_ytd",
    "cand_loans": "loans_made_by_candidate_ytd",
    "indiv_itemized": "individual_itemized_contributions_ytd",
    # spending through Sep 30 (cycle-to-date) and cash in the bank on Sep 30 - the latter is a
    # point-in-time balance, not a running total (2026-10-02, new features)
    "disbursements": "total_disbursements_ytd",
    "cash_on_hand": "cash_on_hand_end_period",
    # money moved between a candidate's OWN authorized committees (2026-10-09) - see
    # fetch_cycle: subtracted when 2+ committees are summed, or it is counted twice
    "transfers_out_auth": "transfers_to_other_authorized_committee_ytd",
}


def api_key():
    env = dict(l.strip().split("=", 1) for l in open(".env", encoding="utf-8-sig")
               if "=" in l and not l.lstrip().startswith("#"))
    k = env.get("FEC_API_KEY", "").strip()
    if not k:
        raise SystemExit("FEC_API_KEY missing from .env")
    return k


def get(params, key):
    for attempt in range(8):
        try:
            r = requests.get(API, params={**params, "api_key": key}, timeout=90, headers=H)
        except requests.exceptions.RequestException:
            time.sleep(15); continue
        if r.status_code == 429:
            time.sleep(65); continue
        if r.status_code >= 500:
            time.sleep(15); continue
        if r.status_code != 200:     # never raise_for_status: its message prints the key
            raise RuntimeError(f"FEC API HTTP {r.status_code} for "
                               f"{ {k: v for k, v in params.items() if k != 'api_key'} }")
        return r.json()
    raise RuntimeError(f"FEC API failed: { {k: v for k, v in params.items() if k != 'api_key'} }")


def reports(key, cycle, report_types, report_year):
    """Most-recent (amendment-resolved) reports of the given types -> {committee_id: row}."""
    out = {}
    for rt in report_types:
        page = 1
        while True:
            # `year` (not report_year - silently ignored) is the report-year filter, and the
            # sort must be UNIQUE: thousands of Q3s share coverage_end_date, and offset paging
            # over that tie skipped/duplicated up to 15% of reports (Ossoff's Q3 2020 vanished).
            # beginning_image_number is unique per filing - verified complete (2407/2407).
            j = get({"cycle": cycle, "report_type": rt, "year": report_year,
                     "most_recent": "true", "per_page": 100, "page": page,
                     "sort": "beginning_image_number"}, key)
            for x in j.get("results", []):
                # the API IGNORES report_year: a "YE of the off year" query also returns the
                # ELECTION year's year-end (post-election money - Ossoff 2020 came out at $185M
                # vs $156M full-cycle). Keep only reports whose coverage ENDS in report_year.
                if str(x.get("coverage_end_date", ""))[:4] != str(report_year):
                    continue
                cid = x.get("committee_id")
                prev = out.get(cid)
                # a committee may file both (rare); keep the later coverage
                if prev is None or str(x.get("coverage_end_date")) > str(prev.get("coverage_end_date")):
                    out[cid] = x
            if page >= j.get("pagination", {}).get("pages", 1):
                break
            page += 1
            time.sleep(1.1)          # stay under the per-minute limit
    return out


def linkage(cycle):
    """committee_id -> candidate_id for principal/authorized committees (bulk ccl file)."""
    yy = str(cycle)[-2:]
    url = f"https://www.fec.gov/files/bulk-downloads/{cycle}/ccl{yy}.zip"
    r = requests.get(url, timeout=120, headers=H)
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("latin-1")
        cols = ["cand_id", "cand_election_yr", "fec_election_yr", "cmte_id", "cmte_tp",
                "cmte_dsgn", "linkage_id"]
        df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=cols, dtype=str)
    else:
        # no linkage file for the oldest cycles (1998): the COMMITTEE MASTER carries the same
        # committee -> candidate link (CAND_ID) for candidate committees
        r = requests.get(f"https://www.fec.gov/files/bulk-downloads/{cycle}/cm{yy}.zip",
                         timeout=120, headers=H)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("latin-1")
        cols = ["cmte_id", "cmte_nm", "tres_nm", "st1", "st2", "city", "st", "zip", "cmte_dsgn",
                "cmte_tp", "party", "filing_freq", "org_tp", "connected_org", "cand_id"]
        df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=cols, dtype=str,
                         usecols=range(15))
    df = df[df["cmte_dsgn"].isin(["P", "A"]) & df["cand_id"].astype(str).str[0].isin(["H", "S"])]
    return dict(zip(df["cmte_id"], df["cand_id"]))


def ytd_values(rep):
    return {k: float(rep.get(v) or 0.0) for k, v in YTD.items()}


def fetch_cycle(cycle, key, live):
    link = linkage(cycle)
    off = reports(key, cycle, ["YE"], cycle - 1)                       # off-year, Jan-Dec
    if live:
        # per COMMITTEE: its Q3 if already filed (due Oct 15, some file early), otherwise its
        # Q2 (through Jun 30). Choosing one report type for the whole cycle left every
        # not-yet-filed campaign with only last year's money.
        on = reports(key, cycle, ["Q2", "M7"], cycle)
        on.update(reports(key, cycle, ["Q3", "M10"], cycle))
        stage = "latest of Q2/Q3"
    else:
        on = reports(key, cycle, ["Q3", "M10"], cycle)
        stage = "Q3"
    rows = {}
    for cid in set(off) | set(on):
        cand = link.get(cid)
        if not cand:
            continue
        acc = rows.setdefault(cand, dict(cycle=cycle, cand_id=cand, n_committees=0,
                                         **{k: 0.0 for k in YTD}))
        acc["n_committees"] += 1
        # Form 3 "_ytd" columns are CYCLE-TO-DATE for House/Senate committees (verified on
        # Ossoff 2020: Q3 2020 ytd $28.65M = his 2019 + 2020 period sums), so the election-
        # year report ALONE is the total through Sep 30. The off-year year-end is only a
        # fallback for a committee that has no election-year report - never added to it
        # (the first build summed both and double-counted the off year).
        part = on.get(cid) or off.get(cid)
        for k, v in ytd_values(part).items():
            acc[k] += v
        end = str((on.get(cid) or off.get(cid) or {}).get("coverage_end_date", ""))[:10]
        acc["asof"] = max(acc.get("asof", ""), end)
    # INTER-COMMITTEE TRANSFERS (2026-10-09). total_receipts / total_disbursements include money
    # moved between a candidate's authorized committees, so summing two of them counts a transfer
    # once as the sender's spending and again as the receiver's receipts (Christine Jennings
    # 2008: 3 committees, $5.2M as-of vs $2.2M full-cycle). With 2+ committees, net the transfers
    # OUT of each summed committee off both totals. A single committee keeps its gross total:
    # its transfers IN come from a joint fundraising committee we do not sum (Scalise 2022:
    # $16.9M from his JFC), so they are real money raised, not a double count.
    for acc in rows.values():
        t = acc.get("transfers_out_auth", 0.0)
        if acc["n_committees"] > 1 and t > 0:
            acc["receipts"] = max(acc["receipts"] - t, 0.0)
            acc["disbursements"] = max(acc["disbursements"] - t, 0.0)
    print(f"  {cycle}: {len(off)} off-year YE + {len(on)} election-year {stage} reports -> "
          f"{len(rows)} candidates")
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    key = api_key()
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set(old["cycle"].unique()) if len(old) else set()
    todo = [c for c in CYCLES if args.all or c not in have or c == CYCLES[-1]]
    kept = old[~old["cycle"].isin(todo)] if len(old) else pd.DataFrame()
    for cyc in todo:
        rows = fetch_cycle(cyc, key, live=(cyc == CYCLES[-1]))
        kept = pd.concat([kept, pd.DataFrame(rows)], ignore_index=True)
        kept.to_csv(OUT, index=False)                      # checkpoint every cycle
    kept["self_fund"] = kept["cand_contrib"] + kept["cand_loans"]
    kept.to_csv(OUT, index=False)
    print(f"saved -> {OUT}  ({len(kept)} rows)")


if __name__ == "__main__":
    main()
