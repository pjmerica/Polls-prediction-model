"""FEC campaign-finance summaries (House + Senate) -> data/fec_summary.csv.

Source: FEC bulk `webl{yy}.zip` ("current campaigns" one-row-per-candidate summaries),
no API key required, format stable 1998-2026. Committed once per cycle like all static data;
re-run near an election to refresh the CURRENT cycle's numbers (candidates file quarterly).

    python fetch_fec.py            # all cycles missing from the CSV + always refresh current
    python fetch_fec.py --all      # force re-download everything

## Leakage / cutoff design (read this)
Historical cycle files are END-OF-CYCLE totals (coverage through Dec 31) — they include
~6 weeks of money raised after our Sep-30 information cutoff, and post-election winner
money. Raw totals therefore carry mild look-ahead. The FEATURES built from this file are
deliberately RATIO-shaped (share of race money, composition percentages): both candidates'
totals truncate at the SAME date, so ratios are comparable between a Dec-31 training row
and a mid-campaign predict row, and money *shares* are empirically stable over a cycle.
Raw `fund_receipts_ln` is included but is the feature to distrust first.
Per-report as-of-Sep-30 totals (the fully clean version) need the FEC API with a real key
(free: api.data.gov signup) — on the roadmap.

## What's NOT here
- Average donation / small-dollar share: webl has no donor counts or unitemized split;
  needs the API (itemized bulk files are multi-GB). Roadmap, pending an API key.
- Governors: the FEC only covers federal races. State campaign-finance is 50 systems;
  see CONCERNS.md for the FollowTheMoney investigation.
"""
import argparse
import io
import os
import zipfile

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (research; polling model)"}
CYCLES = list(range(1998, 2027, 2))
OUT = "data/fec_summary.csv"

COLS = ["cand_id", "cand_name", "ici", "pty_cd", "party", "receipts", "trans_from_auth",
        "disb", "trans_to_auth", "coh_bop", "coh_cop", "cand_contrib", "cand_loans",
        "other_loans", "cand_loan_repay", "other_loan_repay", "debts", "indiv_contrib",
        "state", "district", "spec", "prim", "runoff", "gen", "gen_pct",
        "pac_contrib", "party_contrib", "coverage_end", "indiv_refunds", "cmte_refunds"]

def fetch_cycle(cycle):
    yy = str(cycle)[2:]
    u = f"https://www.fec.gov/files/bulk-downloads/{cycle}/webl{yy}.zip"
    r = requests.get(u, timeout=120, headers=H)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=COLS,
                     dtype=str, on_bad_lines="skip")
    df["cycle"] = cycle
    for c in ["receipts", "indiv_contrib", "pac_contrib", "party_contrib",
              "cand_contrib", "cand_loans"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["office"] = df["cand_id"].str[0].map({"H": "House", "S": "Senate"})
    df["self_fund"] = df["cand_contrib"] + df["cand_loans"]
    keep = df[df["office"].notna()][
        ["cycle", "cand_id", "cand_name", "ici", "party", "state", "district",
         "office", "receipts", "indiv_contrib", "pac_contrib", "party_contrib",
         "self_fund", "coverage_end"]]
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    old = pd.read_csv(OUT, dtype={"district": str}) if os.path.exists(OUT) else pd.DataFrame()
    have = set(old["cycle"].unique()) if len(old) else set()
    current = CYCLES[-1]
    todo = [c for c in CYCLES if args.all or c not in have or c == current]

    frames = [old[~old["cycle"].isin(todo)]] if len(old) else []
    for cyc in todo:
        try:
            df = fetch_cycle(cyc)
            frames.append(df)
            print(f"  OK  {cyc}: {len(df)} candidates "
                  f"(coverage up to {df['coverage_end'].max()})")
        except Exception as e:
            print(f"  SKIP {cyc}: {type(e).__name__}: {e}")

    allf = pd.concat(frames, ignore_index=True).sort_values(["cycle", "state", "district"])
    allf.to_csv(OUT, index=False)
    print(f"saved -> {OUT}  ({len(allf)} rows, cycles "
          f"{int(allf['cycle'].min())}-{int(allf['cycle'].max())})")

if __name__ == "__main__":
    main()
