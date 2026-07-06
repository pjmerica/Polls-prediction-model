"""One-time fetch of monthly macro series -> data/macro_monthly.csv.

This data is STATIC historical record (past months never change), so pull it ONCE
and commit the CSV. The model reads the CSV; it does NOT re-pull on every run.

    python fetch_macro.py

Source: **DBnomics** (https://db.nomics.world), a free aggregator of official statistics
with no API key. We use it instead of FRED because FRED's CSV host (`fredgraph.csv`) was
unreachable from both the dev sandbox and the user's machine, whereas DBnomics responds.
DBnomics does not carry FRED itself, so we pull from the upstream agencies (BLS, BEA, EIA,
Federal Reserve, U. Michigan, DOL) that FRED republishes.

Each series is resolved to monthly. Any series that fails to fetch is SKIPPED with a note
(the CSV is still produced). Output: long format [date, value, metric].
"""
import os, requests
import pandas as pd

# Pull the FULL available history of each series (CPI/unemployment ~1947, gas ~1990, etc.).
# It's static reference data, so we keep everything in case we get older polls later.
# The model only *uses* the cycles it can match to polls (2018+), but the data is all here.
START = None   # set to e.g. "2016-01-01" to truncate; None = each series' full range
OUT = "data/macro_monthly.csv"
H = {"User-Agent": "Mozilla/5.0 (research)"}
BASE = "https://api.db.nomics.world/v22/series"

# metric -> (provider, dataset, series_code, frequency)
#   freq: M monthly/weekly (mean to month), Q quarterly (ffill), A annual (ffill)
# Codes confirmed to return data from DBnomics as of build time; unverified ones will skip.
SERIES = {
    "unemployment": ("BLS", "ln",  "LNS14000000",   "M"),  # unemployment rate, SA %
    "cpi":          ("BLS", "cu",  "CUSR0000SA0",   "M"),  # CPI-U, SA index -> YoY downstream
    "gas":          ("EIA", "PET", "EMM_EPMR_PTE_NUS_DPG.M", "M"),  # regular gas $/gal
    "fed_funds":    ("FED", "H15", "RIFSPFF_N.M",   "M"),  # effective fed funds rate %
    # --- additional series: kept if they resolve, skipped (with a note) if not ---
    "cpi_core":     ("BLS", "cu",  "CUSR0000SA0L1E", "M"), # core CPI (ex food & energy)
    "unemp_u6":     ("BLS", "ln",  "LNS13327709",   "M"),  # U-6 underemployment %
}

def fetch(provider, dataset, code, timeout=45):
    u = f"{BASE}/{provider}/{dataset}/{code}?observations=1"
    r = requests.get(u, timeout=timeout, headers=H)
    docs = r.json().get("series", {}).get("docs", [])
    if not docs:
        raise RuntimeError(f"no data ({r.status_code})")
    d = docs[0]
    s = pd.Series(d["value"], index=pd.PeriodIndex(d["period"], freq="M").to_timestamp()
                  if len(d["period"][0]) == 7 else pd.to_datetime(d["period"]))
    return pd.to_numeric(s, errors="coerce").dropna()

# Presidential approval now comes from data/approval_monthly.csv, produced by
# fetch_approval.py (Gallup via UCSB American Presidency Project, 1993->present).
# Run `python fetch_approval.py` first if that file is missing.
APPROVAL_CSV = "data/approval_monthly.csv"

def build():
    os.makedirs("data", exist_ok=True)
    frames, ok, skipped = [], [], []
    for metric, (prov, ds, code, freq) in SERIES.items():
        try:
            s = fetch(prov, ds, code)
            s = s.resample("MS").ffill() if freq in ("Q", "A") else s.resample("MS").mean()
            if START is not None:
                s = s[s.index >= START]
            if s.empty:
                raise RuntimeError("empty after window")
            out = s.reset_index(); out.columns = ["date", "value"]; out["metric"] = metric
            frames.append(out); ok.append(metric)
            print(f"  OK   {metric:14} {len(out)} months  ({s.index.min().date()}..{s.index.max().date()})")
        except Exception as e:
            skipped.append(metric); print(f"  SKIP {metric:14} {prov}/{ds}/{code}  ({e})")

    if not os.path.exists(APPROVAL_CSV):
        raise FileNotFoundError(f"{APPROVAL_CSV} not found - run `python fetch_approval.py` first")
    ap = pd.read_csv(APPROVAL_CSV, parse_dates=["date"])
    if START is not None:
        ap = ap[ap["date"] >= START]
    frames.append(ap); ok.append("approval")
    print(f"  OK   approval       {len(ap)} months  ({ap['date'].min().date()}..{ap['date'].max().date()})  <- {APPROVAL_CSV}")

    allm = pd.concat(frames, ignore_index=True).sort_values(["metric", "date"])
    allm.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}  ({len(allm)} rows)")
    print(f"metrics OK ({len(ok)}): {ok}")
    if skipped:
        print(f"skipped ({len(skipped)}): {skipped}  (model just won't see these)")

if __name__ == "__main__":
    build()
