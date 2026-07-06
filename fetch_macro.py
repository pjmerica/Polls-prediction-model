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
    "sentiment":    ("SCSMICH", "MICS", "ICS",      "M"),  # UMich Index of Consumer Sentiment (1978+)
}

# Generic-ballot monthly D-R margin, produced by `python fetch_generic_ballot.py --monthly`
# (raw_polls House-G-US polls 1998-2022 + VoteHub 2024-12+; the 2024 cycle window has no
# machine-readable per-poll source -> those months are absent = NaN features, documented).
GENERIC_CSV = "data/generic_ballot_monthly.csv"

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

# DBnomics' BLS mirror lags (it stopped at 2025-01 as of 2026-07), so BLS series get an
# overlay of the last ~10 years straight from the BLS public API (no key; 25 req/day cap,
# one request covers all our BLS series). API data wins where the two overlap.
def fetch_bls_recent(series_ids, years_back=9, timeout=60):
    end = pd.Timestamp.now().year
    r = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                      json={"seriesid": list(series_ids),
                            "startyear": str(end - years_back), "endyear": str(end)},
                      headers={"Content-Type": "application/json", **H}, timeout=timeout)
    j = r.json()
    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API: {j.get('status')} {j.get('message')}")
    out = {}
    for s in j["Results"]["series"]:
        rows = []
        for d in s["data"]:
            if not d["period"].startswith("M") or d["period"] == "M13":
                continue
            v = pd.to_numeric(d["value"], errors="coerce")   # '-' = BLS missing placeholder
            if pd.notna(v):
                rows.append((pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), float(v)))
        out[s["seriesID"]] = pd.Series(dict(rows)).sort_index()
    return out

# Presidential approval now comes from data/approval_monthly.csv, produced by
# fetch_approval.py (Gallup via UCSB American Presidency Project, 1993->present).
# Run `python fetch_approval.py` first if that file is missing.
APPROVAL_CSV = "data/approval_monthly.csv"

def build():
    os.makedirs("data", exist_ok=True)
    bls_codes = [code for (prov, ds, code, f) in SERIES.values() if prov == "BLS"]
    try:
        bls_recent = fetch_bls_recent(bls_codes)
        print(f"  BLS API overlay: {len(bls_recent)} series, latest "
              f"{max(s.index.max() for s in bls_recent.values()).date()}")
    except Exception as e:
        bls_recent = {}
        print(f"  BLS API overlay unavailable ({e}) - DBnomics only")

    frames, ok, skipped = [], [], []
    for metric, (prov, ds, code, freq) in SERIES.items():
        try:
            s = fetch(prov, ds, code)
            if code in bls_recent:                     # fresh months win over the lagged mirror
                s = bls_recent[code].combine_first(s)
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

    if os.path.exists(GENERIC_CSV):
        gb = pd.read_csv(GENERIC_CSV, parse_dates=["date"])
        if START is not None:
            gb = gb[gb["date"] >= START]
        frames.append(gb); ok.append("generic_ballot")
        print(f"  OK   generic_ballot {len(gb)} months  ({gb['date'].min().date()}..{gb['date'].max().date()})  <- {GENERIC_CSV}")
    else:
        print(f"  SKIP generic_ballot ({GENERIC_CSV} missing - run `python fetch_generic_ballot.py --monthly`)")

    allm = pd.concat(frames, ignore_index=True).sort_values(["metric", "date"])
    allm.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}  ({len(allm)} rows)")
    print(f"metrics OK ({len(ok)}): {ok}")
    if skipped:
        print(f"skipped ({len(skipped)}): {skipped}  (model just won't see these)")

if __name__ == "__main__":
    build()
