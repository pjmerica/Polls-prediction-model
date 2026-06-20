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

START = "2016-01-01"
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

# Presidential approval (no clean DBnomics series): documented monthly table (538/Gallup avg).
APPROVAL = {
    **{f"2016-{m:02d}": v for m, v in zip(range(1,13), [46,48,50,51,51,51,50,50,52,54,57,57])},
    **{f"2017-{m:02d}": v for m, v in zip(range(1,13), [45,43,40,41,40,39,38,37,38,38,38,39])},
    **{f"2018-{m:02d}": v for m, v in zip(range(1,13), [40,40,41,42,42,43,43,42,43,44,43,43])},
    **{f"2019-{m:02d}": v for m, v in zip(range(1,13), [40,44,43,42,42,42,42,42,43,42,43,45])},
    **{f"2020-{m:02d}": v for m, v in zip(range(1,13), [44,44,45,46,43,41,41,42,43,45,45,45])},
    **{f"2021-{m:02d}": v for m, v in zip(range(1,13), [55,55,54,54,54,52,50,49,46,43,43,43])},
    **{f"2022-{m:02d}": v for m, v in zip(range(1,13), [43,42,42,42,41,39,38,42,43,43,42,43])},
    **{f"2023-{m:02d}": v for m, v in zip(range(1,13), [43,42,42,42,40,41,41,42,40,40,40,39])},
    **{f"2024-{m:02d}": v for m, v in zip(range(1,13), [40,38,38,39,39,38,38,40,42,42,40,40])},
}

def build():
    os.makedirs("data", exist_ok=True)
    frames, ok, skipped = [], [], []
    for metric, (prov, ds, code, freq) in SERIES.items():
        try:
            s = fetch(prov, ds, code)
            s = s.resample("MS").ffill() if freq in ("Q", "A") else s.resample("MS").mean()
            s = s[s.index >= START]
            if s.empty:
                raise RuntimeError("empty after window")
            out = s.reset_index(); out.columns = ["date", "value"]; out["metric"] = metric
            frames.append(out); ok.append(metric)
            print(f"  OK   {metric:14} {len(out)} months  ({s.index.min().date()}..{s.index.max().date()})")
        except Exception as e:
            skipped.append(metric); print(f"  SKIP {metric:14} {prov}/{ds}/{code}  ({e})")

    ap = pd.DataFrame([(pd.Timestamp(k + "-01"), v, "approval") for k, v in APPROVAL.items()],
                      columns=["date", "value", "metric"])
    ap = ap[ap["date"] >= START]; frames.append(ap); ok.append("approval")

    allm = pd.concat(frames, ignore_index=True).sort_values(["metric", "date"])
    allm.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}  ({len(allm)} rows)")
    print(f"metrics OK ({len(ok)}): {ok}")
    if skipped:
        print(f"skipped ({len(skipped)}): {skipped}  (model just won't see these)")

if __name__ == "__main__":
    build()
