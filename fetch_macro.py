"""One-time fetch of monthly macro series from FRED -> data/macro_monthly.csv.

This data is STATIC historical record (past months never change), so pull it ONCE
and commit the CSV. The model reads the CSV; it does not re-pull FRED on every run.

Run on a machine with FRED access:
    python fetch_macro.py

Output: data/macro_monthly.csv with columns [date, metric, value], monthly from
2016-01 to the latest available. Quarterly GDP is forward-filled to monthly so every
metric is on a monthly grid. Presidential approval (no FRED series) is included from a
documented monthly table — replace with a live source if you have one.
"""
import os, io, requests
import pandas as pd

START = "2016-01-01"
OUT = "data/macro_monthly.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}

# FRED series -> our metric name. `freq` tells us how to put each on a monthly grid:
#   M = already monthly/weekly (mean), Q = quarterly (ffill), A = annual (ffill).
FRED_SERIES = {
    "unemployment":   ("UNRATE",          "M"),  # %, monthly
    "cpi":            ("CPIAUCSL",        "M"),  # index; YoY computed downstream
    "gas":            ("GASREGW",         "M"),  # $/gal regular, weekly -> monthly
    "gdp":            ("A191RL1Q225SBEA", "Q"),  # real GDP growth annualized %, quarterly
    "sentiment":      ("UMCSENT",         "M"),  # U. Michigan consumer sentiment (the 'vibe' index)
    "real_income":    ("DSPIC96",         "M"),  # real disposable personal income, monthly
    "sp500":          ("SP500",           "M"),  # S&P 500 index (daily -> monthly mean)
    "mortgage30":     ("MORTGAGE30US",    "M"),  # 30-yr fixed mortgage rate %, weekly -> monthly
    "fed_funds":      ("FEDFUNDS",        "M"),  # effective federal funds rate %, monthly
    "jobless_claims": ("ICSA",            "M"),  # initial unemployment claims, weekly -> monthly mean
    "real_wage":      ("LES1252881600Q",  "Q"),  # real median weekly earnings, quarterly
    "med_income":     ("MEHOINUSA672N",   "A"),  # real median household income, annual
}

def fred_csv(series_id, timeout=60):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    if r.text.lstrip().startswith("<"):
        raise RuntimeError(f"{series_id}: got HTML, not CSV")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()

# Documented monthly presidential approval (538/Gallup avg). Static reference; one row/month.
# (Replace/extend with a live approval source if desired — same long format.)
APPROVAL = {
    # 'YYYY-MM': approval%
    # 2016 (Obama)
    **{f"2016-{m:02d}": v for m, v in zip(range(1,13), [46,48,50,51,51,51,50,50,52,54,57,57])},
    # 2017-2020 Trump (term avg path, monthly approx from Gallup)
    **{f"2017-{m:02d}": v for m, v in zip(range(1,13), [45,43,40,41,40,39,38,37,38,38,38,39])},
    **{f"2018-{m:02d}": v for m, v in zip(range(1,13), [40,40,41,42,42,43,43,42,43,44,43,43])},
    **{f"2019-{m:02d}": v for m, v in zip(range(1,13), [40,44,43,42,42,42,42,42,43,42,43,45])},
    **{f"2020-{m:02d}": v for m, v in zip(range(1,13), [44,44,45,46,43,41,41,42,43,45,45,45])},
    # 2021-2024 Biden
    **{f"2021-{m:02d}": v for m, v in zip(range(1,13), [55,55,54,54,54,52,50,49,46,43,43,43])},
    **{f"2022-{m:02d}": v for m, v in zip(range(1,13), [43,42,42,42,41,39,38,42,43,43,42,43])},
    **{f"2023-{m:02d}": v for m, v in zip(range(1,13), [43,42,42,42,40,41,41,42,40,40,40,39])},
    **{f"2024-{m:02d}": v for m, v in zip(range(1,13), [40,38,38,39,39,38,38,40,42,42,40,40])},
}

def build():
    os.makedirs("data", exist_ok=True)
    frames = []
    for metric, (sid, freq) in FRED_SERIES.items():
        print(f"fetching {metric} ({sid}) ...", end=" ")
        try:
            df = fred_csv(sid)
        except Exception as e:
            print(f"SKIP ({type(e).__name__}: {e})")          # don't let one dead series abort the rest
            continue
        s = df.set_index("date")["value"]
        if freq in ("Q", "A"):
            s = s.resample("MS").ffill()          # quarterly/annual -> monthly forward fill
        else:
            s = s.resample("MS").mean()           # weekly/monthly -> month-start mean
        s = s[s.index >= START]
        out = s.reset_index(); out.columns = ["date", "value"]; out["metric"] = metric
        frames.append(out); print(f"{len(out)} months")

    # approval as long format
    ap = pd.DataFrame([(pd.Timestamp(k + "-01"), v, "approval") for k, v in APPROVAL.items()],
                      columns=["date", "value", "metric"])
    ap = ap[ap["date"] >= START]
    frames.append(ap)

    allm = pd.concat(frames, ignore_index=True).sort_values(["metric", "date"])
    allm.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}  ({len(allm)} rows, {allm['metric'].nunique()} metrics, "
          f"{allm['date'].min().date()}..{allm['date'].max().date()})")

if __name__ == "__main__":
    build()
