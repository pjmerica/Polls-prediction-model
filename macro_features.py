"""Build per-cycle macro / political-climate features from the STATIC monthly CSV.

Reads `data/macro_monthly.csv` (produced once by `fetch_macro.py`; committed, never re-pulled).
For each election cycle, uses the **expanding window from 2016-01 up to that cycle's election
eve** (2018 -> 2016..2018-11, 2020 -> 2016..2020-11, etc.) and condenses each metric's monthly
series into trajectory stats: eve / mean / max / min / std / trend / last12_delta. The model
then decides which of these matter (heavy regularization drops the rest).

No network. If the CSV is missing, raises a clear error telling you to run fetch_macro.py.
"""
import os
import numpy as np
import pandas as pd

CYCLES = [2018, 2020, 2022, 2024]
PRES_PARTY = {2018: "REP", 2020: "REP", 2022: "DEM", 2024: "DEM"}
CSV_PATH = "data/macro_monthly.csv"
WINDOW_START = "2016-01-01"

# metrics that should enter the model as YoY % change of an index, not the raw level
YOY_METRICS = {"cpi"}   # cpi index -> inflation YoY%

def _eve(cycle):
    return pd.Timestamp(f"{cycle}-11-01")   # election-eve cutoff (first of Nov of the cycle)

def _stats(s, prefix):
    """eve / mean / max / min / std / trend / last12_delta for a monthly Series (date-indexed)."""
    v = s.dropna().values.astype(float)
    if len(v) == 0:
        keys = ["eve", "mean", "max", "min", "std", "trend", "last12_delta"]
        return {f"{prefix}_{k}": np.nan for k in keys}
    x = np.arange(len(v))
    trend = float(np.polyfit(x, v, 1)[0]) if len(v) >= 2 else 0.0
    last12 = float(v[-1] - v[-13]) if len(v) >= 13 else 0.0
    return {f"{prefix}_eve": float(v[-1]), f"{prefix}_mean": float(v.mean()),
            f"{prefix}_max": float(v.max()), f"{prefix}_min": float(v.min()),
            f"{prefix}_std": float(v.std()), f"{prefix}_trend": trend,
            f"{prefix}_last12_delta": last12}

def load_monthly(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python fetch_macro.py` once (on a machine with FRED access) "
            f"to create the static monthly macro CSV, then commit it.")
    m = pd.read_csv(path, parse_dates=["date"])
    return m

def build_macro(path=CSV_PATH):
    """Return {cycle: {feature: value}} from the saved monthly CSV (expanding 2016->eve window)."""
    m = load_monthly(path)
    metrics = sorted(m["metric"].unique())
    rows = {}
    for cyc in CYCLES:
        eve = _eve(cyc)
        f = {}
        for metric in metrics:
            s = (m[m["metric"] == metric]
                 .set_index("date")["value"].sort_index())
            s = s[(s.index >= WINDOW_START) & (s.index <= eve)]
            if metric in YOY_METRICS:                      # index -> YoY %
                s = (s / s.shift(12) - 1.0) * 100.0
                name = "inflation"
            else:
                name = metric
            f.update(_stats(s, name))
        rows[cyc] = f
    return rows

if __name__ == "__main__":
    try:
        macro = build_macro()
    except FileNotFoundError as e:
        print(e); raise SystemExit(1)
    df = pd.DataFrame(macro).T
    feats = sorted({c for c in df.columns})
    print(f"cycles: {list(macro)} | features per cycle: {len(feats)}")
    show = [c for c in df.columns if c.endswith("_max") or c.endswith("_last12_delta")]
    print(df[sorted(show)].round(2).to_string())
