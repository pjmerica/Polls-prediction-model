"""Build per-cycle macro / political-climate features from the STATIC monthly CSV.

Reads `data/macro_monthly.csv` (produced once by `fetch_macro.py`; committed, never re-pulled).
For each election cycle, the trajectory stats are computed over **that cycle's own window =
the prior election eve up to this election eve** (2018 -> 2016-11..2018-11, 2020 -> 2018-11..
2020-11, 2022 -> 2020-11..2022-11, 2024 -> 2022-11..2024-11). So `max`/`mean`/`std`/`trend`
reflect *each cycle's own* conditions, not the all-time max since 2016. The CSV still holds the
full 2016->now history; only the per-cycle slice is summarized.

Stats per metric: (a) full-window: eve / mean / max / min / std / trend / last12_delta, plus
(b) recency cuts — avg / max / trend over the **last 3, 6, and 12 months** before election eve.
The model decides which matter (heavy regularization drops the rest).

No network. If the CSV is missing, raises a clear error telling you to run fetch_macro.py.
"""
import os
import numpy as np
import pandas as pd

CYCLES = [2018, 2020, 2022, 2024]
PRES_PARTY = {2018: "REP", 2020: "REP", 2022: "DEM", 2024: "DEM"}
CSV_PATH = "data/macro_monthly.csv"
# Each cycle's window starts the day after the PRIOR election eve. For the first modeled cycle
# (2018) the prior election is 2016, so its window starts 2016-11.
PRIOR_EVE = {2018: "2016-11-01", 2020: "2018-11-01", 2022: "2020-11-01", 2024: "2022-11-01"}

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

# Recency cuts: the final N months of the cycle's window, ending at election eve.
RECENCY_CUTS = {"3mo": 3, "6mo": 6, "12mo": 12}

def _recency_stats(s, prefix):
    """avg / max / trend over the last 3, 6, and 12 months of the (already cycle-windowed) series `s`."""
    out = {}
    for label, n in RECENCY_CUTS.items():
        v = s.dropna().values.astype(float)[-n:]
        if len(v) == 0:
            out[f"{prefix}_avg_{label}"] = np.nan
            out[f"{prefix}_max_{label}"] = np.nan
            out[f"{prefix}_trend_{label}"] = np.nan
            continue
        x = np.arange(len(v))
        out[f"{prefix}_avg_{label}"] = float(v.mean())
        out[f"{prefix}_max_{label}"] = float(v.max())
        out[f"{prefix}_trend_{label}"] = float(np.polyfit(x, v, 1)[0]) if len(v) >= 2 else 0.0
    return out

def load_monthly(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python fetch_macro.py` once (on a machine with FRED access) "
            f"to create the static monthly macro CSV, then commit it.")
    m = pd.read_csv(path, parse_dates=["date"])
    return m

def build_macro(path=CSV_PATH):
    """Return {cycle: {feature: value}} — stats over each cycle's own (prior-eve -> eve) window."""
    m = load_monthly(path)
    metrics = sorted(m["metric"].unique())
    rows = {}
    for cyc in CYCLES:
        start = pd.Timestamp(PRIOR_EVE[cyc]); eve = _eve(cyc)
        f = {}
        for metric in metrics:
            full = (m[m["metric"] == metric].set_index("date")["value"].sort_index())
            if metric in YOY_METRICS:
                # compute YoY on the FULL series (needs 12 prior months), THEN slice to the cycle window
                full = (full / full.shift(12) - 1.0) * 100.0
                name = "inflation"
            else:
                name = metric
            s = full[(full.index > start) & (full.index <= eve)]   # this cycle's window only
            f.update(_stats(s, name))            # full-cycle stats (max/min/std/trend/...)
            f.update(_recency_stats(s, name))    # last-3/6/12-month avg/max/trend cuts
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
