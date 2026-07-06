"""Single source of truth for election-cycle constants.

Everything that used to be hardcoded in several places (CYCLES, PRES_PARTY, the per-cycle
macro windows, natl_env) lives here. Extending to a new cycle = touch THIS file only.

Cycles are even-year general elections. Coverage starts 1998 because that's where the
frozen 538 raw_polls file (data/raw_polls_538.csv) starts. Odd-year races (VA/NJ etc.)
are intentionally excluded (tiny poll counts, their own macro windows — not worth it).
"""
import os
import pandas as pd

# ---- modeled cycles ----
CYCLES = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

# cycles used for hyperparameter TUNING vs honest EVALUATION (nested scheme):
# tune on the old cycles, evaluate leave-one-cycle-out on the modern ones the tuner never saw.
TUNE_CYCLES = [c for c in CYCLES if c < 2018]
EVAL_CYCLES = [c for c in CYCLES if c >= 2018]

# party holding the White House on election day
PRES_PARTY = {
    1998: "DEM", 2000: "DEM",             # Clinton
    2002: "REP", 2004: "REP", 2006: "REP", 2008: "REP",   # G.W. Bush
    2010: "DEM", 2012: "DEM", 2014: "DEM", 2016: "DEM",   # Obama
    2018: "REP", 2020: "REP",             # Trump 45
    2022: "DEM", 2024: "DEM",             # Biden
    2026: "REP",                          # Trump 47
}

def eve(cycle):
    """Election-eve cutoff: Nov 1 of the cycle (see METHODOLOGY.md)."""
    return pd.Timestamp(f"{cycle}-11-01")

def prior_eve(cycle):
    """Start of the cycle's own macro window = the previous even-year election eve."""
    return pd.Timestamp(f"{cycle - 2}-11-01")

# ---- national environment: generic-ballot DEM-REP margin, last 30 days before election ----
# 1996-2016: computed from the frozen 538 daily historical file (committed).
# 2018-2024: frozen values computed earlier from a (now unreachable) Internet Archive snapshot
#            of the 538 generic-ballot average; the daily file above ends in 2016.
# 2026+:     must be supplied at predict time (e.g. RealClearPolling average) — no free
#            machine-readable source is wired up yet.
_GB_HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "generic_ballot_hist_538.csv")
_NATL_ENV_FROZEN = {2018: 7.8, 2020: 7.5, 2022: 0.7, 2024: 0.1}

def natl_env():
    """{cycle: DEM-REP generic-ballot margin averaged over the 30 days before election eve}."""
    out = dict(_NATL_ENV_FROZEN)
    gb = pd.read_csv(_GB_HIST, low_memory=False)
    gb = gb[gb["subgroup"].astype(str).str.lower() == "all polls"].copy()
    gb["date"] = pd.to_datetime(gb["modeldate"], errors="coerce")
    gb["margin"] = (pd.to_numeric(gb["dem_estimate"], errors="coerce")
                    - pd.to_numeric(gb["rep_estimate"], errors="coerce"))
    gb = gb.dropna(subset=["date", "margin"])
    for cyc in CYCLES:
        if cyc in out:
            continue
        e = eve(cyc)
        win = gb[(gb["date"] > e - pd.Timedelta(days=30)) & (gb["date"] <= e)]
        if len(win):
            out[cyc] = round(float(win["margin"].mean()), 2)
    return out

if __name__ == "__main__":
    ne = natl_env()
    print("cycles:", CYCLES)
    print("tune:", TUNE_CYCLES, "| eval:", EVAL_CYCLES)
    print("natl_env:", {k: ne.get(k) for k in CYCLES})
    missing = [c for c in CYCLES if c not in ne]
    print("missing natl_env:", missing or "none")
