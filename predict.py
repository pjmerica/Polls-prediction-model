"""Predict win probabilities for FUTURE races from the polling-agg raw poll feed.

    python predict.py [--cycle 2026] [--natl-env 1.5] [--polls path.csv ...] [--out preds.csv]

Inputs (ALL local / already-committed — no network):
- Raw polls: the polling-agg repo's data/raw/{nyt_polls,wikipedia_polls}.csv (default paths
  point at the sibling checkout). Only columns a bare poll feed has are used:
  pollster, candidate, party, stage, sample_size, end_date, implied_prob (pct/100), race_id.
- data/model_xgb.json + data/model_features.json — trained/tuned by model.ipynb (all 14
  cycles, params tuned on 1998-2016).
- data/macro_monthly.csv, data/races.csv, data/res_*.csv — frozen fundamentals.
- Historical polls (polls_long_with_results.csv) — ONLY to compute pollster house effects.

Known gaps (documented in CONCERNS.md):
- --natl-env must be looked up manually (e.g. RealClearPolling generic-ballot average,
  DEM minus REP). If omitted, natl_env_cand is NaN (XGBoost routes missing).
- approval/BLS macro series currently end early 2025; late-window macro stats are stale.

Dedup: the polling-agg raw files contain internal repeats and NYT/Wikipedia cross-source
duplicates (audited 2026-07-05: ~1.3k internal, ~2.7k cross). Collapsed here on
(pollster, end_date, race, candidate), preferring the NYT row.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
from cycles import natl_env as natl_env_hist
from macro_features import build_macro

HERE = os.path.dirname(os.path.abspath(__file__))
POLLING_AGG_RAW = os.path.join(HERE, "..", "Polling Agg",
                               "Polling agg and Prediction markets", "data", "raw")
DEFAULT_POLLS = [os.path.join(POLLING_AGG_RAW, "nyt_polls.csv"),
                 os.path.join(POLLING_AGG_RAW, "wikipedia_polls.csv")]

OFFICE_CODE = {"SEN": "Senate", "H": "House", "GOV": "Governor"}

def election_date(cycle):
    """First Tuesday after the first Monday in November."""
    d = pd.Timestamp(f"{cycle}-11-01")
    monday = d + pd.Timedelta(days=(7 - d.dayofweek) % 7)   # first Monday (Mon=0)
    return monday + pd.Timedelta(days=1)

def parse_race_id(rid):
    """'2026-SEN-MI' / '2026-H-AL-01' / '2026-SEN-OH-S' -> (year, office, state, district)."""
    parts = str(rid).split("-")
    if len(parts) < 3 or parts[1] not in OFFICE_CODE:
        return None
    year = int(parts[0])
    office = OFFICE_CODE[parts[1]]
    state = parts[2].upper()
    district = ""
    if office == "House" and len(parts) > 3:
        district = F.pdist(parts[3])
    return year, office, state, district

def load_agg_polls(paths, cycle):
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p, low_memory=False)
        df["_src_priority"] = i          # earlier path wins dedup ties (NYT first)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    parsed = raw["race_id"].map(parse_race_id)
    ok = parsed.notna()
    raw = raw[ok].copy()
    raw[["year", "office", "state", "district"]] = pd.DataFrame(
        parsed[ok].tolist(), index=raw.index)
    raw = raw[raw["year"] == cycle]
    raw = raw[raw["stage"].astype(str).str.lower().str.contains("general", na=False)]
    raw = raw[~raw["candidate"].map(F.is_junk_answer)]

    d = pd.DataFrame({
        "year": raw["year"].astype(int),
        "state": raw["state"],
        "office": raw["office"],
        "district": raw["district"],
        "candidate": raw["candidate"],
        "party_std": raw["party"].map(F.npar),
        "pct": pd.to_numeric(raw["implied_prob"], errors="coerce") * 100.0,
        "end_date": raw["end_date"],
        "sample_size": pd.to_numeric(raw["sample_size"], errors="coerce"),
        "pollster": raw["pollster"],
        "poll_id": raw.get("poll_id"),
        "_src_priority": raw["_src_priority"],
    })
    d["election_date"] = election_date(cycle)
    d["cand_key"] = d["candidate"].map(F.norm_name)
    d = d.dropna(subset=["pct", "cand_key"])

    # dedup: internal repeats + NYT/Wikipedia cross-source duplicates
    before = len(d)
    d = (d.sort_values("_src_priority")
           .drop_duplicates(subset=["pollster", "end_date", "year", "state",
                                    "office", "district", "cand_key"], keep="first")
           .drop(columns="_src_priority"))
    print(f"polls loaded: {before} rows -> {len(d)} after dedup "
          f"({before - len(d)} duplicates removed)")

    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].map(F.dist_str).radd("-").where(d["district"].map(F.dist_str) != "", ""))
    return F.prepare_polls(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--natl-env", type=float, default=None,
                    help="generic-ballot DEM-REP margin (e.g. RealClearPolling average)")
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "data", "model_features.json")) as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(HERE, "data", "model_xgb.json"))
    print(f"model: trained on cycles {meta['trained_on_cycles'][0]}-{meta['trained_on_cycles'][-1]}, "
          f"{len(meta['features'])} features")

    d = load_agg_polls(args.polls, args.cycle)
    if len(d) == 0:
        raise SystemExit(f"no general-election polls found for cycle {args.cycle}")

    # house effect from HISTORICAL training polls, applied to the new cycle's pollsters
    hist = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
    hist = F.prepare_polls(hist[hist["has_result"] == 1])
    hist["race_id"] = (hist["year"].astype(str) + "_" + hist["state"] + "_" + hist["office"]
                       + hist["district"].radd("-").where(hist["district"] != "", ""))
    house = F.compute_house_effect(hist, sorted(hist["year"].unique()))

    macro = build_macro(cycles=[args.cycle])
    ne = dict(natl_env_hist())
    if args.natl_env is not None:
        ne[args.cycle] = args.natl_env
        print(f"natl_env({args.cycle}) = {args.natl_env:+.1f} (given via --natl-env)")
    else:
        # the one live fetch in the project: current-cycle info can't be frozen by definition
        from fetch_generic_ballot import get_natl_env
        v = get_natl_env(args.cycle)
        if v is not None:
            ne[args.cycle] = v
            print(f"natl_env({args.cycle}) = {v:+.1f} (Wikipedia aggregator mean; "
                  f"override with --natl-env)")
        else:
            print("WARNING: generic-ballot fetch failed and --natl-env not given; "
                  "natl_env_cand will be missing (NaN)")

    funds = F.load_fundamentals()
    cand = F.build_candidate_table(d, macro, ne, funds, house=house)

    X = cand.reindex(columns=meta["features"])
    cand["win_prob"] = model.predict_proba(X)[:, 1]
    # within-race normalized probability (raw probs are per-candidate, not a race simplex)
    cand["win_prob_norm"] = (cand["win_prob"]
                             / cand.groupby("race_id")["win_prob"].transform("sum"))

    out_cols = ["race_id", "state", "office", "district", "candidate", "party",
                "n_polls", "poll_avg", "poll_lead", "prior_margin_cand",
                "is_incumbent", "win_prob", "win_prob_norm"]
    out = cand[out_cols].sort_values(["race_id", "win_prob"], ascending=[True, False])
    out_path = args.out or os.path.join(HERE, f"predictions_{args.cycle}.csv")
    out.to_csv(out_path, index=False)

    picks = out.loc[out.groupby("race_id")["win_prob"].idxmax()]
    close = picks[np.isclose(picks["win_prob_norm"], 0.5, atol=0.10)]
    print(f"\nraces predicted: {out['race_id'].nunique()} "
          f"({dict(picks.groupby('office').size())})")
    print(f"saved -> {out_path}")
    print("\nclosest races (leader's normalized prob within 40-60%):")
    print(close[["race_id", "candidate", "party", "poll_avg", "poll_lead", "win_prob_norm"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
