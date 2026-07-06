"""Predict VICTORY MARGINS for future races from the polling-agg raw poll feed.

Completely separate model from the win/lose classifier (user decision): this loads the
artifact saved by margin_model.ipynb (data/margin_model_xgb.json), while predict.py loads
the classifier from model.ipynb. The poll loading/dedup/feature code is shared (imported),
so the two scripts can never diverge on inputs.

    python predict_margin.py [--cycle 2026] [--natl-env 5.8] [--polls ...] [--out preds.csv]

Output: predicted margin in percentage points vs the best opponent (positive = wins by that
much), per candidate, plus the implied race winner.
"""
import argparse
import json
import os

import pandas as pd
import xgboost as xgb

import features as F
from cycles import natl_env as natl_env_hist
from macro_features import build_macro
from predict import DEFAULT_POLLS, load_agg_polls   # same feed, same dedup

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--natl-env", type=float, default=None)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "data", "margin_model_features.json")) as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(HERE, "data", "margin_model_xgb.json"))
    print(f"margin model: trained on cycles {meta['trained_on_cycles'][0]}-"
          f"{meta['trained_on_cycles'][-1]}, {len(meta['features'])} features")

    d = load_agg_polls(args.polls, args.cycle)
    if len(d) == 0:
        raise SystemExit(f"no general-election polls found for cycle {args.cycle}")

    hist = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
    hist = F.prepare_polls(hist[hist["has_result"] == 1])
    hist["race_id"] = (hist["year"].astype(str) + "_" + hist["state"] + "_" + hist["office"]
                       + hist["district"].radd("-").where(hist["district"] != "", ""))
    house = F.compute_house_effect(hist, sorted(hist["year"].unique()))

    macro = build_macro(cycles=[args.cycle])
    ne = dict(natl_env_hist())
    if args.natl_env is not None:
        ne[args.cycle] = args.natl_env
    else:
        from fetch_generic_ballot import get_natl_env
        v = get_natl_env(args.cycle)
        if v is not None:
            ne[args.cycle] = v
            print(f"natl_env({args.cycle}) = {v:+.1f} (Wikipedia aggregator mean)")
        else:
            print("WARNING: natl_env unavailable; feature will be NaN")

    funds = F.load_fundamentals()
    fec = F.load_fec()
    cand = F.build_candidate_table(d, macro, ne, funds, house=house, fec=fec)

    missing = [f for f in meta["features"] if f not in cand.columns]
    assert not missing, f"artifact expects features absent from the built table: {missing[:8]}"
    X = cand.reindex(columns=meta["features"])
    cand["pred_margin"] = model.predict(X)

    # poll-bias robustness sweep (see predict.py / HANDOFF.md): margins under a uniform
    # 3-point national poll shift each way
    for label, dem_shift in [("R3", -3.0), ("D3", +3.0)]:
        ds = d.copy()
        sgn = ds["party_std"].map({"DEM": 1, "REP": -1}).fillna(0)
        ds["pct"] = ds["pct"] + sgn * dem_shift / 2
        cs = F.build_candidate_table(ds, macro, ne, funds, house=house, fec=fec)
        cs["p"] = model.predict(cs.reindex(columns=meta["features"]))
        mm = cs.set_index(["race_id", "cand_key"])["p"]
        cand[f"pred_margin_{label}"] = [mm.get((r, c)) for r, c in
                                        zip(cand["race_id"], cand["cand_key"])]

    out_cols = ["race_id", "state", "office", "district", "candidate", "party",
                "n_polls", "poll_avg", "poll_lead", "avg_margin_over_time",
                "prior_margin_cand", "is_incumbent", "pred_margin",
                "pred_margin_R3", "pred_margin_D3"]
    out = cand[out_cols].sort_values(["race_id", "pred_margin"], ascending=[True, False])
    out_path = args.out or os.path.join(HERE, f"margin_predictions_{args.cycle}.csv")
    out.to_csv(out_path, index=False)

    picks = out.loc[out.groupby("race_id")["pred_margin"].idxmax()]
    print(f"\nraces: {out['race_id'].nunique()} | saved -> {out_path}")
    print("\ntightest predicted margins:")
    tight = picks.reindex(picks["pred_margin"].abs().sort_values().index).head(12)
    print(tight[["race_id", "candidate", "party", "poll_lead", "pred_margin"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
