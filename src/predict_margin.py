"""Predict VICTORY MARGINS for future races from the polling-agg raw poll feed.

Completely separate model from the win model (user decision): this loads the RACE-LEVEL margin
model (data/race_margin_model_xgb.json, models/poll/race_margin_model.py; audit item 7, served
since 2026-10-27). It predicts each candidate's vote share, and the published margin is that
share minus the best rival's predicted share in the same race (src/race_level.py), so exactly one
candidate is ahead and the top two's margins cancel. Before 2026-10-27 it loaded
margin_model.ipynb's per-candidate margin regressor (data/margin_model_xgb.json), no longer
served. The poll loading/dedup/feature code is shared with predict.py (imported), so the two
scripts can never diverge on inputs.

    python predict_margin.py [--cycle 2026] [--natl-env 5.8] [--polls ...] [--out preds.csv]

Output: predicted margin in percentage points vs the best opponent (positive = wins by that
much), per candidate, plus pred_share (the predicted vote share it is derived from).
"""
import argparse
import json
import os

import pandas as pd
import xgboost as xgb

import features as F
from cycles import natl_env as natl_env_hist
from macro_features import build_macro
from race_level import race_margins
from predict import DEFAULT_POLLS, load_agg_polls, patch_redistricted_priors

import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
from paths import ROOT   # one definition for the whole repo (paths.py)
import paths as _paths   # module handle: `out` is a very common local
                         # variable name in this repo, so never import it bare

HERE = ROOT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--natl-env", type=float, default=None)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "data", "race_margin_model_features.json")) as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(HERE, "data", "race_margin_model_xgb.json"))

    def shares_and_margins(table):
        t = table[["race_id"]].copy()
        # float64: XGBoost predicts in float32, and the margins are differences of shares
        t["share"] = model.predict(table.reindex(columns=meta["features"])).astype("float64")
        return t["share"], race_margins(t, "share")
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
    bias = F.compute_bias_priors(hist)   # prior-cycles poll-bias by state (2026 uses <=2024)

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
    fec = F.load_fec(extended=True)
    # bio_office_level shipped 2026-07-29: load bios at serve time or the feature is all-NaN for
    # live races (train/serve skew); the artifact now expects it (assert below would fire).
    bios = F.load_candidate_bios()
    # primary_results too (2026-10-14): margin_model.ipynb trains WITH it, and this script
    # left it out, so primary_margin / opp_primary_margin / primary_margin_diff were all-NaN
    # at serve time - the train/serve skew predict.py and explain_2026.py already guard against
    primres = F.load_primary_results()
    cand = patch_redistricted_priors(
        F.build_candidate_table(d, macro, ne, funds, house=house, fec=fec, bias_priors=bias,
                                candidate_bios=bios, primary_results=primres))

    missing = [f for f in meta["features"] if f not in cand.columns]
    assert not missing, f"artifact expects features absent from the built table: {missing[:8]}"
    X = cand.reindex(columns=meta["features"])
    cand["pred_share"], cand["pred_margin"] = shares_and_margins(cand)

    # poll-bias robustness sweep (see predict.py / HANDOFF.md): margins under a uniform
    # 3-point national poll shift each way
    for label, dem_shift in [("R3", -3.0), ("D3", +3.0)]:
        ds = d.copy()
        sgn = ds["party_std"].map({"DEM": 1, "REP": -1}).fillna(0)
        ds["pct"] = ds["pct"] + sgn * dem_shift / 2
        cs = patch_redistricted_priors(
            F.build_candidate_table(ds, macro, ne, funds, house=house, fec=fec, bias_priors=bias,
                                    candidate_bios=bios, primary_results=primres))
        cs["p"] = shares_and_margins(cs)[1]
        mm = cs.set_index(["race_id", "cand_key"])["p"]
        cand[f"pred_margin_{label}"] = [mm.get((r, c)) for r, c in
                                        zip(cand["race_id"], cand["cand_key"])]

    out_cols = ["race_id", "state", "office", "district", "candidate", "cand_key", "party",
                "display_party",
                "n_polls", "poll_avg", "poll_lead", "avg_margin_over_time",
                "prior_margin_cand", "is_incumbent", "pred_share", "pred_margin",
                "pred_margin_R3", "pred_margin_D3"]
    out = cand[out_cols].sort_values(["race_id", "pred_margin"], ascending=[True, False])
    out_path = args.out or _paths.out(f"margin_predictions_{args.cycle}.csv")
    out.to_csv(out_path, index=False)

    # meta sidecar, same fields as predict.py's: the publish gate (tests/test_publish.py) checks
    # that the win and margin files come from the same run and poll feed, and the fill rates
    import datetime
    meta_out = dict(
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        polls_max_end_date=str(d["end_date"].max().date()),
        n_poll_rows=int(len(d)), n_races=int(out["race_id"].nunique()),
        natl_env=ne.get(args.cycle),
        run_id=os.environ.get("MODEL_RUN_ID"),
        feature_fill={f: round(float(X[f].notna().mean()), 4) for f in meta["features"]},
    )
    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)

    picks = out.loc[out.groupby("race_id")["pred_margin"].idxmax()]
    print(f"\nraces: {out['race_id'].nunique()} | saved -> {out_path}")
    print("\ntightest predicted margins:")
    tight = picks.reindex(picks["pred_margin"].abs().sort_values().index).head(12)
    print(tight[["race_id", "candidate", "party", "poll_lead", "pred_margin"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
