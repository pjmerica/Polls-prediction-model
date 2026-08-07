# -*- coding: utf-8 -*-
"""Out-of-fold win + margin predictions for every candidate, saved to CSV.

    py -X utf8 analysis/score_candidate_history.py [--candidate "Collins"] [--state ME]

Reproduces the EXPANDING-WINDOW protocol both notebooks use for their headline numbers -
each cycle is predicted by a model trained only on STRICTLY EARLIER cycles - and writes
every candidate-row's prediction to data/oof_predictions.csv.

Why this exists: the notebooks compute these predictions, print aggregate metrics, and throw
the per-row predictions away. Any question of the form "how has the model done on <this
candidate/race> historically" then needs the whole eval re-run by hand. Now it doesn't.

The dead-matchup filter (CONCERNS #38) is applied here exactly as in the notebooks, so these
numbers correspond to the shipped models, not the pre-2026-08-06 ones.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features as F
from cycles import CYCLES, EVAL_CYCLES, natl_env
from macro_features import build_macro
from paths import ROOT


def load_filtered_polls():
    """The notebooks' shared load path, dead matchups included."""
    d = pd.read_csv(os.path.join(ROOT, "polls_long_with_results.csv"), low_memory=False)
    gen = d[d["stage"] == "general"]
    q = gen.groupby(["race_id", "question_id"])["has_result"].agg(["sum", "size"])
    dead = set(q[(q["sum"] > 0) & (q["sum"] < q["size"])].index)
    drop = pd.Series([k in dead for k in zip(d["race_id"], d["question_id"])],
                     index=d.index) & (d["stage"] == "general")
    print(f"dead-matchup rows dropped: {int(drop.sum())} ({len(dead)} questions)")
    d = d[~drop]
    d = d[d["has_result"] == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None, help="substring filter for the printout")
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "oof_predictions.csv"))
    args = ap.parse_args()

    d = load_filtered_polls()
    macro = build_macro()
    NATL = natl_env()   # returns the whole {cycle: margin} map
    FUNDS = F.load_fundamentals()
    FEC = F.load_fec()
    BIAS = F.compute_bias_priors(d)
    BIOS = F.load_candidate_bios()

    base = F.build_candidate_table(d, macro, NATL, FUNDS, house_train_years=CYCLES,
                                   fec=FEC, bias_priors=BIAS, candidate_bios=BIOS)

    win_feats = json.load(open(os.path.join(ROOT, "data", "model_features.json")))
    win_feats = win_feats if isinstance(win_feats, list) else win_feats["features"]
    marg_feats = json.load(open(os.path.join(ROOT, "data", "margin_model_features.json")))
    marg_feats = marg_feats if isinstance(marg_feats, list) else marg_feats["features"]

    wparams = json.load(open(os.path.join(ROOT, "data", "model_xgb.json")))
    del wparams  # artifacts are boosters; refit per fold below with the tuned params

    rows = []
    # Every cycle that has at least one earlier cycle to train on - not just EVAL_CYCLES.
    # The notebooks only score 2018-2024 because that is the headline window, but the same
    # expanding-window protocol is valid for any cycle with history behind it, and asking
    # "how did the model do on <candidate>" usually means their whole career.
    scorable = [y for y in sorted(CYCLES) if y > min(CYCLES)]
    for test_y in scorable:
        # EXPANDING WINDOW: train strictly on earlier cycles - the honest protocol.
        house = F.compute_house_effect(d, [y for y in CYCLES if y < test_y])
        adj = F.candidate_poll_adj(d, house).rename("poll_adj").reset_index()
        c = base.drop(columns="poll_adj").merge(adj, on=["race_id", "cand_key"], how="left")

        tr = c[c["year"] < test_y]
        te = c[c["year"] == test_y].copy()
        if not len(tr) or not len(te):
            continue

        wm = xgb.XGBClassifier(max_depth=2, learning_rate=0.02, n_estimators=300,
                               min_child_weight=8, subsample=1.0, colsample_bytree=1.0,
                               reg_lambda=1, reg_alpha=0, eval_metric="logloss")
        wm.fit(tr[win_feats], tr["won"].astype(int))
        te["p_win"] = wm.predict_proba(te[win_feats])[:, 1]
        te["p_win_norm"] = te.groupby("race_id")["p_win"].transform(lambda s: s / s.sum())

        mtr = tr.dropna(subset=["vote_margin"]) if "vote_margin" in tr else tr.iloc[0:0]
        if len(mtr):
            mm = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=300,
                                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=5)
            mm.fit(mtr[marg_feats], mtr["vote_margin"])
            te["pred_margin"] = mm.predict(te[marg_feats])
        rows.append(te)

    out = pd.concat(rows, ignore_index=True)
    keep = [c for c in ["year", "race_id", "candidate", "cand_key", "party_std", "poll_avg",
                        "poll_lead", "n_polls", "won", "vote_pct", "vote_margin",
                        "p_win", "p_win_norm", "pred_margin"] if c in out.columns]
    out[keep].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} candidate-rows across {out['race_id'].nunique()} races")

    if args.candidate:
        sel = out[out["candidate"].astype(str).str.contains(args.candidate, case=False, na=False)]
        if args.state:
            sel = sel[sel["race_id"].astype(str).str.contains(f"_{args.state}_")]
        if len(sel):
            print()
            print(sel[keep].to_string(index=False))
        else:
            print(f"\nno rows matched candidate~{args.candidate!r} state={args.state}")


if __name__ == "__main__":
    main()
