"""SHAP explanations for the 2026 predictions -> model_explanations_2026.json.

For every predicted race, explains WHY each model predicts what it does, via the top-10
features by |SHAP| — one block for the WIN model, one for the (separate) MARGIN model.
Powers the dashboard's per-race "Explain" modal (Model vs Markets tab).

Framing: we explain the LEADING DEMOCRAT's row (highest win_prob among DEM candidates),
because the tab's headline number is the race's Democratic win probability. If a race has
no Democrat, the overall leader is explained instead (party is recorded either way).

Units: the win model's SHAP values are in LOG-ODDS (XGBoost's native output for a
classifier) — sign/direction and relative size are what matter for the chart; the base and
final values are converted to probabilities for display. The margin model's SHAP values
are directly in PERCENTAGE POINTS of victory margin.

Inputs are identical to predict.py (same loaders, same redistricting patch), so the
explained prediction matches the published one when run in the same refresh.

    python explain_2026.py [--cycle 2026] [--natl-env 5.8]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
from cycles import natl_env as natl_env_hist
from macro_features import build_macro
from predict import DEFAULT_POLLS, load_agg_polls, patch_redistricted_priors

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets")

# ---- human-readable feature names for the dashboard ----
FRIENDLY = {
    "poll_avg": "Polling average (%)",
    "poll_last": "Most recent poll (%)",
    "poll_last30": "Polling avg, final 30 days (%)",
    "poll_std": "Poll-to-poll variability",
    "n_polls": "# of polls (candidate)",
    "n_polls_over50": "# of polls above 50%",
    "frac_polls_over50": "Share of polls above 50%",
    "race_total_polls": "# of polls (race)",
    "avg_sample": "Avg poll sample size",
    "min_days": "Days since last poll",
    "poll_lead": "Polling lead vs best opponent (pts)",
    "poll_share": "Share of race polling",
    "n_cands": "# of candidates",
    "is_dem": "Democrat",
    "is_rep": "Republican",
    "is_senate": "Senate race",
    "is_gov": "Governor race",
    "prior_margin_cand": "Party's margin here last election (pts)",
    "is_incumbent": "Incumbent",
    "is_inc_party_race": "Seat has a party incumbent",
    "twoparty_margin_cand": "Two-party polled margin (pts)",
    "abs_gap": "Polled gap size (pts)",
    "tossup": "Toss-up race (<3pt gap)",
    "undecided": "Undecided share (%)",
    "gap_x_recency": "Polling lead x recency",
    "natl_env_cand": "National environment (generic ballot, pts)",
    "bias_prior_cand": "Historical state poll bias (pts)",
    "poll_momentum": "Polling momentum (final-60d slope)",
    "poll_adj": "House-effect-adjusted poll avg (%)",
    "n_lead_changes": "# of lead changes over campaign",
    "lead_changed": "Lead ever changed",
    "avg_margin_over_time": "Avg polled margin over campaign (pts)",
    "margin_volatility": "Polled-margin volatility",
    "min_margin": "Worst polled margin (pts)",
    "margin_trend": "Polled-margin trend",
    "is_president_party": "President's party",
    "fund_receipts_ln": "Fundraising total (log $)",
    "fund_share": "Share of race fundraising",
    "fund_indiv_pct": "Individual-donor share of funds",
    "fund_pac_pct": "PAC share of funds",
    "fund_party_pct": "Party-committee share of funds",
    "fund_self_pct": "Self-funded share",
    "fund_smalldollar_pct": "Small-dollar donor share",
}
_METRIC = {"unemployment": "Unemployment", "inflation": "Inflation", "cpi_core": "Core CPI",
           "gas": "Gas price", "fed_funds": "Fed funds rate", "unemp_u6": "U-6 underemployment",
           "approval": "Presidential approval", "sentiment": "Consumer sentiment",
           "generic_ballot": "Generic ballot"}
_SUFFIX = {"eve": "latest", "mean": "cycle avg", "max": "cycle max", "min": "cycle min",
           "std": "volatility", "trend": "trend", "last12_delta": "12-mo change",
           "avg_3mo": "3-mo avg", "max_3mo": "3-mo max", "trend_3mo": "3-mo trend",
           "avg_6mo": "6-mo avg", "max_6mo": "6-mo max", "trend_6mo": "6-mo trend",
           "avg_12mo": "12-mo avg", "max_12mo": "12-mo max", "trend_12mo": "12-mo trend"}

def friendly(f):
    if f in FRIENDLY:
        return FRIENDLY[f]
    for m, mlabel in sorted(_METRIC.items(), key=lambda kv: -len(kv[0])):
        if f.startswith(m + "_"):
            suf = f[len(m) + 1:]
            return f"{mlabel} ({_SUFFIX.get(suf, suf)})"
    return f

def top_shap(feats, values, shap_row, base, pred, k=10):
    """Top-k features by |SHAP| -> [{f, label, val, shap}] + base/pred."""
    order = np.argsort(-np.abs(shap_row))[:k]
    out = []
    for i in order:
        v = values[i]
        out.append(dict(
            f=feats[i], label=friendly(feats[i]),
            val=(None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)),
            shap=round(float(shap_row[i]), 4)))
    return dict(base=round(float(base), 4), pred=round(float(pred), 4), feats=out)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--natl-env", type=float, default=None)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    args = ap.parse_args()

    metas, models = {}, {}
    for name, mf, cls in [("win", "model", xgb.XGBClassifier),
                          ("margin", "margin_model", xgb.XGBRegressor)]:
        with open(os.path.join(HERE, "data", f"{mf}_features.json")) as f:
            metas[name] = json.load(f)
        m = cls()
        m.load_model(os.path.join(HERE, "data", f"{mf}_xgb.json"))
        models[name] = m

    # same inputs as predict.py
    d = load_agg_polls(args.polls, args.cycle)
    hist = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
    hist = F.prepare_polls(hist[hist["has_result"] == 1])
    hist["race_id"] = (hist["year"].astype(str) + "_" + hist["state"] + "_" + hist["office"]
                       + hist["district"].radd("-").where(hist["district"] != "", ""))
    house = F.compute_house_effect(hist, sorted(hist["year"].unique()))
    bias = F.compute_bias_priors(hist)
    macro = build_macro(cycles=[args.cycle])
    ne = dict(natl_env_hist())
    if args.natl_env is not None:
        ne[args.cycle] = args.natl_env
    else:
        from fetch_generic_ballot import get_natl_env
        v = get_natl_env(args.cycle)
        if v is not None:
            ne[args.cycle] = v
    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    cand = patch_redistricted_priors(
        F.build_candidate_table(d, macro, ne, funds, house=house, fec=fec, bias_priors=bias))

    Xw = cand.reindex(columns=metas["win"]["features"])
    Xm = cand.reindex(columns=metas["margin"]["features"])
    cand["win_prob"] = models["win"].predict_proba(Xw)[:, 1]
    cand["pred_margin"] = models["margin"].predict(Xm)

    sv_w = shap.TreeExplainer(models["win"])(Xw)       # log-odds space
    sv_m = shap.TreeExplainer(models["margin"])(Xm)    # margin points

    out = {}
    for rid, g in cand.groupby("race_id"):
        dems = g[g["party"] == "DEM"]
        row = (dems.loc[dems["win_prob"].idxmax()] if len(dems)
               else g.loc[g["win_prob"].idxmax()])
        i = cand.index.get_loc(row.name)
        win = top_shap(metas["win"]["features"], Xw.iloc[i].values,
                       sv_w.values[i], sigmoid(sv_w.base_values[i]),
                       row["win_prob"])
        margin = top_shap(metas["margin"]["features"], Xm.iloc[i].values,
                          sv_m.values[i], sv_m.base_values[i],
                          row["pred_margin"])
        out[rid] = dict(candidate=row["candidate"], party=row["party"],
                        win=win, margin=margin)

    payload = dict(cycle=args.cycle,
                   note="SHAP top-10 per model, explaining the leading Democrat "
                        "(or overall leader if no Democrat). Win model in log-odds "
                        "(base/pred shown as probabilities); margin model in points.",
                   generated_at=pd.Timestamp.now().isoformat(), races=out)
    p1 = os.path.join(HERE, f"model_explanations_{args.cycle}.json")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    p2 = os.path.join(AGG, "data", "processed", f"model_explanations_{args.cycle}.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"explanations: {len(out)} races -> {os.path.basename(p1)} (+ polling-agg copy)")

if __name__ == "__main__":
    main()
