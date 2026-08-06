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

from paths import ROOT, AGG   # one definition for the whole repo (paths.py)

HERE = ROOT

# ---- human-readable feature names for the dashboard ----
FRIENDLY = {
    "poll_avg": "Polling average (%)",
    "poll_last": "Most recent poll (%)",
    "poll_last30": "Polling avg, final 30 days (%)",
    "poll_last7": "Polling avg, final 7 days (%)",
    "n_polls_last7": "# of polls in the final 7 days",
    "poll_lead_last7": "Polling lead, final 7 days (pts)",
    "poll_std": "Poll-to-poll variability",
    "n_polls": "# of polls (candidate)",
    "n_polls_over50": "# of polls above 50%",
    "frac_polls_over50": "Share of polls above 50%",
    "race_total_polls": "poll rows in race (field-summed)",
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
    "poll_momentum": "Polling momentum (slope of all polls)",
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
    # SHIPPED to the general model 2026-07-29 but never added here, so the Explain modal
    # rendered it with an empty tooltip (246 occurrences). It lives in the SHARED map rather
    # than explain_primary's local override because BOTH models use the feature now.
    "bio_office_level": "Office experience level",
}
# one-line plain-English explanations, shown on hover in the dashboard modal
DESC = {
    "poll_avg": "Average of all the candidate's general-election polls this cycle.",
    "poll_last": "The candidate's share in the most recent dated poll.",
    "poll_last30": "Average over polls taken in the final 30 days before the election.",
    "poll_last7": "Average over polls taken in the final week before the election - the "
                  "freshest read on the race. Blank when no poll landed in that window.",
    "n_polls_last7": "How many polls of this candidate landed in the final week.",
    "poll_lead_last7": "Final-week polling lead over the best opponent. Unlike the overall "
                       "polling lead, this ignores early-campaign polls entirely.",
    "poll_std": "How much the candidate's poll numbers bounce around between surveys.",
    "n_polls": "How many polls include this candidate.",
    "n_polls_over50": "How many polls put the candidate above 50%.",
    "frac_polls_over50": "Fraction of the candidate's polls that put them above 50%.",
    "race_total_polls": "Sum of per-candidate poll rows across the race (one survey of N "
                        "candidates counts as N rows, so this exceeds the survey count "
                        "shown in the table).",
    "avg_sample": "Average sample size of the candidate's polls (bigger = more precise).",
    "min_days": "Days between the candidate's latest poll and election day (staleness).",
    "poll_lead": "Candidate's polling average minus the best opponent's.",
    "poll_share": "Candidate's slice of all polling support in the race.",
    "n_cands": "Number of polled candidates in the race.",
    "is_dem": "1 if the candidate is a Democrat.",
    "is_rep": "1 if the candidate is a Republican.",
    "is_senate": "1 if this is a Senate race.",
    "is_gov": "1 if this is a Governor race.",
    "prior_margin_cand": "How much the candidate's party won/lost this seat by last time "
                         "(redrawn districts use 2x the new map's Cook PVI instead).",
    "is_incumbent": "1 if the candidate currently holds this seat.",
    "is_inc_party_race": "1 if either candidate's party currently holds the seat.",
    "twoparty_margin_cand": "Polled Dem-minus-Rep margin, signed toward this candidate.",
    "abs_gap": "Size of the polled gap between the two parties, ignoring direction.",
    "tossup": "1 if the polled gap is under 3 points.",
    "undecided": "Share of voters not yet committed to any polled candidate.",
    "gap_x_recency": "Polling lead discounted by how stale the latest poll is - "
                     "a fresh lead counts more than an old one.",
    "natl_env_cand": "National generic-ballot mood (last 30 days), signed toward the "
                     "candidate's party.",
    "bias_prior_cand": "How much polls in this state have historically over/under-stated "
                       "the candidate's party (prior cycles only).",
    # Changed 2026-08-03 from a final-60-day window to ALL dated polls (see
    # F.poll_momentum_slope). The old label survived the change and kept telling
    # users "final 60 days" for a feature that no longer works that way.
    "poll_momentum": "Least-squares slope of ALL the candidate's dated polls - "
                     "rising or falling.",
    # poll_adj dropped as a feature 2026-07-12 (see features.py); label kept harmlessly
    # in case an older artifact is loaded.
    "poll_adj": "Poll average after correcting each pollster's historical partisan lean.",
    "n_lead_changes": "How many times the race's front-runner changed during the campaign.",
    "lead_changed": "1 if the front-runner changed at least once.",
    "avg_margin_over_time": "The candidate's average polled lead/deficit across the "
                            "whole campaign.",
    "margin_volatility": "How unstable the candidate's polled margin has been.",
    "min_margin": "The candidate's worst polled margin at any point in the campaign.",
    "margin_trend": "Whether the candidate's polled margin is improving or worsening.",
    "is_president_party": "1 if the candidate's party holds the White House "
                          "(midterms usually punish it).",
    "fund_receipts_ln": "Total campaign money raised (log scale).",
    "fund_share": "Candidate's share of ALL money raised in the race - "
                  "donors are forecasters with skin in the game.",
    "fund_indiv_pct": "Share of the candidate's money from individual donors.",
    "fund_pac_pct": "Share from PACs/committees - PAC money tends to flow to "
                    "likely winners.",
    "fund_party_pct": "Share from party committees - parties triage toward "
                      "winnable races.",
    "fund_self_pct": "Share the candidate gave/loaned themselves.",
    "fund_smalldollar_pct": "Share of individual money from small (<$200) donors.",
    "bio_office_level": "Highest public office the candidate held BEFORE this election "
                        "(4 = federal, 3 = statewide, 2 = state legislature, 1 = local, "
                        "0 = none). A proxy for name recognition and donor networks. "
                        "As-of-year, so the same person reads a lower level in an earlier "
                        "cycle - a first-time candidate is 0 even if they later won office.",
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

def describe(f):
    if f in DESC:
        return DESC[f]
    for m, mlabel in sorted(_METRIC.items(), key=lambda kv: -len(kv[0])):
        if f.startswith(m + "_"):
            suf = f[len(m) + 1:]
            return (f"{mlabel}, {_SUFFIX.get(suf, suf)}, over the ~2 years before the "
                    f"election. National condition - same value for every race this cycle.")
    return ""

def top_shap(feats, values, shap_row, base, pred, k=10):
    """Top-k features by |SHAP| -> [{f, label, val, shap}] + base/pred."""
    order = np.argsort(-np.abs(shap_row))[:k]
    out = []
    for i in order:
        v = values[i]
        out.append(dict(
            f=feats[i], label=friendly(feats[i]), desc=describe(feats[i]),
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
        # display_party = real affiliation (e.g. Osborn shows IND though modeled in the
        # DEM two-party slot); fall back to model party if the column isn't present
        disp = row["display_party"] if "display_party" in cand.columns else row["party"]
        out[rid] = dict(candidate=row["candidate"], party=disp,
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
