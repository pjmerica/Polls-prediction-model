"""SHAP explanations for the 2026 predictions -> model_explanations_2026.json.

For every predicted race, explains WHY each model predicts what it does, via the top-10
features by |SHAP| — one block for the WIN model, one for the (separate) MARGIN model.
Powers the dashboard's per-race "Explain" modal (Model vs Markets tab).

Framing: we explain the LEADING DEMOCRAT's row (highest win_prob among DEM candidates),
because the tab's headline number is the race's Democratic win probability. If a race has
no Democrat, the overall leader is explained instead (party is recorded either way).

Race-level models (2026-10-27, audit item 7): both blocks explain the candidate AGAINST THEIR
STRONGEST RIVAL (`vs` in each block), because that is what both published numbers are.
- Win: the ranker's within-race softmax. Against one rival it is sigmoid((s_i - s_j) / T), so each
  bar is (SHAP_i - SHAP_j) / T in LOG-ODDS of the head-to-head; base = 0.5 (an even race, moved
  only by the minor-candidate offset in src/race_level.py) and pred = the head-to-head chance,
  which IS the published win_prob in a two-candidate race (a bigger field takes some of it: the
  page shows the published number beside it).
- Margin: the published margin is predicted share minus the best rival's predicted share, so each
  bar is SHAP_i - SHAP_j of the share model in PERCENTAGE POINTS; base = 0 and the bars add up
  exactly to the published margin.
Each feature also carries the rival's value (`vs_val`).

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
from race_level import load_ensemble, minor_flag, race_margins, race_win_probs
from predict import DEFAULT_POLLS, load_agg_polls, patch_redistricted_priors

import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
from paths import ROOT, AGG   # one definition for the whole repo (paths.py)
import paths as _paths   # module handle: `out` is a very common local
                         # variable name in this repo, so never import it bare

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
    "is_incumbent": "Candidate's party holds this seat",
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
    # 2026-10-01 / 10-02 feature batches
    "poll_avg_dpoll": "Avg in Dem-sponsored polls (%)",
    "poll_avg_rpoll": "Avg in Rep-sponsored polls (%)",
    "poll_avg_npoll": "Avg in nonpartisan polls (%)",
    "poll_lead_dpoll": "Lead in Dem-sponsored polls (pts)",
    "poll_lead_rpoll": "Lead in Rep-sponsored polls (pts)",
    "poll_lead_npoll": "Lead in nonpartisan polls (pts)",
    "n_polls_dpoll": "# of Dem-sponsored polls",
    "n_polls_rpoll": "# of Rep-sponsored polls",
    "is_personal_inc": "Candidate is the sitting officeholder",
    "open_seat": "Open seat",
    "pres_lean_cand": "State presidential lean (pts)",
    "fund_pac_rel": "PAC share vs party norm",
    "poll_avg_60": "Polling average, last 60 days",
    "poll_lead_60": "Polling lead, last 60 days (pts)",
    "poll_std_60": "Poll-to-poll variability, last 60 days",
    "n_polls_60": "# of polls, last 60 days",
    "race_total_polls_60": "Polls in the race, last 60 days",
    "pres_approval_cand": "Presidential approval (for the president's party)",
    "pres_unemp_cand": "Unemployment change (for the president's party)",
    "pres_inflation_cand": "Inflation (for the president's party)",
    "pres_sentiment_cand": "Consumer sentiment (for the president's party)",
    "dist_pres_lean_cand": "District presidential lean (pts)",
    "poll_avg_rw": "Recency-weighted polling avg (%)",
    "poll_lead_rw": "Recency-weighted polling lead (pts)",
    "pollster_sd": "Pollster disagreement (pts)",
    "n_pollsters": "# of pollsters in race",
    "midterm_pres_party": "President's party in a midterm",
    "midterm_approval_cand": "Midterm x presidential approval",
    "fund_cash_share": "Share of race cash on hand",
    "fund_spend_share": "Share of race spending",
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
    # NOT person-level incumbency. features.py computes this as
    # (incumbent_party == candidate's party), because races.csv carries incumbent_PARTY and
    # no incumbent NAME - there is no per-person incumbency anywhere in the inputs. So every
    # candidate of the holding party reads 1, including primary fields where 3-7 candidates
    # of that party are all flagged. The label and this text must not promise more than that
    # (fixed 2026-08-08; the old text said "the candidate currently holds this seat", which
    # shipped to 128 live race explanations).
    "is_incumbent": "1 if the candidate's PARTY currently holds this seat (party-level, "
                    "not personal incumbency).",
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
                        "(4 = elected federal, 3 = elected statewide, 2 = state legislature, "
                        "1 = local elected, 0.75 = appointed federal post, "
                        "0.5 = appointed state or local post, 0 = none). A proxy for name recognition and donor networks. "
                        "As-of-year, so the same person reads a lower level in an earlier "
                        "cycle - a first-time candidate is 0 even if they later won office.",
    "poll_avg_dpoll": "Average share in polls sponsored by Democrats or Democratic groups.",
    "poll_avg_rpoll": "Average share in polls sponsored by Republicans or Republican groups.",
    "poll_avg_npoll": "Average share in polls with no partisan sponsor.",
    "poll_lead_dpoll": "Lead over the best opponent, Democratic-sponsored polls only.",
    "poll_lead_rpoll": "Lead over the best opponent, Republican-sponsored polls only.",
    "poll_lead_npoll": "Lead over the best opponent, nonpartisan polls only.",
    "n_polls_dpoll": "How many Democratic-sponsored polls include the candidate.",
    "n_polls_rpoll": "How many Republican-sponsored polls include the candidate.",
    "is_personal_inc": "1 if this person (not just their party) holds the seat - won it in a "
                       "previous election or sits there by appointment.",
    "open_seat": "1 if no candidate in the race is the sitting officeholder.",
    "pres_lean_cand": "How much more the state voted for this candidate's party than the "
                      "nation did in the last presidential election (state-level, even for House).",
    "fund_pac_rel": "Share of the candidate's money from PACs, minus the median for their party "
                    "that cycle (Democrats largely stopped taking corporate PAC money after 2018).",
    "poll_avg_60": "The candidate's polling average over the last 60 days before the election only.",
    "poll_lead_60": "The candidate's lead or deficit over the last 60 days of polls.",
    "poll_std_60": "How much the candidate's polls moved over the last 60 days.",
    "n_polls_60": "How many polls of the candidate ended in the last 60 days.",
    "race_total_polls_60": "How many polls of this race ended in the last 60 days.",
    "pres_approval_cand": "The president's approval minus 50, counted for the president's party and "
                          "against the other major party.",
    "pres_unemp_cand": "The last year's change in unemployment, counted against the president's party "
                       "when it rises.",
    "pres_inflation_cand": "Inflation above 2%, counted against the president's party.",
    "pres_sentiment_cand": "Consumer sentiment against its long-run average, counted for the "
                           "president's party when high.",
    "dist_pres_lean_cand": "How much more the DISTRICT voted for this candidate's party than the nation "
                           "did in the last presidential election, on this year's district lines "
                           "(blank for Senate and Governor races, which use the state lean).",
    "poll_avg_rw": "Polling average where a poll's weight halves every 14 days, so recent "
                   "polls count most.",
    "poll_lead_rw": "Recency-weighted polling lead over the best opponent.",
    "pollster_sd": "How much different pollsters disagree about this candidate.",
    "n_pollsters": "How many different pollsters have surveyed the race.",
    "midterm_pres_party": "1 if the candidate is from the president's party in a midterm year.",
    "midterm_approval_cand": "Midterm penalty scaled by presidential approval: (approval - 50), "
                             "positive for the president's party and negative for the other "
                             "party, in midterm years only.",
    "fund_cash_share": "Candidate's cash on hand as a share of the race total (latest FEC report).",
    "fund_spend_share": "Candidate's spending to date as a share of the race total.",
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

def _num(v):
    return None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)


def top_shap(feats, values, shap_row, base, pred, k=10, vs=None, vs_values=None):
    """Top-k features by |SHAP| -> [{f, label, val, vs_val, shap}] + base/pred (+ vs)."""
    order = np.argsort(-np.abs(shap_row))[:k]
    out = []
    for i in order:
        out.append(dict(
            f=feats[i], label=friendly(feats[i]), desc=describe(feats[i]),
            val=_num(values[i]),
            vs_val=(_num(vs_values[i]) if vs_values is not None else None),
            shap=round(float(shap_row[i]), 4)))
    blk = dict(base=round(float(base), 4), pred=round(float(pred), 4), feats=out)
    if vs is not None:
        blk["vs"] = vs
    return blk

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
    for name, mf, kind in [("win", "race_win_model", "ranker"),
                           ("margin", "race_margin_model", "regressor")]:
        with open(os.path.join(HERE, "data", f"{mf}_features.json")) as f:
            metas[name] = json.load(f)
        # the averaged top settings (audit item 6): SHAP of the mean = mean of the members' SHAP
        models[name] = load_ensemble(metas[name], os.path.join(HERE, "data"), kind)

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
    # candidate_bios + primary_results MUST match predict.py's build (2026-10-08): without them
    # bio_office_level and the primary-strength features were NaN here, so every explainer
    # described a different input than the one actually predicted (119 races showed n/a
    # experience, and the explainer's raw score did not match the published win prob).
    cand = patch_redistricted_priors(
        F.build_candidate_table(d, macro, ne, funds, house=house, fec=fec, bias_priors=bias,
                                candidate_bios=F.load_candidate_bios(),
                                primary_results=F.load_primary_results()))

    Xw = cand.reindex(columns=metas["win"]["features"])
    Xm = cand.reindex(columns=metas["margin"]["features"])
    temp = float(metas["win"]["softmax_temp"])
    offset = float(metas["win"].get("minor_offset", 0.0))
    cand["score"] = models["win"].predict(Xw)
    cand["win_prob"] = race_win_probs(cand, "score", temp, offset)
    # the minor-candidate offset (race_level.py) is a constant, not a feature: it starts the
    # head-to-head base away from 0.5 when exactly one of the two is a minor candidate
    cand["_off"] = offset * minor_flag(cand)
    cand["share"] = models["margin"].predict(Xm).astype("float64")
    cand["pred_margin"] = race_margins(cand, "share")
    # consistency check against the published predictions (same model, same inputs)
    _pp = os.path.join(HERE, "outputs", "predictions_2026.csv")
    if os.path.exists(_pp):
        _p = pd.read_csv(_pp)[["race_id", "candidate", "win_prob"]]
        _j = cand[["race_id", "candidate", "win_prob"]].merge(_p, on=["race_id", "candidate"],
                                                             suffixes=("", "_pub"))
        # Louisiana House publishes the seat chance incl. the runoff (src/la_runoff.py); the
        # explanation describes the first-round ranking, so those races are left out of the check
        _j = _j[~_j["race_id"].str.contains("_LA_House")]
        if len(_j):
            _gap = float((_j["win_prob"] - _j["win_prob_pub"]).abs().max())
            print(f"explainer vs published win_prob: max gap {_gap:.4f} over {len(_j)} candidates")

    sv_w_vals, _ = models["win"].shap(Xw)       # ranker score space
    sv_m_vals, _ = models["margin"].shap(Xm)    # vote-share points

    out = {}
    for rid, g in cand.groupby("race_id"):
        dems = g[g["party"] == "DEM"]
        row = (dems.loc[dems["win_prob"].idxmax()] if len(dems)
               else g.loc[g["win_prob"].idxmax()])
        i = cand.index.get_loc(row.name)
        others = g.drop(index=row.name)
        win = margin = None
        if len(others):
            # strongest rival on each model's own scale: highest win chance / predicted share
            jw = cand.index.get_loc(others["win_prob"].idxmax())
            jm = cand.index.get_loc(others["share"].idxmax())
            d_off = cand["_off"].iloc[i] - cand["_off"].iloc[jw]
            win = top_shap(metas["win"]["features"], Xw.iloc[i].values,
                           (sv_w_vals[i] - sv_w_vals[jw]) / temp, sigmoid(d_off),
                           sigmoid((cand["score"].iloc[i] - cand["score"].iloc[jw]) / temp + d_off),
                           vs=cand["candidate"].iloc[jw], vs_values=Xw.iloc[jw].values)
            margin = top_shap(metas["margin"]["features"], Xm.iloc[i].values,
                              sv_m_vals[i] - sv_m_vals[jm], 0.0, row["pred_margin"],
                              vs=cand["candidate"].iloc[jm], vs_values=Xm.iloc[jm].values)
        # display_party = real affiliation (e.g. Osborn shows IND though modeled in the
        # DEM two-party slot); fall back to model party if the column isn't present
        disp = row["display_party"] if "display_party" in cand.columns else row["party"]
        out[rid] = dict(candidate=row["candidate"], party=disp,
                        win=win, margin=margin)

    payload = dict(cycle=args.cycle,
                   note="SHAP top-10 per model, explaining the leading Democrat "
                        "(or overall leader if no Democrat) against the strongest rival (vs). "
                        "Win: log-odds of the head-to-head (base 0.5, pred = head-to-head "
                        "chance); margin: points, adding up to the published margin.",
                   generated_at=pd.Timestamp.now().isoformat(), races=out)
    p1 = _paths.out(f"model_explanations_{args.cycle}.json")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    p2 = os.path.join(AGG, "data", "processed", f"model_explanations_{args.cycle}.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"explanations: {len(out)} races -> {os.path.basename(p1)} (+ polling-agg copy)")

if __name__ == "__main__":
    main()
