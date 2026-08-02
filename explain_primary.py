# -*- coding: utf-8 -*-
"""Per-race SHAP explanations for the PRIMARY nominee model -> the dashboard's
Primary vs Markets Explain modal.

Explains the model's TOP candidate (predicted nominee) in every 2026 primary race, top-10
features by |SHAP|. Mirrors explain_2026.py (win model part) and reuses its friendly-name
machinery; adds labels for primary-only features.

    py -X utf8 explain_primary.py
Writes primary_explanations_2026.json + the polling-agg data/processed copy.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
import features_primary as FP
from explain_2026 import FRIENDLY, DESC, sigmoid, top_shap
from predict import DEFAULT_POLLS
from predict_primary import load_primary_feed

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets")

FRIENDLY.update({
    "is_dem_primary": "Democratic primary",
    "is_defending_party": "Party defends this seat",
    "is_pres_party": "President's party primary",
    "undecided": "Undecided share",
    "n_cands": "Candidates in field",
    "race_total_polls": "Poll-candidate rows in field",
})
FRIENDLY.update({
    "hist_prior_runs": "Prior general-election runs",
    "hist_prior_wins": "Prior general-election wins",
    "hist_ever_won": "Has won a general before",
    "hist_best_general_pct": "Best past general result (%)",
    "hist_last_general_pct": "Last general result (%)",
    "hist_years_since_last_run": "Years since last run",
    "hist_prior_primary_wins": "Prior primary wins",
    # bio_office_level intentionally NOT overridden here (2026-08-02): it now lives in the
    # shared FRIENDLY/DESC in explain_2026, since BOTH models use the feature. Keeping a
    # local copy is how the two drifted - the general modal had no description at all.
    "bio_in_office": "Currently holds office",
    "bio_prior_candidacy": "Past candidacies (bio)",
})
DESC.update({
    "hist_prior_runs": "How many Senate/Governor/House general elections this candidate "
                       "ran in before this cycle (results archives, 1998+).",
    "hist_prior_wins": "How many of those prior generals they WON.",
    "hist_ever_won": "1 if they have ever won a tracked general election.",
    "hist_best_general_pct": "Their best vote share in any prior general.",
    "hist_last_general_pct": "Vote share in their most recent prior general.",
    "hist_years_since_last_run": "Years since their last general-election run.",
    "hist_prior_primary_wins": "Prior primary victories (2018+ scraped results).",
    # bio_office_level description lives in explain_2026.DESC - see the note above.
    "bio_in_office": "Bio says they currently hold the office ('present').",
    "bio_prior_candidacy": "Bio mentions past candidacies our archives never tracked.",
})
DESC.update({
    "is_dem_primary": "1 = Democratic primary, 0 = Republican. Lets the model learn "
                      "party-specific primary dynamics.",
    "is_defending_party": "This party currently holds the seat (defending side). "
                          "Unknown = missing.",
    "is_pres_party": "The primary belongs to the sitting president's party - "
                     "establishment/anti-establishment dynamics differ.",
    "undecided": "100 minus the field's combined polling average - big undecided shares "
                 "mean late movement is possible.",
    "n_cands": "Number of polled candidates in this party's field.",
    "race_total_polls": "Sum of per-candidate poll rows across the field (one survey of N "
                        "candidates = N rows, so this exceeds the survey count shown in "
                        "the table). A proxy for how well-measured the race is.",
})

# SURVEYED-POPULATION SPLITS (labelled 2026-08-02). These 18 features are the same six poll
# aggregates recomputed per surveyed-population class, so they were generated rather than
# hand-written - previously ALL of them rendered in the Explain modal as a raw column name
# with an empty tooltip (poll_last30_lv alone appeared 83 times). A race rarely has all three
# classes, so the absent ones are NaN and the modal simply omits the value.
_POP_LABEL = {"lv": "likely voters", "rv": "registered voters", "a": "all adults"}
_POP_BASE = {
    "poll_avg":    ("Polling average", "Average of the candidate's polls"),
    "poll_last":   ("Most recent poll", "The candidate's share in their most recent poll"),
    "poll_last30": ("Polling avg, final 30 days",
                    "Average over polls taken in the final 30 days"),
    "poll_std":    ("Poll-to-poll variability",
                    "How much the candidate's numbers bounce between surveys"),
    "n_polls":     ("# of polls", "How many polls include this candidate"),
    "poll_lead":   ("Polling lead vs best opponent",
                    "Candidate's average minus the best other candidate's"),
}
for _base, (_lab, _desc) in _POP_BASE.items():
    for _tag, _pop in _POP_LABEL.items():
        _unit = "" if _base.startswith("n_") else (" (pts)" if "lead" in _base else " (%)")
        FRIENDLY[f"{_base}_{_tag}"] = f"{_lab} - {_pop}{_unit}"
        DESC[f"{_base}_{_tag}"] = (f"{_desc}, counting ONLY polls of {_pop}. "
                                   f"Blank when no poll of this race sampled {_pop}.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    args = ap.parse_args()

    import shap
    with open(os.path.join(HERE, "data", "primary_model_features.json")) as f:
        meta = json.load(f)
    # RANKER (2026-07-29): the primary model is now a learning-to-rank model. SHAP explains the
    # raw rank SCORE (its bars' direction + relative size are what matter - unchanged framing).
    # The headline "pred" shown is the WITHIN-RACE SOFTMAX probability - the SAME number the
    # dashboard's win_prob_norm shows, so the Explain modal and the table now AGREE (they used
    # to disagree: the old classifier's raw independent prob, e.g. 95%, vs the table's
    # divide-by-sum normalized prob, e.g. 51%).
    model = xgb.XGBRanker()
    model.load_model(os.path.join(HERE, "data", "primary_model_xgb.json"))
    temp = float(meta.get("softmax_temp", 1.0))

    d = load_primary_feed(args.polls, args.cycle)
    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    from candidate_history import CandidateHistory
    cand = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                                  hist=CandidateHistory(), bios=FP.load_candidate_bios())
    X = cand.reindex(columns=meta["features"])
    cand["score"] = model.predict(X)
    # within-race softmax = the dashboard's win_prob_norm (identical formula to predict_primary)
    def _softmax(s):
        v = np.asarray(s, dtype=float) / temp
        e = np.exp(v - np.nanmax(v))
        return e / e.sum()
    cand["p"] = cand.groupby("race_id", group_keys=False)["score"].transform(_softmax)
    # field confidence, ranker-consistent: leader's softmax prob above a uniform 1/n split
    field_conf = cand.groupby("race_id")["p"].transform(lambda s: s.max() - 1.0 / len(s))

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)

    out = {}
    for rid, g in cand.groupby("race_id"):
        i = g["p"].idxmax()          # explain the predicted nominee (top softmax prob)
        row = cand.loc[i]
        # SHAP bars are in score space; base = mean score -> shown as the field-average share
        # (softmax of a flat field = 1/n) so base->pred reads as "generic candidate -> this
        # candidate's within-race probability". pred is the dashboard's win_prob_norm.
        n = len(g)
        blk = top_shap(meta["features"], X.loc[i].values, sv.values[cand.index.get_loc(i)],
                       1.0 / n, float(row["p"]))
        out[rid] = dict(candidate=row["candidate"], party=row["party"], win=blk,
                        field_confidence=round(float(field_conf.loc[i]), 4))

    payload = dict(cycle=args.cycle,
                   note="SHAP top-10 for the PRIMARY nominee model, explaining the "
                        "predicted nominee. Log-odds bars; base/pred as probabilities.",
                   generated_at=pd.Timestamp.now().isoformat(), races=out)
    p1 = os.path.join(HERE, f"primary_explanations_{args.cycle}.json")
    p2 = os.path.join(AGG, "data", "processed", f"model_primary_explanations_{args.cycle}.json")
    for p in (p1, p2):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    print(f"primary explanations: {len(out)} races -> {os.path.basename(p1)} (+ polling-agg copy)")

if __name__ == "__main__":
    main()
