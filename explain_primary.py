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
    "race_total_polls": "Total polls in race",
})
FRIENDLY.update({
    "hist_prior_runs": "Prior general-election runs",
    "hist_prior_wins": "Prior general-election wins",
    "hist_ever_won": "Has won a general before",
    "hist_best_general_pct": "Best past general result (%)",
    "hist_last_general_pct": "Last general result (%)",
    "hist_years_since_last_run": "Years since last run",
    "hist_prior_primary_wins": "Prior primary wins",
    "bio_office_level": "Office experience level",
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
    "bio_office_level": "From their Wikipedia bio: 4=federal office, 3=statewide, "
                        "2=state legislature, 1=local office, 0=none detected.",
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
    "race_total_polls": "Total poll rows across the whole field - how well-measured the "
                        "race is.",
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    args = ap.parse_args()

    import shap
    with open(os.path.join(HERE, "data", "primary_model_features.json")) as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(HERE, "data", "primary_model_xgb.json"))

    d = load_primary_feed(args.polls, args.cycle)
    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    from candidate_history import CandidateHistory
    cand = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                                  hist=CandidateHistory(), bios=FP.load_candidate_bios())
    X = cand.reindex(columns=meta["features"])
    cand["p"] = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    base = float(np.ravel(sv.base_values)[0])

    out = {}
    for rid, g in cand.groupby("race_id"):
        i = g["p"].idxmax()          # explain the predicted nominee
        row = cand.loc[i]
        blk = top_shap(meta["features"], X.loc[i].values, sv.values[cand.index.get_loc(i)],
                       sigmoid(base), float(row["p"]))
        out[rid] = dict(candidate=row["candidate"], party=row["party"], win=blk)

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
