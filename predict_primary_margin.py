# -*- coding: utf-8 -*-
"""Predict 2026 PRIMARY victory MARGINS from the polling-agg raw feed.

    py -X utf8 predict_primary_margin.py [--cycle 2026] [--polls ...] [--out preds.csv]

The primary-side sibling of predict_margin.py, wired the same way: a completely separate
model from the nominee ranker (predict_primary.py loads the XGBRanker; this loads the
XGBRegressor saved by models/poll/primary_margin_model.py). The poll loading / dedup / feature
code is IMPORTED from predict_primary, so the two scripts can never diverge on inputs -
the same never-fork rule the general pair follows.

Output: primary_margin_predictions_2026.csv - predicted margin in percentage points vs the
best OTHER candidate in the same primary field (positive = wins the primary by that much),
one row per candidate, plus the implied winner per race.

Read the margins with the model's own error bar in mind: held-out MAE is ~17 points (see
data/primary_margin_model_features.json). Primary margins are roughly twice as hard as general
margins - the target's std is 40.8 points because primaries are blowouts far more often than
two-party general elections. A predicted +25 is "probably a comfortable win", not a precise
forecast.
"""
import argparse
import datetime
import json
import os

import pandas as pd
import xgboost as xgb

import features as F
import features_primary as FP
from paths import ROOT, AGG   # noqa: F401  (one definition for the whole repo)
from predict import DEFAULT_POLLS
from predict_primary import load_primary_feed

HERE = ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "data", "primary_margin_model_features.json")) as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(HERE, "data", "primary_margin_model_xgb.json"))
    print(f"primary margin model: {len(meta['features'])} features, "
          f"{meta['n_races']} training races, cycles {meta['trained_on_cycles'][0]}-"
          f"{meta['trained_on_cycles'][-1]}")

    d = load_primary_feed(args.polls, args.cycle)
    print(f"primary races: {d['race_id'].nunique()} | poll rows {len(d)}")

    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    from candidate_history import CandidateHistory
    cand = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                                  hist=CandidateHistory(), bios=FP.load_candidate_bios())
    cand = cand[~cand["candidate"].map(F.is_junk_answer)].copy()

    missing = [f for f in meta["features"] if f not in cand.columns]
    assert not missing, f"artifact expects features absent from the table: {missing[:8]}"
    cand["pred_margin"] = model.predict(cand.reindex(columns=meta["features"]))

    # The regressor scores each candidate independently, so a field's predicted margins need
    # not be mutually consistent (two candidates can both read positive). The RANK is what is
    # meaningful; margin_pick flags the per-race argmax so the dashboard can show whether the
    # margin model agrees with the nominee ranker - exactly what models_agree does on the
    # general tab.
    cand["margin_pick"] = 0
    cand.loc[cand.groupby("race_id")["pred_margin"].idxmax(), "margin_pick"] = 1

    # UNCONTESTED-FIELD flag. A race with one polled candidate has poll_lead == 0.0 by
    # construction (there is no "best other"), so the model is extrapolating a margin from the
    # candidate's own level alone - Husted +98, Cotton +82. Those are plausible for genuinely
    # unopposed incumbents, but they are not margin PREDICTIONS in the sense the eval measured,
    # and the training target always had a real opponent. Flagged rather than dropped: the
    # dashboard can hide or mark them, and a field can also be a single candidate simply
    # because nobody polled the rest.
    cand["n_polled_cands"] = cand.groupby("race_id")["candidate"].transform("count")
    cand["uncontested_field"] = (cand["n_polled_cands"] < 2).astype(int)
    n_unc = int(cand.loc[cand["uncontested_field"] == 1, "race_id"].nunique())
    if n_unc:
        print(f"uncontested/one-polled-candidate fields: {n_unc} races flagged "
              f"(pred_margin there extrapolates from level alone)")

    surveys = d.groupby("race_id").apply(
        lambda g: g.groupby(["pollster", "end_date"]).ngroups, include_groups=False)
    cand["n_surveys"] = cand["race_id"].map(surveys).fillna(0).astype(int)

    out_cols = ["race_id", "state", "office", "district", "party", "candidate",
                "election_date", "n_polls", "n_surveys", "poll_avg", "poll_lead",
                "pred_margin", "margin_pick", "n_polled_cands", "uncontested_field"]
    out = cand[out_cols].sort_values(["race_id", "pred_margin"], ascending=[True, False])
    out_path = args.out or os.path.join(HERE, f"primary_margin_predictions_{args.cycle}.csv")
    out.to_csv(out_path, index=False)

    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(dict(generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                       polls_max_end_date=str(d["end_date"].max().date()),
                       n_rows=int(len(out)), n_races=int(out["race_id"].nunique()),
                       heldout_mae=meta["eval_expanding_window"]), f, indent=1)

    print(f"saved -> {out_path} ({out['race_id'].nunique()} races)")
    picks = out[(out["margin_pick"] == 1)
                & (out["uncontested_field"] == 0)].nlargest(10, "pred_margin")
    print("\nlargest predicted primary blowouts (contested fields only):")
    print(picks[["race_id", "candidate", "poll_lead", "pred_margin"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
