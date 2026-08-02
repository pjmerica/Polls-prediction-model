# -*- coding: utf-8 -*-
"""#6: is the FUNDAMENTALS model better than the POLL model on thin-poll races?

    py -X utf8 analysis/fundamentals_vs_polls_thin.py

This is the question that decides whether the no-polling model is useful or merely
interesting. analysis/poll_volume_breakpoint.ipynb established that the PRIMARY poll model is
~9-12 points overconfident below 3 distinct surveys. A fundamentals model cannot be fooled by
one bad survey - but it is also much weaker overall (race_acc ~.42 vs ~.90). Those two facts
do not settle anything on their own: a model that is always wrong is not "well calibrated",
it is just confidently useless.

So we compare them HEAD TO HEAD on the same held-out races, split by survey count, on two
axes that answer different questions:
  * race_acc   - does it pick the right winner?           (is it USEFUL?)
  * calib gap  - predicted minus actual for its own pick  (does it KNOW when it is unsure?)

A blend is only justified if the fundamentals model wins, or draws, on thin races.
Both models are scored expanding-window (train strictly before the test cycle), never
in-sample, using each one's own shipped artifact parameters.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import ROOT  # noqa: E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

import features as F  # noqa: E402
import features_primary as FP  # noqa: E402
import primary_model as PM  # noqa: E402


def _survey_counts(long_polls):
    return long_polls.groupby("race_id").apply(
        lambda g: g.groupby(["pollster", "end_date"]).ngroups, include_groups=False)


def primary_head_to_head():
    from candidate_history import CandidateHistory
    d = pd.read_csv(_os.path.join(ROOT, "data", "primary_polls_long.csv"), low_memory=False)
    d = F.prepare_polls(d)
    d["district"] = d["district"].map(F.dist_str)
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    c = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                               hist=CandidateHistory(), bios=FP.load_candidate_bios())
    c = c[~c["candidate"].map(F.is_junk_answer)].copy()
    c["n_surveys"] = c["race_id"].map(_survey_counts(d))

    poll = json.load(open(_os.path.join(ROOT, "data", "primary_model_features.json")))
    fund = json.load(open(_os.path.join(ROOT, "data",
                                        "fundamentals_model_primary_features.json")))
    out = []
    for ty in (2022, 2024):
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(tr) or not len(te):
            continue
        te["p_poll"] = PM.eval_fold(tr, te, poll["features"], poll["xgb_params"],
                                    temp=poll["softmax_temp"])
        te["p_fund"] = PM.eval_fold(tr, te, fund["features"], fund["xgb_params"],
                                    temp=fund["softmax_temp"])
        out.append(te)
    return pd.concat(out, ignore_index=True)


def general_head_to_head():
    from cycles import CYCLES, natl_env
    from macro_features import build_macro
    d = pd.read_csv(_os.path.join(ROOT, "polls_long_with_results.csv"), low_memory=False)
    d = d[d["has_result"] == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    c = F.build_candidate_table(d, build_macro(), natl_env(), funds,
                                house_train_years=CYCLES, fec=fec,
                                bias_priors=F.compute_bias_priors(d),
                                candidate_bios=F.load_candidate_bios())
    c["n_surveys"] = c["race_id"].map(_survey_counts(d))

    poll = json.load(open(_os.path.join(ROOT, "data", "model_features.json")))
    fund = json.load(open(_os.path.join(ROOT, "data",
                                        "fundamentals_model_general_features.json")))
    pp = {k: v for k, v in poll["xgb_params"].items() if k not in ("random_state", "n_jobs")}
    fp = {k: v for k, v in fund["xgb_params"].items() if k not in ("random_state", "n_jobs")}
    out = []
    for ty in (2018, 2020, 2022, 2024):
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(tr) or not len(te):
            continue
        for tag, feats, prm in (("poll", poll["features"], pp), ("fund", fund["features"], fp)):
            m = xgb.XGBClassifier(**prm, random_state=42, n_jobs=-1)
            m.fit(tr[feats], tr["won"].astype(int))
            te[f"p_{tag}"] = m.predict_proba(te[feats])[:, 1]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def compare(te, label):
    """Per survey-count bucket: which model picks better, and which knows when it is unsure."""
    rows = []
    for lo, hi, name in ((1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, 10**6, "11+")):
        sub = te[(te["n_surveys"] >= lo) & (te["n_surveys"] <= hi)]
        if not sub["race_id"].nunique():
            continue
        r = dict(bucket=name, races=sub["race_id"].nunique())
        for tag in ("poll", "fund"):
            lead = sub.loc[sub.groupby("race_id")[f"p_{tag}"].idxmax()]
            r[f"{tag}_acc"] = lead["won"].mean()
            r[f"{tag}_gap"] = lead[f"p_{tag}"].mean() - lead["won"].mean()
            r[f"{tag}_brier"] = brier_score_loss(sub["won"].astype(int), sub[f"p_{tag}"])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("bucket")
    print(f"\n=== {label}: poll model vs fundamentals model, by distinct surveys ===")
    print("  acc = called-winner accuracy | gap = predicted - actual (+ = overconfident)")
    print(t.round(3).to_string())
    thin = t.loc[[b for b in t.index if b in ("1", "2-3")]]
    if len(thin):
        better = (thin["fund_acc"] > thin["poll_acc"]).any()
        print(f"\n  On thin races the fundamentals model "
              f"{'BEATS' if better else 'does NOT beat'} the poll model on accuracy.")
    return t


def main():
    gt = general_head_to_head()
    compare(gt, "GENERAL")
    pt = primary_head_to_head()
    compare(pt, "PRIMARY")
    print("\nVerdict logic: a blend is only worth building if the fundamentals model wins or "
          "draws on the thin buckets. Losing on BOTH axes there means the poll model, even "
          "overconfident, is still the better estimate and the fix belongs in its calibration "
          "(see CONCERNS.md #26), not in a second model.")


if __name__ == "__main__":
    main()
