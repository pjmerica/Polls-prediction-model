# -*- coding: utf-8 -*-
"""Ablation: does the per-QUESTION `undecided_q` beat the existing `undecided`?

Added 2026-08-14. Background in CONCERNS (#50 pooling, and the undecided write-up):

    undecided    = 100 - sum(poll_avg over candidates), clipped at 0     [current]
    undecided_q  = mean over questions of (100 - that question's sum)    [candidate]

The current one averages each candidate over all polls FIRST and subtracts once, which is
only valid when every candidate appears in every poll. Where a pollster tests several
separate head-to-heads, the per-candidate averages sum far past 100 (ME-Sen 2026: 315.7) and
the clip drops the race onto exactly 0.0 - indistinguishable from "no undecideds". That is
29% of 2026 races vs 0.6% of training races.

Measured before this ablation:
    correlation(old, new)   training 0.948   |   2026 serve 0.399
i.e. they agree where the model LEARNED and disagree where it PREDICTS. That is the
train/serve skew this ablation exists to price.

Run:  py -X utf8 analysis/undecided_ablation.py
Honest by construction: expanding window (train strictly on cycles before the test cycle),
fixed hyperparameters from the shipped artifact, only the feature set differs between arms.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: F401,E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

import features as F  # noqa: E402
from cycles import CYCLES, EVAL_CYCLES, natl_env  # noqa: E402
from macro_features import build_macro  # noqa: E402


def build():
    d = pd.read_csv(paths.root("polls_long_with_results.csv"), low_memory=False)
    d = d[d["has_result"] == 1].copy()
    d = F.prepare_polls(d)
    macro = build_macro()
    MACRO = sorted({k for cyc in macro for k in macro[cyc]})
    FUNDS = F.load_fundamentals()
    FEC = F.load_fec(extended=True)
    BIAS = F.compute_bias_priors(d)
    BIOS = F.load_candidate_bios()
    PRI = F.load_primary_results()
    base = F.build_candidate_table(d, macro, natl_env(), FUNDS, house_train_years=CYCLES,
                                   fec=FEC, bias_priors=BIAS, candidate_bios=BIOS,
                                   primary_results=PRI)
    feats = F.feature_list(MACRO, fund=True, candidate_bios=True, primary_results=True)
    base = add_margin_target(base)
    return base, feats


def add_margin_target(c):
    """Same derivation the margin notebook uses: own vote_pct minus the BEST OTHER
    candidate's (the runner-up if you led, the leader otherwise)."""
    c = c.copy()
    mx = c.groupby("race_id")["vote_pct"].transform("max")
    m2 = c.groupby("race_id")["vote_pct"].transform(
        lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else np.nan)
    c["margin_actual"] = c["vote_pct"] - np.where(c["vote_pct"] >= mx, m2, mx)
    return c


def evaluate(base, feats, params, target, label):
    """Expanding window: train on cycles strictly before the test cycle."""
    rows = []
    for ty in EVAL_CYCLES:
        tr = base[base["year"] < ty]
        te = base[base["year"] == ty]
        if not len(te) or not len(tr):
            continue
        if target == "won":
            m = xgb.XGBClassifier(**params)
            m.fit(tr[feats], tr["won"].astype(int))
            p = m.predict_proba(te[feats])[:, 1]
            pick = te.assign(p=p).loc[lambda x: x.groupby("race_id")["p"].idxmax()]
            rows.append(dict(cycle=ty, n=len(te),
                             brier=float(np.mean((p - te["won"].astype(int)) ** 2)),
                             race_acc=float(pick["won"].mean())))
        else:
            m = xgb.XGBRegressor(**params)
            tr2 = tr.dropna(subset=[target]); te2 = te.dropna(subset=[target])
            m.fit(tr2[feats], tr2[target])
            pr = m.predict(te2[feats])
            rows.append(dict(cycle=ty, n=len(te2),
                             mae=float(np.mean(np.abs(pr - te2[target]))),
                             r2=float(1 - np.sum((pr - te2[target]) ** 2)
                                      / np.sum((te2[target] - te2[target].mean()) ** 2))))
    t = pd.DataFrame(rows).set_index("cycle")
    print(f"\n=== {label} ===")
    print(t.round(4).to_string())
    print("MEAN:", {k: round(v, 4) for k, v in t.drop(columns="n").mean().items()})
    return t


def main():
    base, feats = build()
    print(f"rows {len(base)} | races {base['race_id'].nunique()} | features {len(feats)}")
    for col in ("undecided", "undecided_q"):
        s = base[col]
        print(f"  {col:14} mean {s.mean():6.2f}  zero {int((s == 0).sum()):4}  "
              f"NaN {int(s.isna().sum()):4}")

    win_params = json.load(open(paths.data("model_features.json")))["xgb_params"]
    mar_params = json.load(open(paths.data("margin_model_features.json")))["xgb_params"]

    keep = [f for f in feats if f != "undecided"]
    arms = {
        "A current (undecided)":      [f for f in feats],
        "B corrected (undecided_q)":  keep + ["undecided_q"],
        "C both":                     [f for f in feats] + ["undecided_q"],
        "D neither":                  keep,
    }
    for target, params, name in (("won", win_params, "WIN"),
                                 ("margin_actual", mar_params, "MARGIN")):
        if target != "won" and target not in base.columns:
            print(f"\n(skipping {name}: no {target} column)")
            continue
        print(f"\n{'='*70}\n{name} MODEL\n{'='*70}")
        for label, fl in arms.items():
            evaluate(base, fl, params, target, f"{name} / {label}")


if __name__ == "__main__":
    main()
