# -*- coding: utf-8 -*-
"""Train + evaluate the PRIMARY nominee model; save production artifacts.

    py -X utf8 primary_model.py

Scheme (documented in METHODOLOGY.md):
- Data: data/primary_polls_long.csv (build_primary_dataset.py) - contested, labeled,
  regular partisan primaries; DEM/REP; jungle/top-two states excluded.
- Tuning: LOCO over the OLD cycles (<= 2020) only - small grid, MAE-of-Brier scored.
- Honest eval: EXPANDING-WINDOW on cycles >= 2022 (train strictly before the test cycle),
  matching the general model's primary scheme. Report vs the poll-leader baseline.
- Macro ablation: the model is trained WITHOUT macro by default (a few hundred races can't
  support 144 macro columns); this script measures the with-macro variant each run and
  prints the comparison so the choice stays evidence-based.
- Artifacts: data/primary_model_xgb.json + data/primary_model_features.json (trained on
  ALL labeled cycles, no-macro feature set).

This is a SCRIPT, not a notebook: full run is minutes (unlike the 30-min general
notebooks), so the printed record lands in the terminal/commit message instead.
"""
import itertools
import json
import os
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)

import features as F
import features_primary as FP
from macro_features import build_macro_asof

HERE = os.path.dirname(os.path.abspath(__file__))

def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float((tpr - fpr).max())

def race_acc(te, col):
    pick = te.loc[te.groupby("race_id")[col].idxmax()]
    return pick["won"].mean()

def eval_fold(tr, te, feats, params):
    m = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **params)
    m.fit(tr[feats], tr["won"].astype(int))
    return m.predict_proba(te[feats])[:, 1]

def loco_brier(c, feats, params, cycles):
    bs = []
    for ty in cycles:
        tr, te = c[c["year"] != ty], c[c["year"] == ty]
        if not len(te) or tr["won"].nunique() < 2:
            continue
        p = eval_fold(tr, te, feats, params)
        bs.append(brier_score_loss(te["won"].astype(int), p))
    return float(np.mean(bs)) if bs else np.inf

def expanding_eval(c, feats, params, eval_years, label):
    rows = []
    for ty in eval_years:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(te) or not len(tr):
            continue
        te["p"] = eval_fold(tr, te, feats, params)
        y = te["won"].astype(int)
        rows.append(dict(
            cycle=ty, n_races=te["race_id"].nunique(), n_cand=len(te),
            AUC=roc_auc_score(y, te["p"]) if y.nunique() > 1 else np.nan,
            AUC_PR=average_precision_score(y, te["p"]),
            KS=ks(y, te["p"]),
            Brier=brier_score_loss(y, te["p"]),
            race_acc=race_acc(te, "p"),
            pollleader_acc=race_acc(te, "poll_avg"),
        ))
    ev = pd.DataFrame(rows).set_index("cycle")
    print(f"\n=== EXPANDING-WINDOW eval ({label}) ===")
    print(ev.round(3).to_string())
    print("MEAN:", ev[["AUC", "AUC_PR", "KS", "Brier", "race_acc", "pollleader_acc"]]
          .mean().round(3).to_dict())
    return ev

def main():
    d = pd.read_csv(os.path.join(HERE, "data", "primary_polls_long.csv"), low_memory=False)
    d = F.prepare_polls(d)
    d["district"] = d["district"].map(F.dist_str)
    print(f"rows {len(d)} | races {d['race_id'].nunique()} | cycles "
          f"{sorted(d['year'].unique())}")

    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    c = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"])
    assert c["won"].notna().all(), "training rows must all have labels"
    FEATS = FP.feature_list_primary()
    print(f"candidate rows {len(c)} | features {len(FEATS)} | "
          f"base nominee rate {c['won'].mean():.3f}")

    years = sorted(c["year"].unique())
    tune_years = [y for y in years if y <= 2020]
    eval_years = [y for y in years if y >= 2022]
    print("tune cycles:", tune_years, "| eval cycles (expanding):", eval_years)

    # ---- small nested grid search (LOCO over the old cycles, Brier-scored) ----
    GRID = dict(max_depth=[1, 2], learning_rate=[0.03, 0.05, 0.1],
                n_estimators=[100, 200], min_child_weight=[4, 8, 15],
                subsample=[0.7, 1.0], colsample_bytree=[0.5, 1.0],
                reg_lambda=[1, 5, 20])
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    random.seed(0); random.shuffle(combos); combos = combos[:48]
    best = None
    for combo in combos:
        p = dict(zip(keys, combo))
        b = loco_brier(c, FEATS, p, tune_years)
        if best is None or b < best[0]:
            best = (b, p)
    PARAMS = best[1]
    print(f"\nbest tune-cycle LOCO Brier: {best[0]:.4f}")
    print("PARAMS =", PARAMS)

    ev = expanding_eval(c, FEATS, PARAMS, eval_years,
                        "no fund, no macro - PRIMARY headline")

    # ---- fund ablation (leakage evidence: cycle-END FEC totals include post-primary
    # money, so fund_share partly encodes the training label; see feature_list_primary) --
    evf = expanding_eval(c, FP.feature_list_primary(fund=True), PARAMS, eval_years,
                         "WITH fund - ablation only (leak-suspect)")
    print(f"fund ablation: race-acc {evf['race_acc'].mean():.3f} vs {ev['race_acc'].mean():.3f} "
          f"(identical picks expected), Brier gain {ev['Brier'].mean()-evf['Brier'].mean():+.4f} "
          f"= the leak-suspect juice; artifact stays NO-fund")

    # ---- macro ablation (evidence for keeping macro out) ----
    cm = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                                macro_asof=build_macro_asof)
    macro_feats = sorted(set(cm.columns) - set(c.columns))
    evm = expanding_eval(cm, FP.feature_list_primary(macro_feats), PARAMS, eval_years,
                         f"WITH macro ({len(macro_feats)} extra cols) - ablation only")
    dif = evm["Brier"].mean() - ev["Brier"].mean()
    print(f"\nmacro ablation: Brier {'WORSE' if dif > 0 else 'better'} by {abs(dif):.4f} "
          f"with macro (noise-level on ~{c['race_id'].nunique()} races) -> artifact stays NO-macro")

    # ---- production artifact: train on ALL cycles, no-macro feature set ----
    prod = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **PARAMS)
    prod.fit(c[FEATS], c["won"].astype(int))
    prod.save_model(os.path.join(HERE, "data", "primary_model_xgb.json"))
    with open(os.path.join(HERE, "data", "primary_model_features.json"), "w") as f:
        json.dump(dict(features=FEATS, xgb_params=PARAMS,
                       trained_on_cycles=[int(y) for y in years],
                       n_races=int(c["race_id"].nunique()),
                       target="won = became the party's general-election nominee",
                       eval_expanding_window={str(k): {m: (round(float(v), 4) if v == v else None)
                                                       for m, v in r.items()}
                                              for k, r in ev.iterrows()}), f, indent=1)
    print("\nsaved data/primary_model_xgb.json + data/primary_model_features.json")

if __name__ == "__main__":
    main()
