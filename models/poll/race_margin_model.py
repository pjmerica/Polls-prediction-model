# -*- coding: utf-8 -*-
"""RACE-LEVEL margin model (2026-10-27, audit item 7): predicts each candidate's VOTE SHARE, and
the published margin is derived inside the race - share minus the best rival's predicted share.

    py -X utf8 models/poll/race_margin_model.py

Why: margin_model.ipynb regresses every candidate's "own share minus best opponent" on its own,
so a race's margins never had to fit together - in the 2026-10-27 run 51 two-candidate races'
margins did not cancel, 5 races had two predicted winners and 1 had none. Deriving all margins
from one set of shares fixes that by construction: exactly one candidate is ahead (ties aside),
the top two's margins are equal and opposite, and third parties / same-party top-two fields /
independents need no special case.

Scheme (mirrors race_win_model.py):
- Same training rows as model.ipynb / margin_model.ipynb (race_win_model.training_table) and the
  margin model's feature list (data/margin_model_features.json).
- Target: vote_pct (share of the full results field). The published number, and the tuning
  score, is the DERIVED margin, scored against margin_actual = vote_pct - best_other_pct.
- Tuning: LOCO over the 1998-2016 TUNE cycles only, each fold training on the other tune cycles;
  stage 1 screens N_CONFIGS settings on one seed, stage 2 re-scores the TOP_K on 3 seeds.
- Held-out 2018-24, expanding window, reported as a sanity check (user 2026-10-26: method fixes
  are judged on correctness, not on a metric comparison with the per-candidate model).
- Artifacts: data/race_margin_model_xgb.json + data/race_margin_model_features.json (all cycles).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT  # noqa: E402

import contextlib
import io
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from cycles import TUNE_CYCLES, EVAL_CYCLES
from race_level import race_margins
from race_win_model import training_table

N_CONFIGS = 60
TOP_K = 8
TUNE_SEEDS = (42, 7, 123)
GRID = dict(max_depth=[2, 3, 4], learning_rate=[0.02, 0.03, 0.05],
            n_estimators=[200, 400, 800], min_child_weight=[8, 15, 30],
            subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 1.0],
            reg_lambda=[1, 5, 20], reg_alpha=[0, 1])


def fit_share(tr, feats, params, seed=42):
    m = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=-1, **params)
    m.fit(tr[feats], tr["vote_pct"])
    return m


def mae(te, col="_margin"):
    return float((te[col] - te["margin_actual"]).abs().mean())


def loco_tune(c, feats, params, seed):
    out = []
    tune = c[c["year"].isin(TUNE_CYCLES)]
    for ty in TUNE_CYCLES:
        tr, te = tune[tune["year"] != ty], tune[tune["year"] == ty].copy()
        if not len(te):
            continue
        te["_share"] = fit_share(tr, feats, params, seed).predict(te[feats])
        te["_margin"] = race_margins(te, "_share")
        out.append(mae(te))
    return float(np.mean(out))


def coherence(te, col="_margin"):
    """The checks tests/test_publish.py makes on the published file, on held-out predictions."""
    two = te.groupby("race_id").filter(lambda g: len(g) == 2)
    cancel = two.groupby("race_id")[col].sum().abs().max() if len(two) else 0.0
    ahead = (te[col] > 0).groupby(te["race_id"]).sum()
    return dict(max_two_cand_sum=round(float(cancel), 6),
                races_not_one_winner=int((ahead != 1).sum()))


def main():
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        c = training_table()
    c["margin_actual"] = c["vote_pct"] - c["best_other_pct"]
    meta_m = json.load(open(os.path.join(ROOT, "data", "margin_model_features.json")))
    FEATS = meta_m["features"]
    print(f"training rows {len(c)} | races {c['race_id'].nunique()} | features {len(FEATS)} "
          f"| built in {time.time() - t0:.0f}s", flush=True)

    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    random.seed(0); random.shuffle(combos)
    combos = [dict(zip(keys, cb)) for cb in combos[:N_CONFIGS]]
    t1 = time.time()
    stage1 = sorted(((loco_tune(c, FEATS, p, 42), p) for p in combos), key=lambda t: t[0])
    print(f"stage 1: {len(combos)} settings in {time.time() - t1:.0f}s; best tune MAE "
          f"{[round(s[0], 3) for s in stage1[:5]]}", flush=True)
    stage2 = sorted(((float(np.mean([loco_tune(c, FEATS, p, s) for s in TUNE_SEEDS])), p)
                     for _, p in stage1[:TOP_K]), key=lambda t: t[0])
    best, PARAMS = stage2[0]
    print(f"stage 2 ({len(TUNE_SEEDS)}-seed): {[round(s[0], 3) for s in stage2]}")
    print(f"PARAMS = {PARAMS}  (tune MAE of derived margin {best:.3f}; runner-up "
          f"{stage2[1][0]:.3f})", flush=True)

    # ---- held-out 2018-24, expanding window, on the published (derived) margin ----
    rows, preds = [], []
    for ty in EVAL_CYCLES:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        te["_share"] = fit_share(tr, FEATS, PARAMS).predict(te[FEATS])
        te["_margin"] = race_margins(te, "_share")
        lead = te.loc[te.groupby("race_id")["_margin"].idxmax()]
        rows.append(dict(cycle=ty, n_races=te["race_id"].nunique(), MAE_margin=mae(te),
                         MAE_share=float((te["_share"] - te["vote_pct"]).abs().mean()),
                         race_acc=float(lead["won"].mean()), **coherence(te)))
        preds.append(te)
    ev = pd.DataFrame(rows).set_index("cycle")
    print("\n=== held-out 2018-24, expanding window (sanity check) ===")
    print(ev.round(3).to_string())
    means = ev[["MAE_margin", "MAE_share", "race_acc"]].mean().round(4)
    print("mean:", means.to_dict())

    # ---- production artifact: all cycles ----
    m = fit_share(c, FEATS, PARAMS)
    m.save_model(os.path.join(ROOT, "data", "race_margin_model_xgb.json"))
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    imp.rename("gain_importance").rename_axis("feature").reset_index().to_csv(
        os.path.join(ROOT, "models", "poll", "race_margin_feature_importance.csv"), index=False)
    with open(os.path.join(ROOT, "data", "race_margin_model_features.json"), "w") as f:
        json.dump(dict(features=FEATS, model_type="xgbregressor", target="vote_pct",
                       published="margin = predicted share minus the best rival's predicted share "
                                 "in the same race (race_margins)",
                       xgb_params=PARAMS,
                       trained_on_cycles=sorted(int(y) for y in c["year"].unique()),
                       tune=dict(cycles=TUNE_CYCLES, metric="MAE of the derived margin",
                                 stage2=[[round(s[0], 4), s[1]] for s in stage2]),
                       eval_expanding_window=dict(mean=means.to_dict(),
                                                  per_cycle=ev.round(4).reset_index().to_dict("records"))),
                  f, indent=1)
    print(f"\nsaved data/race_margin_model_xgb.json + _features.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
