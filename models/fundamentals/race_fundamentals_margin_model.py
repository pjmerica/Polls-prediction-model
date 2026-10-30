# -*- coding: utf-8 -*-
"""RACE-LEVEL no-polls MARGIN model (2026-10-29, user: "do the no polls margin model too"): the
margin for the 2026 races no pollster covers, as race_margin_model.py gives it for polled ones.

    py -X utf8 models/fundamentals/race_fundamentals_margin_model.py [--extend]

Scheme (race_margin_model.py's, on race_fundamentals_model.py's rows and features):
- Training rows: race_fundamentals_model.training_table() - every general race 1998-2024, polled
  (full candidate fields) and unpolled (the D and R candidates; data/res_*.csv). Unpolled rows get
  their vote share from the results archive (fundamentals_on_unpolled.results_universe "percent",
  share of the full field, like the polled rows' vote_pct). Races with any share missing are dropped.
- Features: the no-polls win model's list (no poll features).
- Target: vote_pct. The published number is the DERIVED margin (race_level.race_margins: share
  minus the best rival's predicted share), so every race has one leader and the top two's margins
  are equal and opposite.
- Scoring: MAE of the derived margin against the actual margin, on races where it is known - two
  or more candidates in the table. An unpolled race with ONE D/R candidate (688: minor-party or
  write-in opposition the archive drops) still trains the share, but its margin is not scored.
- Tuning: LOCO over the 1998-2016 TUNE cycles, each fold training on the other tune cycles;
  random draws + race_level.refine_search hill-climb + the top settings re-scored on 3 seeds; the
  served model averages the top 3 settings' shares (as race_margin_model.py).
- Held-out 2018-24, expanding window, graded on unpolled and polled races, next to the naive
  margin (the candidate's own prior margin) as a reference point.
- Artifacts: data/fundamentals_margin_model*.json + data/fundamentals_margin_model_features.json,
  read by predict_unpolled.py.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))
_sys.path[:0] = [_HERE, _ROOT, _os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "analysis"),
                 _os.path.join(_ROOT, "models", "poll")]
from paths import ROOT  # noqa: E402

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

import features as F  # noqa: E402
import fundamentals_model as FM  # noqa: E402
import fundamentals_on_unpolled as FU  # noqa: E402
import race_fundamentals_model as RF  # noqa: E402
from cycles import TUNE_CYCLES, EVAL_CYCLES  # noqa: E402
from race_level import Ensemble, race_margins, refine_search, sample_grid  # noqa: E402

N_CONFIGS = 120      # random draws from GRID; then refinement
TOP_K = 10
TUNE_SEEDS = (42, 7, 123)
GRID = dict(max_depth=[2, 3, 4, 5, 6],
            learning_rate=[0.02, 0.03, 0.05, 0.1],
            n_estimators=[200, 400, 800, 1600],
            min_child_weight=[5, 10, 20, 40],
            subsample=[0.7, 0.85, 1.0],
            colsample_bytree=[0.4, 0.6, 0.8, 1.0],
            reg_lambda=[1, 5, 20],
            reg_alpha=[0, 1])
ART = os.path.join(ROOT, "data", "fundamentals_margin_model_features.json")


def training_table():
    c = RF.training_table().copy()
    r = FU.results_universe()
    pct = r.set_index(["race_id", "cand_key"])["percent"]
    un = c["polled"] == 0
    c.loc[un, "vote_pct"] = [pct.get(k) for k in zip(c.loc[un, "race_id"], c.loc[un, "cand_key"])]
    c["vote_pct"] = pd.to_numeric(c["vote_pct"], errors="coerce")
    bad = c.loc[c["vote_pct"].isna(), "race_id"].unique()
    c = c[~c["race_id"].isin(bad)].copy()
    n = c.groupby("race_id")["race_id"].transform("size")
    c["margin_actual"] = race_margins(c, "vote_pct").where(n > 1)    # unknown with one candidate
    return c, len(bad)


def fit_share(tr, feats, params, seed=42):
    m = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=-1, **params)
    m.fit(tr[feats], tr["vote_pct"])
    return m


def mae(te, col="_margin"):
    k = te["margin_actual"].notna()
    return float((te.loc[k, col] - te.loc[k, "margin_actual"]).abs().mean())


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


def grade(te):
    k = te[te["margin_actual"].notna()]
    lead = k.loc[k.groupby("race_id")["_margin"].idxmax()]
    naive = k["prior_margin_cand"].notna()
    return dict(n_races=int(k["race_id"].nunique()), MAE_margin=mae(te),
                MAE_share=float((te["_share"] - te["vote_pct"]).abs().mean()),
                race_acc=float(lead["won"].mean()),
                MAE_naive_prior_margin=float((k.loc[naive, "prior_margin_cand"]
                                              - k.loc[naive, "margin_actual"]).abs().mean()),
                naive_coverage=float(naive.mean()))


def main():
    t0 = time.time()
    c, n_bad = training_table()
    FEATS = [f for f in F.general_features("win") if not FM.is_poll_feature(f)]
    print(f"rows {len(c)} | races {c['race_id'].nunique()} (polled {c.loc[c.polled == 1, 'race_id'].nunique()}, "
          f"unpolled {c.loc[c.polled == 0, 'race_id'].nunique()}; {n_bad} dropped for a missing share; "
          f"{int(c.groupby('race_id')['margin_actual'].first().isna().sum())} one-candidate) | "
          f"features {len(FEATS)} | {time.time() - t0:.0f}s", flush=True)

    combos = ([s[1] for s in json.load(open(ART))["tune"]["stage2"]] if "--extend" in _sys.argv
              else sample_grid(GRID, N_CONFIGS, seed=0))
    t1 = time.time()
    stage1 = sorted(((loco_tune(c, FEATS, p, 42), p) for p in combos), key=lambda t: t[0])
    print(f"stage 1: {len(combos)} settings in {time.time() - t1:.0f}s; best tune MAE "
          f"{[round(s[0], 3) for s in stage1[:5]]}", flush=True)
    n1 = len(stage1)
    stage1 = refine_search(stage1, GRID, lambda q: loco_tune(c, FEATS, q, 42), lambda t: t[0])
    print(f"refinement: {len(stage1) - n1} one-step neighbours of the top 5; best now "
          f"{[round(s[0], 3) for s in stage1[:5]]}", flush=True)
    stage2 = sorted(((float(np.mean([loco_tune(c, FEATS, p, s) for s in TUNE_SEEDS])), p)
                     for _, p in stage1[:TOP_K]), key=lambda t: t[0])
    print(f"stage 2 ({len(TUNE_SEEDS)}-seed): {[round(s[0], 3) for s in stage2]}")
    print(f"PARAMS = {stage2[0][1]}  (tune MAE {stage2[0][0]:.3f}; runner-up {stage2[1][0]:.3f})", flush=True)
    ENS = [p for _, p in stage2[:3]]

    def fit_ens(tr):
        return Ensemble([fit_share(tr, FEATS, p) for p in ENS])

    # ---- held-out 2018-24, expanding window ----
    rows = []
    for ty in EVAL_CYCLES:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        te["_share"] = fit_ens(tr).predict(te[FEATS])
        te["_margin"] = race_margins(te, "_share")
        for tag, sub in (("unpolled", te[te.polled == 0]), ("polled", te[te.polled == 1])):
            rows.append(dict(cycle=ty, set=tag, **grade(sub)))
    ev = pd.DataFrame(rows)
    print("\n=== held-out 2018-24, expanding window (sanity check) ===")
    print(ev.set_index(["set", "cycle"]).round(3).to_string())
    means = ev.groupby("set")[["MAE_margin", "MAE_share", "race_acc", "MAE_naive_prior_margin"]].mean().round(4)
    print(means.to_string())

    # ---- production: all cycles ----
    ens = fit_ens(c)
    files = []
    for i, m in enumerate(ens.models):
        fn = "fundamentals_margin_model.json" if i == 0 else f"fundamentals_margin_model_{i}.json"
        m.save_model(os.path.join(ROOT, "data", fn))
        files.append(fn)
    imp = pd.Series(np.mean([m.feature_importances_ for m in ens.models], axis=0),
                    index=FEATS).sort_values(ascending=False)
    imp.rename("gain_importance").rename_axis("feature").reset_index().to_csv(
        os.path.join(_HERE, "margin_feature_importance.csv"), index=False)
    print("top features:", imp.head(8).round(3).to_dict())
    with open(ART, "w") as f:
        json.dump(dict(features=FEATS, model_type="xgbregressor", target="vote_pct",
                       published="margin = predicted share minus the best rival's predicted share "
                                 "in the same race (race_margins)",
                       ensemble_files=files, xgb_params=ENS,
                       trained_on_cycles=sorted(int(y) for y in c["year"].unique()),
                       tune=dict(cycles=TUNE_CYCLES, metric="MAE of the derived margin (races with 2+ candidates)",
                                 stage2=[[round(s[0], 4), s[1]] for s in stage2]),
                       eval_expanding_window=dict(mean={k: r.to_dict() for k, r in means.iterrows()},
                                                  per_cycle=ev.round(4).to_dict("records"))),
                  f, indent=1)
    print(f"\nsaved {files} + fundamentals_margin_model_features.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
