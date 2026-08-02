# -*- coding: utf-8 -*-
"""Train + evaluate the PRIMARY MARGIN model; save production artifacts.

    py -X utf8 models/poll/primary_margin_model.py

The fourth model. Sibling of margin_model.ipynb (general margin) exactly as primary_model.py
is the sibling of model.ipynb: same features_primary pipeline, same expanding-window honest
eval, but a REGRESSION target instead of a within-race ranking.

TARGET: `vote_margin` = the candidate's actual primary vote share minus the best OTHER
candidate's, in percentage points. Positive = won by that much. Labels come from the committed
results archives (data/primary_results_hist.csv + data/house_primary_results_hist.csv), which
are near-complete: the per-race pct sums average 99.8, so "best other" is computed against the
FULL results field, not just the subset that happened to be polled. That distinction matters -
using only the modelled subset would compare the front-runner against whoever else was polled
rather than whoever actually finished second.

WHY THIS IS HARDER THAN THE GENERAL MARGIN MODEL (measured before building, so the bar is
honest rather than retrofitted):
  - target std is 37.9 pts, range -92 to +92. Primaries are blowouts far more often than
    general elections, where the two-party structure compresses margins.
  - the calibrated-poll baseline is MAE 15.81 pts (vs ~7.45 for the general model). The
    calibration slope is 1.19: primary poll leads systematically UNDERSTATE the final margin,
    because front-runners consolidate late as also-rans fade.
  - ~591 labelled candidate rows across 179 races, against ~4,400 rows for the general margin
    model. Thin.
SUCCESS TEST, fixed in advance: beat MAE 15.81 (the linear calibration of poll_lead) on the
held-out cycles. If it cannot, that is a real finding about primary margins, not a bug - and
the script prints both baselines every run so the comparison can never be quietly dropped.

No fund features (same cycle-end FEC leak as the primary win model - proven 2026-08-02: the
eventual nominee is the top fundraiser 92.4% of the time vs the poll leader winning 69.6%).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
from paths import ROOT  # noqa: E402

import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

import features as F  # noqa: E402
import features_primary as FP  # noqa: E402

HERE = ROOT
_HERE_DIR = _os.path.dirname(_os.path.abspath(__file__))


def load_results():
    """Both primary results archives -> {(race_id, cand_key): pct} + per-race field totals."""
    frames = []
    for fn in ("primary_results_hist.csv", "house_primary_results_hist.csv"):
        p = os.path.join(HERE, "data", fn)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, low_memory=False))
    res = pd.concat(frames, ignore_index=True)
    res = res.dropna(subset=["race_id", "cand_key", "pct"])
    return res


def build_table():
    """Candidate table + the margin target, restricted to races with a trustworthy field."""
    from candidate_history import CandidateHistory
    d = pd.read_csv(os.path.join(HERE, "data", "primary_polls_long.csv"), low_memory=False)
    d = F.prepare_polls(d)
    d["district"] = d["district"].map(F.dist_str)
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    c = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                               hist=CandidateHistory(), bios=FP.load_candidate_bios())
    c = c[~c["candidate"].map(F.is_junk_answer)].copy()

    res = load_results()
    # BEST-OTHER is computed over the RESULTS field, not the modelled subset. A polled
    # candidate who dropped out before the primary has no result row, and the runner-up is
    # frequently someone nobody polled - taking the max over the modelled subset would
    # silently compare the front-runner against the wrong person.
    key = res["race_id"].astype(str) + "|" + res["cand_key"].astype(str)
    pct_map = dict(zip(key, res["pct"]))
    best_other_map = {}
    for rid, g in res.groupby("race_id"):
        s = g.sort_values("pct", ascending=False)
        top, second = s["pct"].iloc[0], (s["pct"].iloc[1] if len(s) > 1 else 0.0)
        for r in s.itertuples():
            best_other_map[f"{rid}|{r.cand_key}"] = second if r.pct == top else top

    k = c["race_id"].astype(str) + "|" + c["cand_key"].astype(str)
    c["vote_pct"] = k.map(pct_map)
    c["vote_margin"] = c["vote_pct"] - k.map(best_other_map)

    # only races whose results field is essentially complete (pct sums ~100); a truncated
    # results table would put "best other" against a phantom.
    field_sum = res.groupby("race_id")["pct"].sum()
    good = set(field_sum[field_sum >= 95].index)
    c = c[c["race_id"].isin(good)]
    return c.dropna(subset=["vote_margin"]).copy()


def baselines(te):
    """Raw poll_lead, and a linear calibration of it fitted on the TRAINING rows only."""
    return te["poll_lead"]


def expanding_eval(c, feats, params, eval_years, label):
    rows = []
    for ty in eval_years:
        tr, te = c[c["year"] < ty], c[c["year"] == ty]
        tr = tr.dropna(subset=["vote_margin"])
        te = te.dropna(subset=["vote_margin"])
        if len(tr) < 30 or not len(te):
            continue
        m = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(tr[feats], tr["vote_margin"])
        pred = m.predict(te[feats])

        # baseline 1: the raw polling lead, used directly as a margin prediction
        raw = te["poll_lead"].fillna(0.0)
        # baseline 2: poll_lead linearly calibrated on the TRAINING rows (leak-free)
        trl = tr.dropna(subset=["poll_lead"])
        lin = LinearRegression().fit(trl[["poll_lead"]], trl["vote_margin"])
        cal = lin.predict(te[["poll_lead"]].fillna(0.0))

        rows.append(dict(cycle=ty, n=len(te), n_races=te["race_id"].nunique(),
                         MAE_model=mean_absolute_error(te["vote_margin"], pred),
                         MAE_poll=mean_absolute_error(te["vote_margin"], raw),
                         MAE_calib=mean_absolute_error(te["vote_margin"], cal),
                         R2=r2_score(te["vote_margin"], pred)))
    ev = pd.DataFrame(rows).set_index("cycle")
    print(f"\n=== EXPANDING-WINDOW eval ({label}) ===")
    print(ev.round(2).to_string())
    mean = ev[["MAE_model", "MAE_poll", "MAE_calib", "R2"]].mean()
    print("MEAN:", mean.round(3).to_dict())
    verdict = ("BEATS" if mean["MAE_model"] < mean["MAE_calib"] else "does NOT beat")
    print(f"  -> the model {verdict} the calibrated-poll baseline "
          f"({mean['MAE_model']:.2f} vs {mean['MAE_calib']:.2f} MAE)")
    return ev


def main():
    c = build_table()
    FEATS = FP.feature_list_primary()          # no fund - same leak as the win model
    years = sorted(c["year"].unique())
    tune_years = [y for y in years if y <= 2020]
    eval_years = [y for y in years if y >= 2022]
    print(f"labelled candidate rows {len(c)} | races {c['race_id'].nunique()} "
          f"| features {len(FEATS)}")
    print(f"target: mean {c['vote_margin'].mean():.1f} std {c['vote_margin'].std():.1f} "
          f"range [{c['vote_margin'].min():.0f}, {c['vote_margin'].max():.0f}]")
    print("tune cycles:", tune_years, "| eval cycles:", eval_years)

    GRID = dict(max_depth=[1, 2, 3], learning_rate=[0.03, 0.05, 0.1],
                n_estimators=[100, 200, 300], min_child_weight=[4, 8, 15],
                subsample=[0.7, 1.0], colsample_bytree=[0.5, 1.0],
                reg_lambda=[1, 5, 20])
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    random.seed(0)
    random.shuffle(combos)
    best = None
    for combo in combos[:60]:
        p = dict(zip(keys, combo))
        maes = []
        for ty in tune_years:                  # LOCO over the OLD cycles only
            tr, te = c[c["year"] != ty], c[c["year"] == ty]
            if len(tr) < 30 or not len(te):
                continue
            m = xgb.XGBRegressor(**p, random_state=42, n_jobs=-1)
            m.fit(tr[FEATS], tr["vote_margin"])
            maes.append(mean_absolute_error(te["vote_margin"], m.predict(te[FEATS])))
        if maes:
            a = float(np.mean(maes))
            if best is None or a < best[0]:
                best = (a, p)
    PARAMS = best[1]
    print(f"\nbest tune-cycle LOCO MAE: {best[0]:.3f}")
    print("PARAMS =", PARAMS)

    ev = expanding_eval(c, FEATS, PARAMS, eval_years, "PRIMARY MARGIN headline")

    prod = xgb.XGBRegressor(**PARAMS, random_state=42, n_jobs=-1)
    prod.fit(c[FEATS], c["vote_margin"])
    prod.save_model(os.path.join(HERE, "data", "primary_margin_model_xgb.json"))
    with open(os.path.join(HERE, "data", "primary_margin_model_features.json"), "w") as f:
        json.dump(dict(features=FEATS, xgb_params=PARAMS, model_type="xgbregressor",
                       target=("vote_margin = actual primary vote pct minus the best OTHER "
                               "candidate's, from the results archives"),
                       trained_on_cycles=[int(y) for y in years],
                       n_rows=int(len(c)), n_races=int(c["race_id"].nunique()),
                       eval_expanding_window={str(k): {m: round(float(v), 4)
                                                       for m, v in r.items()}
                                              for k, r in ev.iterrows()}), f, indent=1)

    g = prod.get_booster().get_score(importance_type="gain")
    imp = pd.Series({f: g.get(f, 0.0) for f in FEATS}).sort_values(ascending=False)
    tot = imp.sum()
    pd.DataFrame({"feature": imp.index, "gain": imp.values,
                  "gain_pct": (100 * imp / tot).round(3).values,
                  "rank": range(1, len(imp) + 1)}).to_csv(
        os.path.join(_HERE_DIR, "primary_margin_feature_importance.csv"), index=False)
    print("\ntop 10 by gain:")
    for i, (f, v) in enumerate(imp.head(10).items(), 1):
        print(f"  {i:>2}. {f:<26} {100*v/tot:>5.2f}%")
    print("\nsaved data/primary_margin_model_xgb.json + _features.json"
          " + models/poll/primary_margin_feature_importance.csv")


if __name__ == "__main__":
    main()
