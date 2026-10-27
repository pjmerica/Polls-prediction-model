# -*- coding: utf-8 -*-
"""RACE-LEVEL win model (2026-10-26): XGBRanker + within-race softmax, side by side with the
per-candidate classifier in model.ipynb. Served by src/predict.py since 2026-10-27.

    py -X utf8 models/poll/race_win_model.py

Why: model.ipynb's XGBClassifier scores each candidate on its own, so a race's raw win
probabilities summed to 0.80-1.42 in the 2026 run (GA-Gov: Bottoms .67 AND Jackson .55) and
predict.py divided them by their sum - a number the notebook never evaluated. The primary model
hit the same flaw on 2026-07-29 and moved to a ranker; this is the general-election version.
The softmax output is the ONE probability: what is evaluated here, saved and published
(predict.py serves it since 2026-10-27, audit item 7).

Scheme (also fixes polling-agg audit M4/M5):
- Same training rows as model.ipynb; features.general_features("win") (the list model.ipynb uses).
- Tuning: LOCO over the 1998-2016 TUNE cycles ONLY - each tune fold trains on the other tune
  cycles, never on 2018-24 (model.ipynb's tuner trained on them). Score = log loss of the
  softmaxed probabilities, with the temperature fitted on the same tune-cycle predictions.
  Stage 1 screens N_CONFIGS settings on one seed; stage 2 re-scores the TOP_K on 3 seeds.
- Temperature (2026-10-27): fitted by log loss on out-of-sample predictions for FULL-FIELD
  cycles (2018 on; 1998-2016 polls carry only the top two candidates, so a temperature fitted
  there over-weights minor-party candidates). Production: LOCO predictions of 2018-24. Eval:
  each test cycle uses only earlier full-field cycles. See main().
- Held-out check: expanding-window 2018-24 on the published number, with a calibration table
  (user 2026-10-26: method fixes are judged on correctness, not on a metric comparison).
- Artifacts: data/race_win_model_xgb.json + data/race_win_model_features.json (all cycles).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT  # noqa: E402

import io
import contextlib
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import features as F
from cycles import CYCLES, TUNE_CYCLES, EVAL_CYCLES, natl_env
from macro_features import build_macro
from race_level import MINOR_POLL_MAX, softmax, race_win_probs

N_CONFIGS = 60
TOP_K = 8
TUNE_SEEDS = (42, 7, 123)
TEMP_GRID = np.round(np.exp(np.linspace(np.log(0.1), np.log(3.0), 40)), 4)
GRID = dict(max_depth=[1, 2, 3], learning_rate=[0.02, 0.05, 0.1],
            n_estimators=[150, 300, 600], min_child_weight=[0.5, 2, 8],
            subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 1.0],
            reg_lambda=[1, 5, 20])


# ---- training rows: identical construction to model.ipynb cells 4-12 ----------------------
def training_table():
    d = pd.read_csv(os.path.join(ROOT, "polls_long_with_results.csv"), low_memory=False)
    gen = d[d["stage"] == "general"]
    q = gen.groupby(["race_id", "question_id"])["has_result"].agg(["sum", "size"])
    dead = set(q[(q["sum"] > 0) & (q["sum"] < q["size"])].index)
    drop = (pd.Series([k in dead for k in zip(d["race_id"], d["question_id"])], index=d.index)
            & (d["stage"] == "general"))
    d = d[~drop]
    d = d[d["has_result"] == 1].copy()
    d = d[d.groupby("race_id")["won"].transform("max") == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    return F.build_candidate_table(d, build_macro(), natl_env(), F.load_fundamentals(),
                                   house_train_years=CYCLES, fec=F.load_fec(extended=True),
                                   bias_priors=F.compute_bias_priors(d),
                                   candidate_bios=F.load_candidate_bios(),
                                   primary_results=F.load_primary_results())


# ---- ranker + softmax ------------------------------------------------------------------------
# the one softmax, shared with predict.py / explain_2026.py (src/race_level.py)
race_softmax = softmax
softmax_col = race_win_probs


def fit_ranker(tr, feats, params, seed=42):
    tr = tr.sort_values("race_id")
    grp = tr.groupby("race_id", sort=True).size().values
    m = xgb.XGBRanker(objective="rank:pairwise", random_state=seed, n_jobs=-1, **params)
    m.fit(tr[feats], tr["won"].astype(int), group=grp)
    return m


def fit_temp(df, score_col="_score"):
    """Temperature with the lowest log loss on these rows' softmaxed scores."""
    y = df["won"].astype(int)
    best = None
    for T in TEMP_GRID:
        p = softmax_col(df, score_col, T).clip(1e-6, 1 - 1e-6)
        ll = log_loss(y, p)
        if best is None or ll < best[1]:
            best = (float(T), ll)
    return best


OFFSET_GRID = np.round(np.linspace(-8, 0, 33), 2)


def fit_calibration(df, score_col="_score"):
    """Temperature AND minor-candidate offset (src/race_level.py) with the lowest log loss on
    these rows. Returns (temp, offset, log loss)."""
    y = df["won"].astype(int)
    best = None
    for T in TEMP_GRID:
        for d in OFFSET_GRID:
            p = race_win_probs(df, score_col, T, d).clip(1e-6, 1 - 1e-6)
            ll = log_loss(y, p)
            if best is None or ll < best[2]:
                best = (float(T), float(d), ll)
    return best


def loco_tune_scores(c, feats, params, seed):
    """LOCO over the tune cycles, training ONLY on the other tune cycles."""
    rows = []
    tune = c[c["year"].isin(TUNE_CYCLES)]
    for ty in TUNE_CYCLES:
        tr, te = tune[tune["year"] != ty], tune[tune["year"] == ty].copy()
        if not len(te):
            continue
        te["_score"] = fit_ranker(tr, feats, params, seed).predict(te[feats])
        rows.append(te)
    return pd.concat(rows)


def score_config(c, feats, params, seeds):
    lls, temps = [], []
    for s in seeds:
        T, ll = fit_temp(loco_tune_scores(c, feats, params, s))
        lls.append(ll); temps.append(T)
    return float(np.mean(lls)), float(np.median(temps))


def race_acc(te, col):
    return te.loc[te.groupby("race_id")[col].idxmax()]["won"].mean()


def metrics(te, col):
    y = te["won"].astype(int)
    p = te[col].clip(1e-6, 1 - 1e-6)
    return dict(Brier=brier_score_loss(y, p), LogLoss=log_loss(y, p),
                AUC=roc_auc_score(y, p), race_acc=race_acc(te, col))


def main():
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        c = training_table()
    FEATS = F.general_features("win")
    print(f"training rows {len(c)} | races {c['race_id'].nunique()} | features {len(FEATS)} "
          f"| built in {time.time() - t0:.0f}s")

    # ---- tuning (tune cycles only, log loss of the softmaxed probability) ----
    art = os.path.join(ROOT, "data", "race_win_model_features.json")
    if "--reuse-params" in _sys.argv and os.path.exists(art):
        prev = json.load(open(art))
        PARAMS, stage2 = prev["xgb_params"], [(tuple(s[0]) if isinstance(s[0], list) else (s[0], None), s[1])
                                              for s in prev["tune"]["stage2"]]
        print(f"PARAMS reused from {os.path.basename(art)}: {PARAMS}")
    else:
        keys = list(GRID)
        combos = list(itertools.product(*[GRID[k] for k in keys]))
        random.seed(0); random.shuffle(combos)
        combos = [dict(zip(keys, cb)) for cb in combos[:N_CONFIGS]]
        t1 = time.time()
        stage1 = sorted(((score_config(c, FEATS, p, (42,)), p) for p in combos), key=lambda t: t[0][0])
        print(f"stage 1: {len(combos)} settings in {time.time() - t1:.0f}s; best tune log loss "
              f"{[round(s[0][0], 4) for s in stage1[:5]]}", flush=True)
        stage2 = sorted(((score_config(c, FEATS, p, TUNE_SEEDS), p) for _, p in stage1[:TOP_K]),
                        key=lambda t: t[0][0])
        (best_ll, _), PARAMS = stage2[0]
        print(f"stage 2 ({len(TUNE_SEEDS)}-seed): {[round(s[0][0], 4) for s in stage2]}")
        print(f"PARAMS = {PARAMS}  (tune log loss {best_ll:.4f}; runner-up "
              f"{stage2[1][0][0]:.4f})", flush=True)

    # ---- temperature (2026-10-27) ----
    # The 1998-2016 polls (538's file) carry only the top two candidates per race, so a temperature
    # fitted there has never seen a minor-party candidate; from 2018 the NYT/Wikipedia feeds carry
    # full fields (46-60% of races have 3+ candidates), as 2026's do. Fitted on 1998-2016 the
    # temperature was 0.97 and held-out candidates polling under 10% were given 9 expected wins
    # (0 won), favourites over 90% read 96.6% (98.5% won). A temperature fitted on 2018-24 alone
    # still left them 7.1 expected wins and made the 70-90% band overconfident (80% predicted, 76%
    # won): a softmax over ranker scores gives every minor candidate a floor. So the temperature
    # and a log-scale OFFSET for candidates polling under 10% (race_level.MINOR_POLL_MAX) are
    # fitted together on out-of-sample predictions for FULL-FIELD cycles: in production on 2018-24
    # LOCO predictions; in the honest eval each test cycle uses only earlier full-field cycles
    # (2018: none, so the tune-cycle temperature and no offset).
    def seeded(frames):
        # one race_id per seed: pooling the seeds under the same race_id would softmax over three
        # copies of every candidate
        return pd.concat([f.assign(race_id=lambda t, s=s: t["race_id"] + f"#{s}") for s, f in frames])

    tune_pred = seeded([(s, loco_tune_scores(c, FEATS, PARAMS, s)) for s in TUNE_SEEDS])
    TUNE_TEMP, tune_ll = fit_temp(tune_pred)
    n_cand = c.groupby("race_id")["race_id"].transform("size")
    FULL = sorted(int(y) for y in c.loc[n_cand > 2, "year"].unique())
    print(f"tune-cycle temperature {TUNE_TEMP} (log loss {tune_ll:.4f}); full-field cycles {FULL}")

    # ---- honest eval: expanding-window 2018-24 on the PUBLISHED number ----
    rows, preds = [], []
    for ty in EVAL_CYCLES:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        te["_score"] = fit_ranker(tr, FEATS, PARAMS).predict(te[FEATS])
        prior = [p for p in preds if p["year"].iloc[0] in FULL]
        T_ty, d_ty = fit_calibration(pd.concat(prior))[:2] if prior else (TUNE_TEMP, 0.0)
        te["p_race"] = race_win_probs(te, "_score", T_ty, d_ty)
        rows.append(dict(cycle=ty, temp=T_ty, minor_offset=d_ty, **metrics(te, "p_race")))
        preds.append(te)
    ev = pd.DataFrame(rows).set_index("cycle")
    print("\n=== held-out 2018-24, expanding window (sanity check) ===")
    print(ev.round(4).to_string())
    P = pd.concat(preds)
    P["bin"] = pd.cut(P["p_race"], [0, .02, .1, .3, .5, .7, .9, 1.0])
    print("\ncalibration (all candidates):")
    print(P.groupby("bin", observed=True).agg(n=("won", "size"), pred=("p_race", "mean"),
                                              won=("won", "mean")).round(3).to_string())
    low = P[P["poll_avg"] < 10]
    print(f"candidates polling under 10%: {len(low)}, expected wins {low['p_race'].sum():.1f}, "
          f"won {int(low['won'].sum())}")

    # ---- production: all cycles; temperature from LOCO predictions of the full-field cycles ----
    loco = []
    for s in TUNE_SEEDS:
        for ty in FULL:
            tr, te = c[c["year"] != ty], c[c["year"] == ty].copy()
            te["_score"] = fit_ranker(tr, FEATS, PARAMS, s).predict(te[FEATS])
            loco.append((s, te))
    TEMP, OFFSET, full_ll = fit_calibration(seeded(loco))
    print(f"\nproduction softmax temperature = {TEMP}, minor offset = {OFFSET} "
          f"(LOCO log loss on {FULL}: {full_ll:.4f})")
    m = fit_ranker(c, FEATS, PARAMS)
    m.save_model(os.path.join(ROOT, "data", "race_win_model_xgb.json"))
    means = ev[["Brier", "LogLoss", "AUC", "race_acc"]].mean().round(4)
    with open(os.path.join(ROOT, "data", "race_win_model_features.json"), "w") as f:
        json.dump(dict(features=FEATS, model_type="xgbranker", objective="rank:pairwise",
                       xgb_params=PARAMS, softmax_temp=TEMP, minor_offset=OFFSET,
                       minor_poll_max=MINOR_POLL_MAX,
                       temp_fit=dict(cycles=FULL, scheme="temperature + minor offset fitted on LOCO "
                                     "predictions of the full-field cycles",
                                     tune_cycle_temp=TUNE_TEMP),
                       trained_on_cycles=sorted(int(y) for y in c["year"].unique()),
                       tune=dict(cycles=TUNE_CYCLES, metric="log loss of softmaxed probability",
                                 stage2=[[round(s[0][0], 4), s[1]] for s in stage2]),
                       eval_expanding_window=dict(mean=means.to_dict(),
                                                  per_cycle=ev.round(4).reset_index().to_dict("records")),
                       note="race-level win model: within-race softmax of ranker scores is the "
                            "one probability. Served by src/predict.py since 2026-10-27."),
                  f, indent=1)
    print(f"\nsaved data/race_win_model_xgb.json + _features.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
