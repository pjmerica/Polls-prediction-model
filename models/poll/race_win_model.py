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
from race_level import MINOR_POLL_MAX, Ensemble, refine_search, sample_grid, softmax, race_win_probs

N_CONFIGS = 600      # random draws from GRID (60 until 10-28, 300 on 10-28); then refinement
N_ENSEMBLE = 5       # the served model averages the top 5 stage-2 settings...
N_MODEL_SEEDS = 3    # ...each fitted with 3 seeds (15 members). 2026-10-28, audit M5: with the top 3
                     # on one seed (10-27), a re-tune with fresh randomness moved 25 of 175 races by
                     # 5+ win pts (docs/RETUNE_STABILITY.md, analysis/retune_stability.py)
TOP_K = 15           # settings re-scored on 3 seeds (8 until 10-28, 12 on 10-28)
TUNE_SEEDS = (42, 7, 123)
TEMP_GRID = np.round(np.exp(np.linspace(np.log(0.1), np.log(3.0), 40)), 4)
# 2026-10-29 (final search, user: "even wider and more options"): every edge the 10-28 picks sat on
# extended (all 5 at reg_lambda 0.25 and colsample 0.4, the lowest), finer steps throughout, and L1
# (reg_alpha) searched for the first time. Depth 1 and subsample 1.0 are hard limits.
# Later on 10-29 (--extend): all 5 picks used 900 trees, the most, so 1,350 / 1,800 added, and
# min_child_weight 16 above the one pick at 8. The hill-climb scored 51 new neighbours and kept the
# same 5 settings: more trees did not help.
GRID = dict(max_depth=[1, 2, 3],
            learning_rate=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
            n_estimators=[100, 150, 200, 300, 450, 600, 900, 1350, 1800],
            min_child_weight=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16],
            subsample=[0.6, 0.7, 0.8, 0.9, 1.0],
            colsample_bytree=[0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
            reg_lambda=[0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 20],
            reg_alpha=[0, 0.1, 0.5, 1])


# ---- training rows: identical construction to model.ipynb cells 4-12 ----------------------
def training_polls(keep_unwon=False):
    """The training poll rows (model.ipynb cells 4-12), before the candidate table is built.
    keep_unwon=True keeps the races whose winner was never polled (audit item 13: they are graded,
    scored as every polled candidate losing, though a softmax over the polled field cannot be
    TRAINED on them)."""
    d = pd.read_csv(os.path.join(ROOT, "polls_long_with_results.csv"), low_memory=False)
    gen = d[d["stage"] == "general"]
    q = gen.groupby(["race_id", "question_id"])["has_result"].agg(["sum", "size"])
    dead = set(q[(q["sum"] > 0) & (q["sum"] < q["size"])].index)
    drop = (pd.Series([k in dead for k in zip(d["race_id"], d["question_id"])], index=d.index)
            & (d["stage"] == "general"))
    d = d[~drop]
    d = d[d["has_result"] == 1].copy()
    if not keep_unwon:
        d = d[d.groupby("race_id")["won"].transform("max") == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    return d


def training_table(d=None):
    """Candidate table for training; pass `d` (training_polls(), possibly truncated) to reuse it."""
    d = training_polls() if d is None else d
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


def fit_ensemble(tr, feats, params_list, seed=42, n_seeds=None):
    """Every setting fitted with n_seeds seeds (seed, seed + 1009, ...): subsample / colsample make
    each fit one draw, so averaging seeds removes that noise as averaging settings removes the
    tuner's."""
    n_seeds = N_MODEL_SEEDS if n_seeds is None else n_seeds
    return Ensemble([fit_ranker(tr, feats, p, seed + 1009 * k) for p in params_list for k in range(n_seeds)])


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
        # --extend: no random draws; the hill-climb starts from the saved stage-2 settings, so a
        # grid widened at an edge the last search's picks sat on is explored from where they are
        combos = ([s[1] for s in json.load(open(art))["tune"]["stage2"]] if "--extend" in _sys.argv
                  else sample_grid(GRID, N_CONFIGS, seed=0))
        t1 = time.time()
        stage1 = sorted(((score_config(c, FEATS, p, (42,)), p) for p in combos), key=lambda t: t[0][0])
        print(f"stage 1: {len(combos)} settings in {time.time() - t1:.0f}s; best tune log loss "
              f"{[round(s[0][0], 4) for s in stage1[:5]]}", flush=True)
        n1 = len(stage1)
        stage1 = refine_search(stage1, GRID, lambda q: score_config(c, FEATS, q, (42,)), lambda t: t[0][0])
        print(f"refinement: {len(stage1) - n1} one-step neighbours of the top 5; best now "
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

    # the served model: the average of the top N_ENSEMBLE stage-2 settings (audit item 6)
    ENS = [p for _, p in stage2[:N_ENSEMBLE]]
    print(f"ensemble of {len(ENS)} settings: {ENS}", flush=True)

    def loco_ens(pool, cycles, seed):
        out = []
        for ty in cycles:
            tr, te = pool[pool["year"] != ty], c[c["year"] == ty].copy()
            if len(te):
                te["_score"] = fit_ensemble(tr, FEATS, ENS, seed).predict(te[FEATS])
                out.append(te)
        return pd.concat(out)

    tune = c[c["year"].isin(TUNE_CYCLES)]
    tune_pred = seeded([(s, loco_ens(tune, TUNE_CYCLES, s)) for s in TUNE_SEEDS])
    TUNE_TEMP, tune_ll = fit_temp(tune_pred)
    n_cand = c.groupby("race_id")["race_id"].transform("size")
    FULL = sorted(int(y) for y in c.loc[n_cand > 2, "year"].unique())
    print(f"tune-cycle temperature {TUNE_TEMP} (log loss {tune_ll:.4f}); full-field cycles {FULL}")

    # races whose winner no pollster covered (audit item 13): graded, all polled candidates lost
    with contextlib.redirect_stdout(io.StringIO()):
        dk = training_polls(keep_unwon=True)
        un_ids = set(dk.loc[dk.groupby("race_id")["won"].transform("max") == 0, "race_id"])
        # built on the FULL poll set so poll-bias priors match the training table's, then filtered
        unwon = training_table(dk) if un_ids else c.iloc[0:0]
        unwon = unwon[unwon["race_id"].isin(un_ids)]

    # ---- honest eval: expanding-window 2018-24 on the PUBLISHED number ----
    rows, preds = [], []
    for ty in EVAL_CYCLES:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        ens = fit_ensemble(tr, FEATS, ENS)
        te["_score"] = ens.predict(te[FEATS])
        prior = [p for p in preds if p["year"].iloc[0] in FULL]
        T_ty, d_ty = fit_calibration(pd.concat(prior))[:2] if prior else (TUNE_TEMP, 0.0)
        te["p_race"] = race_win_probs(te, "_score", T_ty, d_ty)
        ue = unwon[unwon["year"] == ty].copy()
        if len(ue):
            ue["_score"] = ens.predict(ue[FEATS])
            ue["p_race"] = race_win_probs(ue, "_score", T_ty, d_ty)
        both = pd.concat([te, ue]) if len(ue) else te
        rows.append(dict(cycle=ty, temp=T_ty, minor_offset=d_ty, unwon_races=int(ue["race_id"].nunique()) if len(ue) else 0,
                         **metrics(te, "p_race"),
                         **{f"{k}_all": v for k, v in metrics(both, "p_race").items() if k in ("Brier", "LogLoss")}))
        preds.append(te)
    ev = pd.DataFrame(rows).set_index("cycle")
    print("\n=== held-out 2018-24, expanding window (sanity check; *_all adds the races whose winner was never polled) ===")
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
    TEMP, OFFSET, full_ll = fit_calibration(seeded([(s, loco_ens(c, FULL, s)) for s in TUNE_SEEDS]))
    print(f"\nproduction softmax temperature = {TEMP}, minor offset = {OFFSET} "
          f"(LOCO log loss on {FULL}: {full_ll:.4f})")
    ens = fit_ensemble(c, FEATS, ENS)
    files = []
    for i, m in enumerate(ens.models):
        fn = "race_win_model_xgb.json" if i == 0 else f"race_win_model_xgb_{i}.json"
        m.save_model(os.path.join(ROOT, "data", fn))
        files.append(fn)
    # gain per feature, averaged over the members (audit M7: the per-candidate notebook's file
    # is not this model's; zero-gain features carry nothing into the explanations)
    imp = pd.Series(np.mean([m.feature_importances_ for m in ens.models], axis=0),
                    index=FEATS).sort_values(ascending=False)
    imp.rename("gain_importance").rename_axis("feature").reset_index().to_csv(
        os.path.join(ROOT, "models", "poll", "race_win_feature_importance.csv"), index=False)
    print(f"features with zero gain: {int((imp == 0).sum())} of {len(imp)}")
    means = ev[["Brier", "LogLoss", "AUC", "race_acc", "Brier_all", "LogLoss_all"]].mean().round(4)
    with open(os.path.join(ROOT, "data", "race_win_model_features.json"), "w") as f:
        json.dump(dict(features=FEATS, model_type="xgbranker", objective="rank:pairwise",
                       ensemble_files=files, xgb_params=ENS, softmax_temp=TEMP, minor_offset=OFFSET,
                       minor_poll_max=MINOR_POLL_MAX,
                       temp_fit=dict(cycles=FULL, scheme="temperature + minor offset fitted on LOCO "
                                     "predictions of the full-field cycles",
                                     tune_cycle_temp=TUNE_TEMP),
                       trained_on_cycles=sorted(int(y) for y in c["year"].unique()),
                       tune=dict(cycles=TUNE_CYCLES, metric="log loss of softmaxed probability",
                                 stage2=[[round(s[0][0], 4), s[1]] for s in stage2]),
                       eval_expanding_window=dict(mean=means.to_dict(),
                                                  per_cycle=ev.round(4).reset_index().to_dict("records")),
                       note="race-level win model: within-race softmax of the averaged ranker scores "
                            "(top-3 settings) is the one probability. Served by src/predict.py."),
                  f, indent=1)
    print(f"\nsaved {files} + race_win_model_features.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
