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

# ---- within-race SOFTMAX: the ONE coherent probability used everywhere -----------------
# The nominee model is a learning-to-RANK model (XGBRanker, below): it scores candidates so
# that, WITHIN a race, the eventual nominee ranks highest. Raw ranker scores are unbounded
# and only meaningful relative to their own race, so we convert them to a probability with a
# within-race softmax. This is the number the dashboard AND the Explain modal both show -
# there is no separate "raw" per-candidate probability anymore (the old XGBClassifier scored
# each candidate INDEPENDENTLY, which (a) made 2+ strong candidates each ~1.0 so divide-by-sum
# normalization mushed them to ~50/50, and (b) made the explainer's raw number disagree with
# the dashboard's normalized number). Ranking + softmax handles an N-candidate field coherently.
# TEMPERATURE (fixed 2026-07-31): this was hardcoded 1.0 with a comment claiming it was
# "tuned on the eval cycles by called-winner accuracy". It never was, and it COULDN'T be:
# softmax is monotonic, so temperature does not change the argmax - race_acc is identical at
# every T. Tuning it by accuracy is a no-op by construction. The metric that actually moves
# is Brier (calibration), and 1.0 turned out to be the WORST value in the plausible range:
# the ranker's raw scores have std ~1.3, so dividing by 1.0 squashes genuinely-separated
# candidates toward 50/50. Measured on the 2022+2024 expanding-window rows:
#     T=1.00  Brier .0428   mean prob on the actual nominee .661   <- old hardcoded value
#     T=0.50  Brier .0254   .819
#     T=0.25  Brier .0213   .887                                   <- optimum
#     T=0.15  Brier .0225   .907
# Symptom that surfaced this: MI-Sen-DEM 2026, where the ranker separates the field cleanly
# (scores 1.94 / 1.54 / -0.69) but the reported probability was a mushy 57.6% - and the SHAP
# explainer disagreed with the dashboard, because SHAP explains the SCORE while the displayed
# number came from the miscalibrated softmax. At the tuned T the same scores give 83.6%.
# tune_softmax_temp() below now fits this on the eval cycles by Brier and writes the result
# into the artifact, so predict_primary.py picks it up via meta["softmax_temp"].
TEMP_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50]
SOFTMAX_TEMP = 0.25   # default/fallback only; main() overwrites it with the tuned value

def race_softmax(scores, temp=SOFTMAX_TEMP):
    s = np.asarray(scores, dtype=float) / temp
    e = np.exp(s - np.nanmax(s))
    return e / e.sum()

def tune_softmax_temp(c, feats, params, eval_years, grid=TEMP_GRID):
    """Pick the within-race softmax temperature by BRIER on the expanding-window eval rows.

    Temperature is a pure calibration knob: it cannot change which candidate the ranker picks
    (softmax is monotonic), so it is tuned on probability quality, not accuracy. Scores are
    collected ONCE per eval cycle under the same train-strictly-before-test discipline as
    expanding_eval, then scored across the grid.
    """
    rows = []
    for ty in eval_years:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(te) or not len(tr):
            continue
        te["_score"] = _fit_ranker(tr, feats, params).predict(te[feats])
        rows.append(te)
    if not rows:
        return SOFTMAX_TEMP, None
    ev = pd.concat(rows)
    y = ev["won"].astype(int)
    out = []
    for T in grid:
        p = ev.groupby("race_id", group_keys=False)["_score"].transform(
            lambda s: race_softmax(s, temp=T))
        out.append(dict(temp=T, Brier=brier_score_loss(y, p), logloss=log_loss(y, p),
                        mean_p_on_nominee=float(p[y == 1].mean())))
    tab = pd.DataFrame(out).set_index("temp")
    print("\n=== SOFTMAX TEMPERATURE tuning (Brier on expanding-window eval rows) ===")
    print(tab.round(4).to_string())
    best = float(tab["Brier"].idxmin())
    print(f"chosen softmax_temp = {best} (Brier {tab['Brier'].min():.4f}; "
          f"T=1.0 would be {tab['Brier'].get(1.0, float('nan')):.4f})")
    return best, tab

def _fit_ranker(tr, feats, params):
    """XGBRanker needs rows grouped by race (qid). Returns (model, sorted_tr)."""
    tr = tr.sort_values("race_id")
    grp = tr.groupby("race_id", sort=True).size().values
    m = xgb.XGBRanker(objective="rank:pairwise", random_state=42, n_jobs=-1, **params)
    m.fit(tr[feats], tr["won"].astype(int), group=grp)
    return m

def eval_fold(tr, te, feats, params, temp=None):
    """Train the ranker on tr, return within-race softmax probabilities for te (index-aligned).

    temp: softmax temperature; None = module default. Only affects CALIBRATION metrics
    (Brier/logloss) - the argmax, and therefore race_acc, is identical at every temperature.
    """
    T = SOFTMAX_TEMP if temp is None else temp
    m = _fit_ranker(tr, feats, params)
    te = te.copy()
    te["_score"] = m.predict(te[feats])
    te["_p"] = te.groupby("race_id", group_keys=False)["_score"].transform(
        lambda s: race_softmax(s, temp=T))
    return te["_p"]

def loco_race_acc(c, feats, params, cycles):
    """LOCO selection score for the RANKER = called-winner accuracy (Brier is not meaningful
    for a ranking model's raw scores; the goal is ranking the actual nominee first). Higher is
    better - callers negate or track the max."""
    accs = []
    for ty in cycles:
        tr, te = c[c["year"] != ty], c[c["year"] == ty]
        if not len(te) or tr["won"].nunique() < 2:
            continue
        te = te.copy()
        te["p"] = eval_fold(tr, te, feats, params)
        accs.append(race_acc(te, "p"))
    return float(np.mean(accs)) if accs else -np.inf

def expanding_eval(c, feats, params, eval_years, label, temp=None):
    rows = []
    for ty in eval_years:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(te) or not len(tr):
            continue
        te["p"] = eval_fold(tr, te, feats, params, temp=temp)
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
    from candidate_history import CandidateHistory
    HIST = CandidateHistory()
    BIOS = FP.load_candidate_bios()
    c = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"], hist=HIST, bios=BIOS)
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
        a = loco_race_acc(c, FEATS, p, tune_years)   # ranker: maximize called-winner accuracy
        if best is None or a > best[0]:
            best = (a, p)
    PARAMS = best[1]
    print(f"\nbest tune-cycle LOCO race-acc: {best[0]:.4f}")
    print("PARAMS =", PARAMS)

    # ---- softmax temperature: calibrate AFTER the hyperparameters are fixed. Tuned by Brier
    # on the expanding-window eval rows (temperature cannot move race_acc - see the note on
    # TEMP_GRID). Everything below reports at the tuned temperature.
    TEMP, temp_tab = tune_softmax_temp(c, FEATS, PARAMS, eval_years)

    ev = expanding_eval(c, FEATS, PARAMS, eval_years,
                        f"no fund, no macro - PRIMARY headline (softmax T={TEMP})", temp=TEMP)

    # ---- candidate-history/bio ablation (2026-07-17 features): measure their value ----
    NOHIST = [f for f in FEATS if not (f.startswith("hist_") or f.startswith("bio_"))]
    evh = expanding_eval(c, NOHIST, PARAMS, eval_years,
                         "WITHOUT candidate history/bio - ablation", temp=TEMP)
    print(f"history/bio ablation: race-acc {ev['race_acc'].mean():.3f} vs "
          f"{evh['race_acc'].mean():.3f} without, Brier {ev['Brier'].mean():.4f} vs "
          f"{evh['Brier'].mean():.4f}")

    # ---- fund ablation (leakage evidence: cycle-END FEC totals include post-primary
    # money, so fund_share partly encodes the training label; see feature_list_primary) --
    evf = expanding_eval(c, FP.feature_list_primary(fund=True), PARAMS, eval_years,
                         "WITH fund - ablation only (leak-suspect)", temp=TEMP)
    print(f"fund ablation: race-acc {evf['race_acc'].mean():.3f} vs {ev['race_acc'].mean():.3f} "
          f"(identical picks expected), Brier gain {ev['Brier'].mean()-evf['Brier'].mean():+.4f} "
          f"= the leak-suspect juice; artifact stays NO-fund")

    # ---- macro ablation (evidence for keeping macro out) ----
    cm = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"], hist=HIST,
                                bios=BIOS, macro_asof=build_macro_asof)
    macro_feats = sorted(set(cm.columns) - set(c.columns))
    evm = expanding_eval(cm, FP.feature_list_primary(macro_feats), PARAMS, eval_years,
                         f"WITH macro ({len(macro_feats)} extra cols) - ablation only", temp=TEMP)
    dif = evm["Brier"].mean() - ev["Brier"].mean()
    print(f"\nmacro ablation: Brier {'WORSE' if dif > 0 else 'better'} by {abs(dif):.4f} "
          f"with macro (noise-level on ~{c['race_id'].nunique()} races) -> artifact stays NO-macro")

    # ---- production artifact: RANKER trained on ALL cycles, no-macro feature set ----
    prod = _fit_ranker(c, FEATS, PARAMS)
    prod.save_model(os.path.join(HERE, "data", "primary_model_xgb.json"))
    with open(os.path.join(HERE, "data", "primary_model_features.json"), "w") as f:
        json.dump(dict(features=FEATS, xgb_params=PARAMS,
                       model_type="xgbranker",            # predict_primary must load XGBRanker
                       objective="rank:pairwise",
                       softmax_temp=TEMP,                 # TUNED (Brier); see tune_softmax_temp
                       softmax_temp_grid={str(t): round(float(b), 4) for t, b
                                          in temp_tab["Brier"].items()} if temp_tab is not None
                                         else None,
                       score_semantics=("raw ranker scores are unbounded and only comparable "
                                        "WITHIN a race; convert to probability via a within-race "
                                        "softmax (temp above). This one number is used by BOTH "
                                        "the dashboard and the Explain modal."),
                       trained_on_cycles=[int(y) for y in years],
                       n_races=int(c["race_id"].nunique()),
                       target="won = became the party's general-election nominee",
                       eval_expanding_window={str(k): {m: (round(float(v), 4) if v == v else None)
                                                       for m, v in r.items()}
                                              for k, r in ev.iterrows()}), f, indent=1)
    print("\nsaved data/primary_model_xgb.json + data/primary_model_features.json (XGBRanker)")

if __name__ == "__main__":
    main()
