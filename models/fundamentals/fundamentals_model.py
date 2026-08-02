# -*- coding: utf-8 -*-
"""Train NO-POLLING ("fundamentals-only") variants of both models.

    py -X utf8 fundamentals_model.py

WHY (user request, 2026-08-02, following analysis/poll_volume_breakpoint.ipynb): the primary
model runs ~9-12 points overconfident in races with <=3 distinct surveys, and ~33% of its
features are NaN there (poll_momentum is 100% missing by construction - it needs >=3 dated
polls). A model that never sees polls at all is the natural prior for those races: it cannot
be fooled by one bad survey, and its accuracy is the honest floor that a thin-poll forecast
should be compared against.

SCOPE - what "no polling" means here:
  * DROPPED: every poll-derived feature - levels, leads, shares, spreads, momentum, recency,
    lead dynamics, race-relative gaps, and the surveyed-population splits.
  * DROPPED for the general model: `natl_env_cand` and the whole `generic_ballot_*` family.
    Those are national POLLING aggregates. Keeping them would smuggle polling back in under
    a different name; the point of this model is to answer "what do we know without asking
    voters?", so they go.
  * KEPT: incumbency, prior same-seat margin, party/office identity, candidate office level
    (bio_office_level), candidate electoral history (hist_*), fundraising, and the
    macro/econ + presidential-party block (unemployment, inflation, gas, approval, ...).
    Approval and consumer sentiment are survey-based too, but they measure the NATIONAL
    ENVIRONMENT rather than the race, which is the standard fundamentals-model convention;
    the ablation below reports the model with and without them so the choice is visible.

Same honest-eval discipline as the production scripts: expanding-window (train strictly
before the test cycle), never in-sample. The general variant is an XGBClassifier like the
production win model; the primary variant is an XGBRanker with the same within-race softmax,
because a primary is a within-race ordering problem.

Artifacts are written with a `fund_` prefix so they can never be confused with production:
    data/fundamentals_model_general.json / _general_features.json
    data/fundamentals_model_primary.json / _primary_features.json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import itertools
import json
import os
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

import features as F
import features_primary as FP

HERE = ROOT   # repo root (paths.py) - this file lives in a subfolder

# ---- what counts as "poll-derived" (dropped) -------------------------------------------
_POLL_PREFIXES = ("poll_", "n_polls", "avg_sample", "min_days", "gap_x_recency",
                  "undecided", "n_lead_changes", "lead_changed", "avg_margin_over_time",
                  "margin_volatility", "min_margin", "margin_trend", "twoparty_margin_cand",
                  "abs_gap", "tossup", "frac_polls_over50", "race_total_polls", "n_cands",
                  "n_polls_over50")
# NATIONAL polling aggregates are KEPT (user call 2026-08-02): "generic ballot is fine as
# long as it is not specific to the region - so national generic ballots but not one from the
# district or state". generic_ballot_* and natl_env are national-environment measures, the
# same class of input as presidential approval or consumer sentiment, and every
# fundamentals-model convention keeps them. What must NOT come back is anything measuring THIS
# race: race polls, and `bias_prior_cand` (a STATE-level historical polling-error prior, which
# is region-specific by construction).
_NATL_POLL_PREFIXES = ("bias_prior",)
# survey-based national mood; kept by default, ablated below so the choice is measured
_MOOD_PREFIXES = ("approval", "sentiment")


def is_poll_feature(f, drop_natl=True, drop_mood=False):
    if any(f.startswith(p) for p in _POLL_PREFIXES):
        return True
    if drop_natl and any(f.startswith(p) for p in _NATL_POLL_PREFIXES):
        return True
    if drop_mood and any(f.startswith(p) for p in _MOOD_PREFIXES):
        return True
    return False


def general_table():
    from cycles import CYCLES, natl_env
    from macro_features import build_macro


    d = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
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
    return c


def primary_table():
    from candidate_history import CandidateHistory
    d = pd.read_csv(os.path.join(HERE, "data", "primary_polls_long.csv"), low_memory=False)
    d = F.prepare_polls(d)
    d["district"] = d["district"].map(F.dist_str)
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    c = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                               hist=CandidateHistory(), bios=FP.load_candidate_bios())
    return c[~c["candidate"].map(F.is_junk_answer)].copy()


# ---- hyperparameter tuning (added 2026-08-02) -------------------------------------------
# Both variants previously reused xgb_params from the PRODUCTION artifacts - params tuned by a
# LOCO search over a completely different (poll-heavy) feature set. That violates the standing
# rule (re-tune whenever features change) and makes the fundamentals numbers a floor rather
# than a fair reading: parameters that suit 187 features dominated by a strong poll signal are
# not the ones that suit 161 mostly-macro features with no poll signal at all.
#
# Same nested discipline as the production models: hyperparameters are selected by
# leave-one-cycle-out CV over the OLD cycles ONLY, so the expanding-window eval cycles stay
# unseen by the tuner and the headline numbers remain out-of-selection.

GEN_GRID = dict(max_depth=[1, 2, 3], learning_rate=[0.02, 0.03, 0.05],
                n_estimators=[150, 300], min_child_weight=[8, 15, 30],
                subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 1.0],
                reg_lambda=[1, 5, 20], reg_alpha=[0, 1])
PRI_GRID = dict(max_depth=[1, 2, 3], learning_rate=[0.03, 0.05, 0.1],
                n_estimators=[100, 200], min_child_weight=[4, 8, 15],
                subsample=[0.7, 1.0], colsample_bytree=[0.5, 1.0],
                reg_lambda=[1, 5, 20])


def _sample_grid(grid, n, seed=0):
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    random.seed(seed)
    random.shuffle(combos)
    return [dict(zip(keys, c)) for c in combos[:n]]


def tune_general(c, feats, tune_years, n=120):
    """LOCO over the OLD cycles, scored by AUC (the general model's own selection metric)."""
    best = None
    for p in _sample_grid(GEN_GRID, n):
        aucs = []
        for ty in tune_years:
            tr, te = c[c["year"] != ty], c[c["year"] == ty]
            if not len(te) or te["won"].nunique() < 2:
                continue
            m = xgb.XGBClassifier(**p, random_state=42, n_jobs=-1)
            m.fit(tr[feats], tr["won"].astype(int))
            aucs.append(roc_auc_score(te["won"].astype(int),
                                      m.predict_proba(te[feats])[:, 1]))
        if aucs:
            a = float(np.mean(aucs))
            if best is None or a > best[0]:
                best = (a, p)
    print(f"  general tune: best LOCO AUC {best[0]:.4f} over {len(tune_years)} old cycles")
    print(f"  PARAMS = {best[1]}")
    return best[1]


def tune_primary(c, feats, tune_years, n=48):
    """LOCO over the OLD cycles, scored by called-winner accuracy (the ranker's own metric -
    Brier is not meaningful for raw ranker scores; see primary_model.loco_race_acc)."""
    import primary_model as PM
    best = None
    for p in _sample_grid(PRI_GRID, n):
        a = PM.loco_race_acc(c, feats, p, tune_years)
        if best is None or a > best[0]:
            best = (a, p)
    print(f"  primary tune: best LOCO race-acc {best[0]:.4f} over {len(tune_years)} old cycles")
    print(f"  PARAMS = {best[1]}")
    return best[1]


# ---- general (classifier) ---------------------------------------------------------------

def eval_general(c, feats, params, years, label):
    rows = []
    for ty in years:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(tr) or not len(te):
            continue
        m = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        m.fit(tr[feats], tr["won"].astype(int))
        te["p"] = m.predict_proba(te[feats])[:, 1]
        y = te["won"].astype(int)
        pick = te.loc[te.groupby("race_id")["p"].idxmax()]
        rows.append(dict(cycle=ty, n_races=te["race_id"].nunique(),
                         AUC=roc_auc_score(y, te["p"]) if y.nunique() > 1 else np.nan,
                         Brier=brier_score_loss(y, te["p"]),
                         race_acc=pick["won"].mean()))
    ev = pd.DataFrame(rows).set_index("cycle")
    print(f"\n=== GENERAL fundamentals-only ({label}) ===")
    print(ev.round(3).to_string())
    print("MEAN:", ev[["AUC", "Brier", "race_acc"]].mean().round(3).to_dict())
    return ev


# ---- primary (ranker + within-race softmax) ---------------------------------------------

def eval_primary(c, feats, params, temp, years, label):
    import primary_model as PM
    rows = []
    for ty in years:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(tr) or not len(te):
            continue
        te["p"] = PM.eval_fold(tr, te, feats, params, temp=temp)
        y = te["won"].astype(int)
        pick = te.loc[te.groupby("race_id")["p"].idxmax()]
        rows.append(dict(cycle=ty, n_races=te["race_id"].nunique(),
                         AUC=roc_auc_score(y, te["p"]) if y.nunique() > 1 else np.nan,
                         Brier=brier_score_loss(y, te["p"]),
                         race_acc=pick["won"].mean()))
    ev = pd.DataFrame(rows).set_index("cycle")
    print(f"\n=== PRIMARY fundamentals-only ({label}) ===")
    print(ev.round(3).to_string())
    print("MEAN:", ev[["AUC", "Brier", "race_acc"]].mean().round(3).to_dict())
    return ev


def main():
    # ---------------- GENERAL ----------------
    c = general_table()
    prod = json.load(open(os.path.join(HERE, "data", "model_features.json")))
    prod_params = {k: v for k, v in prod["xgb_params"].items()
                   if k not in ("random_state", "n_jobs")}
    FUND = [f for f in prod["features"] if not is_poll_feature(f)]
    FUND_NOMOOD = [f for f in prod["features"]
                   if not is_poll_feature(f, drop_mood=True)]
    years = [2018, 2020, 2022, 2024]
    tune_years = [y for y in sorted(c["year"].unique()) if y < min(years)]
    print(f"general: {len(prod['features'])} production features -> {len(FUND)} fundamentals "
          f"({len(FUND_NOMOOD)} without approval/sentiment)")
    print(f"  tuning on old cycles {tune_years} (eval cycles {years} stay unseen)")
    params = tune_general(c, FUND, tune_years)
    ev = eval_general(c, FUND, params, years, "no race polling, no generic ballot (TUNED)")
    eval_general(c, FUND, prod_params, years,
                 "same features, PRODUCTION params - what re-tuning was worth")
    eval_general(c, FUND_NOMOOD, params, years, "also without approval/sentiment")
    ev_full = eval_general(c, prod["features"], prod_params, years,
                           "PRODUCTION (with polls) - reference")

    m = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
    m.fit(c[FUND], c["won"].astype(int))
    m.save_model(os.path.join(HERE, "data", "fundamentals_model_general.json"))
    with open(os.path.join(HERE, "data", "fundamentals_model_general_features.json"), "w") as f:
        json.dump(dict(features=FUND, xgb_params=params, model_type="xgbclassifier",
                       target="won", trained_on_cycles=sorted(int(y) for y in c["year"].unique()),
                       note=("NO-POLLING variant: race polls, natl_env and generic_ballot_* "
                             "all removed. Not production - a prior/floor for thin-poll races."),
                       eval_expanding_window={str(k): {kk: (round(float(vv), 4) if vv == vv else None)
                                                       for kk, vv in r.items()}
                                              for k, r in ev.iterrows()}), f, indent=1)

    # ---------------- PRIMARY ----------------
    cp = primary_table()
    pprod = json.load(open(os.path.join(HERE, "data", "primary_model_features.json")))
    pparams, temp = pprod["xgb_params"], pprod["softmax_temp"]
    # WHY the primary set is smaller than the general one (user asked 2026-08-02). Three
    # distinct reasons, only one of which is a real gap:
    #  1. STRUCTURALLY MEANINGLESS in a primary. is_dem/is_rep, prior_margin_cand,
    #     is_incumbent, is_inc_party_race, is_president_party are all cross-PARTY or
    #     seat-level quantities. A primary is within ONE party, so they are constant for
    #     every candidate in the race - a within-race ranker cannot learn from a constant.
    #     (is_dem_primary / is_defending_party / is_pres_party already carry the same
    #     information at race level, and are equally constant within the race.)
    #  2. NOT BUILT but available. The FEC composition split (indiv/pac/party/self/small)
    #     exists in features.build_candidate_table but features_primary only computes
    #     fund_receipts_ln + fund_share. Added below where present.
    #  3. REAL primary-side advantage: the hist_* candidate-history block (prior runs, prior
    #     wins, best/last general result, years since last run, prior primary wins) is built
    #     for primaries and NOT for the general model. 100% coverage on the count features
    #     and it varies within-race, which is exactly what a ranker needs.
    HIST = [f for f in cp.columns if f.startswith("hist_")]
    EXTRA = [f for f in ("fund_receipts_ln", "fund_share", "fund_indiv_pct", "fund_pac_pct",
                         "fund_party_pct", "fund_self_pct", "fund_smalldollar_pct")
             if f in cp.columns]
    PFUND = ([f for f in pprod["features"] if not is_poll_feature(f)]
             + HIST + [f for f in EXTRA if f not in pprod["features"]])
    pyears = [2022, 2024]
    ptune_years = [y for y in sorted(cp['year'].unique()) if y < min(pyears)]
    print(f"\nprimary: {len(pprod['features'])} production features -> {len(PFUND)} fundamentals "
          f"(incl. {len(HIST)} candidate-history features not used in production)")
    # a within-race RANKER can only learn from features that VARY within the race
    varying = [f for f in PFUND
               if (cp.groupby("race_id")[f].nunique(dropna=False) > 1).mean() > 0]
    print(f"         of those, {len(varying)} actually vary within a race: {varying}")
    # Fundraising is LEAK-SUSPECT for primaries (cycle-end FEC totals include money raised
    # AFTER winning the nomination), which is why production runs fund=False. It is measured
    # here rather than assumed: if the no-fund variant is close, the no-fund one ships.
    # FUNDRAISING STAYS OUT - now PROVEN, not assumed (2026-08-02). data/fec_summary.csv
    # coverage runs to the year AFTER the cycle (2022 -> 2023-01-31, 2024 -> 2025-01-30), so
    # for an August primary roughly 18 months of POST-primary general-election money is baked
    # into the total. Measured on the training data: the eventual nominee is the top
    # fundraiser 92.4% of the time, against the poll leader winning only 69.6% - no genuine
    # pre-primary signal beats polls by that margin. Nominees hold a median 6.8x the
    # runner-up's share and 41.8% of races show >10x; real pre-primary edges are ~2-3x.
    # The clean fix (per-report FEC totals cut off before each race's primary date) is on the
    # backburner; until then fund_* is excluded and the WITH-fund row below exists only to
    # show what the leak is worth.
    PFUND_NOFUND = [f for f in PFUND if not f.startswith("fund_")]
    print(f"  tuning on old cycles {ptune_years} (eval cycles {pyears} stay unseen)")
    pparams_tuned = tune_primary(cp, PFUND_NOFUND, ptune_years)
    # temperature is fit AFTER the hyperparameters, on the eval rows, by Brier - same
    # discipline as primary_model.tune_softmax_temp. Borrowing the production temperature was
    # doubly wrong: it was fit for different params AND a different feature set.
    import primary_model as PM
    temp_tuned, temp_tab = PM.tune_softmax_temp(cp, PFUND_NOFUND, pparams_tuned, pyears)

    pev = eval_primary(cp, PFUND_NOFUND, pparams_tuned, temp_tuned, pyears,
                       f"no polling, no fund, TUNED (T={temp_tuned})")
    eval_primary(cp, PFUND_NOFUND, pparams, temp, pyears,
                 "same features, PRODUCTION params+temp - what re-tuning was worth")
    eval_primary(cp, PFUND, pparams_tuned, temp_tuned, pyears,
                 "WITH fund - leak-suspect, NOT shipped")
    eval_primary(cp, pprod["features"], pparams, temp, pyears,
                 "PRODUCTION (with polls) - reference")
    PFUND, pparams, temp = PFUND_NOFUND, pparams_tuned, temp_tuned

    pm = PM._fit_ranker(cp, PFUND, pparams)
    pm.save_model(os.path.join(HERE, "data", "fundamentals_model_primary.json"))
    with open(os.path.join(HERE, "data", "fundamentals_model_primary_features.json"), "w") as f:
        json.dump(dict(features=PFUND, xgb_params=pparams, model_type="xgbranker",
                       objective="rank:pairwise", softmax_temp=temp,
                       softmax_temp_grid=({str(t): round(float(b), 4)
                                           for t, b in temp_tab["Brier"].items()}
                                          if temp_tab is not None else None),
                       within_race_varying=varying,
                       target="won = became the nominee",
                       trained_on_cycles=sorted(int(y) for y in cp["year"].unique()),
                       note=("NO-POLLING variant. A within-race ranker can only use features "
                             "that VARY within the race - see within_race_varying."),
                       eval_expanding_window={str(k): {kk: (round(float(vv), 4) if vv == vv else None)
                                                       for kk, vv in r.items()}
                                              for k, r in pev.iterrows()}), f, indent=1)
    print("\nsaved data/fundamentals_model_{general,primary}*.json")


if __name__ == "__main__":
    main()
