# -*- coding: utf-8 -*-
"""Shared logic for the poll-volume breakpoint notebook.

Kept as a MODULE (not inline notebook code) so the notebook stays readable and so the
scoring can be re-run from a script without nbconvert. See
analysis/poll_volume_breakpoint.ipynb for the narrative and charts.

The one methodological point that matters here: volume is measured as DISTINCT SURVEYS
(pollster + end_date), not poll ROWS. A single survey of an N-candidate primary produces N
rows, so `n_polls` / `race_total_polls` conflate "well-polled race" with "crowded field" -
scoring on rows put the sharpest break in a different place than scoring on surveys, and the
surveys answer is the one that means what we want it to mean.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import features as F                      # noqa: E402
import features_primary as FP             # noqa: E402


def survey_counts(long_polls):
    """race_id -> number of DISTINCT surveys (pollster + end_date)."""
    return long_polls.groupby("race_id").apply(
        lambda g: g.groupby(["pollster", "end_date"]).ngroups, include_groups=False)


def primary_holdout():
    """Expanding-window held-out predictions for the PRIMARY model (train < test cycle).

    Returns the per-candidate frame with `p` (the model's within-race softmax probability)
    and `n_surveys` attached.
    """
    import primary_model as PM
    from candidate_history import CandidateHistory

    d = pd.read_csv(os.path.join(ROOT, "data", "primary_polls_long.csv"), low_memory=False)
    d = F.prepare_polls(d)
    d["district"] = d["district"].map(F.dist_str)
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    c = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                               hist=CandidateHistory(), bios=FP.load_candidate_bios())
    c["n_surveys"] = c["race_id"].map(survey_counts(d))

    with open(os.path.join(ROOT, "data", "primary_model_features.json")) as f:
        meta = json.load(f)
    feats, params, temp = meta["features"], meta["xgb_params"], meta["softmax_temp"]

    out = []
    for ty in sorted(y for y in c["year"].unique() if y >= 2022):
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(tr) or not len(te):
            continue
        te["p"] = PM.eval_fold(tr, te, feats, params, temp=temp)
        out.append(te)
    return pd.concat(out, ignore_index=True)


def general_holdout():
    """Expanding-window held-out predictions for the GENERAL win model."""
    import xgboost as xgb
    from cycles import CYCLES, natl_env
    from macro_features import build_macro

    d = pd.read_csv(os.path.join(ROOT, "polls_long_with_results.csv"), low_memory=False)
    d = d[d["has_result"] == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    bias = F.compute_bias_priors(d)
    c = F.build_candidate_table(d, build_macro(), natl_env(), funds,
                                house_train_years=CYCLES, fec=fec, bias_priors=bias,
                                candidate_bios=F.load_candidate_bios())
    c["n_surveys"] = c["race_id"].map(survey_counts(d))

    with open(os.path.join(ROOT, "data", "model_features.json")) as f:
        meta = json.load(f)
    feats = meta["features"]
    params = {k: v for k, v in meta["xgb_params"].items()
              if k not in ("random_state", "n_jobs")}

    out = []
    for ty in [2018, 2020, 2022, 2024]:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        if not len(tr) or not len(te):
            continue
        m = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        m.fit(tr[feats], tr["won"].astype(int))
        te["p"] = m.predict_proba(te[feats])[:, 1]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def leaders(te):
    """One row per race: the candidate the model ranked first (the race's actual call)."""
    return te.loc[te.groupby("race_id")["p"].idxmax()].copy()


def calibration_by_bucket(lead, buckets=((1, 1), (2, 2), (3, 3), (4, 5), (6, 10),
                                         (11, 20), (21, 10**6)), n_boot=2000, seed=0):
    """Predicted-vs-actual for the model's called winner, split by survey count.

    `gap` = mean predicted - actual win rate. POSITIVE = overconfident. The bootstrap CI is
    what keeps this honest: several buckets hold <15 races, where a 15-point gap is not
    distinguishable from noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in buckets:
        s = lead[(lead["n_surveys"] >= lo) & (lead["n_surveys"] <= hi)]
        if len(s) < 3:
            continue
        gaps = [float(b["p"].mean() - b["won"].mean())
                for b in (s.sample(len(s), replace=True, random_state=int(rng.integers(1 << 31)))
                          for _ in range(n_boot))]
        lo_ci, hi_ci = np.percentile(gaps, [5, 95])
        rows.append(dict(bucket=(f"{lo}" if lo == hi else
                                 (f"{lo}+" if hi > 10**5 else f"{lo}-{hi}")),
                         n_races=len(s), pred=s["p"].mean(), actual=s["won"].mean(),
                         gap=s["p"].mean() - s["won"].mean(), ci_lo=lo_ci, ci_hi=hi_ci))
    return pd.DataFrame(rows).set_index("bucket")


def permutation_test(lead, cut, n_perm=5000, seed=0):
    """Is the thin-vs-thick calibration difference at `cut` real, or a shuffling artifact?

    H0: survey count carries no information about the calibration gap. Shuffle the thin/thick
    labels and rebuild the difference n_perm times; p = share of shuffles at least as extreme
    as observed. This is the guard against reading a breakpoint off <=15-race buckets.
    """
    rng = np.random.default_rng(seed)
    thin = (lead["n_surveys"] <= cut).values
    p_, w_ = lead["p"].values, lead["won"].values

    def diff(mask):
        a, b = mask, ~mask
        if a.sum() < 2 or b.sum() < 2:
            return np.nan
        return ((p_[a].mean() - w_[a].mean()) - (p_[b].mean() - w_[b].mean()))

    obs = diff(thin)
    null = np.array([diff(rng.permutation(thin)) for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    return dict(cut=cut, observed=float(obs), n_thin=int(thin.sum()),
                p_value=float((np.abs(null) >= abs(obs)).mean()))
