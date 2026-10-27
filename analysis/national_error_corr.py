# -*- coding: utf-8 -*-
"""How strongly do a cycle's races miss TOGETHER? -> data/seat_sim_params.json (2026-10-27, audit item 4).

    py -X utf8 analysis/national_error_corr.py

The seat-count simulation (src/simulate_seats.py) keeps every race's published win probability and
adds one shared national error per run: a Gaussian copula where race r's Democrat wins when
    sqrt(rho) * g + sqrt(1 - rho) * e_r  >  -Phi^-1(p_r),   g, e_r ~ N(0, 1)
so each race still wins with probability p_r, but races move together through g. rho is fitted
here by maximum likelihood on out-of-sample predictions: every cycle 1998-2024 predicted by the
race-level win model trained on the OTHER cycles (production settings), D-vs-R races only, with
g integrated out per cycle (Gauss-Hermite). rho = 0 would mean races miss independently; the
national poll misses of 2016-2024 should show up as rho > 0.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "models", "poll")):
    _sys.path.insert(0, _p)
from paths import ROOT  # noqa: E402

import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import race_win_model as R  # noqa: E402
from race_level import race_win_probs  # noqa: E402

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)   # probabilists' Hermite: weight exp(-x^2/2)
GH_W = GH_W / GH_W.sum()
RHO_GRID = np.round(np.arange(0.0, 0.61, 0.01), 2)


def cycle_loglik(t, y, rho):
    """log P(outcomes) for one cycle: t = Phi^-1(p_D), y = 1 if the Democrat won."""
    s, r = np.sqrt(rho), np.sqrt(1 - rho)
    # P(D wins | g) = Phi((t + s g) / r)
    z = (t[None, :] + s * GH_X[:, None]) / r
    lp = np.where(y[None, :] == 1, norm.logcdf(z), norm.logcdf(-z)).sum(axis=1)
    m = lp.max()
    return m + np.log((GH_W * np.exp(lp - m)).sum())


def main():
    meta = json.load(open(os.path.join(ROOT, "data", "race_win_model_features.json")))
    FE, P = meta["features"], meta["xgb_params"]
    T, d = meta["softmax_temp"], meta.get("minor_offset", 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        c = R.training_table()
    rows = []
    for ty in sorted(c["year"].unique()):
        tr, te = c[c["year"] != ty], c[c["year"] == ty].copy()
        te["_score"] = R.fit_ensemble(tr, FE, P if isinstance(P, list) else [P]).predict(te[FE])
        te["p"] = race_win_probs(te, "_score", T, d)
        g = te[te["party"].isin(["DEM", "REP"])]
        both = g.groupby("race_id")["party"].transform(lambda s: {"DEM", "REP"} <= set(s))
        g = g[both]
        pd_ = g[g["party"] == "DEM"].groupby("race_id").agg(p=("p", "sum"), y=("won", "max"))
        pr_ = g[g["party"] == "REP"].groupby("race_id")["p"].sum()
        pd_["p2"] = pd_["p"] / (pd_["p"] + pr_.reindex(pd_.index))      # two-party Dem chance
        pd_["year"] = ty
        rows.append(pd_)
    D = pd.concat(rows)
    D["t"] = norm.ppf(D["p2"].clip(1e-4, 1 - 1e-4))
    ll = {rho: sum(cycle_loglik(g["t"].values, g["y"].astype(int).values, rho)
                   for _, g in D.groupby("year")) for rho in RHO_GRID}
    best = max(ll, key=ll.get)
    # profile-likelihood 95% interval (drop of 1.92)
    ci = [r for r in RHO_GRID if ll[r] >= ll[best] - 1.92]
    # per-cycle check: actual Dem wins vs expected
    per = D.groupby("year").agg(races=("y", "size"), dem_won=("y", "sum"), expected=("p2", "sum"))
    per["miss"] = per["dem_won"] - per["expected"]
    print(per.round(1).to_string())
    print(f"\nrho = {best} (95% interval {min(ci)}-{max(ci)}); log lik at 0: {ll[0.0]:.1f}, at best: {ll[best]:.1f}")
    out = dict(rho=float(best), rho_ci95=[float(min(ci)), float(max(ci))], n_races=int(len(D)),
               cycles=sorted(int(y) for y in D["year"].unique()),
               method="Gaussian copula, one shared national factor; MLE on LOCO race-level win "
                      "predictions (production settings), D-vs-R races, g integrated per cycle",
               per_cycle=per.round(2).reset_index().to_dict("records"),
               generated_by="analysis/national_error_corr.py")
    with open(os.path.join(ROOT, "data", "seat_sim_params.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote data/seat_sim_params.json")


if __name__ == "__main__":
    main()
