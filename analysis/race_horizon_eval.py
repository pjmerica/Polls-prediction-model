# -*- coding: utf-8 -*-
"""Calibration of the PUBLISHED race-level win odds by horizon (2026-10-27, audit item 8)
-> docs/HORIZON_CALIBRATION.md

    py -X utf8 analysis/race_horizon_eval.py

Each held-out cycle 2018-24 is predicted the way the live model would have seen it H days before
the election: only that cycle's polls that ended at least H days out (H = 0, 10, 35, 60). The model
trains on every earlier cycle's full history (as production does), and the temperature and the
minor-candidate offset come from the earlier full-field cycles' H = 0 predictions (as in
models/poll/race_win_model.py). Races with no poll H days out are left out at that horizon.
Sanity report: the published number's calibration by confidence band, not a model comparison.
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

import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

import features as F  # noqa: E402
import race_win_model as R  # noqa: E402
from cycles import CYCLES, EVAL_CYCLES, natl_env  # noqa: E402
from macro_features import build_macro  # noqa: E402
from race_level import race_win_probs  # noqa: E402

HORIZONS = (0, 10, 35, 60)
BINS = [0, .02, .1, .3, .5, .7, .9, .98, 1.0]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def upset_tail(q, k):
    """P(at least k upsets) when favourite i loses with probability q[i] (Poisson-binomial)."""
    dist = [1.0]
    for qi in q:
        dist = [a * (1 - qi) + b * qi for a, b in zip(dist + [0.0], [0.0] + dist)]
    return float(sum(dist[k:]))


def main():
    meta = json.load(open(os.path.join(ROOT, "data", "race_win_model_features.json")))
    FE, P = meta["features"], meta["xgb_params"]
    with contextlib.redirect_stdout(io.StringIO()):
        d = R.training_polls()
        bias = F.compute_bias_priors(d)            # from full history: what was known before Nov
        base = dict(macro=build_macro(), natl_env_map=natl_env(), funds=F.load_fundamentals(),
                    house_train_years=CYCLES, fec=F.load_fec(extended=True), bias_priors=bias,
                    candidate_bios=F.load_candidate_bios(), primary_results=F.load_primary_results())
        full = F.build_candidate_table(d, **base)
    models = {ty: R.fit_ensemble(full[full["year"] < ty], FE, P if isinstance(P, list) else [P])
              for ty in EVAL_CYCLES}
    # calibration per test cycle from earlier full-field cycles' H = 0 predictions
    h0, cal = [], {}
    for ty in EVAL_CYCLES:
        te = full[full["year"] == ty].copy()
        te["_score"] = models[ty].predict(te[FE])
        cal[ty] = R.fit_calibration(pd.concat(h0))[:2] if h0 else (meta["temp_fit"]["tune_cycle_temp"], 0.0)
        h0.append(te)
    rows, calib, favs = [], [], []
    for H in HORIZONS:
        dh = d if H == 0 else d[~d["year"].isin(EVAL_CYCLES) | (d["days_to_elec"] >= H)]
        with contextlib.redirect_stdout(io.StringIO()):
            tab = full if H == 0 else F.build_candidate_table(dh, **base)
        for ty in EVAL_CYCLES:
            te = tab[tab["year"] == ty].copy()
            if not len(te):
                continue
            te["_score"] = models[ty].predict(te[FE])
            te["p"] = race_win_probs(te, "_score", *cal[ty])
            y = te["won"].astype(int)
            lead = te.loc[te.groupby("race_id")["p"].idxmax()]
            rows.append(dict(H=H, cycle=ty, races=te["race_id"].nunique(),
                             Brier=brier_score_loss(y, te["p"]),
                             LogLoss=log_loss(y, te["p"].clip(1e-6, 1 - 1e-6)),
                             race_acc=lead["won"].mean(), fav_prob=lead["p"].mean()))
            calib.append(te[["p", "won", "office"]].assign(H=H))
            favs.append(lead[["p", "won"]].assign(H=H, cycle=ty))
    r = pd.DataFrame(rows)
    summ = r.groupby("H").agg(races=("races", "sum"), Brier=("Brier", "mean"), LogLoss=("LogLoss", "mean"),
                              race_acc=("race_acc", "mean"), fav_prob=("fav_prob", "mean")).round(3)
    c = pd.concat(calib)
    c["band"] = pd.cut(c["p"], BINS)
    cal_t = (c.groupby(["H", "band"], observed=True).agg(n=("won", "size"), predicted=("p", "mean"),
                                                         won=("won", "mean")).round(3).reset_index())
    cal_t[["lo", "hi"]] = [wilson(round(w * n), n) for w, n in zip(cal_t["won"], cal_t["n"])]
    print(summ.to_string())
    print(cal_t.to_string(index=False))
    # audit M8 (2026-10-27): ~50 statewide races a cycle, so pooled numbers can hide a bad cycle and
    # a band's win rate is only known to within its interval; and the confident favourites (57 of
    # them at 95%+ on 10-24) are judged on expected vs actual upsets, cycle by cycle
    fv = pd.concat(favs)
    fv["lost"] = 1 - fv["won"]

    lines = ["# Calibration of the published win odds by horizon", "",
             "Built by `analysis/race_horizon_eval.py` (audit item 8). Held-out 2018-24: each cycle is "
             "predicted with only the polls that ended at least H days before the election, by the "
             "race-level model trained on earlier cycles, with the temperature and minor-candidate offset "
             "fitted on earlier cycles. A forecast H days out is calibrated when, in each band, the "
             "predicted and actual win rates match.", "",
             "## Summary by horizon (mean over 2018, 2020, 2022, 2024)", "",
             "| Days out | Races | Brier | Log loss | Favourite won | Favourite's average chance |",
             "|---|---|---|---|---|---|"]
    for H, s in summ.iterrows():
        lines.append(f"| {H} | {int(s.races)} | {s.Brier:.3f} | {s.LogLoss:.3f} | {s.race_acc:.1%} | {s.fav_prob:.1%} |")
    lines += ["", "## By cycle, 10 days out (the horizon closest to the last week of the campaign)", "",
              "| Cycle | Races | Brier | Favourite won | Favourite's average chance | Upsets expected | Upsets |",
              "|---|---|---|---|---|---|---|"]
    for t in r[r["H"] == 10].itertuples():
        f = fv[(fv["H"] == 10) & (fv["cycle"] == t.cycle)]
        lines.append(f"| {t.cycle} | {t.races} | {t.Brier:.3f} | {t.race_acc:.1%} | {t.fav_prob:.1%} | "
                     f"{(1 - f['p']).sum():.1f} | {int(f['lost'].sum())} |")
    lines += ["", "## Confident favourites: expected vs actual upsets", "",
              "Each race's favourite, by its published chance. \"Chance of this many upsets or more\" is exact "
              "(Poisson-binomial over the favourites' own odds): a small number means the favourites lost more "
              "often than their odds said.", "",
              "| Days out | Favourite's chance | Favourites | Upsets expected | Upsets | Chance of this many upsets or more |",
              "|---|---|---|---|---|---|"]
    for H in HORIZONS:
        for lo_, hi_ in ((0.95, 1.01), (0.90, 0.95), (0.75, 0.90)):
            f = fv[(fv["H"] == H) & (fv["p"] >= lo_) & (fv["p"] < hi_)]
            if not len(f):
                continue
            k = int(f["lost"].sum())
            lines.append(f"| {H} | {lo_:.0%}{'+' if hi_ > 1 else f'-{hi_:.0%}'} | {len(f)} | {(1 - f['p']).sum():.1f} | {k} | "
                         f"{upset_tail(1 - f['p'].values, k):.2f} |")
    lines += ["", "## Calibration by confidence band (every candidate)", "",
              "The 95% interval is the Wilson interval on the band's actual win rate; a band is off only when its "
              "predicted rate falls outside it.", "",
              "| Days out | Band | Candidates | Predicted | Won | 95% interval |", "|---|---|---|---|---|---|"]
    for t in cal_t.itertuples():
        flag = "" if t.lo <= t.predicted <= t.hi else " (outside)"
        lines.append(f"| {t.H} | {t.band.left:.0%}-{t.band.right:.0%} | {t.n} | {t.predicted:.1%} | {t.won:.1%} | "
                     f"{t.lo:.1%}-{t.hi:.1%}{flag} |")
    out = os.path.join(ROOT, "docs", "HORIZON_CALIBRATION.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
