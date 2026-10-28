# -*- coding: utf-8 -*-
"""Does a re-tune move 2026 races? (audit M5 safeguard, 2026-10-27)

    py -X utf8 src/predict.py --dump-table outputs/served_table_2026.pkl --natl-env <published value>
    py -X utf8 analysis/retune_stability.py [--runs 3] [--models win,margin]

The audit's check: "re-tuning with 3 seeds keeps the chosen settings, or no race moves more than
5 pts". Before 10-27 the tuners picked ONE setting from a near-tie, and the 10-22, 10-24 and 10-25
retrains each moved races by settings alone (up to 15 win pts; margins a median 0.6 pts each
time). The served models now average the top 3 settings (src/race_level.py Ensemble); this script
measures whether that is enough.

Each run repeats the trainer's whole tuning procedure with different randomness: a different
sample of the random search (the trainers shuffle the grid with seed 0), different stage-2 seeds
and a different model seed, then the temperature / offset fit (win) the trainer does. It then
scores the SAME served 2026 table (the one predict.py built, saved with --dump-table) with the
current artifacts and with each re-tune, and reports how far each race moves.

Louisiana House is left out of the win comparison: its published odds come from the margin model
(src/la_runoff.py). Writes docs/RETUNE_STABILITY.md and outputs/retune_stability_2026.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models", "poll"))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models", "fundamentals"))
from paths import ROOT  # noqa: E402

import argparse
import contextlib
import io
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd

import race_margin_model as RMM
import race_win_model as RWM
import features as F
from race_level import load_ensemble, race_margins, race_win_probs

DATA = os.path.join(ROOT, "data")
THRESHOLD = 5.0   # pts, the audit's bar


def _grid_sample(grid, n, seed):
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    random.seed(seed); random.shuffle(combos)
    return [dict(zip(keys, cb)) for cb in combos[:n]]


def retune_win(c, feats, search_seed, seeds):
    combos = _grid_sample(RWM.GRID, RWM.N_CONFIGS, search_seed)
    stage1 = sorted(((RWM.score_config(c, feats, p, seeds[:1]), p) for p in combos), key=lambda t: t[0][0])
    stage2 = sorted(((RWM.score_config(c, feats, p, seeds), p) for _, p in stage1[:RWM.TOP_K]),
                    key=lambda t: t[0][0])
    ens_params = [p for _, p in stage2[:RWM.N_ENSEMBLE]]
    n_cand = c.groupby("race_id")["race_id"].transform("size")
    full = sorted(int(y) for y in c.loc[n_cand > 2, "year"].unique())
    frames = []
    for s in seeds:
        out = []
        for ty in full:
            tr, te = c[c["year"] != ty], c[c["year"] == ty].copy()
            te["_score"] = RWM.fit_ensemble(tr, feats, ens_params, s).predict(te[feats])
            out.append(te)
        frames.append(pd.concat(out).assign(race_id=lambda t, s=s: t["race_id"] + f"#{s}"))
    temp, offset, _ = RWM.fit_calibration(pd.concat(frames))
    return dict(params=ens_params, temp=temp, offset=offset,
                model=RWM.fit_ensemble(c, feats, ens_params, seeds[0]))


def retune_margin(c, feats, search_seed, seeds):
    combos = _grid_sample(RMM.GRID, RMM.N_CONFIGS, search_seed)
    stage1 = sorted(((RMM.loco_tune(c, feats, p, seeds[0]), p) for p in combos), key=lambda t: t[0])
    stage2 = sorted(((float(np.mean([RMM.loco_tune(c, feats, p, s) for s in seeds])), p)
                     for _, p in stage1[:RMM.TOP_K]), key=lambda t: t[0])
    ens_params = [p for _, p in stage2[:3]]
    from race_level import Ensemble
    return dict(params=ens_params, model=Ensemble([RMM.fit_share(c, feats, p, seeds[0]) for p in ens_params]))


def retune_fund(c, feats, search_seed, seeds):
    """The no-polls trainer's procedure (models/fundamentals/race_fundamentals_model.main)."""
    import race_fundamentals_model as RF
    from cycles import TUNE_CYCLES
    tune = c[c["year"].isin(TUNE_CYCLES)]
    combos = _grid_sample(RF.GRID, RF.N_CONFIGS, search_seed)

    def score(p, ss):
        return float(np.mean([RF.temp_only(RF.loco(tune, feats, p, TUNE_CYCLES, tune, s))[1] for s in ss]))
    stage1 = sorted(((score(p, seeds[:1]), p) for p in combos), key=lambda t: t[0])
    stage2 = sorted(((score(p, seeds), p) for _, p in stage1[:RF.TOP_K]), key=lambda t: t[0])
    ens_params = [p for _, p in stage2[:RF.N_ENSEMBLE]]
    pc = c[c["polled"] == 1]
    multi = (pc.groupby("race_id").size() > 2).groupby(pc.groupby("race_id")["year"].first()).mean()
    full = sorted(int(y) for y, v in multi.items() if v >= 0.10)
    temp, offset, _ = RF.calib(RF.seeded([(s, RF.loco(c, feats, ens_params, full, c, s)) for s in seeds]))
    return dict(params=ens_params, temp=temp, offset=offset, model=RF.fit_ens(c, feats, ens_params, seeds[0]))


def fund_probs(table, model, feats, temp, offset):
    t = table[["race_id", "party"]].copy()
    t["_score"] = model.predict(table.reindex(columns=feats))
    return race_win_probs(t, "_score", temp, offset, minor=(~t["party"].isin(["DEM", "REP"])).astype(float)).values


def win_probs(table, model, feats, temp, offset):
    t = table[["race_id", "poll_avg"]].copy()
    t["_score"] = model.predict(table.reindex(columns=feats))
    return race_win_probs(t, "_score", temp, offset).values


def margins(table, model, feats):
    t = table[["race_id"]].copy()
    t["_share"] = model.predict(table.reindex(columns=feats)).astype("float64")
    return race_margins(t, "_share").values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(ROOT, "outputs", "served_table_2026.pkl"))
    ap.add_argument("--fund-table", default=os.path.join(ROOT, "outputs", "served_unpolled_2026.pkl"),
                    help="models/fundamentals/predict_unpolled.py --dump-table <path>")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--models", default="win,margin", help="any of win, margin, fund")
    ap.add_argument("--report", default=os.path.join(ROOT, "docs", "RETUNE_STABILITY.md"))
    ap.add_argument("--fixed-search", action="store_true",
                    help="keep the trainer's sample of settings (seed 0) and vary only the seeds: the audit's literal test")
    args = ap.parse_args()
    kinds = args.models.split(",")
    t0 = time.time()
    res, summary = {}, []

    if "win" in kinds or "margin" in kinds:
        served = pd.read_pickle(args.table)
        served = served[~((served["state"] == "LA") & (served["office"] == "House"))].reset_index(drop=True)
        with contextlib.redirect_stdout(io.StringIO()):
            c = RWM.training_table()
        c["margin_actual"] = c["vote_pct"] - c["best_other_pct"]
        print(f"training rows {len(c)}; served 2026 rows {len(served)} ({served['race_id'].nunique()} races, "
              f"LA House left out); built in {time.time() - t0:.0f}s", flush=True)
    if "win" in kinds:
        meta = json.load(open(os.path.join(DATA, "race_win_model_features.json")))
        feats = meta["features"]
        res["win"] = served[["race_id", "candidate", "party"]].copy()
        res["win"]["win_now"] = win_probs(served, load_ensemble(meta, DATA, "ranker"), feats,
                                          float(meta["softmax_temp"]), float(meta.get("minor_offset", 0.0)))
    if "margin" in kinds:
        mmeta = json.load(open(os.path.join(DATA, "race_margin_model_features.json")))
        mfeats = mmeta["features"]
        res["margin"] = served[["race_id", "candidate", "party"]].copy()
        res["margin"]["margin_now"] = margins(served, load_ensemble(mmeta, DATA, "regressor"), mfeats)
    if "fund" in kinds:
        import race_fundamentals_model as RF
        fserved = pd.read_pickle(args.fund_table)
        with contextlib.redirect_stdout(io.StringIO()):
            fc = RF.training_table()
        fmeta = json.load(open(os.path.join(DATA, "fundamentals_model_general_features.json")))
        ffeats = fmeta["features"]
        res["fund"] = fserved[["race_id", "candidate", "party"]].copy()
        res["fund"]["fund_now"] = fund_probs(fserved, load_ensemble(fmeta, DATA, "ranker"), ffeats,
                                             float(fmeta["softmax_temp"]), float(fmeta.get("minor_offset", 0.0)))
        print(f"no-polls training rows {len(fc)}; served unpolled rows {len(fserved)} "
              f"({fserved['race_id'].nunique()} races)", flush=True)

    for run in range(1, args.runs + 1):
        search_seed = 0 if args.fixed_search else 100 + run
        seeds = tuple(1000 * run + k for k in (1, 2, 3))
        if "win" in kinds:
            t1 = time.time()
            w = retune_win(c, feats, search_seed, seeds)
            res["win"][f"win_{run}"] = win_probs(served, w["model"], feats, w["temp"], w["offset"])
            same = sum(p in meta["xgb_params"] for p in w["params"])
            summary.append(dict(model="win", run=run, settings_shared=f"{same} of {len(w['params'])}",
                                temp=round(w["temp"], 4), offset=w["offset"], minutes=round((time.time() - t1) / 60, 1)))
            print(summary[-1], flush=True)
        if "margin" in kinds:
            t1 = time.time()
            m = retune_margin(c, mfeats, search_seed, seeds)
            res["margin"][f"margin_{run}"] = margins(served, m["model"], mfeats)
            same = sum(p in mmeta["xgb_params"] for p in m["params"])
            summary.append(dict(model="margin", run=run, settings_shared=f"{same} of {len(m['params'])}",
                                minutes=round((time.time() - t1) / 60, 1)))
            print(summary[-1], flush=True)
        if "fund" in kinds:
            t1 = time.time()
            f_ = retune_fund(fc, ffeats, search_seed, seeds)
            res["fund"][f"fund_{run}"] = fund_probs(fserved, f_["model"], ffeats, f_["temp"], f_["offset"])
            same = sum(p in fmeta["xgb_params"] for p in f_["params"])
            summary.append(dict(model="fund", run=run, settings_shared=f"{same} of {len(f_['params'])}",
                                temp=round(f_["temp"], 4), offset=f_["offset"], minutes=round((time.time() - t1) / 60, 1)))
            print(summary[-1], flush=True)
        for k, r in res.items():
            r.to_csv(os.path.join(ROOT, "outputs", f"retune_stability_{k}_2026.csv"), index=False)

    write_report(res, summary, kinds, args.runs, args.report)
    print(f"done in {(time.time() - t0) / 60:.0f} min")


def _moves(res, kind, runs):
    """Per race: the largest move of any candidate across the re-tunes against the served model,
    and between any two re-tunes (the procedure's own spread), in points."""
    scale = 1 if kind == "margin" else 100
    cols = [f"{kind}_{r}" for r in range(1, runs + 1)]
    d = res[["race_id"]].copy()
    d["move"] = np.max([np.abs(res[c] - res[f"{kind}_now"]) for c in cols], axis=0) * scale
    d["spread"] = (res[cols].max(axis=1) - res[cols].min(axis=1)) * scale
    if kind == "margin":
        flips = max(int(((res[c] > 0) != (res[f"{kind}_now"] > 0)).groupby(res["race_id"]).any().sum())
                    for c in cols)
    else:
        lead_now = res.loc[res.groupby("race_id")[f"{kind}_now"].idxmax()].set_index("race_id")["candidate"]
        flips = max(int((res.loc[res.groupby("race_id")[c].idxmax()].set_index("race_id")["candidate"]
                         != lead_now).sum()) for c in cols)
    g = d.groupby("race_id")
    return g["move"].max().sort_values(ascending=False), g["spread"].max(), flips


TITLES = {"win": "Win model (win-chance pts)", "margin": "Margin model (margin pts)",
          "fund": "No-polls model (win-chance pts)"}


def write_report(res, summary, kinds, runs, path):
    lines = ["# Re-tune stability of the 2026 forecast", "",
             "Built by `analysis/retune_stability.py` (audit M5). Each run repeats the trainer's whole tuning "
             "procedure with different randomness (another sample of the random search, other stage-2 and model "
             "seeds, the temperature and offset refitted for the win and no-polls models) and scores the same "
             f"served 2026 table. The audit's bar: no race moves more than {THRESHOLD:.0f} pts. \"Move\" is against "
             "the served model; \"spread\" is between the re-tunes themselves. Louisiana House is left out of the "
             "polled comparison (its odds come from the margin model and the runoff simulation).", "",
             "| Model | Run | Settings shared with the served ensemble | Notes |", "|---|---|---|---|"]
    for s in summary:
        note = f"temperature {s['temp']}, minor offset {s['offset']}" if "temp" in s else ""
        lines.append(f"| {s['model']} | {s['run']} | {s['settings_shared']} | {note} |")
    for kind in kinds:
        mv, spread, flips = _moves(res[kind], kind, runs)
        lines += ["", f"## {TITLES[kind]}", "",
                  f"- Races: {len(mv)}. Against the served model: largest move {mv.max():.1f}, median {mv.median():.2f}, "
                  f"races over {THRESHOLD:.0f}: {(mv > THRESHOLD).sum()}, over 2: {(mv > 2).sum()}.",
                  f"- Between re-tunes: largest spread {spread.max():.1f}, median {spread.median():.2f}, "
                  f"races over {THRESHOLD:.0f}: {(spread > THRESHOLD).sum()}.",
                  f"- {'Predicted-winner' if kind == 'margin' else 'Favourite'} changes in the worst run: {flips}.", "",
                  "| Race | Largest move | Spread between re-tunes |", "|---|---|---|"]
        lines += [f"| {r} | {v:.1f} | {spread[r]:.1f} |" for r, v in mv.head(12).items()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
