# -*- coding: utf-8 -*-
"""RACE-LEVEL no-polls model (2026-10-27, audit items 14 and 7): serves the 2026 races no pollster
covers (models/fundamentals/predict_unpolled.py), as the race-level win model does for polled ones.

    py -X utf8 models/fundamentals/race_fundamentals_model.py [--table-only]

Why:
- The old no-polls model (fundamentals_model.py --general-only) was a per-candidate classifier
  divided by its race total - the shape audit item 7 replaced for polled races.
- It trained on POLLED races only, then served UNPOLLED ones (329 of 506 in 2026, mostly safe
  House seats), and its odds were never graded on unpolled races (audit item 14).

Scheme:
- Training rows: every general race in the results archives, polled AND unpolled, built the way
  predict_unpolled.py builds a 2026 row (blank poll fields, house={}, bias_priors={}, primary
  results, office level 0 for a major-party nominee no source covers). Polled races carry their
  real candidate fields (the poll model's training table) with the poll-derived columns unused.
  Unpolled races come from data/res_*.csv (two-party, one winner; analysis/fundamentals_on_unpolled).
- Features: the served win model's list minus every poll feature (fundamentals_model.is_poll_feature)
  - the same set the old model used.
- XGBRanker + within-race softmax (src/race_level.py); the temperature and an offset for minor-party
  candidates (party not DEM/REP; a Dem-slot independent counts as DEM) are fitted together on
  out-of-sample predictions for the full-field cycles 2018-24, as in race_win_model.py.
- Tuning: LOCO over the 1998-2016 TUNE cycles only, log loss of the softmaxed probability.
- Held-out 2018-24 (expanding window), graded on UNPOLLED races - the population it serves - and on
  polled races, with calibration. Sanity only (user 2026-10-26: correctness, not metric comparisons).
- Artifacts: data/fundamentals_model_general.json + data/fundamentals_model_general_features.json
  (model_type "xgbranker"), read by predict_unpolled.py.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))
for p in (_ROOT, _os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "models", "poll"),
          _os.path.join(_ROOT, "analysis"), _HERE):
    _sys.path.insert(0, p)
from paths import ROOT  # noqa: E402

import contextlib  # noqa: E402
import io  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

import features as F  # noqa: E402
from cycles import TUNE_CYCLES, EVAL_CYCLES  # noqa: E402
from macro_features import build_macro  # noqa: E402
from cycles import natl_env  # noqa: E402
from race_level import Ensemble, race_win_probs  # noqa: E402
import fundamentals_model as FM  # noqa: E402
import fundamentals_on_unpolled as FU  # noqa: E402
from race_win_model import fit_ranker, TEMP_GRID  # noqa: E402

N_CONFIGS = 60
TOP_K = 8
TUNE_SEEDS = (42, 7, 123)
OFFSET_GRID = np.round(np.linspace(-8, 0, 33), 2)
GRID = dict(max_depth=[1, 2, 3], learning_rate=[0.02, 0.05, 0.1],
            n_estimators=[150, 300, 600], min_child_weight=[2, 8, 30],
            subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 1.0],
            reg_lambda=[1, 5, 20])
TABLE = os.path.join(ROOT, "data", "raw", "fundamentals_training_table.pkl")   # gitignored cache


def minor(df):
    """Minor-party candidates: neither the DEM nor the REP slot (src/race_level.py's offset)."""
    return (~df["party"].isin(["DEM", "REP"])).astype(float)


def fill_bio(c):
    # predict_unpolled.py's rule: a major-party nominee with no office found in any source is 0
    blank = c["party"].isin(["DEM", "REP"]) & c["bio_office_level"].isna()
    c.loc[blank, "bio_office_level"] = 0.0
    return c


def unpolled_rows(polled_ids):
    """Unpolled archive races, built as predict_unpolled.table builds a 2026 row."""
    res = FU.results_universe()
    u = res[~res["race_id"].isin(polled_ids)].copy()
    fake = pd.DataFrame({
        "year": u["year"].astype(int), "state": u["state"], "office": u["office"],
        "district": u["district"], "candidate": u["candidate_name"], "cand_key": u["cand_key"],
        "party_std": u["party_std"], "pct": np.nan, "end_date": pd.NaT, "sample_size": np.nan,
        "pollster": "none", "won": u["won"].astype(int),
        "poll_id": ("nopoll-" + u["race_id"]).values,
        "election_date": pd.to_datetime(u["year"].astype(str) + "-11-05"),
        "days_to_elec": np.nan, "race_id": u["race_id"],
    })
    return F.build_candidate_table(fake, build_macro(), natl_env(), F.load_fundamentals(), house={},
                                   fec=F.load_fec(extended=True), bias_priors={},
                                   candidate_bios=F.load_candidate_bios(),
                                   primary_results=F.load_primary_results())


def training_table():
    if "--rebuild" not in _sys.argv and os.path.exists(TABLE):
        return pd.read_pickle(TABLE)
    with contextlib.redirect_stdout(io.StringIO()):
        p = FM.general_table()
        u = unpolled_rows(set(p["race_id"]))
    p["polled"], u["polled"] = 1, 0
    c = fill_bio(pd.concat([p, u], ignore_index=True))
    c.to_pickle(TABLE)
    return c


def calib(df, score_col="_score"):
    """(temp, minor offset, log loss) with the lowest log loss on these rows."""
    y, mn = df["won"].astype(int), minor(df)
    best = None
    for T in TEMP_GRID:
        for d in OFFSET_GRID:
            ll = log_loss(y, race_win_probs(df, score_col, T, d, minor=mn).clip(1e-6, 1 - 1e-6))
            if best is None or ll < best[2]:
                best = (float(T), float(d), ll)
    return best


def temp_only(df, score_col="_score"):
    y = df["won"].astype(int)
    return min(((float(T), log_loss(y, race_win_probs(df, score_col, T).clip(1e-6, 1 - 1e-6)))
                for T in TEMP_GRID), key=lambda t: t[1])


def fit_ens(tr, feats, params, seed=42):
    """params: one setting (tuning) or a list of them - the served average (audit item 6)."""
    plist = params if isinstance(params, list) else [params]
    return Ensemble([fit_ranker(tr, feats, p, seed) for p in plist])


def loco(c, feats, params, cycles, train_pool, seed=42):
    out = []
    for ty in cycles:
        tr, te = train_pool[train_pool["year"] != ty], c[c["year"] == ty].copy()
        if not len(te):
            continue
        te["_score"] = fit_ens(tr, feats, params, seed).predict(te[feats])
        out.append(te)
    return pd.concat(out)


def seeded(frames):
    return pd.concat([f.assign(race_id=lambda t, s=s: t["race_id"] + f"#{s}") for s, f in frames])


def metrics(te, col="p"):
    y = te["won"].astype(int)
    p = te[col].clip(1e-6, 1 - 1e-6)
    lead = te.loc[te.groupby("race_id")[col].idxmax()]
    return dict(n_races=te["race_id"].nunique(), Brier=brier_score_loss(y, p), LogLoss=log_loss(y, p),
                AUC=roc_auc_score(y, p) if y.nunique() > 1 else np.nan, race_acc=lead["won"].mean())


def main():
    t0 = time.time()
    c = training_table()
    FEATS = [f for f in F.general_features("win") if not FM.is_poll_feature(f)]
    # full-field cycles: at least 10% of POLLED races have 3+ candidates (2018 on; 2000 has a lone
    # one). Minor candidates enter training only through the polled races' poll fields
    pc = c[c["polled"] == 1]
    multi = (pc.groupby("race_id").size() > 2).groupby(pc.groupby("race_id")["year"].first()).mean()
    FULL = sorted(int(y) for y, v in multi.items() if v >= 0.10)
    print(f"rows {len(c)} | races {c['race_id'].nunique()} (polled {c.loc[c.polled == 1, 'race_id'].nunique()}, "
          f"unpolled {c.loc[c.polled == 0, 'race_id'].nunique()}) | features {len(FEATS)} | full-field "
          f"cycles {FULL} | {time.time() - t0:.0f}s", flush=True)
    print("unpolled races by cycle:", c[c.polled == 0].groupby("year")["race_id"].nunique().to_dict())
    if "--table-only" in _sys.argv:
        fill = c[FEATS].notna().mean()
        print("fill (unpolled rows):", c.loc[c.polled == 0, FEATS].notna().mean().round(2)
              .sort_values().head(15).to_dict())
        print("fill (all rows) lowest:", fill.round(2).sort_values().head(10).to_dict())
        return

    # ---- tuning: LOCO over the tune cycles, trained on the tune cycles only ----
    tune = c[c["year"].isin(TUNE_CYCLES)]
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    random.seed(0); random.shuffle(combos)
    combos = [dict(zip(keys, cb)) for cb in combos[:N_CONFIGS]]

    def score(p, seeds):
        return float(np.mean([temp_only(loco(tune, FEATS, p, TUNE_CYCLES, tune, s))[1] for s in seeds]))
    t1 = time.time()
    stage1 = sorted(((score(p, (42,)), p) for p in combos), key=lambda t: t[0])
    print(f"stage 1: {len(combos)} settings in {time.time() - t1:.0f}s; best {[round(s[0], 4) for s in stage1[:5]]}",
          flush=True)
    stage2 = sorted(((score(p, TUNE_SEEDS), p) for _, p in stage1[:TOP_K]), key=lambda t: t[0])
    best, PARAMS = stage2[0]
    print(f"stage 2: {[round(s[0], 4) for s in stage2]}\nPARAMS = {PARAMS} (tune log loss {best:.4f}; "
          f"runner-up {stage2[1][0]:.4f})", flush=True)
    ENS = [p for _, p in stage2[:3]]          # the served model averages the top 3 settings
    TUNE_TEMP = temp_only(seeded([(s, loco(tune, FEATS, ENS, TUNE_CYCLES, tune, s)) for s in TUNE_SEEDS]))[0]

    # ---- held-out 2018-24, expanding window; graded on unpolled and polled races ----
    rows, preds = [], []
    for ty in EVAL_CYCLES:
        tr, te = c[c["year"] < ty], c[c["year"] == ty].copy()
        te["_score"] = fit_ens(tr, FEATS, ENS).predict(te[FEATS])
        prior = [p for p in preds if p["year"].iloc[0] in FULL]
        T, d = calib(pd.concat(prior))[:2] if prior else (TUNE_TEMP, 0.0)
        te["p"] = race_win_probs(te, "_score", T, d, minor=minor(te))
        for tag, sub in (("unpolled", te[te.polled == 0]), ("polled", te[te.polled == 1])):
            rows.append(dict(cycle=ty, set=tag, temp=T, minor_offset=d, **metrics(sub)))
        preds.append(te)
    ev = pd.DataFrame(rows)
    print("\n=== held-out 2018-24, expanding window (sanity check) ===")
    print(ev.set_index(["set", "cycle"]).round(4).to_string())
    means = ev.groupby("set")[["Brier", "LogLoss", "AUC", "race_acc"]].mean().round(4)
    print(means.to_string())
    P = pd.concat(preds)
    U = P[P.polled == 0].copy()
    U["bin"] = pd.cut(U["p"], [0, .02, .1, .3, .5, .7, .9, .98, 1.0])
    print("\ncalibration on UNPOLLED races:")
    print(U.groupby("bin", observed=True).agg(n=("won", "size"), pred=("p", "mean"),
                                              won=("won", "mean")).round(3).to_string())
    mn = P[minor(P) == 1]
    print(f"minor-party candidates: {len(mn)}, expected wins {mn['p'].sum():.1f}, won {int(mn['won'].sum())}")

    # ---- production: all cycles; calibration from LOCO predictions of the full-field cycles ----
    lo = [(s, loco(c, FEATS, ENS, FULL, c, s)) for s in TUNE_SEEDS]
    TEMP, OFFSET, ll = calib(seeded(lo))
    print(f"\nproduction temperature {TEMP}, minor offset {OFFSET} (LOCO log loss on {FULL}: {ll:.4f})")
    ens = fit_ens(c, FEATS, ENS)
    files = []
    for i, m in enumerate(ens.models):
        fn = "fundamentals_model_general.json" if i == 0 else f"fundamentals_model_general_{i}.json"
        m.save_model(os.path.join(ROOT, "data", fn))
        files.append(fn)
    FM.write_importance(ens.models[0], FEATS, os.path.join(_HERE, "general_feature_importance.csv"),
                        "general fundamentals (race-level, first of the averaged settings)")
    per = {f"{r.set}_{r.cycle}": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r._asdict().items()
                                   if k in ("n_races", "Brier", "LogLoss", "AUC", "race_acc")}
           for r in ev.itertuples()}
    with open(os.path.join(ROOT, "data", "fundamentals_model_general_features.json"), "w") as f:
        json.dump(dict(features=FEATS, model_type="xgbranker", objective="rank:pairwise",
                       ensemble_files=files, xgb_params=ENS, softmax_temp=TEMP, minor_offset=OFFSET,
                       minor="party not DEM/REP (race_fundamentals_model.minor)",
                       trained_on_cycles=sorted(int(y) for y in c["year"].unique()),
                       training_rows=dict(polled_races=int(c.loc[c.polled == 1, "race_id"].nunique()),
                                          unpolled_races=int(c.loc[c.polled == 0, "race_id"].nunique())),
                       tune=dict(cycles=TUNE_CYCLES, metric="log loss of softmaxed probability",
                                 stage2=[[round(s[0], 4), s[1]] for s in stage2]),
                       eval_expanding_window=dict(mean={k: r.to_dict() for k, r in means.iterrows()},
                                                  per_cycle=per),
                       note="Race-level no-polls model (2026-10-27): ranker + within-race softmax; serves "
                            "the unpolled 2026 races via predict_unpolled.py."), f, indent=1)
    print(f"saved data/fundamentals_model_general.json + _features.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
