"""Training fill rates per feature -> data/feature_fill_train.json (2026-10-26, audit item 1).

The publish gate (tests/test_publish.py) compares each feature's share of non-missing values at
serve time (predict.py / predict_margin.py write it to their _meta.json) with its share in
training. predict.py's own assert only checks that the COLUMNS exist; a feature that exists but
is empty for every live race (the old poll_momentum gap) passes it and predicts garbage.

Builds the training candidate table the way models/poll/model.ipynb cells 2-6 do (same filters,
same build_candidate_table call) and records fill rates over the held-out cycles 2018-2024,
which are the closest to how 2026 is filled. Re-run after a retrain or a feature change:

    py tools/feature_fill_reference.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import paths  # noqa: E402,F401  - puts src/, pipeline/ and models/ on sys.path

import features as F                       # noqa: E402
from cycles import CYCLES, EVAL_CYCLES, natl_env   # noqa: E402
from macro_features import build_macro     # noqa: E402

OUT = os.path.join(ROOT, "data", "feature_fill_train.json")
# the served (race-level) models; their feature lists are the per-candidate notebooks' lists
ARTIFACTS = {"win": "race_win_model_features.json", "margin": "race_margin_model_features.json"}


def training_table():
    d = pd.read_csv("polls_long_with_results.csv", low_memory=False)
    gen = d[d["stage"] == "general"]
    q = gen.groupby(["race_id", "question_id"])["has_result"].agg(["sum", "size"])
    dead = set(q[(q["sum"] > 0) & (q["sum"] < q["size"])].index)
    drop = pd.Series([k in dead for k in zip(d["race_id"], d["question_id"])],
                     index=d.index) & (d["stage"] == "general")
    d = d[~drop]
    d = d[d["has_result"] == 1].copy()
    d = d[d.groupby("race_id")["won"].transform("max") == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    return F.build_candidate_table(
        d, build_macro(), natl_env(), F.load_fundamentals(), house_train_years=CYCLES,
        fec=F.load_fec(extended=True), bias_priors=F.compute_bias_priors(d),
        candidate_bios=F.load_candidate_bios(), primary_results=F.load_primary_results())


def main():
    base = training_table()
    recent = base[base["year"].isin(EVAL_CYCLES)]
    out = {"cycles": sorted(int(y) for y in recent["year"].unique()), "n_rows": int(len(recent))}
    for name, fn in ARTIFACTS.items():
        feats = json.load(open(os.path.join(ROOT, "data", fn)))["features"]
        out[name] = {f: round(float(recent[f].notna().mean()), 4) if f in recent else None
                     for f in feats}
    # the no-polls model trains on its own table (polled races, poll features dropped)
    from fundamentals_model import general_table
    fund = general_table()
    fund = fund[fund["year"].isin(EVAL_CYCLES)]
    feats = json.load(open(os.path.join(ROOT, "data", "fundamentals_model_general_features.json")))["features"]
    out["fundamentals"] = {f: round(float(fund[f].notna().mean()), 4) if f in fund else None
                           for f in feats}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {OUT}: {out['n_rows']} candidate rows, cycles {out['cycles']}")


if __name__ == "__main__":
    main()
