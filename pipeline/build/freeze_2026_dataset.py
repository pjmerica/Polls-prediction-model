# -*- coding: utf-8 -*-
"""Freeze a self-contained 2026-cycle dataset for our records (2026-07-29).

The 2026 cycle is normally read LIVE from the polling-agg raw feed at predict time - nothing
about it is committed, so a run is only reproducible while that feed exists and is unchanged.
This script snapshots the cycle into three committed files, mirroring how the HISTORICAL
cycles live in polls_long_with_results.csv:

  data/polls_2026_long.csv       RAW 2026 poll rows (one per poll-answer), post dedup /
                                 party-override / dropout / stale filtering - the same `d`
                                 predict.py scores. Append-ready to the training set once
                                 results land (add vote_pct / won and set has_result=1).
  data/candidate_table_2026.csv  the built CANDIDATE FEATURE TABLE (one row per candidate per
                                 race, every model feature incl bio_office_level) - the exact
                                 INPUT the model scores. Winners/margins are blank (cycle open).
  predictions_2026.csv           the model OUTPUT (already written by predict.py) - kept as-is.

Reuses predict.py's own feed-loading + table-building so the frozen data is byte-for-byte what
the model actually saw. The ONE non-freezable input is natl_env (live current-cycle generic
ballot - undefined to freeze); it is recorded in the meta JSON as the value used.

    py -X utf8 freeze_2026_dataset.py [--natl-env X]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import argparse, json, os
import pandas as pd

import features as F
import predict as P
from cycles import natl_env as natl_env_hist
from macro_features import build_macro

HERE = ROOT   # repo root (paths.py) - this file lives in a subfolder
CYCLE = 2026

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natl-env", type=float, default=None)
    ap.add_argument("--polls", nargs="*", default=P.DEFAULT_POLLS)
    args = ap.parse_args()

    # ---- 1. RAW long polls: exactly predict.py's `d` (dedup/override/dropout/stale done) ----
    d = P.load_agg_polls(args.polls, CYCLE)
    if len(d) == 0:
        raise SystemExit(f"no {CYCLE} polls found - is the polling-agg feed present?")
    raw_path = os.path.join(HERE, "data", "polls_2026_long.csv")
    d.to_csv(raw_path, index=False)
    print(f"froze raw polls -> {raw_path}: {len(d)} rows, "
          f"{d['race_id'].nunique()} races, {d['cand_key'].nunique()} candidates")

    # ---- 2. CANDIDATE FEATURE TABLE: rebuild it the way predict.py does ----
    hist = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
    hist = F.prepare_polls(hist[hist["has_result"] == 1])
    hist["race_id"] = (hist["year"].astype(str) + "_" + hist["state"] + "_" + hist["office"]
                       + hist["district"].radd("-").where(hist["district"] != "", ""))
    house = F.compute_house_effect(hist, sorted(hist["year"].unique()))
    bias = F.compute_bias_priors(hist)

    macro = build_macro(cycles=[CYCLE])
    ne = dict(natl_env_hist())
    if args.natl_env is not None:
        ne[CYCLE] = args.natl_env
    else:
        from fetch_generic_ballot import get_natl_env


        v = get_natl_env(CYCLE)
        if v is not None:
            ne[CYCLE] = v
    used_ne = ne.get(CYCLE)

    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    bios = F.load_candidate_bios()
    cand = P.patch_redistricted_priors(
        F.build_candidate_table(d, macro, ne, funds, house=house, fec=fec, bias_priors=bias,
                                candidate_bios=bios))
    tbl_path = os.path.join(HERE, "data", "candidate_table_2026.csv")
    cand.to_csv(tbl_path, index=False)
    bio_cov = cand["bio_office_level"].notna().mean() if "bio_office_level" in cand.columns else 0.0
    print(f"froze feature table -> {tbl_path}: {len(cand)} candidate-rows, "
          f"{cand.shape[1]} columns, bio_office_level coverage {bio_cov:.1%}")

    # ---- 3. meta record ----
    meta = {
        "cycle": CYCLE,
        "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "raw_polls": os.path.basename(raw_path),
        "feature_table": os.path.basename(tbl_path),
        "predictions": "predictions_2026.csv",
        "n_poll_rows": int(len(d)),
        "n_races": int(d["race_id"].nunique()),
        "n_candidate_rows": int(len(cand)),
        "natl_env_used": used_ne,
        "note": ("raw polls are append-ready to polls_long_with_results.csv once results land "
                 "(add vote_pct/won, has_result=1). natl_env is the live value used, not "
                 "reproducible offline - recorded here for the record."),
    }
    meta_path = os.path.join(HERE, "data", "dataset_2026_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {meta_path} (natl_env used: {used_ne})")

if __name__ == "__main__":
    main()
