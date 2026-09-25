# -*- coding: utf-8 -*-
"""Recompute the result-label columns of polls_long_with_results.csv in place.

    py -X utf8 tools/repair_result_labels.py            # dry run: report only
    py -X utf8 tools/repair_result_labels.py --apply    # rewrite the CSV

Why (2026-09-25): the build kept ONE arbitrary results row per candidate, so fusion-state
candidates (NY/CT/SC) got a single minor-party line as their vote share and ranked-choice
races (ME/AK) mixed rounds. Nicknames (Bob/Robert Menendez) also dropped four race winners.
See src/results_labels.py for the full story. The poll columns are untouched; only
won / vote_pct / res_party / res_candidate / race_winning_pct / best_other_pct / has_result
are recomputed, plus survey-identity duplicates (features.drop_duplicate_surveys, which the
live predictor already applies) are removed so training and serving see the same polls.

Rebuilding the whole file through build_dataset.ipynb would also re-download the live NYT
feed and churn every 2026 row; this touches only the labels. The notebook now calls the same
module, so a future rebuild produces the same labels.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths  # noqa: E402,F401

import argparse
import os

import numpy as np
import pandas as pd

import features as F  # noqa: E402
import results_labels as RL  # noqa: E402

CSV = os.path.join(paths.ROOT, "polls_long_with_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    old = pd.read_csv(CSV, low_memory=False)
    cols = list(old.columns)
    labels = RL.aggregate_candidates(RL.load_result_lines(), verbose=True)
    new = RL.attach_labels(old, labels, verbose=True)
    new, ndup = F.drop_duplicate_surveys(new, label="training polls")
    new = RL.drop_off_date_polls(new, verbose=True)
    if "best_other_pct" not in cols:
        cols.append("best_other_pct")
    new = new[cols]

    # ---- report ----
    o = old.drop_duplicates(["race_id", "cand_key"]).set_index(["race_id", "cand_key"])
    n = new.drop_duplicates(["race_id", "cand_key"]).set_index(["race_id", "cand_key"])
    j = o[["won", "vote_pct", "has_result"]].join(
        n[["won", "vote_pct", "has_result", "best_other_pct", "res_party"]], rsuffix="_new")
    chg_v = j[(j["vote_pct"] - j["vote_pct_new"]).abs() > 0.01]
    chg_w = j[(j["won"].fillna(-1) != j["won_new"].fillna(-1))]
    gained = j[(j["has_result"] == 0) & (j["has_result_new"] == 1)]
    print(f"\nrows {len(old)} -> {len(new)}  (survey duplicates dropped: {ndup})")
    print(f"candidate-races with vote_pct changed: {len(chg_v)}")
    print(chg_v.assign(d=chg_v["vote_pct_new"] - chg_v["vote_pct"])
          .sort_values("d", key=abs, ascending=False)
          [["vote_pct", "vote_pct_new", "res_party", "won_new"]].round(2).head(25).to_string())
    print(f"\ncandidate-races with won changed: {len(chg_w)}")
    print(chg_w[["won", "won_new", "vote_pct_new"]].to_string())
    print(f"\ncandidate-races newly labelled: {len(gained)}")

    lab = new[new["has_result"] == 1].drop_duplicates(["race_id", "cand_key"])
    w = lab.groupby("race_id")["won"].sum()
    print(f"\nlabelled races {len(w)}; with no polled winner: {(w == 0).sum()}; "
          f"with >1 winner: {(w > 1).sum()}")
    mt = lab["vote_pct"] - lab["best_other_pct"]
    agree = ((mt > 0).astype(int) == lab["won"])[mt.notna()].mean()
    print(f"sign(vote_pct - best_other_pct) agrees with won: {agree:.4f}")
    assert (w <= 1).all(), "a race has two winners - label aggregation is broken"
    assert agree > 0.97, "margin target disagrees with win labels too often"

    if args.apply:
        new.to_csv(CSV, index=False)
        print(f"\nwrote {CSV}")
    else:
        print("\n(dry run - pass --apply to write)")


if __name__ == "__main__":
    main()
