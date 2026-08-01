# -*- coding: utf-8 -*-
"""One-shot repair: recompute every PRECOMPUTED cand_key column with the fixed norm_name.

    py -X utf8 scripts_rekey_cand_key.py [--apply]

Why this exists (2026-08-01): several committed CSVs cache `cand_key` as a COLUMN rather than
deriving it at load time. norm_name changed (intra-word punctuation is now deleted instead of
spaced - see features.norm_name), so those cached keys are frozen under the old, buggy
normalizer. A stale key is worse than a wrong feature: it silently breaks the joins that
attach results/FEC/bios/history to a candidate, and it silently MERGED distinct politicians
(Christy Smith + Cindy Hyde-Smith both keyed 'smith c').

`cand_key` is a pure function of the name column, so recomputing it is lossless - no scraping,
no network, nothing else in the file is touched. Dry-run by default; pass --apply to write.
"""
import argparse
import glob
import os

import pandas as pd

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
# (path, name-column) - the name column each file's cand_key is derived FROM.
NAME_COLS = ["name", "candidate", "candidate_name"]


def targets():
    out = []
    paths = sorted(glob.glob(os.path.join(HERE, "data", "*.csv")))
    paths += [os.path.join(HERE, "polls_long_with_results.csv")]
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            head = pd.read_csv(p, nrows=0).columns
        except Exception:
            continue
        if "cand_key" not in head:
            continue
        namecol = next((c for c in NAME_COLS if c in head), None)
        if namecol:
            out.append((p, namecol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    args = ap.parse_args()

    total = 0
    for path, namecol in targets():
        d = pd.read_csv(path, low_memory=False)
        fresh = d[namecol].map(F.norm_name)
        stored = d["cand_key"]
        # compare as strings so NaN==NaN counts as equal, not as a diff
        diff = stored.astype(str) != fresh.astype(str)
        n = int(diff.sum())
        if not n:
            continue
        total += n
        rel = os.path.relpath(path, HERE)
        print(f"{rel:46s} {n:6d} / {len(d):6d} rows re-keyed")
        for r in d.loc[diff, [namecol]].drop_duplicates().head(3).itertuples():
            nm = getattr(r, namecol)
            print(f"      {nm!r}: {stored[r.Index]!r} -> {fresh[r.Index]!r}")
        if args.apply:
            d["cand_key"] = fresh
            d.to_csv(path, index=False)

    print()
    print(f"TOTAL rows re-keyed: {total}")
    print("WROTE the files." if args.apply else "DRY RUN - pass --apply to write.")


if __name__ == "__main__":
    main()
