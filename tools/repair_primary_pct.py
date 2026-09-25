# -*- coding: utf-8 -*-
"""Fix impossible `pct` values in the historical PRIMARY results tables (2026-09-25).

    py -X utf8 tools/repair_primary_pct.py            # dry run: report only
    py -X utf8 tools/repair_primary_pct.py --apply    # rewrite the CSVs

The Wikipedia scrape misread the percentage column in a handful of tables while the vote
counts are right: GA-1-DEM 2012 had the pct column swapped (Russo 55,880 votes at 45.7% vs
Messinger 15,390 at 54.3%), NH-Sen-REP 1998 summed to 123.8%. `pct` is the primary MARGIN
model's target, so these went straight into training. Winner flags were checked and are all
the top vote-getter.

A table is repaired (pct := votes share of the table) only when its pct is IMPOSSIBLE:
  - the table sums above 100.5, or
  - a candidate has a clearly higher pct (>0.05) with FEWER votes than another.
Ties and tables that sum below 100 are left alone: those are real omissions (Nevada's "None
of these candidates", blank-vote lines), where pct is right and a votes share would not be.

Two tables are the wrong CONTEST and are dropped instead of repaired:
  - 2018_PA_Governor_DEM holds the Lt. Governor primary (Fetterman/Stack/Cozzone).
  - 2012_TX_House-27_DEM mixes the first round (votes) with the runoff (Harrison 60.6%),
    so its winner flag cannot be trusted from this table.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths  # noqa: E402

import argparse
import os
import re

import numpy as np
import pandas as pd

FILES = ["primary_results_hist.csv", "primary_results_deep_hist.csv",
         "house_primary_results_hist.csv"]
WRONG_CONTEST = {"2018_PA_Governor_DEM", "2012_TX_House-27_DEM"}
NONCAND = re.compile(r"(?i)^(blank votes?|all others|others|write-?ins?|none of these.*|"
                     r"scattering|over ?votes|under ?votes)$")


def impossible(g):
    c = g[~g["candidate"].astype(str).str.strip().str.match(NONCAND)]
    v = pd.to_numeric(c["votes"], errors="coerce")
    if v.isna().any() or v.sum() <= 0 or len(c) < 2:
        return False
    p, vv = c["pct"].values, v.values
    over = np.nansum(p) > 100.5
    inv = any(p[i] > p[j] + 0.05 and vv[i] < vv[j]
              for i in range(len(c)) for j in range(len(c)))
    return bool(over or inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for fn in FILES:
        path = os.path.join(paths.DATA, fn)
        r = pd.read_csv(path, low_memory=False)
        drop = r["race_id"].isin(WRONG_CONTEST)
        fixed = []
        for (rid, seq), g in r[~drop].groupby(["race_id", "table_seq"]):
            if not impossible(g):
                continue
            real = g[~g["candidate"].astype(str).str.strip().str.match(r"(?i)^blank votes?$")]
            v = pd.to_numeric(real["votes"], errors="coerce")
            r.loc[real.index, "pct"] = (v / v.sum() * 100).round(2)
            fixed.append(rid)
        print(f"{fn}: repaired {len(fixed)} tables {fixed}; dropped wrong-contest "
              f"{sorted(set(r.loc[drop, 'race_id']))}")
        if args.apply:
            r[~drop].to_csv(path, index=False)
    if not args.apply:
        print("(dry run - pass --apply to write)")


if __name__ == "__main__":
    main()
