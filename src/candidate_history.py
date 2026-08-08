# -*- coding: utf-8 -*-
"""Candidate electoral HISTORY features: has this candidate won before, how did they do
in past generals, have they won primaries before.

Sources (all committed, offline):
- data/res_{senate,house,governor}.csv  general-election candidacies 1998-2024
  (vote_pct + won per candidate per cycle)
- data/primary_results_hist.csv (+ _2026)  primary winners 2018+

LEAK SAFETY: every lookup takes the candidate's CURRENT cycle and uses strictly-prior
rows only (r.cycle < cycle). Matching key = (norm_name, state): name collisions within a
state are possible (audited by check_candidate_history.py's collision pass - measured
small) and cross-state political moves are missed (accepted).

Used by the PRIMARY model (candidate quality matters most in primaries); the general
model already carries incumbency/prior-margin at the SEAT level.
"""
import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
import paths as _P  # noqa: E402

import os
from collections import defaultdict

import numpy as np
import pandas as pd

import features as F

# repo-root-relative (paths.py); this file lives in src/ since 2026-08-08
HERE = _P.ROOT
DATA = _P.DATA

def _general_rows():
    frames = []
    for fn, office in [("res_senate.csv", "Senate"), ("res_house.csv", "House"),
                       ("res_governor.csv", "Governor")]:
        r = pd.read_csv(os.path.join(DATA, fn), low_memory=False)
        r = r[r["stage"].astype(str).str.lower() == "general"]
        out = pd.DataFrame({
            "cycle": pd.to_numeric(r["cycle"], errors="coerce"),
            "state": r["state_abbrev"].astype(str).str.upper(),
            "office": office,
            "cand_key": r["candidate_name"].map(F.norm_name),
            "vote_pct": pd.to_numeric(r["percent"], errors="coerce"),
            "won": r["winner"].astype(str).str.lower().isin(["true", "1"]).astype(int),
        })
        frames.append(out)
    g = pd.concat(frames).dropna(subset=["cycle", "cand_key"])
    g["cycle"] = g["cycle"].astype(int)
    return g

def _primary_wins():
    rows = []
    for fn in ["primary_results_hist.csv", "primary_results_2026.csv"]:
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d["is_winner"]]
        parts = d["race_id"].astype(str).str.split("_", expand=True)
        d = d.assign(cycle=pd.to_numeric(parts[0], errors="coerce"),
                     state=parts[1])
        rows.append(d[["cycle", "state", "cand_key"]].dropna())
    if not rows:
        return pd.DataFrame(columns=["cycle", "state", "cand_key"])
    out = pd.concat(rows)
    out["cycle"] = out["cycle"].astype(int)
    return out

class CandidateHistory:
    """Index once, look up many. history(cycle, state, cand_key) -> feature dict."""

    FEATURES = ["hist_prior_runs", "hist_prior_wins", "hist_ever_won",
                "hist_best_general_pct", "hist_last_general_pct",
                "hist_years_since_last_run", "hist_prior_primary_wins"]

    def __init__(self):
        g = _general_rows()
        self._gen = defaultdict(list)     # (state, ck) -> [(cycle, pct, won)]
        for r in g.itertuples():
            self._gen[(r.state, r.cand_key)].append((r.cycle, r.vote_pct, r.won))
        pw = _primary_wins()
        self._pri = defaultdict(list)     # (state, ck) -> [cycle, ...]
        for r in pw.itertuples():
            self._pri[(r.state, r.cand_key)].append(r.cycle)

    def history(self, cycle, state, cand_key):
        raw_prior = [x for x in self._gen.get((state, cand_key), []) if x[0] < cycle]
        # FUSION-VOTING dedup (caught by the Espaillat known-truth check): NY/CT list one
        # candidate on several ballot lines per election -> multiple rows per cycle.
        # Collapse to one run per cycle: won if ANY line won, pct = max line
        # (conservative; fusion pct is split across lines but max is the dominant line).
        by_cycle = {}
        for cyc, pct, won in raw_prior:
            cur = by_cycle.get(cyc)
            if cur is None:
                by_cycle[cyc] = [cyc, pct, won]
            else:
                cur[1] = max(p for p in (cur[1], pct) if p == p) if (
                    cur[1] == cur[1] or pct == pct) else float("nan")
                cur[2] = max(cur[2], won)
        prior = [tuple(v) for v in by_cycle.values()]
        pri = [c for c in self._pri.get((state, cand_key), []) if c < cycle]
        if not prior:
            return dict(hist_prior_runs=0, hist_prior_wins=0, hist_ever_won=0,
                        hist_best_general_pct=np.nan, hist_last_general_pct=np.nan,
                        hist_years_since_last_run=np.nan,
                        hist_prior_primary_wins=len(pri))
        pcts = [p for _, p, _ in prior if p == p]
        last = max(prior, key=lambda x: x[0])
        wins = sum(w for _, _, w in prior)
        return dict(
            hist_prior_runs=len(prior),
            hist_prior_wins=int(wins),
            hist_ever_won=int(wins > 0),
            hist_best_general_pct=(max(pcts) if pcts else np.nan),
            hist_last_general_pct=(last[1] if last[1] == last[1] else np.nan),
            hist_years_since_last_run=float(cycle - last[0]),
            hist_prior_primary_wins=len(pri),
        )

if __name__ == "__main__":
    h = CandidateHistory()
    print("indexed:", len(h._gen), "(state,candidate) general histories |",
          len(h._pri), "primary-winner histories")
    demo = h.history(2026, "MI", F.norm_name("Haley Stevens"))
    print("Haley Stevens @2026:", demo)
