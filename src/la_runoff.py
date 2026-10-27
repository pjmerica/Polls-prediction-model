"""Louisiana House: chance to WIN THE SEAT, first round + runoff (2026-10-27).

After Louisiana v. Callais the state suspended its May House primaries: Nov 3 is an all-party
first round and, if nobody passes 50%, the top two meet on Dec 12. The race-level win model ranks
who finishes FIRST on Nov 3 (LA-6: Banks D 51% while Republicans split ~39% of the vote three ways
and the two Democrats hold ~27%), which is not who holds the seat. This simulates both rounds
from the margin model's predicted first-round vote shares (src/predict_margin.py's model):

1. First round: each candidate's predicted share plus independent noise (SD SHARE_SD points, the
   held-out per-candidate share error), rescaled so the field sums to 100 (the undecided split in
   proportion).
2. Anyone over 50% wins outright.
3. Runoff between the top two: each finalist keeps the votes of their party's first-round
   candidates (same-party finalists split the rest evenly), the other parties' votes split
   evenly, plus a runoff swing of SD RUNOFF_SD margin points (about 4 on each share).

The published win probability for a polled Louisiana House race is the share of simulations a
candidate wins.
"""
import numpy as np
import pandas as pd

SHARE_SD = 4.0      # ~1.25 x the race margin model's held-out share MAE (3.2 pts), normal-equivalent
RUNOFF_SD = 8.0     # runoff MARGIN noise: ~4 pts on each finalist's share (turnout, consolidation)
RUNS = 20000


def seat_probs(df, share_col="share", party_col="party", runs=RUNS, seed=2026):
    """Per-candidate chance to win the seat; df holds one Louisiana House race per race_id."""
    out = pd.Series(np.nan, index=df.index)
    rng = np.random.default_rng(seed)
    for rid, g in df.groupby("race_id"):
        n = len(g)
        if n == 1:
            out[g.index] = 1.0
            continue
        base = np.clip(g[share_col].to_numpy(float), 0.1, None)
        party = g[party_col].astype(str).to_numpy()
        sims = np.clip(base[None, :] + rng.normal(0, SHARE_SD, (runs, n)), 0.1, None)
        sims = 100 * sims / sims.sum(axis=1, keepdims=True)
        order = np.argsort(-sims, axis=1)
        a, b = order[:, 0], order[:, 1]
        rows = np.arange(runs)
        outright = sims[rows, a] > 50
        pa, pb = party[a], party[b]
        same_a = party[None, :] == pa[:, None]
        same_b = party[None, :] == pb[:, None]
        other = ~(same_a | same_b)
        va = np.where(pa == pb, sims[rows, a], (sims * same_a).sum(axis=1)) + 0.5 * (sims * other).sum(axis=1)
        vb = np.where(pa == pb, sims[rows, b], (sims * same_b).sum(axis=1)) + 0.5 * (sims * other).sum(axis=1)
        if (pa == pb).any():                         # same-party final: the rest splits evenly
            rest = 100 - sims[rows, a] - sims[rows, b]
            va = np.where(pa == pb, sims[rows, a] + rest / 2, va)
            vb = np.where(pa == pb, sims[rows, b] + rest / 2, vb)
        a_wins = (va - vb + rng.normal(0, RUNOFF_SD, runs)) > 0
        winner = np.where(outright | a_wins, a, b)
        out[g.index] = np.bincount(winner, minlength=n)[:n] / runs
    return out
