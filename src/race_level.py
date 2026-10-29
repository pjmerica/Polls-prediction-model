"""Race-level outputs (2026-10-27, audit item 7): the one formula each, shared by training
(models/poll/race_win_model.py, race_margin_model.py) and serving (predict.py, predict_margin.py,
explain_2026.py), so the published number is the number that was evaluated.

- race_win_probs: within-race softmax of the race ranker's scores (plus the minor-candidate
  offset) -> win probabilities that sum to 1 in every race.
- race_margins: each candidate's predicted vote share minus the best rival's predicted share ->
  one candidate ahead per race, the top two's margins equal and opposite.
"""
import numpy as np


def softmax(scores, temp):
    s = np.asarray(scores, dtype=float) / temp
    e = np.exp(s - np.nanmax(s))
    return e / e.sum()


# Candidates polling under this share get the fitted minor offset (2026-10-27): on held-out 2018-24
# none of 478 such candidates won, but a softmax over ranker scores alone gave them 7 expected wins,
# and no temperature fixes both ends (lower it and the 70-90% band turns overconfident).
MINOR_POLL_MAX = 10.0


def minor_flag(df):
    return (df["poll_avg"] < MINOR_POLL_MAX).astype(float)


def race_win_probs(df, score_col, temp, minor_offset=0.0, minor=None):
    """Within-race softmax of score / temp, plus minor_offset (log scale) for minor candidates.
    df needs race_id and, when minor_offset != 0 and no `minor` mask is given, poll_avg (the
    no-polls model passes its own mask: party not DEM/REP)."""
    z = df[score_col] / temp
    if minor_offset:
        z = z + minor_offset * (minor if minor is not None else minor_flag(df))
    z = z - z.groupby(df["race_id"]).transform("max")
    e = np.exp(z)
    return e / e.groupby(df["race_id"]).transform("sum")


class Ensemble:
    """The served model = the AVERAGE of the top tuned settings (2026-10-27, audit item 6 / M5):
    picking one winner from a near-tie flipped settings on retrain after retrain and moved races
    by settings alone (TX-Sen, ME-Sen on 10-27). Scores (ranker) or shares (regressor) are the
    mean over the members; SHAP values of a mean are the mean of the members' SHAP values."""

    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def shap(self, X):
        import shap
        vals = [shap.TreeExplainer(m)(X) for m in self.models]
        return (np.mean([v.values for v in vals], axis=0),
                np.mean([v.base_values for v in vals], axis=0))


def grid_neighbors(params, grid):
    """Every setting one grid step away from `params` in one hyperparameter (2026-10-28): the
    local refinement after a trainer's random search. Random search finds the right region;
    this finds the best point in it, at a fraction of the cost of more random draws."""
    out = []
    for k, vals in grid.items():
        if params.get(k) not in vals:
            continue
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(params)
                q[k] = vals[j]
                out.append(q)
    return out


def _pkey(p):
    return tuple(sorted(p.items()))


def refine_search(stage1, grid, score, sort_key, top=5, rounds=8):
    """stage1: [(score, params), ...] from the random search, sorted best first. Scores every
    untried one-step neighbour of the `top` best; repeats from the new top while the top set keeps
    changing (a hill-climb, at most `rounds` passes - 2026-10-29: one pass let the best walk only one
    step from where the random sample landed). Returns the merged list, sorted best first."""
    out = sorted(stage1, key=sort_key)
    seen = {_pkey(p) for _, p in out}
    for _ in range(rounds):
        before = [_pkey(p) for _, p in out[:top]]
        new = []
        for _, p in out[:top]:
            for q in grid_neighbors(p, grid):
                k = _pkey(q)
                if k not in seen:
                    seen.add(k)
                    new.append(q)
        if not new:
            break
        out = sorted(out + [(score(q), q) for q in new], key=sort_key)
        if [_pkey(p) for _, p in out[:top]] == before:
            break
    return out


def sample_grid(grid, n, seed=0):
    """n distinct random settings from `grid`, reproducible for a seed, without building the full
    product (the 2026-10-29 win grid has ~1.3 million combinations)."""
    import random
    rng = random.Random(seed)
    keys = list(grid)
    total = 1
    for k in keys:
        total *= len(grid[k])
    n = min(n, total)
    out, seen = [], set()
    while len(out) < n:
        p = {k: rng.choice(grid[k]) for k in keys}
        k_ = _pkey(p)
        if k_ not in seen:
            seen.add(k_)
            out.append(p)
    return out


def load_ensemble(meta, data_dir, kind):
    """Load a race-level artifact's members (meta["ensemble_files"]; a lone model before 10-27).
    kind: "ranker" or "regressor"."""
    import os
    import xgboost as xgb
    files = meta.get("ensemble_files") or [meta["model_file"]]
    models = []
    for f in files:
        m = xgb.XGBRanker() if kind == "ranker" else xgb.XGBRegressor()
        m.load_model(os.path.join(data_dir, f))
        models.append(m)
    return Ensemble(models)


def race_margins(df, share_col):
    """Each candidate's share minus the best OTHER candidate's share in the same race.

    A one-candidate field (never published: tests/test_publish.py requires two in polled races;
    one 1998-2016 training race) is measured against the remainder of the vote."""
    s = df[share_col]
    g = s.groupby(df["race_id"])
    top1 = g.transform("max")
    top2 = g.transform(lambda v: v.nlargest(2).iloc[-1] if len(v) > 1 else np.nan)
    n_top = (s == top1).groupby(df["race_id"]).transform("sum")
    # the leader is measured against the runner-up; everyone else against the leader. Tied
    # leaders are both measured against the other leader (margin 0).
    best_other = top1.where((s < top1) | (n_top > 1), top2)
    return (s - best_other).fillna(2 * s - 100)
