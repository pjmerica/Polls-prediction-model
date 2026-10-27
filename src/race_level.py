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


def race_win_probs(df, score_col, temp, minor_offset=0.0):
    """Within-race softmax of score / temp, plus minor_offset (log scale) for minor candidates.
    df needs race_id and, when minor_offset != 0, poll_avg."""
    z = df[score_col] / temp
    if minor_offset:
        z = z + minor_offset * minor_flag(df)
    z = z - z.groupby(df["race_id"]).transform("max")
    e = np.exp(z)
    return e / e.groupby(df["race_id"]).transform("sum")


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
