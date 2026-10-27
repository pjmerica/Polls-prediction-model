"""Publish gate (2026-10-26, audit item 1): checks on the files a refresh just wrote to outputs/.

src/refresh_dashboard.py runs `pytest tests` after predicting and before copying anything into
polling-agg, so a failure here stops the publish (locally and in polling-agg's Model refresh
workflow). Run by hand after a refresh:  py -m pytest tests -q -rxX

Hard checks block. The model-shape checks at the bottom (raw odds summing to 1, margins
cancelling, one winner per race, win and margin favourites agreeing) are expected to fail while
the win and margin models score each candidate separately (audit M1-M3); the race-level models
(audit item 7) fix them by construction, and then they become hard checks.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")
UNPOLLED_FAILED = os.environ.get("MODEL_UNPOLLED_FAILED") == "1"
KEY = ["race_id", "cand_key"]

# generic ballot (DEM minus REP) outside this range is a broken fetch, not a national mood
NATL_ENV_RANGE = (-15.0, 15.0)
# a feature filled in at least this share of training rows must not be empty at serve time...
FILL_MIN_TRAIN = 0.05
# ...nor fall this far below its training fill rate (the old poll_momentum gap: 1.0 -> 0.0)
FILL_MAX_DROP = 0.5


def _csv(name):
    return pd.read_csv(os.path.join(OUT, name), low_memory=False)


def _meta(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def win():
    return _csv("predictions_2026.csv")


@pytest.fixture(scope="module")
def margin():
    return _csv("margin_predictions_2026.csv")


@pytest.fixture(scope="module")
def fund():
    if UNPOLLED_FAILED:
        pytest.skip("no-polls step failed this run: polling-agg keeps its last good copy")
    return _csv("fundamentals_predictions_2026.csv")


@pytest.fixture(scope="module")
def metas():
    m = {"win": _meta("predictions_2026_meta.json"),
         "margin": _meta("margin_predictions_2026_meta.json")}
    if not UNPOLLED_FAILED:
        m["fundamentals"] = _meta("fundamentals_predictions_2026_meta.json")
    return m


@pytest.fixture(scope="module")
def roster():
    r = pd.read_csv(os.path.join(DATA, "general_roster_2026.csv"))
    return r[(r["dropped"].fillna(0) != 1) & (r["write_in"].fillna(0) != 1)]


# ---- the files themselves -------------------------------------------------------------------
@pytest.mark.parametrize("which", ["win", "fund"])
def test_win_probabilities_sane(which, request):
    d = request.getfixturevalue(which)
    for col in ("win_prob", "win_prob_norm"):
        assert d[col].notna().all(), f"{which}: missing {col}"
        assert d[col].between(0, 1).all(), f"{which}: {col} outside [0, 1]"
    sums = d.groupby("race_id")["win_prob_norm"].sum()
    assert np.allclose(sums, 1, atol=1e-6), f"{which}: race odds don't sum to 1: {sums[~np.isclose(sums, 1)].head().to_dict()}"
    dup = d[d.duplicated(KEY, keep=False)]
    assert dup.empty, f"{which}: candidate listed twice: {dup[KEY].values[:5].tolist()}"


def test_margins_sane(margin):
    assert margin["pred_margin"].notna().all(), "missing pred_margin"
    assert margin["pred_margin"].abs().le(100).all(), "margin beyond 100 pts"
    assert not margin.duplicated(KEY).any(), "candidate listed twice in the margin file"


def test_polled_races_have_two_candidates(win):
    n = win.groupby("race_id").size()
    assert (n >= 2).all(), f"polled race with one candidate: {list(n[n < 2].index)}"


def test_one_candidate_races_are_uncontested(fund, roster):
    # a no-polls race may have one candidate only when the roster lists only one
    n = fund.groupby("race_id").size()
    on_ballot = roster.groupby("race_id").size()
    bad = [r for r in n[n < 2].index if on_ballot.get(r, 0) > 1]
    assert not bad, f"contested races published with one candidate: {bad}"


def test_win_and_margin_score_the_same_candidates(win, margin):
    a, b = set(zip(win["race_id"], win["cand_key"])), set(zip(margin["race_id"], margin["cand_key"]))
    assert a == b, f"only in win: {sorted(a - b)[:6]}; only in margin: {sorted(b - a)[:6]}"


def test_every_roster_race_scored_once(win, fund, roster):
    polled, unpolled = set(win["race_id"]), set(fund["race_id"])
    assert not polled & unpolled, f"scored by both models: {sorted(polled & unpolled)[:6]}"
    want = set(roster["race_id"])
    assert polled | unpolled == want, (f"unscored: {sorted(want - polled - unpolled)[:6]}; "
                                       f"not on the roster: {sorted((polled | unpolled) - want)[:6]}")


def test_no_withdrawn_candidate_published(win, margin, roster, request):
    do = pd.read_csv(os.path.join(DATA, "dropped_out_2026.csv"))
    gone = set(zip(do["race_id"], do["cand_key"]))
    full = pd.read_csv(os.path.join(DATA, "general_roster_2026.csv"))
    gone |= set(zip(full.loc[full["dropped"].fillna(0) == 1, "race_id"],
                    full.loc[full["dropped"].fillna(0) == 1, "cand_key"]))
    frames = {"win": win, "margin": margin}
    if not UNPOLLED_FAILED:
        frames["fund"] = request.getfixturevalue("fund")
    for name, d in frames.items():
        hit = sorted(set(zip(d["race_id"], d["cand_key"])) & gone)
        assert not hit, f"{name}: withdrawn candidates published: {hit[:6]}"


# ---- one run, one feed ----------------------------------------------------------------------
def test_files_come_from_one_run(metas, win, margin):
    ids = {k: m.get("run_id") for k, m in metas.items()}
    assert all(ids.values()), f"run id missing (run through src/refresh_dashboard.py): {ids}"
    assert len(set(ids.values())) == 1, f"files from different runs: {ids}"
    if "fundamentals" in metas:
        assert metas["fundamentals"].get("polled_run_id") == ids["win"], \
            "the no-polls model read another run's polled races"
    w, m = metas["win"], metas["margin"]
    for k in ("polls_max_end_date", "n_poll_rows", "natl_env"):
        assert w.get(k) == m.get(k), f"win and margin read different inputs: {k} {w.get(k)} vs {m.get(k)}"
    assert w["n_races"] == win["race_id"].nunique() and m["n_races"] == margin["race_id"].nunique()


def test_generic_ballot_in_range(metas):
    for k, m in metas.items():
        v = m.get("natl_env")
        assert v is not None, f"{k}: generic ballot missing"
        assert NATL_ENV_RANGE[0] <= v <= NATL_ENV_RANGE[1], f"{k}: generic ballot {v:+.1f}"
    vals = {round(m["natl_env"], 6) for m in metas.values()}
    assert len(vals) == 1, f"models used different generic-ballot numbers: {vals}"


def test_feature_fill_matches_training(metas):
    with open(os.path.join(DATA, "feature_fill_train.json")) as f:
        train = json.load(f)
    bad = []
    for k, m in metas.items():
        live = m.get("feature_fill")
        assert live, f"{k}: no feature_fill in its meta"
        for feat, t in train[k].items():
            if t is None or t < FILL_MIN_TRAIN:
                continue
            v = live.get(feat)
            assert v is not None, f"{k}: {feat} not recorded at serve time"
            if v == 0 or v < t - FILL_MAX_DROP:
                bad.append(f"{k}:{feat} train {t:.2f} live {v:.2f}")
    assert not bad, "features empty at serve time: " + "; ".join(bad[:12])


# ---- inputs ---------------------------------------------------------------------------------
def test_candidate_bios_rebuild_matches(tmp_path):
    # candidate_bios.csv is BUILT; a hand edit or a source change that was never rebuilt shows here
    import paths  # noqa: F401  (pipeline folders on sys.path)
    import build_office_level_table as B
    out = tmp_path / "candidate_bios.csv"
    B.main(out=str(out))
    def text(p):   # line endings aside (a Windows checkout may hold CRLF)
        with open(p, encoding="utf-8") as f:
            return f.read().replace("\r\n", "\n")
    assert text(out) == text(os.path.join(DATA, "candidate_bios.csv")), \
        "data/candidate_bios.csv differs from a rebuild: run pipeline/build/build_office_level_table.py"


# ---- model shape (expected to fail until the race-level models, audit item 7) ---------------
SHAPE = pytest.mark.xfail(reason="per-candidate win and margin models (audit M1-M3); "
                                 "the race-level models fix this - then drop the xfail")


@SHAPE
def test_raw_win_odds_sum_to_one(win):
    s = win.groupby("race_id")["win_prob"].sum()
    off = s[~s.between(0.9, 1.1)]
    assert off.empty, f"{len(off)} races: {off.round(2).to_dict()}"


@SHAPE
def test_two_candidate_margins_cancel(margin):
    two = margin.groupby("race_id").filter(lambda g: len(g) == 2)
    s = two.groupby("race_id")["pred_margin"].sum()
    off = s[s.abs() > 1]
    assert off.empty, f"{len(off)} races: {off.round(1).to_dict()}"


@SHAPE
def test_one_predicted_winner_per_race(margin):
    n = (margin["pred_margin"] > 0).groupby(margin["race_id"]).sum()
    assert (n == 1).all(), f"two winners: {list(n[n > 1].index)}; none: {list(n[n == 0].index)}"


@SHAPE
def test_win_and_margin_favourites_agree(win, margin):
    wf = win.loc[win.groupby("race_id")["win_prob"].idxmax()].set_index("race_id")["cand_key"]
    mf = margin.loc[margin.groupby("race_id")["pred_margin"].idxmax()].set_index("race_id")["cand_key"]
    split = list(wf.index[wf != mf.reindex(wf.index)])
    assert not split, f"{len(split)} races: {split}"
