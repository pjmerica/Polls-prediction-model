"""Publish gate (2026-10-26, audit item 1): checks on the files a refresh just wrote to outputs/.

src/refresh_dashboard.py runs `pytest tests` after predicting and before copying anything into
polling-agg, so a failure here stops the publish (locally and in polling-agg's Model refresh
workflow). Run by hand after a refresh:  py -m pytest tests -q -rxX

Hard checks block. The model-shape checks at the bottom (odds summing to 1, margins cancelling,
one winner per race) became hard checks on 2026-10-27 with the race-level win and margin models
(audit item 7), which make them true by construction. The win and margin models picking different
favourites is reported, not failed (test_models_split_is_reported; user 2026-10-29): they are
trained separately, and those races are flagged on the pages (audit item 3).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
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


# fields where several candidates of one party are real co-candidates in November: top-two
# (CA, WA), top-four (AK) and Louisiana's all-party first round
MULTI_PARTY_STATES = {"CA", "WA", "AK", "LA"}


@pytest.mark.parametrize("which", ["win", "fund"])
def test_one_nominee_per_party(which, request):
    # elsewhere two Democrats (or Republicans) in a race are leftover hypothetical matchups, and
    # the pages add a party's candidates up (audit item 2), so the gate keeps them out
    d = request.getfixturevalue(which)
    d = d[~d["state"].isin(MULTI_PARTY_STATES) & d["party"].isin(["DEM", "REP"])]
    n = d.groupby(["race_id", "party"]).size()
    assert (n == 1).all(), f"{which}: several nominees of one party: {list(n[n > 1].index)[:6]}"


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


def test_every_polled_candidate_is_on_the_roster(win):
    # audit check added 10-25: the 5 off-ballot candidates found by hand on 10-24 (Roth, Beaudion,
    # Mujica, Villee, Schmidt) were polled names with no roster entry. Feed spellings are allowed
    # (src/roster_align.py: same surname plus a nickname, or a surname unique in the race), but
    # each published candidate must match exactly one roster candidate, and no two the same one.
    import roster_align as RA
    m = RA.match_roster(win, RA.load_roster())
    off = m[m["roster_key"].isna()]
    assert off.empty, f"published but not on the roster: {off[['race_id', 'candidate']].values[:8].tolist()}"
    dup = m[m.duplicated(["race_id", "roster_key"], keep=False)]
    assert dup.empty, f"two published candidates are one roster entry: {dup[['race_id', 'candidate']].values[:8].tolist()}"


def test_roster_nominees_won_their_primaries(roster):
    # audit check added 10-25: ME-2 was published with the primary's round-1 leader (Baldacci), not
    # its ranked-choice winner (Dunlap). Every decided party primary's winner must be that party's
    # candidate on the roster, unless a committee replaced them (replacement_nominees_2026.csv) or a
    # checked exception says the winner is not on the ballot (nominee_exceptions_2026.csv).
    import features as F
    import roster_align as RA
    p = pd.read_csv(os.path.join(DATA, "primary_results_2026.csv"))
    p = p[p["is_winner"].astype(bool)].copy()
    p["general"] = p["race_id"].str.rsplit("_", n=1).str[0]
    skip = set()
    for f in ("replacement_nominees_2026.csv", "nominee_exceptions_2026.csv"):
        x = pd.read_csv(os.path.join(DATA, f))
        skip |= set(zip(x["race_id"], x["party"].map(F.npar)))
    by_race = {rid: g for rid, g in roster.groupby("race_id")}
    bad, unknown = [], []
    for (rid, party), grp in p.groupby(["general", "party"]):
        # Senate specials: the primary results drop the "-S" (2026_OH_Senate -> 2026_OH_Senate-S)
        rid = rid if rid in by_race else (rid + "-S" if rid + "-S" in by_race else rid)
        if (rid, F.npar(party)) in skip:
            continue
        if rid not in by_race:
            unknown.append(rid)
            continue
        field = by_race[rid][by_race[rid]["party"].map(F.npar) == F.npar(party)]
        if not any(RA.match_one(x.candidate, x.cand_key, field)[0] for x in grp.itertuples()):
            bad.append((rid, party, grp["candidate"].tolist(), field["candidate"].tolist()))
    assert not unknown, f"primary results for races not on the roster: {sorted(set(unknown))[:8]}"
    assert not bad, f"roster nominee is not the primary winner: {bad[:6]}"


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
    # each aggregator behind it (predict.py's meta, 2026-10-27): every one in range too
    for src, v in metas["win"].get("natl_env_sources") or []:
        assert NATL_ENV_RANGE[0] <= v <= NATL_ENV_RANGE[1], f"generic ballot source {src}: {v:+.1f}"
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


def test_seat_simulation_keeps_the_published_odds(win):
    # the copula keeps each race's odds, so the average seat count must equal the odds' sum
    # (src/simulate_seats.py; Monte Carlo noise at 20,000 runs is ~0.1 seat)
    sim = _meta("seat_sim_2026.json")
    assert sim.get("run_id") == _meta("predictions_2026_meta.json").get("run_id"), "seat sim from another run"
    for office, s in sim["chambers"].items():
        assert abs(s["dem_seats"]["mean"] - s["expected_dem_from_odds"]) < 0.6, \
            f"{office}: simulated {s['dem_seats']['mean']} vs odds {s['expected_dem_from_odds']}"
        assert s["p_dem_control"] + s["p_rep_control"] + s["p_neither"] == pytest.approx(1, abs=1e-3)
        assert s["races"] + sum(s["not_up"].values()) == s["total"], f"{office}: seats don't add up"


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


# ---- model shape (race-level models since 2026-10-27, audit item 7) ------------------------
def test_raw_win_odds_sum_to_one(win):
    s = win.groupby("race_id")["win_prob"].sum()
    off = s[~np.isclose(s, 1, atol=1e-6)]
    assert off.empty, f"{len(off)} races: {off.round(2).to_dict()}"


def test_two_candidate_margins_cancel(margin):
    two = margin.groupby("race_id").filter(lambda g: len(g) == 2)
    s = two.groupby("race_id")["pred_margin"].sum()
    off = s[s.abs() > 1e-6]
    assert off.empty, f"{len(off)} races: {off.round(1).to_dict()}"


def test_margins_come_from_the_shares(margin):
    # every published margin is its predicted share minus the best rival's (src/race_level.py)
    import paths  # noqa: F401  (src/ on sys.path)
    from race_level import race_margins
    gap = (race_margins(margin, "pred_share") - margin["pred_margin"]).abs()
    # 1e-4 pts: the CSV holds shares and margins as printed floats
    assert gap.max() < 1e-4, f"margins not derived from the shares: {margin.loc[gap.idxmax(), KEY].tolist()}"


def test_one_predicted_winner_per_race(margin):
    n = (margin["pred_margin"] > 0).groupby(margin["race_id"]).sum()
    assert (n == 1).all(), f"two winners: {list(n[n > 1].index)}; none: {list(n[n == 0].index)}"


def test_models_split_is_reported(win, margin):
    # The win and margin models are trained separately, so they can pick different favourites in a
    # close race. That is a legitimate result, not a failure (user 2026-10-29): the pages flag those
    # races "models split" and mute the market gap (audit item 3). Reported here, never blocking;
    # what IS checked is that every race has exactly one favourite in each file.
    wf = win.loc[win.groupby("race_id")["win_prob"].idxmax()].set_index("race_id")["cand_key"]
    mf = margin.loc[margin.groupby("race_id")["pred_margin"].idxmax()].set_index("race_id")["cand_key"]
    assert set(wf.index) == set(mf.index), "win and margin files cover different races"
    split = sorted(wf.index[wf != mf.reindex(wf.index)])
    print(f"\nmodels split in {len(split)} races (flagged on the pages): {split}")


def test_no_polls_margins_coherent(fund):
    # the no-polls margin (2026-10-29, race_fundamentals_margin_model.py): blank only in an
    # uncontested race; otherwise derived from the shares, so two-candidate margins cancel and
    # exactly one candidate leads
    if "pred_margin" not in fund.columns:
        pytest.skip("no no-polls margin model in this checkout")
    import paths  # noqa: F401  (src/ on sys.path)
    from race_level import race_margins
    n = fund.groupby("race_id")["race_id"].transform("size")
    assert fund.loc[n > 1, "pred_margin"].notna().all(), "contested no-polls race without a margin"
    assert fund.loc[n == 1, "pred_margin"].isna().all(), "uncontested race given a margin"
    k = fund[n > 1]
    assert k["pred_margin"].abs().le(100).all(), "margin beyond 100 pts"
    gap = (race_margins(k, "pred_share") - k["pred_margin"]).abs()
    assert gap.max() < 1e-4, f"margins not derived from the shares: {k.loc[gap.idxmax(), KEY].tolist()}"
    two = k.groupby("race_id").filter(lambda g: len(g) == 2).groupby("race_id")["pred_margin"].sum()
    assert two.abs().max() < 1e-6, f"two-candidate margins don't cancel: {two[two.abs() > 1e-6].round(1).to_dict()}"
    lead = (k["pred_margin"] > 0).groupby(k["race_id"]).sum()
    assert (lead == 1).all(), f"not one leader: {list(lead[lead != 1].index)[:6]}"
    wf = k.loc[k.groupby("race_id")["win_prob"].idxmax()].set_index("race_id")["cand_key"]
    mf = k.loc[k.groupby("race_id")["pred_margin"].idxmax()].set_index("race_id")["cand_key"]
    split = sorted(wf.index[wf != mf.reindex(wf.index)])
    print(f"\nno-polls win and margin models split in {len(split)} races: {split}")
