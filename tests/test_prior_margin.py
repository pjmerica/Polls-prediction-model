"""prior_margin reads the latest real Democrat-vs-Republican race (2026-10-25).

Run from the repo root:  py -m pytest tests
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import features as F          # noqa: E402
import results_labels as RL   # noqa: E402


@pytest.fixture(scope="module")
def mm():
    return F.load_fundamentals()["margin_map"]


def pm(mm, year, state, office, district=""):
    return F.prior_margin(mm, year, state, office, district)


# races whose previous same-office race was not D vs R step back to the last one that was
@pytest.mark.parametrize("year,state,office,district,expect,why", [
    (2026, "ME", "Senate", "", -8.6, "King (IND) won 2024 -> Collins v Gideon 2020"),
    (2026, "AK", "Senate", "", -12.7, "Murkowski v Tshibaka (R v R) 2022 -> Sullivan v Gross 2020"),
    (2026, "WA", "House", "4", -35.3, "Newhouse v Sessler (R v R) 2024 -> 2022"),
    (2020, "ME", "Senate", "", -37.0, "King 2018 -> Collins v Bellows 2014"),
])
def test_steps_back_past_non_dvr(mm, year, state, office, district, expect, why):
    assert pm(mm, year, state, office, district) == pytest.approx(expect, abs=0.1), why


# DEM-slot independents (model_party_overrides_hist.csv) count as the Democrat
def test_dem_slot_independent_counts_as_democrat(mm):
    # NV-2 2024: Amodei 55.0 vs Kidd (IND, DEM slot) 36.1 - was read as Amodei vs nobody (-55.0)
    assert pm(mm, 2026, "NV", "House", "2") == pytest.approx(-18.9, abs=0.1)


def test_osborn_race_unchanged(mm):
    # Osborn 2024 is already on the DEM line: NE-Sen 2026 keeps his own race, not Sasse 2020
    assert pm(mm, 2026, "NE", "Senate") == pytest.approx(-6.7, abs=0.1)


def test_one_party_seat_keeps_safe_seat_value(mm):
    # no Democrat ran in GA-11 2014: the -100 is a real safe-seat signal and stays
    assert pm(mm, 2016, "GA", "House", "11") == pytest.approx(-100.0, abs=0.1)


def test_jungle_runoffs_kept(mm):
    # a D-vs-R runoff after a jungle round is the seat's real margin and must survive the filter
    for key, v in RL.jungle_runoff_margins(F.DATA_DIR).items():
        assert mm.get(key) == pytest.approx(v), key


def test_no_strong_non_dvr_race_left(mm):
    # every remaining margin comes from a D-vs-R race, a runoff, or a one-party seat
    allres = RL.aggregate_candidates(RL.load_result_lines(F.DATA_DIR))
    ov = F.load_hist_party_overrides()
    allres["p"] = [ov.get((int(y), s, o, F.dist_str(d), k), p) for y, s, o, d, k, p in zip(
        allres["year"], allres["state"], allres["office"], allres["district"],
        allres["cand_key"], allres["res_party"])]
    runoff = RL.jungle_runoff_margins(F.DATA_DIR)
    top2 = (allres.dropna(subset=["vote_pct"]).sort_values("vote_pct", ascending=False)
                  .groupby(["year", "state", "office", "district"]).head(2))
    bad = []
    for key, g in top2.groupby(["year", "state", "office", "district"]):
        v = mm.get(key)
        if key in runoff or v is None or np.isnan(v) or len(g) < 2:
            continue
        if set(g["p"]) == {"DEM", "REP"}:
            continue
        if g["p"].iloc[0] == g["p"].iloc[1] or (
                ~g["p"].isin(["DEM", "REP"]) & (g["vote_pct"] >= F.PRIOR_OTHER_MIN)).any():
            bad.append(key)
    assert not bad, bad[:10]
