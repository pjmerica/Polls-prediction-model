"""District presidential lean (2026-10-27): House races get their own district's lean in
`dist_pres_lean_cand`, next to the state lean `pres_lean_cand` (user: keep both columns). Before,
a House race only had the state's lean, so CO-5 (an R+9 Trump seat in a D+12 state) read as
Democratic territory."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import paths  # noqa: E402,F401  (src/ on sys.path)
import features as F  # noqa: E402


@pytest.fixture(scope="module")
def leans():
    pres = F.load_pres_lean()
    return pres, F.load_dist_pres_lean(pres)


def test_every_2026_district_covered(leans):
    _, d = leans
    n = sum(1 for (y, _, _) in d if y == 2026)
    assert n == 435, f"2026 districts with a lean: {n}"


def test_co5_is_republican_though_colorado_is_not(leans):
    pres, d = leans
    assert F.pres_lean_asof(pres, 2026, "CO") > 5          # the state leans Democratic
    assert d[(2026, "CO", "5")] < -5                         # the district leans Republican


def test_lean_is_district_minus_nation(leans):
    pres, d = leans
    # AK at-large equals the state: the district lean and the state lean must agree
    assert abs(d[(2026, "AK", "1")] - pres[(2024, "AK")]) < 0.5
    assert abs(d[(2022, "AK", "1")] - pres[(2020, "AK")]) < 0.5


def test_documented_gaps_stay_missing(leans):
    _, d = leans
    assert not any(y in (1998, 2000) for (y, _, _) in d)
    assert (2016, "FL", "7") not in d and (2002, "GA", "3") not in d
    assert np.isfinite(d[(2018, "PA", "7")])                 # the 2018 court map is covered
