"""2026 House incumbency on the redrawn maps (2026-10-28).

races.csv keyed incumbency by district NUMBER, so CA-3 read Republican (Kiley's old seat) while
Democrat Ami Bera, redistricted from CA-6, runs there, and brand-new seats inherited a holding party.
Training saw the 2022 redraw's new seats as 'new_seat' (no party), so 2026 must match.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import features as F  # noqa: E402


def test_2026_redraws_are_known():
    for st in ("TX", "NC", "OH", "CA", "UT", "FL", "TN", "LA", "AL"):
        assert F._last_redraw(st, 2026) == 2026, st
    assert F._last_redraw("MO", 2026) == 2022      # votes on its 2022 lines (referendum)
    assert F._last_redraw("CA", 2024) == 2022      # training years unchanged


def test_roster_sets_the_new_districts_incumbent_party():
    m = {(2026, "CA", "House", "3"): "REP", (2026, "TX", "House", "9"): "DEM",
         (2026, "TX", "House", "23"): "REP", (2026, "PA", "House", "7"): "DEM"}
    F._apply_2026_redrawn_incumbents(m)
    assert m[(2026, "CA", "House", "3")] == "DEM"           # Bera, from the old 6th
    assert m[(2026, "CA", "House", "41")] == "DEM"          # Sanchez, from the old 38th
    assert F.npar(m[(2026, "TX", "House", "9")]) == "OTH"   # new seat: no party holds it
    assert F.npar(m[(2026, "CA", "House", "6")]) == "OTH"   # Kiley, now an independent
    assert m[(2026, "TX", "House", "23")] == "REP"          # vacancy keeps races.csv
    assert m[(2026, "PA", "House", "7")] == "DEM"           # not a redrawn state: untouched
