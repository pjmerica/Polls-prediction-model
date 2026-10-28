"""src/roster_align.py: which polled names count as a roster candidate (2026-10-27)."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import features as F  # noqa: E402
import roster_align as RA  # noqa: E402


def field(*names):
    return pd.DataFrame({"candidate": names, "cand_key": [F.norm_name(n) for n in names]})


def test_same_key():
    assert RA.match_one("Rich McCormick", "mccormick r", field("Rich McCormick", "Tony Kozycki")) == ("mccormick r", "key")


def test_nickname():
    # GA-7: polls and results say Anthony, the roster says Tony; OH-Sen special: Bill / William
    assert RA.match_one("Anthony Kozycki", "kozycki a", field("Tony Kozycki", "Rich McCormick")) == ("kozycki t", "nickname")
    assert RA.match_one("Bill Redpath", "redpath b", field("William Redpath", "Jon Husted")) == ("redpath w", "nickname")


def test_unique_surname():
    # NM-1: Ndidiamaka / Didi is no nickname pair, but she is the race's only Okpareke
    assert RA.match_one("Ndidiamaka Okpareke", "okpareke n", field("Didi Okpareke", "Melanie Stansbury"))[1] == "surname"


def test_shared_surname_needs_a_nickname():
    # two roster candidates named Jones: a polled "Clyde Jones" matches neither by surname alone
    f = field("Doug Jones", "Bob Jones")
    assert RA.match_one("Clyde Jones", "jones c", f) == (None, None)
    assert RA.match_one("Robert Jones", "jones r", f) == ("jones b", "nickname")


def test_no_match():
    assert RA.match_one("Haley Stevens", "stevens h", field("Abdul El-Sayed", "Mike Rogers")) == (None, None)


def test_match_roster_flags_unknowns():
    d = pd.DataFrame({"race_id": ["R", "R", "Q"], "candidate": ["Tony Kozycki", "Ghost Person", "A B"],
                      "cand_key": ["kozycki t", "person g", "b a"]})
    roster = pd.DataFrame({"race_id": ["R", "R"], "candidate": ["Tony Kozycki", "Rich McCormick"],
                           "cand_key": ["kozycki t", "mccormick r"]})
    m = RA.match_roster(d, roster).set_index("candidate")
    assert m.loc["Tony Kozycki", "match"] == "key"
    assert m["roster_key"].isna().sum() == 2
