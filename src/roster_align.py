"""Match published candidates to the November roster (2026-10-27, audit check added 10-25).

The roster (data/general_roster_2026.csv, from each state's candidate list) is the authority on
who is on the ballot. The poll feeds print whatever name a pollster used, so a polled candidate's
cand_key can differ from the roster's: "Anthony Kozycki" (polls, results page, bio) is "Tony" on
the roster (GA-7), "Bill" Redpath (polls and his candidacy history since 2008) is "William"
(OH-Sen special). Those feed keys are KEPT: the bio table and candidate history are keyed on them,
and renaming to the roster's spelling would break those joins. A spelling that breaks a join is
fixed in data/name_aliases.csv instead (Ndidiamaka -> Didi Okpareke, NM-1).

match_roster() pairs each published candidate with one roster candidate of the same race: the
same key, or the same surname plus nickname-equivalent first names (features_primary's table,
the one the nickname merge uses), or a surname no other roster candidate in the race shares.
tests/test_publish.py requires every polled candidate to be matched, to a distinct roster entry,
and every roster nominee to match the primary winner.
"""
import os

import pandas as pd

import features as F
import features_primary as FP

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def surname(cand_key):
    """cand_key is "<surname> <first initial>" (features.norm_name)."""
    return str(cand_key).rsplit(" ", 1)[0]


def _first_and_mids(name):
    parts = str(name).replace(".", " ").replace(",", " ").split()
    return (parts[0] if parts else ""), FP._middle_names(parts)


def nickname_equiv(a, b):
    """First names that are the same, a prefix of each other, a known nickname pair, or one
    side's middle name used as the other's first name. The surname is not checked here."""
    fa, ma = _first_and_mids(a)
    fb, mb = _first_and_mids(b)
    return bool(fa and fb) and (FP._first_equiv(fa, fb) or FP._cross_name_equiv(fa, ma, fb, mb))


def load_roster(cycle=2026):
    r = pd.read_csv(os.path.join(HERE, "data", f"general_roster_{cycle}.csv"))
    return r[r["dropped"].fillna(0) != 1]


def match_one(name, cand_key, field):
    """field: the roster rows of one race (candidate, cand_key). Returns the matched roster
    cand_key and how it matched ("key", "nickname", "surname"), or (None, None)."""
    if cand_key in set(field["cand_key"]):
        return cand_key, "key"
    same = field[field["cand_key"].map(surname) == surname(cand_key)]
    nick = same[[nickname_equiv(name, n) for n in same["candidate"]]]
    if len(nick) == 1:
        return nick["cand_key"].iloc[0], "nickname"
    if len(same) == 1:
        return same["cand_key"].iloc[0], "surname"
    return None, None


def match_roster(d, roster):
    """One row per published (race_id, cand_key) with roster_key and how it matched."""
    by_race = {rid: g for rid, g in roster.groupby("race_id")}
    rows = []
    for x in d.drop_duplicates(["race_id", "cand_key"]).itertuples():
        field = by_race.get(x.race_id)
        key, how = (None, None) if field is None else match_one(x.candidate, x.cand_key, field)
        rows.append({"race_id": x.race_id, "cand_key": x.cand_key, "candidate": x.candidate,
                     "roster_key": key, "match": how})
    return pd.DataFrame(rows)
