# -*- coding: utf-8 -*-
"""Build the PRIMARY-election training table: long poll rows + nominee labels.

Sources:
- data/primary_polls_wikipedia.csv  (2018-2024 Senate+Governor primary polling tables from
  Wikipedia race pages, via fetch_primary_polls_wikipedia.py driving the polling-agg 2026
  scraper's parser. NOTE: 538's poll CSVs never carried downballot regular-primary rows in
  ANY era - verified against in-season Wayback captures - so Wikipedia is the only source.)
- data/primary_dates_hist.csv       (per-race primary DATES extracted from the same pages
  by fetch_primary_dates.py; needed for days-to-primary recency features)
- data/res_*.csv                    (general-election candidate lists -> LABELS: the primary
  winner is, by definition, the party's general-election candidate)

Label rule: won = candidate's norm_name appears among that party's general-election
candidates for the same (cycle, state, office, district). No primary returns needed.
Known blind spots (accepted, counted, printed): nominees who later withdrew/were replaced,
write-in nominees, fusion tickets.

Scope (MVP):
- Regular partisan primaries only: stage == 'primary'. Jungle primaries ('jungle primary'),
  runoffs ('primary runoff') and top-two/RCV states (CA, WA, LA, AK) are EXCLUDED - they
  are a different prediction target ('advance', not 'win the nomination').
- DEM and REP fields only.

Output: data/primary_polls_long.csv (committed; small).
    py -X utf8 build_primary_dataset.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import os

import numpy as np
import pandas as pd

import features as F



HERE = ROOT   # repo root (paths.py) - this file lives in a subfolder
DATA = os.path.join(HERE, "data")

EXCLUDE_STATES = {"CA", "WA", "LA", "AK"}   # top-two / jungle / RCV: different target

STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
}

def to_abbr(s):
    s = str(s).strip()
    return s.upper() if len(s) == 2 else STATE_ABBR.get(s, s)

OFFICE = {"senate": "Senate", "governor": "Governor", "house": "House"}

def load_wiki_polls():
    p = pd.read_csv(os.path.join(DATA, "primary_polls_wikipedia.csv"), low_memory=False)
    p = p[p["stage"].astype(str).str.lower() == "primary"].copy()   # no runoffs
    parts = p["race_id"].astype(str).str.split("-", expand=True)    # 2022-SEN-PA
    p["office"] = parts[1].map({"SEN": "Senate", "GOV": "Governor"})
    p["state"] = parts[2]
    p = p[~p["state"].isin(EXCLUDE_STATES)]
    p["party_std"] = p["party"].map(F.npar)
    p = p[p["party_std"].isin(["DEM", "REP"])]
    p = p[~p["candidate"].map(F.is_junk_answer)]

    dates = pd.read_csv(os.path.join(DATA, "primary_dates_hist.csv"))
    dmap = dict(zip(dates["page"], dates["primary_date"]))

    out = pd.DataFrame({
        "year": pd.to_numeric(p["year"], errors="coerce").astype("Int64"),
        "state": p["state"], "office": p["office"], "district": "",
        "party_std": p["party_std"],
        "candidate": p["candidate"],
        "cand_key": p["candidate"].map(F.norm_name),
        "pct": pd.to_numeric(p["implied_prob"], errors="coerce") * 100.0,
        "pollster": p["pollster"],
        "sample_size": pd.to_numeric(p["sample_size"], errors="coerce"),
        "population": (p["population"] if "population" in p.columns else None),
        "end_date": p["end_date"],
        "election_date": p["src_page"].map(dmap),
    }).dropna(subset=["pct", "cand_key", "year"])
    # dedup on the NORMALIZED pollster, and NOT on pct (2026-07-31): the old key used the raw
    # pollster string plus pct, so it only caught byte-identical rows - the same survey under
    # two source spellings ("Mitchell Research" / "Mitchell Research & Communications") with
    # slightly different rounding (27 vs 28) survived as two independent polls. predict_primary
    # got the identical fix; keeping the two keys in sync is the never-fork rule.
    out["_pollster_key"] = out["pollster"].map(F.norm_pollster)
    n_before = len(out)
    out = out.drop_duplicates(subset=["_pollster_key", "end_date", "year", "state", "office",
                                      "party_std", "cand_key"]).drop(columns="_pollster_key")
    if n_before - len(out):
        print(f"  cross-source duplicate poll rows dropped: {n_before - len(out)}")
    return out

def nominee_sets():
    """{(cycle, state, office, district, party): set(norm_name of general candidates)}."""
    frames = []
    for fn, office in [("res_senate.csv", "Senate"), ("res_house.csv", "House"),
                       ("res_governor.csv", "Governor")]:
        r = pd.read_csv(os.path.join(DATA, fn), low_memory=False)
        r = r[r["stage"].astype(str).str.lower() == "general"]
        r["office"] = office
        r["state"] = r["state_abbrev"].map(to_abbr)
        r["district"] = ("" if office != "House"
                         else r["office_seat_name"].map(F.pdist))
        r["party_std"] = r["ballot_party"].map(F.npar)
        r["cand_key"] = r["candidate_name"].map(F.norm_name)
        frames.append(r[["cycle", "state", "office", "district", "party_std", "cand_key"]])
    allr = pd.concat(frames).dropna(subset=["cand_key"])
    allr = allr[allr["party_std"].isin(["DEM", "REP"])]
    noms = {}
    for r in allr.itertuples():
        noms.setdefault((r.cycle, r.state, r.office, r.district, r.party_std),
                        set()).add(r.cand_key)
    return noms

def main():
    d = load_wiki_polls()
    print(f"primary poll rows after scope filters: {len(d)}")
    print(d.groupby("year").size().to_string())

    # contamination guard: some Wikipedia pages leak GENERAL-election polls into primary
    # sections (section-context bleed; seen on NC-2022). A primary poll cannot end after
    # its primary (+2d grace for odd date entry).
    ed = pd.to_datetime(d["election_date"], errors="coerce")
    end = pd.to_datetime(d["end_date"], errors="coerce", format="mixed")
    bad = ed.notna() & end.notna() & (end > ed + pd.Timedelta(days=2))
    if bad.any():
        print(f"dropped {int(bad.sum())} rows dated AFTER their primary "
              f"(general-poll leakage into primary sections)")
        d = d[~bad]

    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", "")
                    + "_" + d["party_std"])

    # merge nickname variants of the SAME candidate within a race ('Bobby' vs 'Robert
    # Charles' split one person's polls across two keys - see features_primary._NICK)
    from features_primary import merge_nickname_aliases
    d, n_merged = merge_nickname_aliases(d)
    print(f"nickname-alias merges: {n_merged} candidate name variants unified")

    noms = nominee_sets()
    key = list(zip(d["year"], d["state"], d["office"], d["district"], d["party_std"]))
    d["won"] = [int(ck in noms.get(k, set())) for ck, k in zip(d["cand_key"], key)]

    # PREFER actual primary-results winners (fetch_primary_results_2026.py --hist) over the
    # nominee-join: a primary winner who later dropped out appears nowhere in the general
    # candidate lists and the REPLACEMENT gets mislabeled (Platner scenario). Results are
    # the truth; the nominee-join stays as fallback for races without parsed results.
    res_path = os.path.join(DATA, "primary_results_hist.csv")
    if os.path.exists(res_path):
        res = pd.read_csv(res_path)
        winners = {r.race_id: set() for r in res.itertuples()}
        for r in res[res["is_winner"]].itertuples():
            winners.setdefault(r.race_id, set()).add(r.cand_key)
        covered = d["race_id"].isin(winners.keys())

        # nickname guard: 'Bobby Charles' (polls) vs 'Robert Charles' (results) share a
        # last name but not a first initial. If the winner's exact key is absent from the
        # race's polled field, fall back to a UNIQUE last-name match within that field.
        def match_winner(rid, field_keys):
            wset = winners.get(rid, set())
            hit = wset & field_keys
            if hit:
                return hit
            out = set()
            for w in wset:
                last = w.split(" ")[0]
                same = {k for k in field_keys if k.split(" ")[0] == last}
                if len(same) == 1:
                    out |= same
            return out
        field_by_race = d.groupby("race_id")["cand_key"].agg(set).to_dict()
        win_by_race = {rid: match_winner(rid, field_by_race[rid])
                       for rid in d["race_id"].unique() if rid in winners}
        d["won_res"] = [int(ck in win_by_race[rid]) if rid in win_by_race else None
                        for ck, rid in zip(d["cand_key"], d["race_id"])]
        both = d[covered].groupby("race_id").agg(
            nom=("won", "max"), res_lbl=("won_res", "max"))
        # races where results found a winner but the nominee-join disagreed on WHO
        dis = 0
        for rid, g in d[covered].groupby("race_id"):
            a = set(g.loc[g["won"] == 1, "cand_key"])
            b = set(g.loc[g["won_res"] == 1, "cand_key"])
            if b and a != b:
                dis += 1
                print(f"  label disagreement {rid}: nominee-join={sorted(a) or '(none)'} "
                      f"vs results={sorted(b)} -> using results")
        use = covered & d["won_res"].notna()
        d.loc[use, "won"] = d.loc[use, "won_res"].astype(int)
        n_races_cov = d.loc[covered, "race_id"].nunique()
        print(f"results-based labels: {n_races_cov} races covered "
              f"({dis} disagreed with the nominee-join)")
        d = d.drop(columns="won_res")

    d["race_has_label"] = d.groupby("race_id")["won"].transform("max")

    races = d.groupby("race_id").agg(n_cands=("cand_key", "nunique"),
                                     labeled=("race_has_label", "max"),
                                     year=("year", "first"))
    print(f"\nprimary races: {len(races)} | with a matched nominee: "
          f"{int(races['labeled'].sum())} ({races['labeled'].mean():.0%})")
    print("races by cycle:", races.groupby("year").size().to_dict())
    print("labeled races by cycle:",
          races[races["labeled"] == 1].groupby("year").size().to_dict())

    # keep LABELED races with a real contest (>=2 polled candidates). Unlabeled races are
    # name-match failures or replaced nominees - dropping them is a selection we accept
    # and count here rather than silently.
    keep = d[(d["race_has_label"] == 1)].copy()
    multi = keep.groupby("race_id")["cand_key"].transform("nunique") >= 2
    keep = keep[multi]
    print(f"\nkept: {keep['race_id'].nunique()} contested labeled races, {len(keep)} poll rows")

    out = os.path.join(DATA, "primary_polls_long.csv")
    keep.drop(columns=["race_has_label"]).to_csv(out, index=False)
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
