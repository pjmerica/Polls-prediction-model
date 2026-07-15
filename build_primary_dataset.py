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
import os

import numpy as np
import pandas as pd

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
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
        "end_date": p["end_date"],
        "election_date": p["src_page"].map(dmap),
    }).dropna(subset=["pct", "cand_key", "year"])
    out = out.drop_duplicates(subset=["pollster", "end_date", "year", "state", "office",
                                      "party_std", "cand_key", "pct"])
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

    noms = nominee_sets()
    key = list(zip(d["year"], d["state"], d["office"], d["district"], d["party_std"]))
    d["won"] = [int(ck in noms.get(k, set())) for ck, k in zip(d["cand_key"], key)]
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
