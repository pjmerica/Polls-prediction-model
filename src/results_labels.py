# -*- coding: utf-8 -*-
"""General-election results -> ONE label row per (race, candidate).

Shared by pipeline/build/build_dataset.ipynb (the training labels in
polls_long_with_results.csv), tools/repair_result_labels.py (the in-place repair of that
file) and features.load_fundamentals (prior-cycle margins). NEVER-FORK: before 2026-09-25
each of those read the 538 results archives its own way, and all of them were wrong in the
same two places.

Why the archives are not one row per candidate:

  1. FUSION VOTING (NY, CT, SC, plus stray write-in lines elsewhere). A candidate appears once
     per BALLOT LINE. Chris Murphy, CT-Sen 2024: DEM 55.83 + WFP 2.75 = 58.58. The build
     kept whichever line came first in the file, so his training `vote_pct` was 2.75 and the
     margin model was taught that a 17-point winner lost by 53. 117 labelled candidates in 60+
     NY/CT races carried a single minor-party line as their vote share.

  2. RANKED-CHOICE ROUNDS (ME, AK). A candidate appears once per ROUND, and the build again
     kept the first row in file order, which mixed rounds within one race (2018 ME-2: Golden's
     round 1 = 45.6 against Poliquin's round 3 = 49.4). We use ROUND 1 for everyone: it is
     what polls measure (first choice) and what every non-RCV race measures (plurality).
     `won` still comes from the archive's winner flag, so a come-from-behind RCV winner keeps
     won=1 with a negative first-round margin - the known, documented disagreement.

Two rows with the same cand_key but different candidate ids in one race are different PEOPLE
(1998 GA-5: John Lewis DEM 78.5 vs a REP Lewis 21.5). Their lines are never summed; the
larger one keeps the key and the collision is reported.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths  # noqa: F401,E402

import os

import numpy as np
import pandas as pd

RESULT_FILES = (("res_senate.csv", "Senate"), ("res_house.csv", "House"),
                ("res_governor.csv", "Governor"))

LABEL_COLS = ["won", "vote_pct", "res_party", "res_candidate", "race_winning_pct",
              "best_other_pct"]


def _race_id(year, state, office, district):
    return (year.astype(int).astype(str) + "_" + state + "_" + office
            + district.radd("-").where(district != "", ""))


def load_result_lines(data_dir=None):
    """Every general-election ballot line, one row each, with a race_id matching the polls'."""
    import features as F   # lazy: features imports this module inside load_fundamentals
    data_dir = data_dir or paths.DATA
    frames = []
    for fn, office in RESULT_FILES:
        r = pd.read_csv(os.path.join(data_dir, fn), low_memory=False)
        r = r[r["stage"].astype(str).str.lower().str.contains("general", na=False)]
        special = r["special"].astype(str).str.lower().isin(["true", "1"])
        out = pd.DataFrame({
            "year": pd.to_numeric(r["cycle"], errors="coerce"),
            "state": r["state_abbrev"].astype(str).str.upper(),
            "office": office,
            "district": (r["office_seat_name"].map(F.pdist) if office == "House" else ""),
            "special": special,
            "res_candidate": r["candidate_name"],
            # the 'party' column is all-null in these files; ballot_party holds DEM/REP/WFP/...
            "line_party": r["ballot_party"].map(F.npar),
            "vote_pct": pd.to_numeric(r["percent"], errors="coerce"),
            "won": r["winner"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
                   .astype(int),
            "rcv_round": pd.to_numeric(r["ranked_choice_round"], errors="coerce"),
            # person identity for fusion sums; the name is the last resort
            "cand_uid": (r["candidate_id"].astype("Int64").astype(str)
                         .where(r["candidate_id"].notna(),
                                r["politician_id"].astype("Int64").astype(str)
                                .where(r["politician_id"].notna(), r["candidate_name"]))),
        })
        out["district"] = out["district"].fillna("")
        # dual-seat fix: a non-House SPECIAL gets its own key (district 'S'); the rare House
        # special+regular collision (NY-19 2022) is dropped rather than mislabeled
        out.loc[out["special"] & (out["office"] != "House"), "district"] = "S"
        out = out[~(out["special"] & (out["office"] == "House"))]
        frames.append(out)
    lines = pd.concat(frames, ignore_index=True)
    lines["cand_key"] = lines["res_candidate"].map(
        lambda s: F.norm_name(s) if isinstance(s, str) else None)
    lines = lines.dropna(subset=["year", "cand_key"])
    lines["year"] = lines["year"].astype(int)
    lines["race_id"] = _race_id(lines["year"], lines["state"], lines["office"],
                                lines["district"])
    return lines


def aggregate_candidates(lines, verbose=False):
    """Ballot lines -> one row per (race_id, cand_key) with the candidate's TOTAL round-1 vote."""
    L = lines.copy()
    # RCV: keep each race's first round only (rows without a round number count as round 1)
    rnd = L["rcv_round"].fillna(1)
    L = L[rnd == rnd.groupby(L["race_id"]).transform("min")]

    # FUSION: sum a person's lines. res_party = their biggest DEM/REP line if they have one
    # (a WFP or Conservative line never outranks the major-party line they share it with).
    L = L.assign(_major=L["line_party"].isin(["DEM", "REP"]).astype(int))
    L = L.sort_values(["_major", "vote_pct"], ascending=False)
    g = L.groupby(["race_id", "cand_key", "cand_uid"], sort=False)
    per = g.agg(year=("year", "first"), state=("state", "first"), office=("office", "first"),
                district=("district", "first"), res_candidate=("res_candidate", "first"),
                res_party=("line_party", "first"), vote_pct=("vote_pct", "sum"),
                won=("won", "max"), n_lines=("vote_pct", "size")).reset_index()

    # same key, different people: keep the larger, report it
    per = per.sort_values("vote_pct", ascending=False)
    clash = per.duplicated(["race_id", "cand_key"], keep=False)
    if verbose and clash.any():
        print("cand_key collisions (different people, same key) - larger kept:")
        print(per[clash].sort_values(["race_id", "vote_pct"])
              [["race_id", "res_candidate", "res_party", "vote_pct", "won"]].to_string(index=False))
    per = per.drop_duplicates(["race_id", "cand_key"], keep="first")

    # race-level context on the aggregated field (NOT the polled subset)
    per["race_winning_pct"] = per.groupby("race_id")["vote_pct"].transform("max")
    import features as F
    per["best_other_pct"] = per.groupby("race_id")["vote_pct"].transform(F.best_other)
    per.loc[per.groupby("race_id")["vote_pct"].transform("size") == 1, "best_other_pct"] = np.nan
    if verbose:
        print(f"result lines {len(lines)} -> candidate-races {len(per)} "
              f"(fusion-summed: {(per['n_lines'] > 1).sum()})")
    return per.drop(columns="n_lines").reset_index(drop=True)


def general_day(year):
    """First Tuesday after the first Monday in November."""
    d = pd.Timestamp(f"{int(year)}-11-01")
    return d + pd.Timedelta(days=(7 - d.dayofweek) % 7 + 1)


# Generals that really were held off the normal day (Hurricane Gustav moved LA-2 and LA-4
# to 2008-12-06); their results rows ARE that election, so their polls are correctly labelled.
POSTPONED_GENERALS = {"2008_LA_House-2", "2008_LA_House-4"}


def drop_off_date_polls(polls, verbose=False):
    """Drop polls of a DIFFERENT election that share the November race's key (2026-09-25).

    Both poll sources file House SPECIALS (HI-1 May 2010, FL-13 Mar 2014, AZ-8 Apr 2018,
    NY-3 Feb 2024, ...) and Senate RUNOFFS (GA Dec 2008, LA) under the regular race. The
    results side keeps only the November general, so a spring special's polls were labelled
    with November's result - HI-1 2010 read Hanabusa as the winner of the special Djou won.
    Keyed on the poll's own election date. Senate/Governor specials (district 'S') keep their
    polls: they are their own race with their own result row (MA-Sen Jan 2010).
    """
    ed = pd.to_datetime(polls["election_date"], format="mixed", errors="coerce").dt.normalize()
    off = ed.notna() & (ed != polls["year"].map(general_day))
    own_race = (polls["office"] != "House") & (polls["district"].astype(str) == "S")
    drop = off & ~own_race & ~polls["race_id"].isin(POSTPONED_GENERALS)
    if verbose:
        print(f"off-date polls dropped (specials/runoffs filed under the November race): "
              f"{int(drop.sum())} rows in {polls.loc[drop, 'race_id'].nunique()} races")
    return polls[~drop]


def surname(cand_key):
    return str(cand_key).split(" ")[0]


def attach_labels(polls, labels, verbose=False):
    """Left-join result labels onto poll rows by (race_id, cand_key).

    NICKNAME FALLBACK: a poll row whose exact key misses gets the label of the ONE results
    DEM/REP candidate in the same race with the same surname AND party, provided no other polled
    candidate already took that results row. 'Robert Menendez' (polls) vs 'Bob Menendez'
    (results) and 'Charles William Young' vs 'Bill Young' silently dropped the WINNER of four
    races from training before this.
    """
    p = polls.drop(columns=[c for c in LABEL_COLS + ["has_result"] if c in polls.columns])
    lab = labels[["race_id", "cand_key"] + LABEL_COLS]
    m = p.merge(lab, on=["race_id", "cand_key"], how="left", indicator=True)
    m["has_result"] = (m["_merge"] == "both").astype(int)
    m = m.drop(columns="_merge")

    exact = set(zip(m.loc[m["has_result"] == 1, "race_id"], m.loc[m["has_result"] == 1, "cand_key"]))
    miss = m[(m["has_result"] == 0) & m["race_id"].isin(labels["race_id"])]
    lb = labels.assign(_sn=labels["cand_key"].map(surname))
    lb = lb[~lb.set_index(["race_id", "cand_key"]).index.isin(list(exact))]
    fb = []
    for (rid, ck, party), _ in miss.groupby(["race_id", "cand_key", "party_std"]):
        if party not in ("DEM", "REP"):   # a minor-party surname match is too loose to trust
            continue
        c = lb[(lb["race_id"] == rid) & (lb["_sn"] == surname(ck)) & (lb["res_party"] == party)]
        if len(c) == 1:
            fb.append((rid, ck, party, c.iloc[0]))
    fb_keys = {}
    for rid, ck, party, row in fb:
        fb_keys.setdefault((rid, row["cand_key"]), []).append((ck, party, row))
    n = 0
    for (rid, rkey), hits in fb_keys.items():
        if len(hits) != 1:          # two different polled names want one result row: skip
            continue
        ck, party, row = hits[0]
        idx = (m["race_id"] == rid) & (m["cand_key"] == ck) & (m["party_std"] == party)
        for col in LABEL_COLS:
            m.loc[idx, col] = row[col]
        m.loc[idx, "has_result"] = 1
        n += int(idx.sum())
        if verbose:
            print(f"  nickname match: {rid}  polls '{ck}' -> results '{row['res_candidate']}'"
                  f" ({int(idx.sum())} rows)")
    if verbose:
        print(f"nickname fallback labelled {n} poll rows")
    return m
