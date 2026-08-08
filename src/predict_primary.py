# -*- coding: utf-8 -*-
"""Predict 2026 PRIMARY nominee probabilities from the polling-agg raw feed.

    py -X utf8 predict_primary.py [--cycle 2026] [--polls path.csv ...] [--out preds.csv]

Same input discipline as predict.py (raw feed columns only, no network), same scope rules
as training (build_primary_dataset.py): regular DEM/REP partisan primaries; jungle/top-two
states (CA, WA, LA, AK) excluded.

Primary DATES (for days-to-primary recency features), in priority order per (state,office):
1. 538-format *_current.csv primary rows' per-race election_date (most precise),
2. polling-agg data/raw/primaries.json per-state date (Ballotpedia),
3. polling-agg data/processed/primary_calendar_2026.json (committed accumulator; dates are
   per-state maxima so may include runoffs - fallback only).

Output: primary_predictions_2026.csv (+ _meta.json sidecar) - one row per candidate per
primary race, with win_prob (raw) and win_prob_norm (within-race simplex).

field_confidence / low_confidence_field (added 2026-07-22): win_prob_norm rescales the raw
per-candidate scores to sum to 1 within a race, which ALWAYS produces a leader - even when
the model cannot separate the field at all. The flag exists to say which of those two is
happening. Motivating case: SD-Governor-REP, where four candidates' raw probabilities summed
to 0.064 and normalizing turned a 3.3% candidate into a 51.8% "front-runner."

    field_confidence = win_prob_norm.max() - 1/n        # n = candidates in the race
    low_confidence_field = field_confidence < 0.10

i.e. how far the leader sits above a uniform field. ~0 means the model is effectively
guessing; the maximum is 1 - 1/n, so the ceiling scales with field size.

NB the definition CHANGED when the model became a ranker (see _field_conf below): it used to
be the raw probability sum with a 0.30 threshold, which is what this docstring described
until 2026-08-05. The RANKING win_prob_norm implies is honest either way - it is the same
ordering the raw scores gave. The flag is about whether to trust the PROBABILITY attached to
it: treat a leader in a low_confidence_field race as "best guess among weak options," not
"the model is confident."
"""
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
import features_primary as FP
from build_primary_dataset import EXCLUDE_STATES, to_abbr
from predict import (DEFAULT_POLLS, REQUIRED_FEED_COLS, parse_race_id,
                     drop_stale_candidates, warn_near_duplicate_names)

import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
from paths import ROOT, AGG   # one definition for the whole repo (paths.py)
import paths as _paths   # module handle: `out` is a very common local
                         # variable name in this repo, so never import it bare

HERE = ROOT

def primary_dates(cycle):
    """{(state, office): Timestamp} + {state: Timestamp} fallbacks."""
    per_race, per_state = {}, {}
    for fn, office in [("senate_current.csv", "Senate"),
                       ("governor_current.csv", "Governor"),
                       ("house_current.csv", "House")]:
        p = os.path.join(HERE, "data", fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, low_memory=False)
        d = d[d["stage"].astype(str) == "primary"]
        d = d[pd.to_numeric(d["cycle"], errors="coerce") == cycle]
        d["st"] = d["state"].map(to_abbr)
        for st, grp in d.groupby("st"):
            ed = pd.to_datetime(grp["election_date"], errors="coerce").dropna()
            if len(ed):
                per_race[(st, office)] = ed.mode().iloc[0]
    pj = os.path.join(AGG, "data", "raw", "primaries.json")
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as f:
            for r in json.load(f).get("races", []):
                st, dt = r.get("state_abbrev"), r.get("date_iso")
                if st and dt and "runoff" not in str(r.get("description", "")).lower():
                    ts = pd.Timestamp(dt)
                    if st not in per_state or ts < per_state[st]:
                        per_state[st] = ts       # earliest non-runoff = the primary
    cal = os.path.join(AGG, "data", "processed", "primary_calendar_2026.json")
    if os.path.exists(cal):
        with open(cal, encoding="utf-8") as f:
            for st, dt in json.load(f).items():
                per_state.setdefault(st, pd.Timestamp(dt))
    # LAST-RESORT static fallback. Both upstream calendars cover 44 states and omit the
    # same four (AR, IL, MS, OH) - they are early-2026 primaries and the scrapers appear
    # to list only UPCOMING races, so these silently dropped off once they had been held.
    # Without a date, days_to_elec is NaN and ALL 14 recency features go NaN with it -
    # including poll_last30/poll_last/gap_x_recency, 44.8% of the primary margin model's
    # gain. 22 races were affected (found 2026-08-05).
    # Dates are cited in the file; IL and OH are corroborated by their own results
    # reporting. Entries here are only used if no scraper supplied the state.
    manual = os.path.join(HERE, "data", f"primary_dates_{cycle}_manual.csv")
    if os.path.exists(manual):
        md = pd.read_csv(manual)
        for _, row in md.iterrows():
            per_state.setdefault(str(row["state"]), pd.Timestamp(row["primary_date"]))
    return per_race, per_state

def load_primary_feed(paths, cycle):
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p, low_memory=False)
        missing = REQUIRED_FEED_COLS - set(df.columns)
        assert not missing, f"feed schema drift in {p}: missing {sorted(missing)}"
        df["_src_priority"] = i
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    # Same source-labelling reconciliation the general feed does (F.normalize_special_race_id):
    # NYT's '2026-SEN-FL-S' and Wikipedia's '2026-SEN-FL' are the same primary.
    raw["race_id"] = raw["race_id"].map(F.normalize_special_race_id)
    parsed = raw["race_id"].map(parse_race_id)
    ok = parsed.notna()
    raw = raw[ok].copy()
    raw[["year", "office", "state", "district"]] = pd.DataFrame(parsed[ok].tolist(),
                                                                index=raw.index)
    raw = raw[raw["year"] == cycle]
    raw = raw[raw["stage"].astype(str).str.lower() == "primary"]   # no runoffs (own round)
    raw = raw[~raw["candidate"].map(F.is_junk_answer)]
    raw = raw[~raw["state"].isin(EXCLUDE_STATES)]

    d = pd.DataFrame({
        "year": raw["year"].astype(int), "state": raw["state"], "office": raw["office"],
        "district": raw["district"].map(F.dist_str),
        "candidate": raw["candidate"],
        "party_std": raw["party"].map(F.npar),
        "pct": pd.to_numeric(raw["implied_prob"], errors="coerce") * 100.0,
        "end_date": raw["end_date"],
        "sample_size": pd.to_numeric(raw["sample_size"], errors="coerce"),
        "population": (raw["population"] if "population" in raw.columns else None),
        "pollster": raw["pollster"], "_src_priority": raw["_src_priority"],
    })
    d = d[d["party_std"].isin(["DEM", "REP"])]
    # Correct known feed misspellings BEFORE keying, or a one-character typo splits a
    # candidate into two (see F.load_name_aliases). Rewrites the display name too, so the
    # dashboard shows the right spelling.
    d["candidate"] = F.apply_name_aliases(d["candidate"])
    d["cand_key"] = d["candidate"].map(F.norm_name)
    d = d.dropna(subset=["pct", "cand_key"])

    drop_path = os.path.join(HERE, "data", "dropped_out_2026.csv")
    if os.path.exists(drop_path):
        do = pd.read_csv(drop_path)
        # SCOPED BY RACE. The file has always carried a race_id column but the filter used
        # to ignore it and drop by cand_key globally - fine while every entry was a true
        # withdrawal, but wrong for a candidate who left ONE race and is active in another
        # (Jared Moskowitz: out in FL-23 after redistricting, running in FL-25). A global
        # drop would have deleted him from both.
        # The file's race_id omits the party suffix ("2026_FL_House-23"), so match on the
        # race_id prefix rather than equality.
        key = d["year"].astype(str) + "_" + d["state"] + "_" + d["office"] + \
              d["district"].map(lambda x: f"-{x}" if str(x) not in ("", "nan", "None") else "")
        mask = pd.Series(False, index=d.index)
        for _, row in do.iterrows():
            mask |= (d["cand_key"] == row["cand_key"]) & (key == row["race_id"])
        n = int(mask.sum())
        if n:
            print(f"dropped-out candidates removed: {n} poll rows")
        unmatched = [r["cand_key"] for _, r in do.iterrows()
                     if not ((d["cand_key"] == r["cand_key"]) & (key == r["race_id"])).any()]
        if unmatched:
            print(f"  NOTE: {len(unmatched)} dropout entries matched no poll rows: {unmatched}")
        d = d[~mask]

    # race_id built BEFORE merge/dedup (both need it) - merge runs before dedup so a
    # cross-source name variant (e.g. NYT "Thomas (Jay) Feely" vs Wikipedia "Jay Feely" for
    # the SAME poll) collapses to one cand_key first; only then can dedup recognize the two
    # rows as the same survey and drop the duplicate, instead of silently double-counting it
    # under one merged name (found 2026-07-22 auditing AZ-01's primary predictions).
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", "")
                    + "_" + d["party_std"])
    d, n_merged = FP.merge_nickname_aliases(d)
    if n_merged:
        print(f"nickname-alias merges: {n_merged} candidate name variants unified")

    # dedup on the NORMALIZED pollster (2026-07-31): keying on the raw string let the same
    # survey survive twice under two source spellings ("Glengariff Group, Inc." 41.4 and
    # "Glengariff Group" 41.0 for 2026-07-11; likewise Mitchell Research, Susquehanna,
    # Rosetta Stone), silently double-counting it in every poll aggregate. MI-Sen-DEM carried
    # 99 poll rows for 36 real surveys before this.
    d["_pollster_key"] = d["pollster"].map(F.norm_pollster)
    n_before = len(d)
    d = (d.sort_values("_src_priority")
           .drop_duplicates(subset=["_pollster_key", "end_date", "year", "state", "office",
                                    "district", "party_std", "cand_key"], keep="first")
           .drop(columns=["_src_priority", "_pollster_key"]))
    if n_before - len(d):
        print(f"cross-source duplicate poll rows dropped: {n_before - len(d)}")

    per_race, per_state = primary_dates(cycle)
    d["election_date"] = [per_race.get((st, of), per_state.get(st))
                          for st, of in zip(d["state"], d["office"])]
    no_date = d["election_date"].isna()
    if no_date.any():
        print(f"WARNING: no primary date for "
              f"{sorted(d.loc[no_date, 'state'].unique())} - recency features NaN there")

    d = drop_stale_candidates(F.prepare_polls(d))

    bad_pct = ~d["pct"].between(0, 100)
    assert bad_pct.mean() < 0.01, "feed pct out of [0,100] - implied_prob scale changed?"
    assert d["race_id"].nunique() >= 10, "suspiciously few primary races parsed"
    warn_near_duplicate_names(d)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "data", "primary_model_features.json")) as f:
        meta = json.load(f)
    # RANKER (2026-07-29): the primary model is a learning-to-rank model, not a classifier -
    # it scores candidates so the nominee ranks highest WITHIN a race; a within-race softmax
    # turns those scores into ONE coherent probability used by both the dashboard and the
    # Explain modal (the old independent-classifier + divide-by-sum gave 2+ strong candidates
    # ~50/50 and made the explainer's raw number disagree with the dashboard's normalized one).
    model = xgb.XGBRanker()
    model.load_model(os.path.join(HERE, "data", "primary_model_xgb.json"))
    SOFTMAX_TEMP = float(meta.get("softmax_temp", 1.0))
    print(f"primary model: XGBRanker, cycles {meta['trained_on_cycles']}, "
          f"{len(meta['features'])} features, {meta['n_races']} training races")

    d = load_primary_feed(args.polls, args.cycle)
    print(f"primary races: {d['race_id'].nunique()} | poll rows {len(d)}")

    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    from candidate_history import CandidateHistory
    cand = FP.build_primary_table(d, fec=fec, inc_map=funds["inc_map"],
                                  hist=CandidateHistory(), bios=FP.load_candidate_bios())

    missing = [f for f in meta["features"] if f not in cand.columns]
    assert not missing, f"artifact expects features absent from the table: {missing[:8]}"
    X = cand.reindex(columns=meta["features"])
    cand["rank_score"] = model.predict(X)          # raw ranker score (unbounded, per-race)
    # within-race softmax -> ONE coherent probability. win_prob and win_prob_norm are now the
    # SAME thing (kept both column names for downstream/dashboard back-compat).
    import numpy as np
    def _softmax(s):
        v = np.asarray(s, dtype=float) / SOFTMAX_TEMP
        e = np.exp(v - np.nanmax(v))
        return e / e.sum()
    cand["win_prob_norm"] = cand.groupby("race_id", group_keys=False)["rank_score"].transform(_softmax)
    cand["win_prob"] = cand["win_prob_norm"]
    # field confidence, redefined for the ranker: how much the leader's softmax prob exceeds a
    # uniform field (1/n). ~0 => the model can't separate the field (best-guess ranking);
    # high => a clear front-runner. Replaces the old "raw prob sum" heuristic.
    def _field_conf(g):
        n = len(g)
        return float(g["win_prob_norm"].max() - 1.0 / n) if n else 0.0
    fc = cand.groupby("race_id").apply(_field_conf, include_groups=False)
    cand["field_confidence"] = cand["race_id"].map(fc)
    cand["low_confidence_field"] = (cand["field_confidence"] < 0.10).astype(int)
    surveys = d.groupby("race_id").apply(
        lambda g: g.groupby(["pollster", "end_date"]).ngroups, include_groups=False)
    cand["n_surveys"] = cand["race_id"].map(surveys).fillna(0).astype(int)

    # POLL AGE. n_polls alone hides the difference between "1 poll last week" and "1 poll
    # last September". The latter is where the model's worst calls come from: it scores a
    # year-old name recognition snapshot as if it were the current field, so a candidate who
    # has since WITHDRAWN can sit at 99% (CT-Gov-REP had Erin Stewart there on a 357-day-old
    # poll while the actual nominee showed 0.4%). Season-to-date, this pattern accounts for
    # several of the model's confident misses - Hogan 100% in MD-Gov-REP, Gowdy 99% in
    # SC-Sen-REP, Crenshaw 98% in TX-2.
    # Surfaced as a column so the staleness is visible rather than inferred from n_polls.
    newest = d.groupby("race_id")["end_date"].max()
    cand["newest_poll"] = cand["race_id"].map(newest)
    asof = pd.Timestamp(datetime.date.today())
    cand["poll_age_days"] = (asof - pd.to_datetime(cand["newest_poll"], errors="coerce")).dt.days
    cand["stale_polling"] = (cand["poll_age_days"] > 90).astype(int)
    n_stale = cand.drop_duplicates("race_id")["stale_polling"].sum()
    if n_stale:
        print(f"STALE POLLING: {int(n_stale)} races have no poll in the last 90 days - "
              f"their fields may have changed (withdrawals, new entrants) since the last "
              f"survey. See the stale_polling / poll_age_days columns.")

    n_low_conf = cand.drop_duplicates("race_id")["low_confidence_field"].sum()
    if n_low_conf:
        print(f"low-confidence fields (leader barely above a uniform 1/n split): "
              f"{int(n_low_conf)} races - the win_prob_norm leader there is a best-guess "
              f"ranking, not a confident pick (see field_confidence column)")

    out_cols = ["race_id", "state", "office", "district", "party", "candidate",
                "election_date", "n_polls", "n_surveys", "poll_avg", "poll_lead",
                "fund_share", "rank_score", "win_prob", "win_prob_norm",
                "field_confidence", "low_confidence_field",
                "poll_age_days", "stale_polling"]
    out = cand[out_cols].sort_values(["race_id", "win_prob_norm"], ascending=[True, False])
    out_path = args.out or _paths.out(f"primary_predictions_{args.cycle}.csv")
    out.to_csv(out_path, index=False)

    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(dict(generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                       polls_max_end_date=str(d["end_date"].max().date()),
                       n_poll_rows=int(len(d)),
                       n_races=int(out["race_id"].nunique())), f, indent=1)

    picks = out.loc[out.groupby("race_id")["win_prob"].idxmax()]
    print(f"saved -> {out_path} ({out['race_id'].nunique()} races)")
    print("\nclosest fields (leader's normalized prob < 60%):")
    close = picks[picks["win_prob_norm"] < 0.60]
    print(close[["race_id", "candidate", "poll_avg", "win_prob_norm"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
