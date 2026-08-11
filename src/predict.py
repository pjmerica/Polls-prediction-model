"""Predict win probabilities for FUTURE races from the polling-agg raw poll feed.

    python predict.py [--cycle 2026] [--natl-env 1.5] [--polls path.csv ...] [--out preds.csv]

Inputs (ALL local / already-committed — no network):
- Raw polls: the polling-agg repo's data/raw/{nyt_polls,wikipedia_polls}.csv (default paths
  point at the sibling checkout). Only columns a bare poll feed has are used:
  pollster, candidate, party, stage, sample_size, end_date, implied_prob (pct/100), race_id.
- data/model_xgb.json + data/model_features.json — trained/tuned by model.ipynb (all 14
  cycles, params tuned on 1998-2016).
- data/macro_monthly.csv, data/races.csv, data/res_*.csv — frozen fundamentals.
- Historical polls (polls_long_with_results.csv) — ONLY to compute pollster house effects.

Known gaps (documented in CONCERNS.md):
- --natl-env must be looked up manually (e.g. RealClearPolling generic-ballot average,
  DEM minus REP). If omitted, natl_env_cand is NaN (XGBoost routes missing).
- approval/BLS macro series currently end early 2025; late-window macro stats are stale.

Dedup: the polling-agg raw files contain internal repeats and NYT/Wikipedia cross-source
duplicates (audited 2026-07-05: ~1.3k internal, ~2.7k cross). Collapsed here on
(pollster, end_date, race, candidate), preferring the NYT row.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
import features_primary as FP   # merge_nickname_aliases - shared with the primary path
from cycles import natl_env as natl_env_hist
from macro_features import build_macro

import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
from paths import ROOT, AGG   # one definition for the whole repo (paths.py)
import paths as _paths   # module handle: `out` is a very common local
                         # variable name in this repo, so never import it bare

HERE = ROOT
POLLING_AGG_RAW = os.path.join(AGG, "data", "raw")
DEFAULT_POLLS = [os.path.join(POLLING_AGG_RAW, "nyt_polls.csv"),
                 os.path.join(POLLING_AGG_RAW, "wikipedia_polls.csv")]

OFFICE_CODE = {"SEN": "Senate", "H": "House", "GOV": "Governor"}

def election_date(cycle):
    """First Tuesday after the first Monday in November."""
    d = pd.Timestamp(f"{cycle}-11-01")
    monday = d + pd.Timedelta(days=(7 - d.dayofweek) % 7)   # first Monday (Mon=0)
    return monday + pd.Timedelta(days=1)

def parse_race_id(rid):
    """'2026-SEN-MI' / '2026-H-AL-01' / '2026-SEN-OH-S' -> (year, office, state, district)."""
    parts = str(rid).split("-")
    if len(parts) < 3 or parts[1] not in OFFICE_CODE:
        return None
    year = int(parts[0])
    office = OFFICE_CODE[parts[1]]
    state = parts[2].upper()
    district = ""
    if office == "House" and len(parts) > 3:
        district = F.pdist(parts[3])
    elif len(parts) > 3 and parts[3].upper() == "S":
        district = "S"   # SPECIAL election (e.g. 2026-SEN-FL-S) — its own race, never merged
    return year, office, state, district

REQUIRED_FEED_COLS = {"race_id", "pollster", "candidate", "party", "stage",
                      "sample_size", "end_date", "implied_prob"}

def load_agg_polls(paths, cycle):
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p, low_memory=False)
        missing = REQUIRED_FEED_COLS - set(df.columns)
        assert not missing, f"feed schema drift in {p}: missing columns {sorted(missing)}"
        df["_src_priority"] = i          # earlier path wins dedup ties (NYT first)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # Reconcile the two sources' special-election labelling BEFORE parsing, or the same
    # contest lands under two race_ids (see F.normalize_special_race_id).
    raw["race_id"] = raw["race_id"].map(F.normalize_special_race_id)
    parsed = raw["race_id"].map(parse_race_id)
    ok = parsed.notna()
    raw = raw[ok].copy()
    raw[["year", "office", "state", "district"]] = pd.DataFrame(
        parsed[ok].tolist(), index=raw.index)
    raw = raw[raw["year"] == cycle]
    raw = raw[raw["stage"].astype(str).str.lower().str.contains("general", na=False)]
    raw = raw[~raw["candidate"].map(F.is_junk_answer)]

    d = pd.DataFrame({
        "year": raw["year"].astype(int),
        "state": raw["state"],
        "office": raw["office"],
        "district": raw["district"],
        "candidate": raw["candidate"],
        "party_std": raw["party"].map(F.npar),
        "pct": pd.to_numeric(raw["implied_prob"], errors="coerce") * 100.0,
        "end_date": raw["end_date"],
        "sample_size": pd.to_numeric(raw["sample_size"], errors="coerce"),
        "pollster": raw["pollster"],
        "poll_id": raw.get("poll_id"),
        # question_id separates the MATCHUPS inside one poll. A single survey routinely
        # tests several hypothetical pairings (Glengariff 2025-05-08 tested five), and
        # without this column every pairing's numbers pool together as if they came from
        # one question - see drop_dead_matchups().
        "question_id": raw.get("question_id"),
        "_src_priority": raw["_src_priority"],
    })
    d["election_date"] = election_date(cycle)
    # Correct known feed misspellings BEFORE keying, or a one-character typo splits a
    # candidate into two (see F.load_name_aliases). Rewrites the display name too, so the
    # dashboard shows the right spelling.
    d["candidate"] = F.apply_name_aliases(d["candidate"])
    d["cand_key"] = d["candidate"].map(F.norm_name)
    d = d.dropna(subset=["pct", "cand_key"])

    # candidate-party corrections. Two distinct columns (see the CSV):
    #   model_party   -> what the MODEL treats them as (party_std): fills the two-party slot
    #                    so poll_lead / two-party margin / normalization work. An independent
    #                    who is the de-facto main challenger (Dan Osborn vs Ricketts) is
    #                    modeled DEM here.
    #   display_party -> their REAL affiliation, shown on the dashboard (Osborn = IND).
    # `display_party` rides through as a separate column; party_std stays the model party.
    d["display_party"] = d["party_std"]
    ov_path = os.path.join(HERE, "data", "candidate_party_overrides.csv")
    if os.path.exists(ov_path):
        ov = pd.read_csv(ov_path)
        rid_pre = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                   + d["district"].map(F.dist_str).radd("-").where(
                       d["district"].map(F.dist_str) != "", ""))
        mmap = {(r.race_id, r.cand_key): r.model_party for r in ov.itertuples()}
        dmap = {(r.race_id, r.cand_key): r.display_party for r in ov.itertuples()}
        keys = list(zip(rid_pre, d["cand_key"]))
        m_fixed = pd.Series([mmap.get(k) for k in keys], index=d.index)
        d_fixed = pd.Series([dmap.get(k) for k in keys], index=d.index)
        d.loc[m_fixed.notna(), "party_std"] = m_fixed[m_fixed.notna()].map(F.npar)
        # display_party keeps the raw affiliation string (IND/LIB/...) for a clean label,
        # not npar's OTH bucket
        d.loc[d_fixed.notna(), "display_party"] = d_fixed[d_fixed.notna()].str.upper().str[:3]
        if m_fixed.notna().sum():
            print(f"candidate-party overrides applied: {int(m_fixed.notna().sum())} poll rows")

    # dropped-out candidates: remove their (now-stale) poll rows so they don't linger as
    # phantom options. The 14-day stale filter usually catches them, but an explicit list
    # is unambiguous and self-documents (e.g. Mike Duggan ended his MI-GOV bid).
    drop_path = os.path.join(HERE, "data", "dropped_out_2026.csv")
    if os.path.exists(drop_path):
        do = pd.read_csv(drop_path)
        rid_pre = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                   + d["district"].map(F.dist_str).radd("-").where(
                       d["district"].map(F.dist_str) != "", ""))
        drop_keys = set(zip(do["race_id"], do["cand_key"]))
        mask = pd.Series([(r, c) in drop_keys for r, c in zip(rid_pre, d["cand_key"])],
                         index=d.index)
        if mask.any():
            print(f"dropped-out candidates removed: {int(mask.sum())} poll rows")
            d = d[~mask]

    # dedup: internal repeats + NYT/Wikipedia cross-source duplicates.
    # Keyed on the NORMALIZED pollster (2026-07-31): the raw string missed exactly the
    # cross-source case this dedup exists for - the same survey filed under two spellings
    # ("Glengariff Group, Inc." vs "Glengariff Group", "Mitchell Research" vs "Mitchell
    # Research & Communications") survived as two independent polls and got double-counted
    # in every aggregate. Found in the primary model (MI-Sen-DEM: 99 rows for 36 real
    # surveys); the general feed had the same bug, 46 rows. Same key as predict_primary.py
    # and build_primary_dataset.py - keeping them in sync is the never-fork rule.
    before = len(d)

    # race_id FIRST, because the nickname merge and both dedup passes are race-scoped.
    # (It used to be built after dedup; moved up 2026-08-08 with the nickname port below.)
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].map(F.dist_str).radd("-")
                      .where(d["district"].map(F.dist_str) != "", ""))

    # Nickname/variant merge, ported from the primary path 2026-08-08 (CONCERNS #21).
    # The same person arrives from two sources as "David W. Jolly" / "David Jolly" and
    # "Edward J. Markey" / "Ed Markey", which splits their polls across two cand_keys. That
    # both DILUTES a leading candidate's poll_avg and defeats dedup: two rows reporting the
    # identical survey no longer share a key, so they double-count. Found while fixing the
    # duplicate-survey bug - the two defects compound, and 33 duplicate rows in the general
    # feed survived both dedup passes purely because the candidate name differed.
    d, n_merged = FP.merge_nickname_aliases(d)
    if n_merged:
        print(f"nickname-alias merges: {n_merged} candidate name variants unified")

    d["_pollster_key"] = d["pollster"].map(F.norm_pollster)
    d = (d.sort_values("_src_priority")
           .drop_duplicates(subset=["_pollster_key", "end_date", "year", "state",
                                    "office", "district", "cand_key"], keep="first")
           .drop(columns=["_src_priority", "_pollster_key"]))
    # ...then survey-identity dedup, for the same survey filed under two different
    # organisation NAMES (see F.drop_duplicate_surveys). Never-fork: the identical second
    # pass runs in predict_primary.py and build_primary_dataset.py.
    d, _ = F.drop_duplicate_surveys(d, label="general feed")
    print(f"polls loaded: {before} rows -> {len(d)} after dedup "
          f"({before - len(d)} duplicates removed)")

    # (race_id is already built above, before the nickname merge and both dedup passes.)
    # Drop primary losers BEFORE the relative staleness rule. Order matters: the loser
    # filter is absolute (a called result), the stale filter is relative to the race's
    # newest poll, and a defeated candidate is never stale by that measure.
    d = drop_primary_losers(d, cycle)
    d = drop_stale_candidates(F.prepare_polls(d))

    # schema sanity: a silent upstream change (pct scale, stage labels, race_id format)
    # must crash here, loudly, instead of producing confident garbage downstream
    bad_pct = ~d["pct"].between(0, 100)
    assert bad_pct.mean() < 0.01, \
        f"feed pct out of [0,100] for {bad_pct.mean():.1%} of rows — implied_prob scale changed?"
    assert d["end_date"].notna().mean() > 0.95, "feed end_date mostly unparseable"
    n_races = d["race_id"].nunique()
    assert n_races >= 20, f"only {n_races} general races parsed — race_id/stage format changed?"
    warn_near_duplicate_names(d)
    return d

def drop_primary_losers(d, cycle):
    """Remove DEAD MATCHUPS - questions involving a candidate who lost their primary.

    Operates on whole questions, not rows: a survey asks several separate head-to-heads
    and each one's numbers are only meaningful against the opponent actually named in it.

    The general feed carries hypothetical matchups polled for months before a primary, so a
    defeated candidate keeps a full set of recent polls and simply never stops being scored.
    `drop_stale_candidates` cannot catch this: it is a RELATIVE rule (>14 days behind the
    race's newest poll) and every candidate in the race, winner and loser alike, is polled
    right up to primary day. Nobody is stale relative to anybody.

    Found 2026-08-06, two days after Michigan voted: the general model still had Haley
    Stevens ahead of Abdul El-Sayed in MI-Sen and Perry Johnson ahead of John James in
    MI-Gov - in both cases ranking the person who LOST the primary above the actual nominee.
    16 defeated candidates across 12 races.

    Uses data/primary_results_2026.csv, which is authoritative once a primary is called: any
    candidate flagged is_winner=0 in a party-primary whose general race we are predicting is
    out of that general election. Winners are untouched, and races with no result yet are
    untouched, so this is a no-op until a primary is actually decided.
    """
    path = os.path.join(HERE, "data", f"primary_results_{cycle}.csv")
    if not os.path.exists(path):
        # LOUD, not silent. This file is the ONLY thing that keeps defeated primary
        # candidates out of the general-election feed, and data/*.csv is gitignored by
        # default - so when it was not force-added, CI checked out a tree without it,
        # this function returned unchanged, and the nightly refresh put Haley Stevens
        # and Perry Johnson back into MI-Sen/MI-Gov. Twice. A silent return made a
        # data-availability problem look like a modelling one.
        print(f"WARNING: {path} not found - primary losers CANNOT be filtered. "
              f"Defeated candidates will be scored as if still running. "
              f"(Is the file committed? data/*.csv is gitignored by default.)")
        return d
    r = pd.read_csv(path)
    if not {"race_id", "cand_key", "is_winner"} <= set(r.columns):
        return d
    # primary race_id is "<general_race_id>_<PARTY>"; strip the party suffix to match
    r["general_race_id"] = r["race_id"].astype(str).str.rsplit("_", n=1).str[0]
    losers = set(zip(r.loc[~r["is_winner"].astype(bool), "general_race_id"],
                     r.loc[~r["is_winner"].astype(bool), "cand_key"]))
    if not losers:
        return d
    is_loser = pd.Series([(rid, ck) in losers
                          for rid, ck in zip(d["race_id"], d["cand_key"])], index=d.index)
    if not is_loser.any():
        return d

    # DROP THE WHOLE MATCHUP, not just the loser's row. A survey asks several separate
    # head-to-heads (question_id); "Stevens 43.7 vs Rogers 44.1" is one question and
    # "El-Sayed 40.8 vs Rogers 46.9" is another. Deleting only Stevens leaves Rogers'
    # 44.1 behind - a number measured against an opponent who is no longer running -
    # and pools it with his real numbers. Rogers polls differently against different
    # Democrats, so that contaminates his average with a matchup that will never happen.
    # Fall back to (poll_id, candidate-set) if question_id is absent.
    if "question_id" in d.columns and d["question_id"].notna().any():
        qkey = d["race_id"].astype(str) + "|" + d["question_id"].astype(str)
    else:
        qkey = d["race_id"].astype(str) + "|" + d["poll_id"].astype(str)
    dead_q = set(qkey[is_loser])
    mask = qkey.isin(dead_q)

    gone = d.loc[is_loser, ["race_id", "candidate"]].drop_duplicates()
    collateral = int(mask.sum() - is_loser.sum())
    print(f"dead matchups removed from the general feed: {int(mask.sum())} poll rows "
          f"({len(dead_q)} questions) - {len(gone)} eliminated candidates across "
          f"{gone['race_id'].nunique()} races, plus {collateral} surviving-candidate rows "
          f"that were only measured against them")
    for _, row in gone.head(12).iterrows():
        print(f"   {row['race_id']}: {row['candidate']}")
    return d[~mask]

def warn_near_duplicate_names(d, threshold=0.88):
    """Print any two candidates in the SAME race whose names are near-identical.

    norm_name deliberately does not fuzzy-match, so a one-character feed typo silently
    splits a candidate into two half-strength entries. That is invisible in the output -
    both halves just look like weakly-polled candidates. Found 2026-08-05 in FOUR races
    at once ("Josh Elliot"/"Josh Elliott" six days before CT's primary; Raffensperger
    split across two races), then two more on the general side.

    This only WARNS. Fixing means adding a row to data/name_aliases.csv, which is a human
    decision: two similar names can be two real people (brothers, a junior/senior pair),
    and auto-merging them would be worse than the split.
    """
    import difflib
    seen = []
    for rid, g in d.groupby("race_id" if "race_id" in d.columns else "cand_key"):
        names = sorted(set(g["candidate"].astype(str)))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                # Only a DIFFERENT cand_key is a real split. norm_name already collapses
                # middle initials, punctuation and hyphens ("Mark R. Warner"/"Mark Warner",
                # "Beto O'Rourke"/"Beto O’Rourke"), so those pairs look near-identical here
                # but are already one candidate - warning about them would be noise that
                # trains the reader to ignore this check.
                if F.norm_name(a) == F.norm_name(b):
                    continue
                if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold:
                    seen.append((rid, a, b))
    if seen:
        print(f"NEAR-DUPLICATE NAMES ({len(seen)}) - likely one candidate split by a feed "
              f"typo. Add to data/name_aliases.csv if they are the same person:")
        for rid, a, b in seen:
            print(f"   {rid}: {a!r} vs {b!r}")
    return seen

def drop_stale_candidates(d, stale_days=14):
    """Drop candidates who stopped being polled before the race moved on.

    'General' feeds carry hypothetical matchups from the primary season (e.g. ME-Sen 2026
    polled Mills-vs-Collins AND Platner-vs-Collins for months). Once the real nominee
    emerges, the losing primary candidates stop appearing in new polls — so anyone whose
    LAST poll is > stale_days older than the race's most recent poll is presumed out.

    Guards (per race): never drop below 2 candidates, and never lose a whole party that
    had polling — in those cases the race is left untouched (sparsely polled races).
    """
    last_cand = d.groupby(["race_id", "cand_key"])["end_date"].transform("max")
    last_race = d.groupby("race_id")["end_date"].transform("max")
    keep = last_cand >= (last_race - pd.Timedelta(days=stale_days))

    kept = d[keep]
    reverted = []
    for rid, orig in d.groupby("race_id"):
        g = kept[kept["race_id"] == rid]
        two_before = orig["cand_key"].nunique() >= 2
        parties_before = {"DEM", "REP"} & set(orig["party_std"])
        if (two_before and g["cand_key"].nunique() < 2) or \
           (parties_before and not parties_before <= set(g["party_std"])):
            reverted.append(rid)
    out = pd.concat([kept[~kept["race_id"].isin(reverted)],
                     d[d["race_id"].isin(reverted)]], ignore_index=True)
    n_dropped = d.groupby(["race_id", "cand_key"]).ngroups - out.groupby(["race_id", "cand_key"]).ngroups
    print(f"stale-candidate filter: dropped {n_dropped} candidates "
          f"(no polls within {stale_days}d of their race's latest; {len(reverted)} races left untouched by guards)")
    return out

def patch_redistricted_priors(c):
    """In 2025-26 REDRAWN districts, prior_margin_cand describes old boundaries. Replace it
    with sign * 2 * current-map Cook PVI (data/district_pvi_current.csv) — a structural
    estimate of the NEW district's margin. Applied at predict time only; training cycles'
    districts always match their own results, so no train-side change is needed."""
    rd_path = os.path.join(HERE, "data", "redistricted_2026.csv")
    pvi_path = os.path.join(HERE, "data", "district_pvi_current.csv")
    if not (os.path.exists(rd_path) and os.path.exists(pvi_path)):
        print("WARNING: redistricting/PVI files missing — redrawn-district priors NOT patched")
        return c
    redrawn = set(pd.read_csv(rd_path)["state"])
    pv = pd.read_csv(pvi_path, dtype={"district": str})
    pv["district"] = pv["district"].fillna("")
    pvi = {(r.state, r.district): r.pvi for r in pv.itertuples()}
    mask = (c["office"] == "House") & c["state"].isin(redrawn)
    sign = c["party"].map({"DEM": 1, "REP": -1}).fillna(0)
    est = pd.Series([pvi.get((s, str(d))) for s, d in zip(c["state"], c["district"])],
                    index=c.index, dtype=float)
    patch = mask & est.notna() & (sign != 0)
    c.loc[patch, "prior_margin_cand"] = (sign * 2.0 * est)[patch]
    print(f"redistricting patch: prior_margin re-estimated from new-map PVI for "
          f"{int(patch.sum())} candidate rows in {c.loc[patch, 'race_id'].nunique()} redrawn districts")
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=2026)
    ap.add_argument("--natl-env", type=float, default=None,
                    help="generic-ballot DEM-REP margin (e.g. RealClearPolling average)")
    ap.add_argument("--polls", nargs="*", default=DEFAULT_POLLS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "data", "model_features.json")) as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(HERE, "data", "model_xgb.json"))
    print(f"model: trained on cycles {meta['trained_on_cycles'][0]}-{meta['trained_on_cycles'][-1]}, "
          f"{len(meta['features'])} features")

    d = load_agg_polls(args.polls, args.cycle)
    if len(d) == 0:
        raise SystemExit(f"no general-election polls found for cycle {args.cycle}")

    # house effect from HISTORICAL training polls, applied to the new cycle's pollsters
    hist = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
    hist = F.prepare_polls(hist[hist["has_result"] == 1])
    hist["race_id"] = (hist["year"].astype(str) + "_" + hist["state"] + "_" + hist["office"]
                       + hist["district"].radd("-").where(hist["district"] != "", ""))
    house = F.compute_house_effect(hist, sorted(hist["year"].unique()))
    bias = F.compute_bias_priors(hist)   # prior-cycles poll-bias by state (2026 uses <=2024)

    macro = build_macro(cycles=[args.cycle])
    ne = dict(natl_env_hist())
    if args.natl_env is not None:
        ne[args.cycle] = args.natl_env
        print(f"natl_env({args.cycle}) = {args.natl_env:+.1f} (given via --natl-env)")
    else:
        # the one live fetch in the project: current-cycle info can't be frozen by definition
        from fetch_generic_ballot import get_natl_env
        v = get_natl_env(args.cycle)
        if v is not None:
            ne[args.cycle] = v
            print(f"natl_env({args.cycle}) = {v:+.1f} (Wikipedia aggregator mean; "
                  f"override with --natl-env)")
        else:
            print("WARNING: generic-ballot fetch failed and --natl-env not given; "
                  "natl_env_cand will be missing (NaN)")

    funds = F.load_fundamentals()
    fec = F.load_fec(extended=True)
    # bio_office_level shipped 2026-07-29: load candidate bios at SERVE time too, or the feature
    # is all-NaN for live races (train/serve skew). load_candidate_bios reads the same committed
    # data/candidate_bios.csv the training run used. (The feature-presence assert below would
    # otherwise fire, since the artifact now expects bio_office_level.)
    bios = F.load_candidate_bios()
    # primary_results MUST be passed here too, or primary_margin/opp_primary_margin/
    # primary_margin_diff are all-NaN at serve time while the artifact was trained on real
    # values - train/serve skew of exactly the kind the feature-presence assert below cannot
    # catch (the COLUMNS would exist, just empty). Same reasoning as candidate_bios above.
    primres = F.load_primary_results()
    cand = F.build_candidate_table(d, macro, ne, funds, house=house, fec=fec, bias_priors=bias,
                                   candidate_bios=bios, primary_results=primres)
    cand = patch_redistricted_priors(cand)

    # guard: every feature the artifact expects must exist in the built table — reindex
    # would otherwise silently fill a whole missing block with NaN and predict garbage
    missing = [f for f in meta["features"] if f not in cand.columns]
    assert not missing, f"artifact expects features absent from the built table: {missing[:8]}"
    X = cand.reindex(columns=meta["features"])
    cand["win_prob"] = model.predict_proba(X)[:, 1]

    # ---- poll-bias robustness sweep (cycle-level poll error is historically ±4-7 pts,
    # shared across ALL races — see HANDOFF.md). Re-predict under a uniform 3-point
    # national margin shift each way; picks that flip are bias-fragile, not edges. ----
    for label, dem_shift in [("R3", -3.0), ("D3", +3.0)]:
        ds = d.copy()
        sgn = ds["party_std"].map({"DEM": 1, "REP": -1}).fillna(0)
        ds["pct"] = ds["pct"] + sgn * dem_shift / 2
        cs = patch_redistricted_priors(
            F.build_candidate_table(ds, macro, ne, funds, house=house, fec=fec, bias_priors=bias,
                                    candidate_bios=bios, primary_results=primres))
        cs["p"] = model.predict_proba(cs.reindex(columns=meta["features"]))[:, 1]
        m = cs.set_index(["race_id", "cand_key"])["p"]
        cand[f"win_prob_{label}"] = [m.get((r, c)) for r, c in
                                     zip(cand["race_id"], cand["cand_key"])]
    base_pick = cand.loc[cand.groupby("race_id")["win_prob"].idxmax()].set_index("race_id")["cand_key"]
    fragile = set()
    for label in ("R3", "D3"):
        p2 = cand.loc[cand.groupby("race_id")[f"win_prob_{label}"].idxmax()].set_index("race_id")["cand_key"]
        fragile |= set(base_pick.index[base_pick != p2.reindex(base_pick.index)])
    cand["bias_fragile"] = cand["race_id"].isin(fragile).astype(int)
    print(f"bias-fragile races (pick flips under a +/-3pt national poll shift): "
          f"{len(fragile)} of {cand['race_id'].nunique()}")
    # within-race normalized probability (raw probs are per-candidate, not a race simplex)
    cand["win_prob_norm"] = (cand["win_prob"]
                             / cand.groupby("race_id")["win_prob"].transform("sum"))

    # distinct SURVEYS the model actually used per race (a poll = one pollster+end_date;
    # n_polls is a per-CANDIDATE row count that sums to more than the survey count).
    surveys = (d.groupby("race_id")
                .apply(lambda g: g.groupby(["pollster", "end_date"]).ngroups,
                       include_groups=False))
    cand["n_surveys"] = cand["race_id"].map(surveys).fillna(0).astype(int)

    out_cols = ["race_id", "state", "office", "district", "candidate", "party",
                "display_party",
                "n_polls", "n_surveys", "poll_avg", "poll_lead", "prior_margin_cand",
                "is_incumbent", "win_prob", "win_prob_norm",
                "win_prob_R3", "win_prob_D3", "bias_fragile"]
    out = cand[out_cols].sort_values(["race_id", "win_prob"], ascending=[True, False])
    out_path = args.out or _paths.out(f"predictions_{args.cycle}.csv")
    out.to_csv(out_path, index=False)

    # meta sidecar: what the model actually consumed. The dashboard shows polls_max_end_date
    # next to predictions_as_of — a big market edge can be a stale-polls artifact, and the
    # generation timestamp alone can't reveal that.
    import datetime
    meta_out = dict(
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        polls_max_end_date=str(d["end_date"].max().date()),
        n_poll_rows=int(len(d)), n_races=int(out["race_id"].nunique()),
        natl_env=ne.get(args.cycle),
    )
    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)

    picks = out.loc[out.groupby("race_id")["win_prob"].idxmax()]
    close = picks[np.isclose(picks["win_prob_norm"], 0.5, atol=0.10)]
    print(f"\nraces predicted: {out['race_id'].nunique()} "
          f"({dict(picks.groupby('office').size())})")
    print(f"saved -> {out_path}")
    print("\nclosest races (leader's normalized prob within 40-60%):")
    print(close[["race_id", "candidate", "party", "poll_avg", "poll_lead", "win_prob_norm"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
