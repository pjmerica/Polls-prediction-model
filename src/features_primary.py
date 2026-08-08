# -*- coding: utf-8 -*-
"""Primary-election feature pipeline: long primary-poll rows -> one row per candidate per
primary race. Shared by training (primary_model.py) and prediction (predict_primary.py) -
same never-fork rule as features.py.

Race = (cycle, state, office[, district], party): a WITHIN-PARTY multi-candidate contest.
Design notes vs the general pipeline (features.py):
- All poll aggregates are plain means of raw polls (same no-weighting rule).
- Poll recency is measured against the PRIMARY's own election_date, which every historical
  row carries (verified: real primary dates incl. moved ones; days_to_elec via
  F.prepare_polls on the long frame).
- No natl_env / bias priors / house effects: within-party races have no partisan channel.
- Macro is OPTIONAL (build_macro_asof windows keyed to each primary's date, publication-lag
  guarded). Default OFF: ~200 training races cannot support 144 macro columns; the
  training script runs the ablation and documents it.
- Fundraising: fund_share is recomputed WITHIN the primary field (that party's candidates
  only) - the general pipeline's race-wide share would leak the other party's money into
  the denominator.
- is_defending_party: this party currently holds the seat (races.csv incumbent_party).
  True candidate-level incumbency isn't derivable from committed data (no incumbent name);
  documented gap.
"""
import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
import paths as _P  # noqa: E402

import numpy as np
import pandas as pd

import features as F
from cycles import PRES_PARTY

# common-nickname equivalence for WITHIN-RACE candidate merging: different sources name
# the same person differently ('Bobby Charles' vs 'Robert Charles' split one ME-Gov-26
# candidate into two, diluting his own poll support). Merging is gated on SAME LAST NAME
# within ONE race, so mild ambiguity in the map is safe (a 'ted smith' and an
# 'edward smith' in the same primary are the same person; Mayra vs Eric Flores are not
# merged because first names are not nickname-equivalent).
_NICK = {}
for canon, nicks in {
    "robert": ["bob", "bobby", "rob", "robbie", "bert"],
    "william": ["bill", "billy", "will", "willie", "liam"],
    "michael": ["mike", "mikey", "mick"],
    "james": ["jim", "jimmy", "jamie"],
    "john": ["jack", "johnny", "jon"],
    "joseph": ["joe", "joey"],
    "richard": ["rick", "ricky", "rich", "dick"],
    "daniel": ["dan", "danny"],
    "david": ["dave", "davey"],
    "thomas": ["tom", "tommy"],
    "christopher": ["chris", "kit"],
    "matthew": ["matt"],
    "anthony": ["tony"],
    "andrew": ["andy", "drew"],
    "steven": ["steve"], "stephen": ["steve"],
    "kenneth": ["ken", "kenny"],
    "edward": ["ed", "eddie", "ted", "teddy", "ned"],
    "theodore": ["ted", "teddy", "theo"],
    "gregory": ["greg"],
    "jeffrey": ["jeff"],
    "nicholas": ["nick"],
    "samuel": ["sam", "sammy"],
    "benjamin": ["ben", "benny"],
    "alexander": ["alex"],
    "timothy": ["tim", "timmy"],
    "charles": ["charlie", "chuck"],
    "ronald": ["ron", "ronnie"],
    "donald": ["don", "donny"],
    "lawrence": ["larry"],
    "raymond": ["ray"],
    "gerald": ["jerry"], "jerome": ["jerry"],
    "katherine": ["kate", "katie", "kathy", "kay"], "kathleen": ["kate", "katie", "kathy"],
    "elizabeth": ["liz", "beth", "betsy", "betty", "eliza"],
    "margaret": ["peggy", "meg", "maggie"],
    "deborah": ["debbie", "deb"],
    "susan": ["sue", "susie"],
    "jennifer": ["jen", "jenny"],
    "patricia": ["pat", "patty", "trish"], "patrick": ["pat"],
    "rebecca": ["becky"],
    "abigail": ["abby"],
    "victoria": ["vicky", "tori"],
    "cynthia": ["cindy"],
    "pamela": ["pam"],
    "sandra": ["sandy"],
    "barbara": ["barb"],
    "frederick": ["fred", "freddie"],
    "leonard": ["len", "lenny", "leo"],
    "walter": ["walt"],
    "harold": ["hal", "harry"], "henry": ["hank", "harry"],
    "albert": ["al"], "alan": ["al"], "alfred": ["al"],
    "eugene": ["gene"],
    "vincent": ["vince", "vinny"],
    "philip": ["phil"], "phillip": ["phil"],
    "stanley": ["stan"],
    "norman": ["norm"],
    "arthur": ["art", "artie"],
    "peter": ["pete"],
    "francis": ["frank", "fran"], "franklin": ["frank"],
}.items():
    group = frozenset([canon] + nicks)
    for n in [canon] + nicks:
        _NICK.setdefault(n, set()).update(group)

def _first_equiv(a, b):
    a, b = a.lower(), b.lower()
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    return b in _NICK.get(a, set())

def _middle_names(parts):
    """Middle tokens of a parsed name, stripped of a trailing parenthetical wrapper and
    punctuation (['Thomas','(Jay)','Feely'] -> ['jay']; ['David','Madison','Cawthorn'] ->
    ['madison']). Single-letter/initial tokens ('A.', 'J') are dropped - they're not a
    usable alternate first name."""
    mids = []
    for p in parts[1:-1]:
        p = p.strip("().").lower()
        if len(p) > 1:
            mids.append(p)
    return mids

def _cross_name_equiv(first_a, mids_a, first_b, mids_b):
    """True if either full name's FIRST name matches a MIDDLE name the other source used as
    the publicly-known first name (NYT tends to print legal first-middle-last; Wikipedia/
    race pages print the name the candidate actually goes by, often their middle name -
    verified 2026-07-22 on real cross-source poll pairs: Thomas (Jay) Feely / Jay Feely,
    David Madison Cawthorn / Madison Cawthorn). Does NOT merge on a shared middle name alone
    when the two first names are otherwise unrelated words - that would be too permissive."""
    fa, fb = first_a.lower(), first_b.lower()
    return fa in mids_b or fb in mids_a

def merge_nickname_aliases(d, name_col="candidate"):
    """Within each race, merge candidates whose FULL names share a last name and have
    nickname-equivalent (or prefix-equivalent) first names, OR where one source's first name
    is the other source's middle name / parenthetical nickname used as their public first name
    (added 2026-07-22 - see _cross_name_equiv). They get one cand_key (the variant with more
    poll rows keeps its name). Returns d with cand_key/candidate fixed and the number of
    merges."""
    d = d.copy()
    merges = 0
    for rid, g in d.groupby("race_id"):
        names = g.groupby(name_col).size().sort_values(ascending=False)
        canon = {}
        seen = []          # [(full_name, first, last, mids)]
        for full in names.index:
            parts = str(full).replace(".", " ").split()
            if len(parts) < 2:
                continue
            first, last = parts[0], parts[-1]
            mids = _middle_names(parts)
            hit = next((s for s in seen if s[2].lower() == last.lower()
                        and (_first_equiv(first, s[1])
                             or _cross_name_equiv(first, mids, s[1], s[3]))), None)
            if hit:
                canon[full] = hit[0]
                merges += 1
            else:
                seen.append((full, first, last, mids))
        if canon:
            m = d["race_id"] == rid
            d.loc[m, name_col] = d.loc[m, name_col].replace(canon)
    d["cand_key"] = d[name_col].map(F.norm_name)
    return d, merges

def load_candidate_bios(path=None):
    """data/candidate_bios.csv -> {(year, office, state, district, party, ck): feats}, plus
    the two fallback indexes the primary lookup needs (see build_primary_table):
      "__person_offices__" : {(ck, state): offices}  - leak-free as-of-year level (shared with
                             features.load_candidate_bios; used identically here)
      "__person_years__"   : {(ck, state): {year: feats}} - the same person's bio in OTHER
                             cycles, for the most-recent-PRIOR-year fallback
    Wikipedia race-page candidate descriptors classified into office levels
    (fetch_candidate_bios.py). district = '' for statewide."""
    import os
    path = path or _P.data("candidate_bios.csv")
    if not os.path.exists(path):
        return {}
    b = pd.read_csv(path, low_memory=False)
    out = {}
    by_person = {}
    for r in b.itertuples():
        di = F.dist_str(r.district)   # crash-safe: handles "S" special-election district + float round-trip
        party = F.npar(r.party)
        if pd.isna(r.cand_key):
            continue
        feats = dict(bio_office_level=int(r.office_level),
                     bio_in_office=int(r.bio_in_office),
                     bio_prior_candidacy=int(r.bio_prior_candidacy))
        yr = int(r.year)
        out[(yr, r.office, r.state, di, party, r.cand_key)] = feats
        # keep the HIGHEST level seen for a person-year (a cycle can hold several rows for
        # one person - e.g. they appear on both the Senate and Governor page)
        pk = (r.cand_key, r.state)
        prev = by_person.setdefault(pk, {}).get(yr)
        if prev is None or feats["bio_office_level"] > prev["bio_office_level"]:
            by_person[pk][yr] = feats
    out["__person_years__"] = by_person
    # reuse the general loader's person-level tenure map (hand-coded + Ballotpedia), which is
    # keyed (cand_key, state) and evaluated as-of-year, so it is leak-free by construction
    gen = F.load_candidate_bios()          # no-arg: reads the same default data/candidate_bios.csv
    out["__person_offices__"] = gen.get("__person_offices__", {}) if gen else {}
    return out

_BIO_NAN = dict(bio_office_level=np.nan, bio_in_office=np.nan, bio_prior_candidacy=np.nan)

def _bio_lookup(bios, yr, of, st, di, party, ck):
    """Bio feats for one candidate-race, or _BIO_NAN.

    Three tiers, all leak-free (2026-08-01 - the primary path previously had ONLY tier 1,
    which is why 453 real candidates read NaN while their bios sat in the file):
      1. exact (year, office, state, district, party, cand_key), with statewide offices
         collapsed to district "" (ports the same "S"-district fix features.py already had);
      2. person-level tenure map, as-of-year (identical to the general pipeline's fallback);
      3. the SAME person's bio from the most recent STRICTLY-PRIOR cycle. Only looks
         backward, so it cannot leak a later-won office - it is the same no-look-ahead rule
         the as-of-year table is built on. This is what rescues odd-year races (2019 KY,
         2019 MS, 2021 VA have ~9 bio rows total because the scraper only covers even years)
         and people like Andrew Cuomo, who has 12 bio rows but none for the 2024 cycle.
    """
    if bios is None:
        return _BIO_NAN
    bio_di = "" if of in ("Senate", "Governor") else di
    hit = bios.get((yr, of, st, bio_di, party, ck))
    if hit is not None:
        return hit
    poff = bios.get("__person_offices__", {})
    if (ck, st) in poff:
        lvl = F._person_asof_level(poff[(ck, st)], yr)
        if lvl is not None:
            return dict(bio_office_level=lvl, bio_in_office=np.nan,
                        bio_prior_candidacy=np.nan)
    years = bios.get("__person_years__", {}).get((ck, st))
    if years:
        prior = [y for y in years if y < yr]
        if prior:
            return years[max(prior)]
    return _BIO_NAN


def build_primary_table(d, fec=None, inc_map=None, macro_asof=None, hist=None, bios=None):
    """d: prepared long frame (F.prepare_polls applied) with columns
    race_id, year, state, office, district, party_std, candidate, cand_key, pct, end_date,
    days_to_elec, sample_size, pollster [, won].
    macro_asof: optional callable date -> {feature: value} (macro_features.build_macro_asof);
    evaluated once per unique election_date and merged onto every candidate row.
    """
    lead_changes = {rid: F.count_lead_changes(g) for rid, g in d.groupby("race_id")}
    margin_dyn = {rid: F.margin_dynamics(g) for rid, g in d.groupby("race_id")}
    has_won = "won" in d.columns

    macro_cache = {}
    def macro_for(ed):
        if macro_asof is None or pd.isna(ed):
            return {}
        k = pd.Timestamp(ed).normalize()
        if k not in macro_cache:
            macro_cache[k] = macro_asof(k)
        return macro_cache[k]

    # surveyed-population splits (LV / RV / A): the same poll aggregates computed per
    # population class. 'v' (unspecified voters) folds into RV; missing labels get no
    # class (excluded from splits, still in the overall aggregates). A race rarely has
    # all three classes - absent class = NaN (XGBoost routes missing).
    has_pop = "population" in d.columns
    POPS = {"lv": ("lv",), "rv": ("rv", "v"), "a": ("a",)}

    rows = []
    for race_id, g in d.groupby("race_id"):
        yr = int(g["year"].iloc[0]); st = g["state"].iloc[0]
        of = g["office"].iloc[0]; di = F.dist_str(g["district"].iloc[0])
        party = g["party_std"].iloc[0]
        ed = g["election_date"].dropna().max()
        incp = inc_map.get((yr, st, of, di)) if inc_map else None
        dyn = margin_dyn.get(race_id, {})
        for ck, gc in g.groupby("cand_key"):
            gc = gc.sort_values("end_date")
            dated = gc.dropna(subset=["end_date"])
            last30 = gc[gc["days_to_elec"] <= 30]
            last7 = gc[gc["days_to_elec"] <= 7]
            pop_feats = {}
            for tag, vals in POPS.items():
                gp = (gc[gc["population"].astype(str).str.lower().isin(vals)]
                      if has_pop else gc.iloc[0:0])
                gp30 = gp[gp["days_to_elec"] <= 30]
                gpd = gp.dropna(subset=["end_date"])
                pop_feats[f"poll_avg_{tag}"] = gp["pct"].mean() if len(gp) else np.nan
                pop_feats[f"poll_last_{tag}"] = (gpd["pct"].iloc[-1] if len(gpd) else np.nan)
                pop_feats[f"poll_last30_{tag}"] = gp30["pct"].mean() if len(gp30) else np.nan
                pop_feats[f"poll_std_{tag}"] = gp["pct"].std() if len(gp) > 1 else np.nan
                pop_feats[f"n_polls_{tag}"] = len(gp)
            # Shared with the general model - see F.poll_momentum_slope. This was a
            # forked 60-day copy until 2026-08-03; the fork meant the general model's
            # switch to all-dated-polls silently skipped both primary models.
            slope = F.poll_momentum_slope(gc)
            fe = fec.get((yr, st, of, di, ck)) if fec is not None else None
            rec = fe["receipts"] if fe else np.nan
            md = dyn.get(ck, {})
            rows.append(dict(
                race_id=race_id, year=yr, state=st, office=of, district=di, party=party,
                cand_key=ck, candidate=gc["candidate"].iloc[0],
                election_date=ed,
                won=(int(gc["won"].iloc[0]) if has_won and pd.notna(gc["won"].iloc[0]) else np.nan),
                poll_avg=gc["pct"].mean(),
                poll_last=(dated["pct"].iloc[-1] if len(dated) else gc["pct"].mean()),
                poll_last30=(last30["pct"].mean() if len(last30) else gc["pct"].mean()),
                # final-week average (2026-07-31). Unlike poll_last30 this does NOT fall back
                # to the all-time mean when the window is empty: a 7-day window is empty for
                # most candidates most of the time, and that fallback is exactly the bug this
                # feature exists to counter (it would silently re-inject the stale 15-month
                # average under a "final week" name). NaN instead - XGBoost routes missing.
                # Rationale: poll_avg is a flat mean over every poll ever taken, so in a race
                # with a year of polling it lags badly (MI-Sen-DEM 2026: poll_avg 32.3 while
                # the candidate had been polling 41-56 for a month). poll_last is fresh but
                # is a single poll; poll_last7 is the fresh-AND-averaged middle ground.
                poll_last7=(last7["pct"].mean() if len(last7) else np.nan),
                n_polls_last7=len(last7),
                poll_std=gc["pct"].std(),
                n_polls=len(gc),
                avg_sample=gc["sample_size"].mean(),
                min_days=gc["days_to_elec"].min(),
                poll_momentum=slope,
                n_lead_changes=lead_changes.get(race_id, 0),
                avg_margin_over_time=md.get("avg_margin_over_time", np.nan),
                margin_volatility=md.get("margin_volatility", np.nan),
                min_margin=md.get("min_margin", np.nan),
                margin_trend=md.get("margin_trend", np.nan),
                is_dem_primary=int(party == "DEM"),
                is_senate=int(of == "Senate"), is_gov=int(of == "Governor"),
                # the party defends this seat (races.csv incumbent_party); unknown = NaN
                is_defending_party=((1 if incp == party else 0)
                                    if incp in ("DEM", "REP") else np.nan),
                is_pres_party=int(party == PRES_PARTY.get(yr)),
                _fund_receipts=(rec if fe else np.nan),
                fund_receipts_ln=(np.log1p(rec) if fe and rec and rec > 0 else np.nan),
                **pop_feats,
                # candidate electoral history (candidate_history.CandidateHistory,
                # strictly-prior-cycle; fact-checked - see check_candidate_history.py)
                **(hist.history(yr, st, ck) if hist is not None else {}),
                # officeholder bio: exact key -> person-level as-of-year -> most recent PRIOR
                # cycle (all leak-free; see _bio_lookup). NaN only when the person is absent
                # from the bio table entirely.
                **(_bio_lookup(bios, yr, of, st, di, party, ck) if bios is not None else {}),
                **macro_for(ed),
            ))
    c = pd.DataFrame(rows)

    # within-field lead per population class (BUGFIX 2026-07-21, see features.best_other:
    # this used to be one race-wide constant subtracted from every candidate, so 2nd place
    # always read poll_lead(_tag) == 0.0 exactly - now per-row best-OTHER-candidate)
    for tag in ("lv", "rv", "a"):
        col = f"poll_avg_{tag}"
        best = c.groupby("race_id")[col].transform(F.best_other)
        c[f"poll_lead_{tag}"] = c[col] - best

    # within-FIELD relatives (the field = this party's candidates = the race group)
    c["field_best"] = c.groupby("race_id")["poll_avg"].transform(F.best_other)
    c["poll_lead"] = c["poll_avg"] - c["field_best"]
    # ...and the same lead computed on the final week only (2026-07-31). poll_lead inherits
    # poll_avg's staleness, which is how one 15-month-old number leaked into five features at
    # once; this is the fresh counterpart. NaN when any side of the comparison has no
    # final-week polls (never a silent 0 - that would read as "tied").
    c["poll_lead_last7"] = c["poll_last7"] - c.groupby("race_id")["poll_last7"].transform(
        F.best_other)
    c["poll_share"] = c["poll_avg"] / c.groupby("race_id")["poll_avg"].transform("sum")
    c["n_cands"] = c.groupby("race_id")["cand_key"].transform("count")
    c["race_total_polls"] = c.groupby("race_id")["n_polls"].transform("sum")
    c["undecided"] = (100 - c.groupby("race_id")["poll_avg"].transform("sum")).clip(lower=0)
    c["gap_x_recency"] = c["poll_lead"] * (1.0 / (1.0 + c["min_days"].clip(lower=0) / 30.0))
    tot = c.groupby("race_id")["_fund_receipts"].transform("sum")
    c["fund_share"] = np.where(tot > 0, c["_fund_receipts"] / tot, np.nan)
    c = c.drop(columns="_fund_receipts")
    return c

def feature_list_primary(macro_feats=(), fund=False):
    """Model inputs. Everything is computable for a future primary from the raw feed +
    committed statics.

    fund=False is the ARTIFACT default (2026-07-15): FEC receipts are CYCLE-END totals,
    and nominees raise most of their money AFTER winning the primary - so fund_share
    partially encodes the training label (leakage) while a mid-cycle 2026 candidate has
    no such money yet (train/serve skew). Measured: dropping fund leaves race-acc
    IDENTICAL (.895) and costs only Brier .035->.046 - the picks never depended on it.
    Revisit only with as-of-primary-date FEC reports. macro_feats: ablation-only."""
    return [
        "poll_avg", "poll_last", "poll_last30", "poll_std", "n_polls", "avg_sample",
        "min_days", "poll_momentum",
        # final-week window (2026-07-31, user request). poll_avg is a flat all-time mean, so
        # in a long primary campaign every feature derived from it (poll_lead, poll_share,
        # poll_std, gap_x_recency) carries year-old polling. These are the fresh counterparts;
        # both are exposed so the model can use the long-run level AND the current one.
        # NOT recency WEIGHTING - the no-weighting rule (features.py:7) still holds; this is
        # an unweighted window exactly like the existing poll_last30.
        "poll_last7", "n_polls_last7", "poll_lead_last7",
        "poll_lead", "poll_share", "n_cands", "race_total_polls", "undecided",
        "gap_x_recency",
        "n_lead_changes", "avg_margin_over_time", "margin_volatility", "min_margin",
        "margin_trend",
        "is_dem_primary", "is_senate", "is_gov", "is_defending_party", "is_pres_party",
        # surveyed-population splits (2026-07-15, user request): the poll aggregates per
        # LV / RV / A class. Deeper per-class variants (momentum, dynamics) are too sparse
        # at ~200 training races - documented, not forgotten.
        "poll_avg_lv", "poll_last_lv", "poll_last30_lv", "poll_std_lv", "n_polls_lv", "poll_lead_lv",
        "poll_avg_rv", "poll_last_rv", "poll_last30_rv", "poll_std_rv", "n_polls_rv", "poll_lead_rv",
        "poll_avg_a", "poll_last_a", "poll_last30_a", "poll_std_a", "n_polls_a", "poll_lead_a",
        # candidate officeholder experience (2026-07-18 overfit review): ONLY
        # bio_office_level survives. A per-cycle 6-seed sweep showed the other 9
        # history features (results-archive track record + bio_prior_candidacy) added
        # nothing on top of this one and slightly WORSENED both eval cycles - classic
        # thin-coverage overfitting on ~200 races. bio_office_level = highest office held
        # (4 fed/3 statewide/2 state-leg/1 local/0), a real durable signal (name
        # recognition, donor networks) with 35% coverage. It alone lifts 2024 AUC-PR
        # .910->.958 and Brier .054->.025 with NO 2022 regression. The hist_*/results
        # machinery + candidate_history.py stay in the repo (fact-checked, may return with
        # more data) but are no longer model features.
        "bio_office_level",
    ] + (["fund_receipts_ln", "fund_share"] if fund else []) + list(macro_feats)
