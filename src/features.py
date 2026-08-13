"""Shared feature pipeline: long poll rows -> one row per candidate per race.

Used by BOTH model.ipynb (training/CV) and predict.py (live 2026+ races), so features are
guaranteed to be computed identically at train and predict time.

Design rules (see CONCERNS.md):
- RAW POLLS ONLY: no poll weighting of any kind (no recency/sample/grade weights). The old
  pipeline weighted by 538's pollster grade, which does not exist for future polls -> that
  was train/serve skew. Averages are plain means now (user decision, 2026-07-05).
- No 538-only columns: numeric_grade / pollscore / partisan-lean are never used.
- Every input must be available in a bare poll feed: state, office, district, candidate,
  party, pollster, end_date, pct, sample_size (+ election_date to compute days_to_elec).
- Missing fundamentals are NaN (XGBoost routes missing natively) - never silently 0.
"""
import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd

import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
import paths  # noqa: F401  - side effect: puts the repo root + pipeline folders on sys.path,
              # so the lazy `import fetch_candidate_bios_ballotpedia` below resolves after the
              # 2026-08-02 reorganisation moved that script into pipeline/fetch/.
from cycles import PRES_PARTY

# Resolved from the REPO ROOT (paths.py), not this file's directory: features.py moved
# into src/ on 2026-08-08 and the old __file__-relative form would point at src/data/.
DATA_DIR = paths.DATA

# ---------------------------------------------------------------- small parsers

def dist_str(v):
    """Normalize a district value to canonical string form ('1', '23', or '').
    Guards against the CSV float round-trip bug ('1' -> 1.0 -> '1.0')."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s in ("", "nan"):
        return ""
    try:
        return str(int(float(s)))
    except ValueError:
        return s

def pdist(v):
    m = re.search(r"(\d+)", str(v))
    return str(int(m.group(1))) if m else ""

def npar(p):
    p = str(p).upper()
    return "DEM" if p.startswith("DEM") else "REP" if p.startswith("REP") else "OTH"

# Intra-word punctuation is DELETED, not spaced (2026-08-01). Before this, the two
# apostrophe characters took different paths and produced different keys for one person:
#   "Beto O’Rourke" (curly) -> NFKD/ascii DROPS it   -> "orourke"  -> key 'orourke b'
#   "Beto O'Rourke"      (straight) survives ascii, then
#                                  [^a-z\s] -> SPACE      -> "o rourke" -> key 'rourke b'
# ...so one candidate became two, splitting his own polling support between two rows (found
# 2026-07-31 in TX-Sen-DEM). The same split hit ASCII vs Unicode hyphens, which matters much
# more: a hyphenated surname ("Ocasio-Cortez") keyed off the LAST token only ('cortez a') when
# the hyphen spaced, but the whole surname ('ocasiocortez a') when it vanished. Deleting
# intra-word punctuation makes both spellings converge on the SAME key and keeps hyphenated
# surnames whole. Word SEPARATORS (whitespace) are still separators.
_APOS = r"\'‘’ʼʻ`´′"          # apostrophes/primes
_DASH = r"\-‐‑‒–—―"           # hyphens/dashes
# Apostrophes and periods are deleted in place. HYPHENS additionally swallow the whitespace
# around them, because sources type stray spaces there and a hyphen joins two surname parts
# into ONE token: the feed spells one candidate "Debbie Mucarsel- Powell" while the results
# archive has "Debbie Mucarsel-Powell", and without eating that space the surname still splits
# ('powell d' vs 'mucarselpowell d'), silently dropping 2024_FL_Senate_DEM from training.
# Periods do NOT swallow space - "Robert F. Kennedy" must stay three tokens, or the middle
# initial glues onto the surname ('fkennedy r').
_PUNCT_IN_WORD = re.compile(rf"[{_APOS}\.]")
_HYPHEN_JOIN = re.compile(rf"\s*[{_DASH}]\s*")

def norm_name(s):
    """Candidate-name join key: strip accents/suffixes -> 'lastname firstinitial'.

    THE shared join key: polls <-> results <-> FEC <-> bios <-> candidate history, in both the
    general and primary pipelines. Any change here re-keys every join, so it is a
    retrain-triggering change (see the _PUNCT_IN_WORD note above)."""
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    # Strip a TRAILING parenthetical party/label: "Amy Klobuchar (DFL)" -> "Amy Klobuchar".
    # Added 2026-08-12. Without this the parenthetical became the last token and therefore the
    # SURNAME, so "Amy Klobuchar (DFL)" keyed as 'dfl a' and "Angie Craig (DFL)" ALSO keyed as
    # 'dfl a' - both a split (one person scored as two candidates, splitting their probability)
    # and a collision (two different people sharing one key). Live effect: MN-Gov and MN-Sen
    # each carried a phantom duplicate of their own front-runner.
    # Anchored to the END and only for a short token, so a real parenthetical nickname used as
    # a name (handled elsewhere by _cross_name_equiv) is untouched, and it never empties the
    # string. Measured before shipping: 0 rows in either training file, 13 rows / 4 names in
    # the live feed - so this is a serve-time fix and does NOT re-key training data.
    s = re.sub(r"\s*\([a-z]{1,4}\)\s*$", " ", s)
    # collapse intra-word punctuation BEFORE the catch-all, so "o'rourke"/"o’rourke" and
    # "smith-jones"/"smith – jones" all land on one spelling instead of several.
    s = _HYPHEN_JOIN.sub("", s)     # hyphens join their two halves into one token
    s = _PUNCT_IN_WORD.sub("", s)   # apostrophes/periods vanish in place
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    parts = [w for w in s.split() if w]
    if not parts:
        return None
    last = parts[-1]
    fi = parts[0][0] if parts[0] != last else ""
    return f"{last} {fi}".strip()

JUNK_ANSWERS = {
    "generic democrat", "generic republican", "generic ballot", "generic candidate",
    "don't know", "dont know", "undecided", "someone else", "other",
    "would not vote", "neither", "no opinion", "refused", "none of these",
    "skipped", "will not vote", "not sure",
}

# Exact-set matching alone let COMBINED and punctuated variants through (found 2026-08-01
# auditing NaN bio_office_level rows): "Other / Undecided", "Undecided/ Other",
# "Other/Undecided", "Others", "Generic Opponent", plus bare "Yes"/"No" from
# approve/disapprove-style questions. 19 of these were sitting in PRIMARY training as if they
# were real candidates - several polling 45-62%, which inflates the race's summed support and
# so distorts poll_share, poll_lead and undecided for the REAL candidates in that race.
# Pattern-based now: strip punctuation, then match the whole normalized string.
_JUNK_RX = re.compile(
    r"^(?:"
    r"(?:other|others|undecided|dont know|do not know|not sure|no opinion|refused|skipped|"
    r"neither|none|none of these|someone else|no one|nobody|no preference|"
    r"would not vote|will not vote|not voting|no answer|na|n a)"
    r"(?:\s+(?:or|and)?\s*(?:other|others|undecided|dont know|not sure|no opinion|none))*"
    r"|generic\s+(?:democrat|republican|opponent|candidate|ballot|challenger|dem|rep|gop)"
    # Article-led hypothetical descriptors: "A Progressive Challenger", "an unnamed
    # Democrat", "the Republican nominee". These are POLL PLACEHOLDERS for an unnamed
    # person, not people. Found 2026-08-05: "A Progressive Challenger" was the model's
    # 94% front-runner in FL-23-DEM, beating a real named candidate.
    r"|(?:a|an|the)\s+(?:\w+\s+){0,2}"
    r"(?:challenger|opponent|candidate|democrat|republican|nominee|progressive|"
    r"moderate|conservative|liberal|independent)"
    r"|(?:unnamed|generic|hypothetical|any|another)\s+\w+"
    r"|(?:more|less)\s+\w+\s+\w+\s+(?:democrat|republican)"   # "more liberal female Democrat"
    r"|yes|no"                              # approve/disapprove style question answers
    r"|write[- ]?ins?"
    r")$")

# Slash-joined compound non-answers: "Don't know/Someone else", "Don't know/Would not vote",
# "Neither/would not vote". Found 2026-08-08 auditing published predictions - 7 of these were
# scored as candidates in the 2026 general feed and 1 in the primary feed, each carrying a
# real poll share (one at 22%) that normalisation then converted into win probability stolen
# from the REAL candidates in that race.
#
# Why _JUNK_RX missed them: it strips punctuation to spaces, so the string arrives as
# "dont know someone else", and its only join clause is `(or|and)` over a short token list -
# it never allowed a bare space to join two full non-answer phrases.
#
# Handled as a SEPARATE pass rather than by loosening _JUNK_RX, because the safe rule here is
# "EVERY slash-separated part is itself a non-answer". Loosening the main regex to match a
# non-answer phrase followed by arbitrary words would swallow real people - "Mike Rounds"
# (a sitting senator, and the reason 'round' can never be a substring rule) and "Tony
# Knowles" (a real former governor, vs "know") both survive only because matching is
# whole-string. Each part is re-checked with the full is_junk_answer logic, so this also
# covers slash-joins of anything the main regex already knows.
_NONANSWER_PART_RX = re.compile(
    r"^(?:dont know|do not know|not sure|no opinion|undecided|someone else|other|others|"
    r"neither|none|refused|skipped|no answer|would not vote|will not vote|not voting|"
    r"no one|nobody|no preference|unsure|dk|na)$")

def _is_nonanswer_part(part):
    # Drop apostrophes with NO space ("don't" -> "dont", matching the regexes' spelling)
    # before turning the remaining punctuation into spaces. Doing it in the other order
    # yields "don t", which matches nothing.
    p = part.strip().lower().replace("'", "").replace("’", "")
    p = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", p)).strip()
    return bool(p) and (bool(_NONANSWER_PART_RX.match(p)) or bool(_JUNK_RX.match(p)))

def is_junk_answer(name):
    """True for poll RESPONSE OPTIONS that are not candidates. These must never become
    candidate rows: they carry real poll percentages, so a missed one both trains as a fake
    candidate and skews every within-race relative feature for the real ones."""
    s = str(name).strip().lower()
    if s in JUNK_ANSWERS:
        return True
    # Slash/pipe-joined compounds ("Don't know/Someone else") - junk only when EVERY part is
    # itself a non-answer, so a real hyphenated or slashed person's name is never caught.
    if re.search(r"[/|]", s):
        parts = [p for p in re.split(r"[/|]", s) if p.strip()]
        if len(parts) > 1 and all(_is_nonanswer_part(p) for p in parts):
            return True
    # Bare "Round" / "Round 2": an RCV tabulation label the Wikipedia primary parser emits as
    # a candidate. Whole-string only - "Mike Rounds" (plural, and a real senator) is safe.
    if re.match(r"^round\s*\d*$", s):
        return True
    s = re.sub(r"[^a-z0-9\s]", " ", s)       # "other / undecided" -> "other   undecided"
    s = re.sub(r"\s+", " ", s).strip()
    if _JUNK_RX.match(s):
        return True
    # Bare trailing-noun variants the anchored regex can't reach: "Other named candidates",
    # "RCV round" (a tabulation artifact, not a person). Deliberately narrow and whole-string.
    return bool(re.match(r"^(?:other named candidates?|all other candidates?|"
                         r"(?:rcv|ranked choice|instant runoff) round \d*|rcv round|"
                         r"someone else entirely|another candidate)$", s))

def drop_duplicate_surveys(d, label=""):
    """Second-pass dedup keyed on SURVEY IDENTITY, not on the pollster's name.

    Added 2026-08-08. `norm_pollster` collapses spelling variants ("Glengariff Group, Inc."
    vs "Glengariff Group"), but it cannot know that two genuinely DIFFERENT strings name the
    same organisation. The live feed is full of these:

        'Saint Anselm College'  ==  'St. Anselm'
        'Remington'             ==  'Remington Research Group'
        'Marquette Law School'  ==  'Marquette University'  ==  'Marquette University Law School'
        'KSTP / SurveyUSA'      ==  'SurveyUSA'
        'UT Tyler'              ==  'University of Texas at Tyler Center for Opinion Research'

    37 such pairs were measurable in the 2026 primary feed, double-counting 740 poll rows
    across 55 races (found 2026-08-08 investigating why Francesca Hong read 99.97% in
    WI-Gov-DEM; the duplicates were not the cause there, but the bug is real).

    WHY NOT A NAME-ALIAS TABLE: because names are the unreliable part. Two rows are the same
    survey when they report the SAME percentage for the SAME candidate in the SAME race on the
    SAME end_date from the SAME sample size - regardless of what the source calls the pollster.
    Sample size is what makes this safe, and it is doing real work: 'Barbara Jordan Public
    Policy Research and Survey Center' and 'Texas Southern University' publish IDENTICAL
    numbers where n matches (that is one survey under a sponsor's name and the field house's
    name) but differ by up to 14 points on 2025-08-12 where n also matches - different
    matchup questions in one poll, which must NOT be merged. Keying on the reported value
    itself keeps those apart, where a name-alias table would have silently merged them.

    Deliberately conservative: rows missing sample_size are left alone (no key, no dedup), and
    the pct must match exactly. This only ever removes a row that is byte-identical in the
    fields that define a survey's result.
    """
    need = {"end_date", "pct", "sample_size", "race_id", "cand_key"}
    if not need.issubset(d.columns):
        return d, 0
    keyed = d["sample_size"].notna() & d["pct"].notna()
    if not keyed.any():
        return d, 0
    sub = d[keyed]
    dup = sub.duplicated(subset=["race_id", "cand_key", "end_date", "pct", "sample_size"],
                         keep="first")
    n = int(dup.sum())
    if n:
        d = d.drop(index=sub.index[dup])
        if label:
            print(f"duplicate SURVEYS dropped ({label}): {n} rows - same race/candidate/date/"
                  f"pct/sample-size under a different pollster NAME")
    return d, n

def best_other(s):
    """Per-row 'best OTHER value in the group' (NaN-safe): for the top value, the runner-up's
    value; for everyone else, the top value. Ties broken by position (first occurrence of the
    max is treated as "the leader"). Used for poll_lead: each candidate's gap to the best
    candidate who ISN'T them, never a single race-wide constant.

    BUGFIX (2026-07-21): the old poll_lead used one constant per race (the runner-up's value)
    subtracted from EVERY candidate, including the runner-up themself -> poll_lead was exactly
    0.0 for 100% of 2nd-place candidates (verified on 2024 training data) and used the WRONG
    comparison point for 3rd place and below. Only the true leader's value was ever correct.
    """
    ok = s.notna()
    if ok.sum() <= 1:
        return pd.Series(np.where(ok, s.fillna(0.0), 0.0), index=s.index)
    vals = s[ok]
    top_idx = vals.idxmax()
    top = vals.loc[top_idx]
    second = vals.drop(top_idx).max()
    out = pd.Series(top, index=s.index)
    out.loc[top_idx] = second
    return out.where(ok)

# ---------------------------------------------------------------- fundamentals

def load_fundamentals():
    """Incumbency + prior-margin lookups from the committed static files in data/.

    Returns dict(inc_map=..., margin_map=...). No network, ever.
    """
    rc = pd.read_csv(os.path.join(DATA_DIR, "races.csv"), low_memory=False)
    off = rc["office_name"].astype(str).str.lower()
    rc["office"] = np.select(
        [off.str.contains("senate"), off.str.contains("house"), off.str.contains("governor")],
        ["Senate", "House", "Governor"], default=None)
    rc["state"] = rc["state_abbrev"].str.upper()
    rc["district"] = ""
    hm = rc["office"] == "House"
    rc.loc[hm, "district"] = rc.loc[hm, "office_seat_name"].map(pdist)
    inc_map = {(r.cycle, r.state, r.office, r.district): npar(r.incumbent_party)
               for r in rc[rc["office"].notna()].itertuples()
               if pd.notna(r.incumbent_party)}

    def _load_res(fn, office):
        r = pd.read_csv(os.path.join(DATA_DIR, fn), low_memory=False)
        r = r[r["stage"].astype(str).str.lower().str.contains("general", na=False)]
        r["office"] = office
        r["state"] = r["state_abbrev"].str.upper()
        r["district"] = "" if office != "House" else r["office_seat_name"].map(pdist)
        r["p"] = r["ballot_party"].map(npar)   # 'party' col is null in these files
        r["pct"] = pd.to_numeric(r["percent"], errors="coerce")
        return r[["cycle", "state", "office", "district", "p", "pct"]]

    allres = pd.concat([_load_res("res_senate.csv", "Senate"),
                        _load_res("res_house.csv", "House"),
                        _load_res("res_governor.csv", "Governor")])
    piv = (allres[allres["p"].isin(["DEM", "REP"])]
           .groupby(["cycle", "state", "office", "district", "p"])["pct"].max().unstack("p"))
    for col in ["DEM", "REP"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv["margin"] = piv["DEM"].fillna(0) - piv["REP"].fillna(0)
    margin_map = {idx: row.margin for idx, row in piv.iterrows()}
    return dict(inc_map=inc_map, margin_map=margin_map)

def prior_margin(margin_map, year, state, office, district):
    """Most recent same-office two-party margin strictly BEFORE `year` (leak-free)."""
    for back in range(2, 9, 2):
        v = margin_map.get((year - back, state, office, district))
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return np.nan

# ---------------------------------------------------------------- primary results (2026-07-22)

def load_primary_results():
    """data/{house,primary}_results_hist.csv -> {(year,state,office,district,party,cand_key):
    dict(is_primary_nominee, primary_margin, primary_uncontested)} for the GENERAL model's
    candidate-quality features (does this candidate's party field show they won a contested
    or lopsided primary?). Sourced from fetch_house_primary_results_hist.py (House, 1998-2024,
    fact-checked - see that script's docstring) + fetch_primary_results_2026.py --hist
    (Senate/Governor). Real coverage 33.5% of the general model's full 14-cycle candidate
    table (measured 2026-07-23 against the true production BASE table - an earlier ~49%
    estimate here was wrong, measured against only the 2018-2024 slice by mistake). The
    rest is NaN, not 0 - a candidate with no primary-results match is UNKNOWN, not "ran
    unopposed". NOTE: this feature was built, ablated on both the win and margin models,
    and DROPPED from production (2026-07-23, honest null result on both - see HANDOFF.md)
    - this loader stays in the codebase but feature_list(primary_results=True) is NOT the
    production default.

    primary_margin: winner's pct minus runner-up's pct (>=0 by construction, a property of
    HOW they won their primary, same value for every candidate who WAS that nominee - this
    is a fact about the nominee's primary, not about the current candidate's own vote share
    in some other race). NaN for a single-candidate (no real contest to measure) race - see
    primary_uncontested for that case instead.
    primary_uncontested: 1 if the primary had one candidate, or a runner-up with <5% (the
    write-in/token-challenger case - verified: 112 of 1735 two-candidate races fit this,
    median genuine 2-candidate runner-up share is ~30%, so 5% cleanly separates real
    contests from non-contests); 0 if genuinely contested; NaN if no primary-results match.
    Only the WINNER of each primary is attributed a value (only nominees reach the general
    election, which is the only place this feature is consumed)."""
    frames = []
    # Order matters on the merge below: LATER files win on a duplicate (year,state,office,
    # party) key. primary_results_deep_hist.csv (2026-08-07) is the widest archive - every
    # Senate/Governor primary 1998-2024 - and supersedes the 2018-only hist file where they
    # overlap. Before it existed, Senate and Governor were 0.0% populated for EVERY pre-2018
    # cycle, because the --hist scrape took its targets from a primary-POLLS file that only
    # went back to 2018. Results never depended on polls; the deep scrape asks the training
    # races directly.
    # primary_results_2026.csv is the CURRENT cycle and MUST be here, or the block is
    # 100% NaN at serve time while the model trained on real values - train/serve skew that
    # the feature-presence assert in predict.py cannot catch (the columns exist, they are
    # just empty). Found 2026-08-07 by measuring serve coverage directly after wiring
    # primary_results into predict.py and still getting 0.0%.
    for fn in ("data/house_primary_results_hist.csv", "data/primary_results_hist.csv",
               "data/primary_results_deep_hist.csv", "data/primary_results_2026.csv"):
        p = os.path.join(os.path.dirname(DATA_DIR), fn)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        return {}
    pr = pd.concat(frames, ignore_index=True)
    # DEDUPE BY race_id. The archives OVERLAP - 2018-2024 Senate/Governor races appear in
    # both primary_results_hist.csv and the newer deep archive - and concatenating them
    # duplicated every candidate row. The runner-up lookup below then found the WINNER's
    # own second copy, so primary_margin came out 0.0 for 232 candidate-rows: Kay Ivey won
    # the 2018 AL-Gov primary 56.1-24.9 and was recorded as winning by nothing.
    # Keep the LAST file's copy (deep archive supersedes), matching the load order above.
    pr = pr.drop_duplicates(subset=["race_id", "candidate"], keep="last")
    parts = pr["race_id"].str.split("_", n=3, expand=True)
    pr["year"] = parts[0].astype(int)
    pr["state"] = parts[1]
    of_di = parts[2].str.split("-", n=1, expand=True)
    pr["office"] = of_di[0]
    pr["district"] = of_di[1].fillna("") if of_di.shape[1] > 1 else ""
    pr["party"] = parts[3]

    out = {}
    for (yr, st, of, di, pty), g in pr.groupby(["year", "state", "office", "district", "party"]):
        if not (g["is_winner"] == True).any():   # noqa: E712 (explicit bool match)
            continue
        winner = g.loc[g["is_winner"] == True].iloc[0]
        n = len(g)
        if n == 1:
            margin, uncontested = np.nan, 1
        else:
            runner_up_pct = g.loc[g.index != winner.name, "pct"].max()
            margin = winner["pct"] - runner_up_pct
            uncontested = int(runner_up_pct < 5)
        out[(yr, st, of, di, pty, winner["cand_key"])] = dict(
            primary_margin=(float(margin) if margin == margin else np.nan),
            primary_uncontested=uncontested,
        )
    return out

# ---------------------------------------------------------------- candidate bios (2026-07-23)

def load_candidate_bios():
    """data/candidate_bios.csv -> {(year,office,state,district,party,cand_key):
    dict(bio_office_level)} for the GENERAL model. Same source the PRIMARY model already
    uses (features_primary.load_candidate_bios) - not imported from there to avoid a
    circular import (features_primary imports features); this is a standalone re-read of
    the same committed file, GENERAL-model party/office keying (district='' for
    Senate/Governor vs primary's within-party-field keying).

    bio_office_level: 4 federal / 3 statewide / 2 state-leg / 1 local / 0 none-detected
    (fetch_candidate_bios.py). Real coverage 58.1% of the general model's full 14-cycle
    candidate table, 67.7% among WINNERS (measured 2026-07-24, AFTER the unbiased-target-
    list fix - an earlier 32.7% figure here predated that fix; the old target list was
    derived from primary-POLL pages and systematically excluded uncontested/safe-seat
    incumbent races, i.e. exactly the highest-office-level candidates). Remaining known
    gaps: a few House pages missing from transient fetch failures (CA 2024, LA/WA several
    cycles - re-running fetch_house_candidate_bios_hist.py retries only missing pages),
    genuinely thin pre-2012 Wikipedia editing depth, and independents filed under an
    "Independents" section the parser's stage tracker doesn't classify as primary
    (Bernie Sanders - open item). 2026 predict-time coverage will likely trail training
    coverage (race pages accrue bio detail in the YEARS AFTER an election) - a real
    train/serve mismatch to weigh against whatever an ablation shows. ABLATION STATUS
    (2026-07-24, post-coverage-fix): win model - calibration metrics (AUC/AUC-PR/KS/Brier)
    all flipped positive vs the pre-fix null, but race-acc still -0.0048 (2020/2024 folds
    regress, confirmed NOT a NaN-dilution artifact via a matched-races-only split); margin
    model - uniformly worse MAE in all 4 eval cycles. NOT wired into production. Only
    office_level is exposed here (not bio_in_office/bio_prior_candidacy) - the PRIMARY
    model's own overfit review found those added nothing once tested per-cycle
    (METHODOLOGY.md)."""
    path = os.path.join(DATA_DIR, "candidate_bios.csv")
    if not os.path.exists(path):
        return {}
    b = pd.read_csv(path, low_memory=False)
    out = {}
    for r in b.itertuples():
        di = dist_str(r.district)
        party = npar(r.party)
        out[(int(r.year), r.office, r.state, di, party, r.cand_key)] = dict(
            bio_office_level=int(r.office_level))

    # PERSON-LEVEL as-of-year fallback (2026-07-29): the exact-key map above is built from
    # candidate_bios.csv, whose rows exist only for races in the TRAINING poll file. Live
    # predict-time candidates (e.g. a 2026 race present in the polling-agg feed but not yet in
    # polls_long_with_results.csv) would miss even when we KNOW their office history. So also
    # build a person-level tenure map from the hand-coded + Ballotpedia sources
    # (offices_json = [[office, start, end_or_null]]) keyed (cand_key, state); build_candidate_
    # table falls back to computing the as-of-year level from it when the exact key misses.
    # Leak-free: only offices whose tenure STARTED strictly before the race year count.
    person = {}
    for fn in ("candidate_bios_manual.csv", "candidate_bios_ballotpedia.csv"):
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            continue
        src = pd.read_csv(p, low_memory=False)
        for r in src.itertuples():
            raw = getattr(r, "offices_json", None)
            try:
                offices = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
            except (json.JSONDecodeError, TypeError):
                offices = []
            key = (norm_name(r.candidate), r.state)
            # manual overrides ballotpedia (manual file read first); [] = verified no office
            if key not in person:
                person[key] = offices
    out["__person_offices__"] = person
    return out

def _person_asof_level(offices, year):
    """Highest office-level among a person's offices whose tenure STARTED strictly before
    `year` (leak-free). offices = [[office_phrase, start, end_or_None], ...]. Returns an int
    level (0 if they held no office before that year) or None if offices is unknown/empty-
    and-therefore-not-a-verified-zero. An EMPTY list here means a hand-verified "no prior
    office" -> level 0 (the manual/ballotpedia sources only store [] when that's confirmed)."""
    from fetch_candidate_bios_ballotpedia import classify_ballotpedia  # lazy: circular import
    if offices is None:
        return None
    levels = [classify_ballotpedia(o[0]) for o in offices
              if len(o) >= 2 and o[1] is not None and int(o[1]) < year]
    return max(levels) if levels else 0

# ---------------------------------------------------------------- FEC fundraising

def fec_cand_key(name):
    """'PELTOLA, MARY (ALIAS)' -> norm_name('mary peltola') -> 'peltola m'."""
    s = re.sub(r"\(.*?\)", "", str(name))
    parts = s.split(",", 1)
    s = (parts[1] + " " + parts[0]) if len(parts) == 2 else s
    return norm_name(s)

def load_fec(path=None, extended=False):
    """data/fec_summary.csv -> {(cycle,state,office,district,cand_key): {receipts,...}}.

    Senate district = ''; FEC House at-large '00' -> '1' (matches our race keys).
    NOTE the cutoff caveat in fetch_fec.py: historical totals run through Dec 31, so
    RATIO features (share/composition) are the trustworthy ones; raw totals are secondary.

    extended=True (BATCH 5+ ONLY — changes feature values, so artifacts and predict must
    flip together, then retrain) additionally merges:
      - data/fec_detail.csv (API): itemized individual money -> small-dollar share
        (unitemized = bulk total individual minus API itemized).
      - data/governor_finance.csv (FollowTheMoney): governor receipts -> fund_receipts_ln
        + fund_share finally exist for Governor rows (composition stays NaN).
    """
    path = path or os.path.join(DATA_DIR, "fec_summary.csv")
    f = pd.read_csv(path, dtype={"district": str})
    f["cand_key"] = f["cand_name"].map(fec_cand_key)
    f["district"] = [("" if o == "Senate"
                      else ("1" if str(di) in ("00", "0", "nan") else str(int(float(di)))))
                     for o, di in zip(f["office"], f["district"])]
    f = f.sort_values("receipts", ascending=False).drop_duplicates(
        ["cycle", "state", "office", "district", "cand_key"])
    out, by_id = {}, {}
    for r in f.itertuples():
        k = (r.cycle, r.state, r.office, r.district, r.cand_key)
        out[k] = dict(receipts=r.receipts, indiv=r.indiv_contrib, pac=r.pac_contrib,
                      party=r.party_contrib, self=r.self_fund, small=np.nan)
        by_id[(r.cycle, r.cand_id)] = k
    if not extended:
        return out

    det_path = os.path.join(DATA_DIR, "fec_detail.csv")
    if os.path.exists(det_path):
        det = pd.read_csv(det_path)
        itemized = det.groupby(["cycle", "cand_id"])["indiv_itemized"].max()
        for (cyc, cid), item in itemized.items():
            k = by_id.get((cyc, cid))
            if k and out[k]["indiv"] and out[k]["indiv"] > 0 and pd.notna(item):
                out[k]["small"] = float(np.clip(1 - item / out[k]["indiv"], 0, 1))

    gov_path = os.path.join(DATA_DIR, "governor_finance.csv")
    if os.path.exists(gov_path):
        g = pd.read_csv(gov_path)
        g["cand_key"] = g["cand_name"].map(fec_cand_key)
        g = g.sort_values("receipts", ascending=False).drop_duplicates(
            ["cycle", "state", "cand_key"])
        for r in g.itertuples():
            if pd.notna(r.receipts):
                out.setdefault((r.cycle, r.state, "Governor", "", r.cand_key),
                               dict(receipts=float(r.receipts), indiv=np.nan, pac=np.nan,
                                    party=np.nan, self=np.nan, small=np.nan))
    return out

FUND_FEATS = ["fund_receipts_ln", "fund_share", "fund_indiv_pct", "fund_pac_pct",
              "fund_party_pct", "fund_self_pct"]
FUND_FEATS_EXT = FUND_FEATS + ["fund_smalldollar_pct"]   # batch 5+ (extended FEC)

# ---------------------------------------------------------------- poll prep

def prepare_polls(d):
    """Coerce types on a long poll frame (one row per poll-candidate). Modifies a copy."""
    d = d.copy()
    d["end_date"] = pd.to_datetime(d["end_date"], errors="coerce", format="mixed")
    d["election_date"] = pd.to_datetime(d["election_date"], errors="coerce", format="mixed")
    d["days_to_elec"] = (d["election_date"] - d["end_date"]).dt.days
    for c in ["pct", "sample_size", "days_to_elec"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # harmonize instrument precision: the live 2026 feed carries ~1-decimal pcts while the
    # 538-era training files carry more — round BOTH paths so train and serve match.
    d["pct"] = d["pct"].round(1)
    d["district"] = d["district"].map(dist_str)
    return d

# Generic corporate/industry descriptor words that sources append inconsistently to the SAME
# pollster ("Mitchell Research" vs "Mitchell Research & Communications", "Rosetta Stone" vs
# "Rosetta Stone Communications"). Stripped only from the END of the name, and never down to
# nothing (see norm_pollster) - so a firm whose name IS a descriptor keeps it. Deliberately
# does NOT include 'research'/'polling'/'strategies'/'insights'/'group': those are load-bearing
# parts of real distinct names (Tulchin Research vs Tulchin; Peak Insights vs Peak).
_POLLSTER_TAIL = r"(communications|company|llc|ltd|corp|corporation|and associates|associates)"

def norm_pollster(p):
    """Normalize pollster names so house effects match across feeds (538 vs NYT/Wikipedia):
    casefold, drop partisan tags, '&'->'and', 'Co.'->'company', strip punctuation, then drop
    trailing generic descriptors (2026-07-31).

    The trailing-descriptor strip exists because the 2026 feed carries the same survey under
    two spellings from two sources, which then survived dedup as two independent polls and
    got double-counted in the averages (found in MI-Sen-DEM: 99 poll rows for 36 real
    surveys). Stripping is anchored to the END and refuses to empty the string, so it merges
    'mitchell research communications' -> 'mitchell research' without touching a pollster
    actually named e.g. 'Communications Co'."""
    s = str(p).casefold().strip()
    s = re.sub(r"\(([dr]|dem|rep)\)", "", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\bco\b\.?", "company", s)
    s = re.sub(r"\binc\b\.?|,", "", s)
    s = re.sub(r"[^a-z0-9/ ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # repeat: "X Research and Communications LLC" -> "X Research"
    while True:
        s2 = re.sub(rf"\s+{_POLLSTER_TAIL}$", "", s).strip()
        s2 = re.sub(r"\s+and$", "", s2).strip()   # left dangling by the strip above
        if s2 == s or not s2:
            break
        s = s2
    return s

# ---------------------------------------------------------------- race dynamics

def count_lead_changes(g):
    """How often the running-mean front-runner flipped over the race's poll dates."""
    g = g.dropna(subset=["end_date", "pct"]).sort_values("end_date")
    prev, changes = None, 0
    for dt in g["end_date"].drop_duplicates().sort_values():
        means = g[g["end_date"] <= dt].groupby("cand_key")["pct"].mean()
        if means.empty:
            continue
        leader = means.idxmax()
        if prev is not None and leader != prev:
            changes += 1
        prev = leader
    return changes

_SPECIAL_SEATS = {(2026, "SEN", "FL"), (2026, "SEN", "OH")}

def normalize_special_race_id(rid):
    """Add the missing '-S' suffix when a seat is ONLY ever contested as a special.

    The two poll sources disagree on labelling. NYT tags every Florida Senate row
    '2026-SEN-FL-S'; Wikipedia writes the same contest as '2026-SEN-FL'. parse_race_id
    correctly treats '-S' as its own race, so the disagreement split ONE contest into
    two - FL Senate showed up twice on the dashboard with the same candidates and the
    same date, differing only by "Alex" vs "Alexander" Vindman (found 2026-08-05).

    Only seats in _SPECIAL_SEATS are rewritten, and only when there is no regular race
    for that seat in the same cycle: FL and OH have NO regular 2026 Senate election, so
    an unsuffixed id can only be the special. A seat holding both a regular and a special
    race in one cycle must NOT be listed here - the suffix is the only thing telling them
    apart, and rewriting would merge two genuinely different contests.
    """
    parts = str(rid).split("-")
    if len(parts) != 3:
        return rid
    try:
        year = int(parts[0])
    except ValueError:
        return rid
    if (year, parts[1].upper(), parts[2].upper()) in _SPECIAL_SEATS:
        return f"{rid}-S"
    return rid

_ALIASES = None

def load_name_aliases():
    """{wrong spelling -> correct spelling} from data/name_aliases.csv (cached).

    The two poll sources spell some names differently, and norm_name deliberately does
    NOT fuzzy-match (that would merge genuinely different people). So a one-character
    feed typo silently splits a candidate in two and halves both halves' poll counts.
    Found 2026-08-05: CT-Gov-DEM carried "Josh Elliot" (7 polls) AND "Josh Elliott"
    (7 polls) as separate candidates six days before the primary; Raffensperger was
    split in TWO different races.

    Hand-maintained and exact-match only - no automatic fuzzy merging, because the cost
    of wrongly merging two real people is much higher than leaving a split in place.
    """
    global _ALIASES
    if _ALIASES is None:
        p = paths.data("name_aliases.csv")
        _ALIASES = {}
        if os.path.exists(p):
            a = pd.read_csv(p)
            _ALIASES = dict(zip(a["wrong"].astype(str), a["correct"].astype(str)))
    return _ALIASES

def apply_name_aliases(s):
    """Map a candidate-name Series through the alias table before any keying."""
    al = load_name_aliases()
    return s.map(lambda x: al.get(str(x), x)) if al else s

def poll_momentum_slope(gc):
    """OLS slope of pct vs time (pts/day) over ALL of a candidate's dated polls.

    THE single definition of poll_momentum - imported by features_primary.py too.
    Do not re-implement: a forked 60-day copy lived in features_primary.py until
    2026-08-03 and silently kept both primary models on the old definition through
    a full retrain, because the retrain was verified against features.py only.

    Changed 2026-08-03 from a 60-day window to all dated polls. The window version
    was NaN for 100% of 2026 serve rows (the nearest primary was >60 days out), so
    the trees were splitting on a feature that never exists in production. The two
    definitions measure the same thing (correlation 0.917 on the 2,396 training rows
    where both exist): serve coverage goes 0% -> 39%, train coverage 54% -> 60%.

    The held-out difference is NOISE-LEVEL - across 5 seeds the win model reads
    .8631 (full) vs .8613 (window) with +/-.002 spread, and the margin model mildly
    prefers the window (MAE 7.39 vs 7.35). The change is justified by SERVE-TIME
    AVAILABILITY, not by accuracy. Carrying BOTH was tested and is strictly worse on
    the win model (.8602) - at r=0.92 they are near-duplicates and the spare one just
    gives the trees a sparser column to overfit.

    NB: callers must NOT reuse their end_date-based `dated` frame here - that one
    feeds poll_last, and aliasing it would silently change poll_last too.
    """
    rows = gc.dropna(subset=["pct", "days_to_elec"])
    if len(rows) < 3:
        return np.nan
    x = -rows["days_to_elec"].values.astype(float)
    y = rows["pct"].values.astype(float)
    if np.ptp(x) <= 0:
        return np.nan
    return np.polyfit(x, y, 1)[0]

def margin_dynamics(g):
    """Per-candidate margin-vs-best-opponent trajectory stats over the campaign."""
    g = g.dropna(subset=["end_date", "pct"]).sort_values("end_date")
    dates = g["end_date"].drop_duplicates().sort_values()
    series = {}
    t0 = dates.min()
    for dt in dates:
        means = g[g["end_date"] <= dt].groupby("cand_key")["pct"].mean()
        if len(means) == 0:
            continue
        elapsed = (dt - t0).days
        for ck, val in means.items():
            others = means.drop(ck)
            best_other = others.max() if len(others) else 0.0
            series.setdefault(ck, []).append((elapsed, val - best_other))
    out = {}
    for ck, pts in series.items():
        m = np.array([p[1] for p in pts], dtype=float)
        x = np.array([p[0] for p in pts], dtype=float)
        trend = np.polyfit(x, m, 1)[0] if (len(m) >= 2 and np.ptp(x) > 0) else 0.0
        out[ck] = dict(avg_margin_over_time=float(np.mean(m)),
                       margin_volatility=float(np.std(m)) if len(m) > 1 else 0.0,
                       min_margin=float(np.min(m)),
                       margin_trend=float(trend))
    return out

# ---------------------------------------------------------------- house effect

def compute_house_effect(d, train_years, shrink_k=5.0):
    """Per-pollster DEM-REP margin deviation vs the race consensus, TRAIN years only.
    Keyed by NORMALIZED pollster name (norm_pollster) so 2026-feed names match history.
    Empirical-Bayes shrunken toward 0 by n/(n+k): a 2-poll pollster's raw 'house effect'
    is mostly noise and used to be applied at full strength."""
    mar = (d[d["party_std"].isin(["DEM", "REP"])]
           .pivot_table(index=["race_id", "poll_id", "pollster", "year"],
                        columns="party_std", values="pct", aggfunc="max").reset_index())
    for col in ["DEM", "REP"]:
        if col not in mar.columns:
            mar[col] = np.nan
    mar["m"] = mar["DEM"] - mar["REP"]
    tm = mar[mar["year"].isin(list(train_years))].copy()
    tm["dev"] = tm["m"] - tm.groupby("race_id")["m"].transform("mean")
    tm["pollster_key"] = tm["pollster"].map(norm_pollster)
    g = tm.groupby("pollster_key")["dev"].agg(["mean", "count"])
    return (g["mean"] * g["count"] / (g["count"] + shrink_k)).to_dict()

def compute_bias_priors(d, shrink_k=8.0):
    """{(cycle, state): shrunken PRIOR-cycles mean signed poll-margin error} + (cycle,'_nat').

    e = polled(D−R) − actual(D−R) per race; positive = polls overstated Democrats there.
    For target cycle Y only cycles < Y contribute (leak-free). State means are shrunk toward
    the national prior mean by n/(n+k). Historically this shifts ±4-7 pts between cycles —
    the single biggest correlated risk (HANDOFF.md). d must carry vote_pct (training frame)."""
    dd = d[d["party_std"].isin(["DEM", "REP"])].dropna(subset=["vote_pct"])
    g = (dd.groupby(["year", "race_id", "party_std"])
           .agg(poll=("pct", "mean"), act=("vote_pct", "first")).reset_index())
    p = g.pivot_table(index=["year", "race_id"], columns="party_std", values=["poll", "act"])
    e = ((p[("poll", "DEM")] - p[("poll", "REP")])
         - (p[("act", "DEM")] - p[("act", "REP")])).dropna()
    err = e.rename("e").reset_index()
    err["state"] = err["race_id"].str.split("_").str[1]
    out = {}
    years = sorted(err["year"].unique())
    for y in years + [years[-1] + 2]:            # +2 covers the next (predict) cycle
        past = err[err["year"] < y]
        if past.empty:
            continue
        nat = float(past["e"].mean())
        out[(y, "_nat")] = nat
        for s, grp in past.groupby("state"):
            n = len(grp)
            out[(y, s)] = float((grp["e"].mean() * n + nat * shrink_k) / (n + shrink_k))
    return out

def candidate_poll_adj(d, house):
    """Per (race_id, cand_key) plain mean of house-effect-adjusted pct.

    Lets CV folds swap in a leak-free poll_adj (house effect from train cycles only)
    without rebuilding the whole candidate table."""
    dd = d[["race_id", "cand_key", "party_std", "pct", "pollster"]].copy()
    dd["sign"] = dd["party_std"].map({"DEM": 1, "REP": -1}).fillna(0)
    dd["adj"] = dd["pct"] - dd["sign"] * dd["pollster"].map(norm_pollster).map(house).fillna(0.0)
    return dd.groupby(["race_id", "cand_key"])["adj"].mean()

# ---------------------------------------------------------------- main builder

def build_candidate_table(d, macro, natl_env_map, funds, house_train_years=None, house=None,
                          fec=None, bias_priors=None, primary_results=None,
                          candidate_bios=None):
    """Collapse prepared long polls `d` -> one row per candidate per race, with features.

    d must have: race_id, year, state, office, district, candidate, cand_key, party_std,
    pct, end_date, days_to_elec, sample_size, pollster. `won` optional (NaN at predict time).
    All poll aggregates are PLAIN averages (no weighting).

    House effect: pass `house_train_years` to compute it from those cycles of `d`, or pass a
    precomputed `house` dict directly (predict time: computed from historical polls, applied
    to the new cycle's polls).

    primary_results: pass load_primary_results() to add primary_margin/primary_uncontested
    (2026-07-22) - how contested/lopsided this candidate's own primary was, 33.5% coverage
    (real matches only; NaN elsewhere, never a silent 0/uncontested guess). ABLATED OUT of
    production 2026-07-23 (honest null result on both win and margin models) - kept in the
    codebase, not the default.

    candidate_bios: pass load_candidate_bios() to add bio_office_level (2026-07-23) - the
    candidate's highest office held (4 fed/3 statewide/2 state-leg/1 local/0 none), 58.1%
    coverage post-target-list-fix (see load_candidate_bios's docstring for the full
    coverage + ablation story; ablated 2026-07-24: NOT production - better calibration on
    the win model but worse pick-accuracy in recent folds, uniformly worse margin MAE).
    """
    if house is None:
        house = compute_house_effect(d, house_train_years or [])
    lead_change_map = {rid: count_lead_changes(g) for rid, g in d.groupby("race_id")}
    margin_dyn_map = {rid: margin_dynamics(g) for rid, g in d.groupby("race_id")}
    inc_map, margin_map = funds["inc_map"], funds["margin_map"]
    has_won = "won" in d.columns

    rows = []
    for race_id, g in d.groupby("race_id"):
        yr = int(g["year"].iloc[0]); st = g["state"].iloc[0]
        of = g["office"].iloc[0];    di = dist_str(g["district"].iloc[0])
        dyn = margin_dyn_map.get(race_id, {})
        for ck, gc in g.groupby("cand_key"):
            gc = gc.sort_values("end_date")
            dated = gc.dropna(subset=["end_date"])          # NaT polls can't be "most recent"
            last30 = gc[gc["days_to_elec"] <= 30]
            last7 = gc[gc["days_to_elec"] <= 7]
            party = gc["party_std"].iloc[0]
            sign = 1 if party == "DEM" else -1 if party == "REP" else 0

            incp = inc_map.get((yr, st, of, di))
            pm = prior_margin(margin_map, yr, st, of, di)

            # POLL MOMENTUM: slope of the candidate's polls over time, over ALL dated
            # polls rather than a final-60-day window. The window version was 0.0%
            # populated at serve time for the general model (a general election is a
            # FIXED date, so until ~September nothing is within 60 days of it) yet
            # ranked 12th of 187 by gain and appeared in 82 dashboard SHAP blocks with
            # a NULL value in every one. See poll_momentum_slope() for the full record.
            slope = poll_momentum_slope(gc)

            adj = gc["pct"] - gc["pollster"].map(lambda p: sign * house.get(norm_pollster(p), 0.0))
            md = dyn.get(ck, {})

            pr = (primary_results.get((yr, st, of, di, party, ck))
                 if primary_results is not None else None)
            # bio_office_level is a candidate PROPERTY, keyed statewide for Senate/Governor.
            # The race district can be "S" for a special election (predict.py keeps
            # 2026-SEN-FL-S / -OH-S as their own races) while the bio table stores statewide
            # Senate as "" - so look bios up with a statewide-collapsed district for those
            # offices (fixed 2026-07-29: FL/OH 2026 special-election Senate candidates - Moody,
            # Brown, Husted, Vindman, Nixon - were missing bio purely on this "S" vs "" mismatch).
            bio_di = "" if of in ("Senate", "Governor") else di
            bio = (candidate_bios.get((yr, of, st, bio_di, party, ck))
                  if candidate_bios is not None else None)
            # PERSON-LEVEL fallback: exact-key miss (common for live predict-time candidates
            # absent from the training poll file) -> compute the as-of-year level from the
            # hand-coded/Ballotpedia tenure map, keyed (cand_key, state), leak-free (2026-07-29).
            if bio is None and candidate_bios is not None:
                poff = candidate_bios.get("__person_offices__", {})
                if (ck, st) in poff:
                    lvl = _person_asof_level(poff[(ck, st)], yr)
                    if lvl is not None:
                        bio = dict(bio_office_level=lvl)

            fe = fec.get((yr, st, of, di, ck)) if fec is not None else None
            rec = fe["receipts"] if fe else np.nan
            fund = dict(
                fund_receipts_ln=(np.log1p(rec) if fe and rec > 0 else np.nan),
                fund_indiv_pct=(fe["indiv"] / rec if fe and rec > 0 else np.nan),
                fund_pac_pct=(fe["pac"] / rec if fe and rec > 0 else np.nan),
                fund_party_pct=(fe["party"] / rec if fe and rec > 0 else np.nan),
                fund_self_pct=(fe["self"] / rec if fe and rec > 0 else np.nan),
                fund_smalldollar_pct=(fe.get("small", np.nan) if fe else np.nan),
                _fund_receipts=(rec if fe else np.nan),
            ) if fec is not None else {}

            rows.append(dict(
                race_id=race_id, year=yr, state=st, office=of, district=di,
                cand_key=ck, candidate=gc["candidate"].iloc[0], party=party,
                # real affiliation for display (defaults to model party unless overridden)
                display_party=(gc["display_party"].iloc[0] if "display_party" in gc.columns
                               else party),
                won=(int(gc["won"].iloc[0]) if has_won and pd.notna(gc["won"].iloc[0]) else np.nan),
                # actual vote share — LABEL for the margin model, never a feature
                vote_pct=(pd.to_numeric(gc["vote_pct"], errors="coerce").iloc[0]
                          if "vote_pct" in gc.columns else np.nan),
                poll_avg=gc["pct"].mean(),
                poll_last=(dated["pct"].iloc[-1] if len(dated) else gc["pct"].mean()),
                poll_last30=(last30["pct"].mean() if len(last30) else gc["pct"].mean()),
                # final-week average (2026-07-31), mirroring features_primary. Unlike
                # poll_last30 this does NOT fall back to the all-time mean on an empty window:
                # a 7-day window is empty for most candidates most of the time, and that
                # fallback would silently re-inject the stale full-campaign average under a
                # "final week" name. NaN instead - XGBoost routes missing natively.
                # Still an unweighted WINDOW, not recency weighting: the no-weighting rule at
                # the top of this file is intact (same basis as the existing poll_last30).
                poll_last7=(last7["pct"].mean() if len(last7) else np.nan),
                n_polls_last7=len(last7),
                poll_std=gc["pct"].std(),
                n_polls=len(gc),
                n_polls_over50=int((gc["pct"] > 50).sum()),
                avg_sample=gc["sample_size"].mean(),
                min_days=gc["days_to_elec"].min(),
                prior_margin_cand=(sign * pm if not (isinstance(pm, float) and np.isnan(pm)) else np.nan),
                # unknown incumbency = NaN (missing), never a silent 0
                is_incumbent=((1 if incp == party else 0) if incp in ("DEM", "REP") else np.nan),
                is_inc_party_race=(1 if incp in ("DEM", "REP") else 0),
                natl_env_cand=(sign * natl_env_map.get(yr, np.nan)),
                # prior-cycles poll-bias prior for this state (leak-free), candidate-signed:
                # positive = polls here historically overstated THIS candidate's party
                bias_prior_cand=(sign * _bp if bias_priors is not None and sign != 0
                                 and (_bp := bias_priors.get((yr, st),
                                                             bias_priors.get((yr, "_nat"))))
                                 is not None else np.nan),
                poll_momentum=slope,
                poll_adj=adj.mean(),
                n_lead_changes=lead_change_map.get(race_id, 0),
                lead_changed=int(lead_change_map.get(race_id, 0) > 0),
                avg_margin_over_time=md.get("avg_margin_over_time", np.nan),
                margin_volatility=md.get("margin_volatility", np.nan),
                min_margin=md.get("min_margin", np.nan),
                margin_trend=md.get("margin_trend", np.nan),
                is_president_party=int(party == PRES_PARTY.get(yr)),
                # how contested/lopsided THIS candidate's own primary was (NaN = no matched
                # primary-results page, not "ran unopposed" - see load_primary_results)
                primary_margin=(pr["primary_margin"] if pr else np.nan),
                primary_uncontested=(pr["primary_uncontested"] if pr else np.nan),
                # highest office ever held (NaN = no matched bio, not "no experience" -
                # see load_candidate_bios)
                bio_office_level=(bio["bio_office_level"] if bio else np.nan),
                **fund,
                **macro.get(yr, {}),
            ))
    c = pd.DataFrame(rows)

    if fec is not None:
        # share of the race's (matched) money — the ratio feature robust to cutoff dates
        tot = c.groupby("race_id")["_fund_receipts"].transform("sum")
        c["fund_share"] = np.where(tot > 0, c["_fund_receipts"] / tot, np.nan)
        c = c.drop(columns="_fund_receipts")

    # race-relative features (all based on the plain poll average)
    # BUGFIX (2026-07-21): field_best used to be one race-wide constant (the runner-up's
    # poll_avg), subtracted from EVERY candidate including the runner-up themself -> every
    # 2nd-place candidate got poll_lead exactly 0.0 (100% of them, verified on 2024 training
    # data), and 3rd-place-and-below got a poll_lead compared against the WRONG opponent
    # (2nd place, not the leader). Only the true front-runner's value was ever correct.
    # Fix: best-OTHER-candidate per row (same pattern already used correctly in
    # margin_model.ipynb's add_margin_target). Feature-value change -> full retrain (rule 1).
    c["field_best"] = c.groupby("race_id")["poll_avg"].transform(best_other)
    c["poll_lead"] = c["poll_avg"] - c["field_best"]
    # same lead computed on the final week only (2026-07-31). poll_lead inherits poll_avg's
    # full-campaign staleness; this is its fresh counterpart. NaN when either side has no
    # final-week polls - never a silent 0, which would read as "tied".
    c["poll_lead_last7"] = c["poll_last7"] - c.groupby("race_id")["poll_last7"].transform(
        best_other)
    # OPPONENT primary strength (2026-08-07, user request). primary_margin above describes
    # how a candidate won their OWN primary; these describe the person across the ballot,
    # and the differential between the two.
    #
    # Why opponent-facing rather than own-facing: the July 2026-07-23 ablation of the
    # own-primary features was a null result on both models (win race-acc -0.0031, margin
    # MAE +0.019), with the recorded reading that "polls already price in primary-contest
    # weakness by general-election time." An ABSOLUTE fact about your own primary is largely
    # redundant with your polling. A RELATIVE one - you cruised while your opponent barely
    # survived - is not obviously priced in the same way, and is what the differential below
    # measures.
    #
    # Leak-safe by construction: a primary always precedes its general election, so nothing
    # here is knowable only after the outcome being predicted.
    # NaN, never 0, when the opponent has no primary-results match: unknown is not
    # "uncontested" (same rule as primary_margin itself - see load_primary_results).
    if "primary_margin" in c.columns:
        # best_other over primary_margin is wrong here (it maximizes); we want the margin
        # belonging to the single strongest-polling OTHER candidate - the real opponent.
        opp_idx = (c.sort_values("poll_avg", ascending=False)
                    .groupby("race_id")["cand_key"].apply(list).to_dict())
        pm_by = {(r, k): v for r, k, v in
                 zip(c["race_id"], c["cand_key"], c["primary_margin"])}

        def _opp_margin(row):
            order = opp_idx.get(row["race_id"], [])
            for k in order:                      # strongest-polling other candidate first
                if k != row["cand_key"]:
                    return pm_by.get((row["race_id"], k), np.nan)
            return np.nan

        c["opp_primary_margin"] = c.apply(_opp_margin, axis=1)
        # Positive = this candidate had the easier primary of the two. This is the feature
        # the request was really about, and the only one of the set that is purely relative.
        c["primary_margin_diff"] = c["primary_margin"] - c["opp_primary_margin"]

    c["poll_share"] = c["poll_avg"] / c.groupby("race_id")["poll_avg"].transform("sum")
    c["n_cands"] = c.groupby("race_id")["cand_key"].transform("count")
    c["race_total_polls"] = c.groupby("race_id")["n_polls"].transform("sum")
    c["frac_polls_over50"] = c["n_polls_over50"] / c["n_polls"]
    c["is_dem"] = (c["party"] == "DEM").astype(int)
    c["is_rep"] = (c["party"] == "REP").astype(int)
    c["is_senate"] = (c["office"] == "Senate").astype(int)
    c["is_gov"] = (c["office"] == "Governor").astype(int)

    dem = c[c["party"] == "DEM"].groupby("race_id")["poll_avg"].max()
    rep = c[c["party"] == "REP"].groupby("race_id")["poll_avg"].max()
    tp = dem - rep
    c["twoparty_margin_cand"] = (c["race_id"].map(tp)
                                 * c["party"].map({"DEM": 1, "REP": -1}).fillna(0))
    c["abs_gap"] = c["race_id"].map(tp.abs())
    c["tossup"] = (c["abs_gap"] < 3).astype(int)
    c["undecided"] = (100 - c.groupby("race_id")["poll_avg"].transform("sum")).clip(lower=0)
    c["gap_x_recency"] = c["poll_lead"] * (1.0 / (1.0 + c["min_days"].clip(lower=0) / 30.0))
    return c

def feature_list(macro_feats, fund=False, primary_results=False, candidate_bios=False):
    """The model's input columns. Everything here is available for future races.
    fund=True appends the FEC fundraising features (pass fec=load_fec() to the builder).
    primary_results=True appends the primary-strength block (pass
    primary_results=load_primary_results() to the builder):
      primary_margin       - how this candidate won their own primary   (2026-07-22)
      opp_primary_margin   - how their general-election OPPONENT won theirs (2026-08-07)
      primary_margin_diff  - the difference; positive = easier primary than the opponent
    primary_uncontested was part of the original block but is EXCLUDED here (user call
    2026-08-07): it had near-zero importance in both models in the July ablation.
    HISTORY: the own-primary-only version was ablated 2026-07-23 and dropped - a null
    result on both models (win race-acc -0.0031, margin MAE +0.019), read at the time as
    "polls already price in primary-contest weakness by general-election time." The
    opponent-facing and differential columns are new and were never part of that test, and
    the training data has since changed (dead-matchup filter, 2026-08-06), so this needs a
    FRESH ablation - do not assume either verdict carries over.
    candidate_bios=True appends bio_office_level (2026-07-23; pass
    candidate_bios=load_candidate_bios() to the builder) - ablate before trusting."""
    return (([] if not fund else list(FUND_FEATS_EXT))
           + ([] if not primary_results else ["primary_margin", "opp_primary_margin",
                                               "primary_margin_diff"])
           + ([] if not candidate_bios else ["bio_office_level"])) + [
        "poll_avg", "poll_last", "poll_last30", "poll_std", "n_polls",
        # poll_last7 / n_polls_last7 / poll_lead_last7 are BUILT in build_candidate_table but
        # deliberately NOT model features here (2026-07-31). They work for the PRIMARY model
        # (feature_list_primary does include them) because primaries are always imminent when
        # predicted - MI's is 4 days out, so the window is populated at serve time. The
        # GENERAL election is a single fixed date: on 2026-07-31 it is 95 days away, so
        # poll_last7 is populated for 39% of TRAINING rows and 0.0% of live 2026 general rows
        # (measured on the feed: 0 of 3370 rows within 7 days; min days_to_elec = 99). Adding
        # it would train splits on a feature that is always-missing in production - the exact
        # train/serve skew that got poll_adj dropped on 2026-07-12.
        # To ship it here, gate it on days-to-election so training only sees it when the race
        # is as close as the race being served, or re-add it in late October when the live
        # window actually fills.
        "n_polls_over50", "frac_polls_over50", "race_total_polls",
        "avg_sample", "min_days",
        "poll_lead", "poll_share", "n_cands",
        "is_dem", "is_rep", "is_senate", "is_gov",
        "prior_margin_cand", "is_incumbent", "is_inc_party_race",
        "twoparty_margin_cand", "abs_gap", "tossup", "undecided", "gap_x_recency",
        # poll_adj (house-effect-adjusted poll avg) DROPPED 2026-07-12: ablation showed it
        # added no out-of-sample value (win AUC/acc unchanged, margin MAE slightly BETTER
        # without it - it was ~redundant with poll_avg) AND it had a train/serve risk (the
        # pollster house-effect table matches only ~67% of 2026-feed pollsters, so it's
        # computed on a different basis for future polls). The `poll_adj` column is still
        # built in build_candidate_table (harmless) but is no longer a model feature.
        "natl_env_cand", "bias_prior_cand", "poll_momentum",
        "n_lead_changes", "lead_changed",
        "avg_margin_over_time", "margin_volatility", "min_margin", "margin_trend",
        "is_president_party",
    ] + list(macro_feats)
