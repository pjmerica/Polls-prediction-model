# -*- coding: utf-8 -*-
"""Build the ONE authoritative, leak-free office-level table -> data/candidate_bios.csv.

Replaces the ad-hoc combine_candidate_bios.py merge with a single tidy master keyed by
(year, office, state, district, party, cand_key), one office_level each, computed as the
highest office the candidate held STRICTLY BEFORE that election year (user requirement
2026-07-25 - no look-ahead: a first-time 2018 candidate reads 0 even if they later became
a Senator).

Two source kinds, both contributing an as-of-year level:
  WIKIPEDIA (data/candidate_bios_{senate,governor,house}.csv): already contemporaneous -
    each row was scraped from that YEAR's own race page, which describes the candidate as
    they were then (verified: Abigail Spanberger reads 0 on her 2018 page, 4 on 2020+). So
    its office_level is used as-is. Preferred source on any overlap.
  BALLOTPEDIA (data/candidate_bios_ballotpedia.csv): person-level, but carries per-office
    TENURE DATES (offices_json, e.g. [["U.S. House ...",2019,2025],["Governor ...",2026,
    null]]). We compute the as-of-year level = max office-level among offices that STARTED
    before `year`. Gap-filler only (Wikipedia wins overlaps). This is what makes Ballotpedia
    rows leak-free and time-varying instead of a frozen peak.

    py -X utf8 build_office_level_table.py
Writes data/candidate_bios.csv (every consumer reads this). Rebuilds from scratch each run.
"""
import json
import os
import re

import pandas as pd

import features as F
from fetch_candidate_bios_ballotpedia import classify_ballotpedia

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WIKI_SOURCES = ["candidate_bios_senate.csv", "candidate_bios_governor.csv",
               "candidate_bios_house.csv"]
OUT = os.path.join(DATA, "candidate_bios.csv")

def _bp_asof_level(offices, year):
    """Highest office-level among offices whose tenure STARTED strictly before `year`.
    offices = [(phrase, start, end_or_None)]. Returns 0 if none qualify (held no office
    before that race), or None if offices is empty/unknown."""
    if not offices:
        return None
    levels = [classify_ballotpedia(phrase) for phrase, start, _ in offices
              if start is not None and start < year]
    return max(levels) if levels else 0

def main():
    # ---- Wikipedia: already per-year contemporaneous, used as-is ----
    wiki_frames = []
    for fn in WIKI_SOURCES:
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            df = pd.read_csv(p, low_memory=False)
            df["src"] = "wikipedia"
            wiki_frames.append(df)
            print(f"{fn}: {len(df)} rows")
    if not wiki_frames:
        raise SystemExit("no Wikipedia source files - nothing to build")
    wiki = pd.concat(wiki_frames, ignore_index=True)
    wiki = wiki.drop_duplicates(
        subset=["year", "office", "state", "district", "party", "cand_key"], keep="first")
    KEY = ["year", "office", "state", "district", "party", "cand_key"]
    wiki["district"] = wiki["district"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    wiki_keys = set(map(tuple, wiki[KEY].astype(str).values))
    # Wikipedia level per key - needed so a Ballotpedia row with a REAL level can override a
    # Wikipedia row that reads 0 ONLY because it came from a results TABLE (blank descriptor,
    # no prior-office info). Found 2026-07-27: Schumer 1998 / Carper 2000 had level-0 Wikipedia
    # table rows outranking their correct Ballotpedia levels. Wikipedia still wins whenever it
    # has a real (>0) level; it only yields when its own value is an uninformative 0.
    wiki_level = {tuple(str(x) for x in k): lvl for k, lvl in
                  zip(wiki[KEY].values, wiki["office_level"].values)}

    # ---- Ballotpedia: expand person-rows to the uncovered per-race rows, as-of-year level ----
    bp_path = os.path.join(DATA, "candidate_bios_ballotpedia.csv")
    # Map BP hits against the FULL uncovered roster (office_level_backfill_targets.csv, written
    # by measure_office_coverage.py), NOT the transient uncovered_candidates.csv scraper-scope
    # file. Fixed 2026-07-28: uncovered_candidates.csv gets rewritten each scrape round to a
    # narrow subset (e.g. winners-only), so mapping against it dropped BP hits for anyone not
    # in the current round's scope (161 -> 100 BP rows between rounds). The full roster covers
    # every uncovered race a BP-resolved person appears in. Falls back to the scope file if the
    # roster is absent.
    bp_rows = []
    if os.path.exists(bp_path):
        bp = pd.read_csv(bp_path, low_memory=False)
        bp = bp[bp["office_level"].notna()]
        off_map = {}
        for r in bp.itertuples():
            try:
                off_map[(r.candidate, r.state)] = json.loads(getattr(r, "offices_json", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                off_map[(r.candidate, r.state)] = []
        prior_map = {(r.candidate, r.state): getattr(r, "bio_prior_candidacy", 0) for r in bp.itertuples()}
        # MANUAL hardcode (Stage 3, 2026-07-27): data/candidate_bios_manual.csv carries the same
        # person-level offices_json (tenure dates) for winners that neither Wikipedia nor
        # Ballotpedia covered - researched by hand with source_note. Merged into off_map so the
        # SAME leak-free as-of-year computation applies. Manual OVERRIDES Ballotpedia on overlap
        # (it's the hand-verified source). An empty [] here means "verified: no prior office"
        # (level 0), which is real data - distinct from a missing key (unknown).
        manual_path = os.path.join(DATA, "candidate_bios_manual.csv")
        manual_keys = set()
        if os.path.exists(manual_path):
            man = pd.read_csv(manual_path, low_memory=False)
            for r in man.itertuples():
                try:
                    off_map[(r.candidate, r.state)] = json.loads(getattr(r, "offices_json", "[]") or "[]")
                    manual_keys.add((r.candidate, r.state))
                except (json.JSONDecodeError, TypeError):
                    pass
            print(f"candidate_bios_manual.csv: {len(manual_keys)} hand-coded people merged")
        # GROUND TRUTH for where a BP hit maps: every candidate-race in the poll feed, keyed by
        # (candidate,state). NOT a transient "uncovered" list - fixed 2026-07-28 after two bugs
        # where per-round scope files (uncovered_candidates.csv rewritten winners-only) and the
        # regenerated roster (excludes now-covered people) each dropped BP hits for anyone not
        # in that particular file. The poll feed is stable and contains every race a person ran,
        # so a BP-resolved person's as-of-year level lands on ALL their races. `won` carried for
        # the mapping only (name matched below on candidate+state, same as before).
        # The roster must span BOTH feeds (2026-08-01). It used to read only the GENERAL poll
        # file, so a hand-coded or Ballotpedia-resolved person who never appears in a general
        # poll got NO rows at all - silently discarding the entry. That hid every primary-only
        # candidate (Eric Barlow, Andy Gipson, Shad White, Nirav Shah, Lynn Fitch, ...) and is
        # why hand-coding them appeared to do nothing. The primary long file carries the same
        # columns after prepare_polls, so the two concatenate directly.
        import features as _F
        _COLS = ["year", "office", "state", "district", "cand_key", "party_std", "candidate"]
        _frames = []
        for _p in ("polls_long_with_results.csv", os.path.join("data", "primary_polls_long.csv")):
            _fp = os.path.join(HERE, _p)
            if os.path.exists(_fp):
                _f = _F.prepare_polls(pd.read_csv(_fp, low_memory=False))
                _frames.append(_f[_COLS])
        # ...and the LIVE primary feed, which is the only place a current-cycle primary-only
        # candidate appears (the two training files stop at the last completed cycle). Without
        # this, hand-coding a 2026 primary candidate silently produced no rows: Eric Barlow,
        # Andy Gipson, Shad White, Megan Degenfelder are in no historical poll file.
        try:
            from predict import DEFAULT_POLLS as _DP
            from predict_primary import load_primary_feed as _lpf
            _live = _lpf(_DP, 2026)
            _frames.append(_live[_COLS])
        except Exception as _e:      # feed absent (fresh clone / CI without polling-agg)
            print(f"  live 2026 primary feed unavailable for the roster ({_e.__class__.__name__})"
                  " - current-cycle primary-only hand-codes will not land")
        _d = pd.concat(_frames, ignore_index=True)
        unc = _d.drop_duplicates(subset=["year", "office", "state", "district", "cand_key",
                                         "party_std"]).copy()
        unc = unc.rename(columns={"candidate": "candidate", "party_std": "party"})
        unc["district"] = unc["district"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        for r in unc.itertuples():
            key = (r.candidate, r.state)
            if key not in off_map:
                continue
            rowkey = tuple(str(x) for x in (r.year, r.office, r.state, r.district, r.party, r.cand_key))
            is_manual = key in manual_keys
            lvl = _bp_asof_level(off_map[key], int(r.year))
            if lvl is None:
                # empty offices: for a MANUAL entry this means "verified: no prior office" ->
                # a real level 0. For Ballotpedia it means unknown -> skip (leak-safe).
                if is_manual:
                    lvl = 0
                else:
                    continue
            if rowkey in wiki_keys:
                # Wikipedia covers it. It wins UNLESS its level is an uninformative 0 (a
                # results-table row with no descriptor) and this source has a real higher
                # level - then let this row through to override in the dedup below. A manual
                # entry overrides a wiki table-zero at level 0 too (hand-verified 0 beats an
                # unknown table 0 - same key, dedup keeps either; harmless).
                if not (wiki_level.get(rowkey, 0) == 0 and (lvl > 0 or is_manual)):
                    continue
            bp_rows.append(dict(
                year=r.year, office=r.office, state=r.state, district=r.district,
                party=r.party, name=r.candidate, cand_key=r.cand_key,
                office_level=int(lvl), bio_in_office=0,
                bio_prior_candidacy=int(prior_map.get(key, 0)),
                src=("manual" if is_manual else "ballotpedia")))
        print(f"candidate_bios_ballotpedia.csv: {len(bp)} profiles -> {len(bp_rows)} "
              f"leak-free as-of-year rows (gap-filling; Wikipedia preferred)")

    combined = pd.concat([wiki, pd.DataFrame(bp_rows)], ignore_index=True) if bp_rows else wiki

    # ---- WIKIPEDIA SELF-CROSS-REFERENCE (2026-07-28, user-requested, leak-free) ----
    # A person's office history is often recorded on SOME of their Wikipedia race pages but
    # not others: Dino Rossi reads "state senator" (2) in 2004/2010/2016 but 0 on his blank
    # 2008/2018 table rows. We propagate a person's own informative Wikipedia levels FORWARD
    # ONLY, which is unambiguously leak-free:
    #   an office HELD (level L>0) as of Wikipedia bio-year Y applies to that person's races in
    #   years >= Y (offices persist forward; you don't un-hold an office).
    # BACKWARD propagation was tried and REMOVED (2026-07-28): a "former US Rep" descriptor in
    # year Y does NOT reveal how far back the office extends - it gave Denny Heck's 2010 race
    # level 4 from a 2024 "former U.S. representative" row, but he wasn't a US Rep until 2013 (a
    # real leak). Forward-only never assumes prior tenure (Rossi 2008->2 still works: forward
    # from his 2004 state-senator row). Only Wikipedia rows feed the reference (contemporaneous,
    # trusted); emitted as src="wiki_xref" so it's auditable and ranks under a direct wiki row.
    w = combined[combined["src"] == "wikipedia"].copy()
    fwd = {}            # (name,state) -> [(year, level)] of offices held as-of that bio-year
    for r in w.itertuples():
        if r.office_level <= 0:
            continue
        fwd.setdefault((r.name, r.state), []).append((int(r.year), int(r.office_level)))
    # existing (key) coverage so we only FILL gaps, never overwrite a real row
    covered_lvl = {tuple(str(x) for x in (r.year, r.office, r.state, r.district, r.party, r.cand_key)):
                   r.office_level for r in combined.itertuples()}
    xref_rows = []
    if os.path.exists(os.path.join(HERE, "polls_long_with_results.csv")):
        allraces = F.prepare_polls(pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"),
                                               low_memory=False))
        allraces = allraces.drop_duplicates(subset=["year", "office", "state", "district",
                                                    "cand_key", "party_std"]).copy()
        allraces = allraces.rename(columns={"party_std": "party"})
        allraces["district"] = allraces["district"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        for r in allraces.itertuples():
            k = (r.candidate, r.state)
            best = 0
            for (y, l) in fwd.get(k, []):
                if y <= r.year:            # office held by bio-year Y, race is in year >= Y
                    best = max(best, l)
            if best <= 0:
                continue
            rk = tuple(str(x) for x in (r.year, r.office, r.state, r.district, r.party, r.cand_key))
            if covered_lvl.get(rk, 0) >= best:   # only fill if it beats what's there (incl table-0)
                continue
            xref_rows.append(dict(
                year=r.year, office=r.office, state=r.state, district=r.district, party=r.party,
                name=r.candidate, cand_key=r.cand_key, office_level=int(best),
                bio_in_office=0, bio_prior_candidacy=0, src="wiki_xref"))
        combined = pd.concat([combined, pd.DataFrame(xref_rows)], ignore_index=True) if xref_rows else combined
        print(f"wikipedia self-cross-reference: {len(xref_rows)} gap rows "
              f"(forward office-persistence, leak-free)")

    # final dedup: on a key collision keep the HIGHER office_level. This lets a Ballotpedia
    # real-level row override a Wikipedia table-zero row (the only case a BP row is emitted for
    # an already-covered key - see the override gate above), while still keeping Wikipedia's
    # value in every normal case (equal or higher wiki level -> a stable sort keeps wiki, which
    # is concatenated first). Ties (both same level) keep Wikipedia via the stable sort.
    # On a key collision keep the higher office_level; on EQUAL level prefer the more-
    # authoritative source (manual > ballotpedia > wikipedia). The src tiebreak matters for
    # VERIFIED ZEROS: a hand-coded manual level-0 (candidate truly held no prior office) must
    # win over a Wikipedia table-zero level-0 (unknown) at the same key, so the surviving row
    # carries src="manual" and measure_office_coverage.py counts it as covered, not table_zero
    # (fixed 2026-07-27: Renzi/Ellmers-type verified first-timers were being read as uncovered).
    # src authority on EQUAL level: hand-verified manual > ballotpedia > direct wikipedia >
    # wiki self-cross-reference (a derived fill, ranks last so a direct row always wins its key).
    _src_rank = {"manual": 0, "ballotpedia": 1, "wikipedia": 2, "wiki_xref": 3}
    combined["_sr"] = combined["src"].map(_src_rank).fillna(4)
    combined = (combined.sort_values(["office_level", "_sr"], ascending=[False, True],
                                     kind="stable")
                        .drop_duplicates(subset=KEY, keep="first")
                        .drop(columns="_sr"))
    combined.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(combined)} rows "
          f"({(combined['src']=='wikipedia').sum()} wiki, {(combined['src']=='ballotpedia').sum()} ballotpedia)")
    print("office_level distribution:", combined["office_level"].value_counts().sort_index().to_dict())

if __name__ == "__main__":
    main()
