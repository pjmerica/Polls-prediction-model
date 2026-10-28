# -*- coding: utf-8 -*-
"""Scrape 2026 PRIMARY RESULTS from Wikipedia race pages -> data/primary_results_2026.csv.

Ground truth for the primary model: the actual primary winner per (race, party) —
directly, not via the post-primary-general-polling proxy (which is blind exactly where
the primary IS the contest, e.g. deep-blue NYC House seats).

Parsing: results wikitables (headers contain Candidate + Votes/%) under primary-stage
section context. Per (race_id, party) the LAST results table in document order wins —
that's the runoff where one exists (verified: TX-Sen-REP round 1 = Cornyn 42.0/Paxton
40.5, runoff = Paxton 63.8 -> nominee Paxton). Tables under non-primary context (e.g.
prior-cycle results in Background sections) are excluded by the stage tracker; mixed-
party tables are skipped as a second guard.

Doubles as the LABELING machinery for adding cycle 2026 to training after November.

    py -X utf8 fetch_primary_results_2026.py            # 2026 (pages from predictions)
    py -X utf8 fetch_primary_results_2026.py --hist     # 2018-2024 (pages from the
                                                        # historical polls scrape) ->
                                                        # data/primary_results_hist.csv
HIST NOTE: general-election candidate lists are a flawed label source - a primary winner
who later drops out (Platner, ME-Sen 2026) appears NOWHERE in them and the replacement
gets mislabeled as the winner. Actual results are the truth; build_primary_dataset
prefers them and reports every disagreement with the nominee-join.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)
import paths as _paths   # module handle: `out` is a very common local
                         # variable name in this repo, so never import it bare

import os
import re
import sys
import time

import pandas as pd

HERE = ROOT   # repo root (paths.py) - this file lives in a subfolder
sys.path.insert(0, AGG)
from bs4 import BeautifulSoup  # noqa: E402
from scrapers.wikipedia_polls import fetch_page, infer_section_context, STATES  # noqa: E402

import features as F  # noqa: E402



URL_SEN = "https://en.wikipedia.org/wiki/{year}_United_States_Senate_election_in_{state}"
URL_GOV = "https://en.wikipedia.org/wiki/{year}_{state}_gubernatorial_election"
URL_HOUSE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of_Representatives"
             "_elections_in_{state}")

PARTY_ROW = {"republican": "REP", "democratic": "DEM"}
AT_LARGE_2026 = {"AK", "DE", "ND", "SD", "VT", "WY"}

# State affiliates whose Wikipedia party label is NOT the bare national name. Exact-match on
# PARTY_ROW silently skipped these, and a skipped party table means the whole race is absent
# from primary_results_2026.csv - which makes drop_primary_losers() a SILENT NO-OP for it.
#
# Found 2026-08-12: Minnesota labels its rows "Democratic (DFL)" (Democratic-Farmer-Labor),
# so the MN-Sen DFL primary never parsed. Flanagan beat Craig 59.0-39.4 on 2026-08-11, and
# the general-election model kept scoring the DEFEATED Craig at 47.6% the next day. North
# Dakota's "Democratic (NPL)" (Nonpartisan League) has the same shape.
_PARTY_PREFIX = (
    ("republican", "REP"),
    ("democratic", "DEM"),   # covers "Democratic (DFL)", "Democratic (NPL)", "Democratic"
    ("democratic-farmer-labor", "DEM"),
    ("democratic–farmer–labor", "DEM"),   # en-dash spelling
    ("dfl", "DEM"),
)

def party_from_label(cell):
    """'Democratic (DFL)' -> 'DEM'. Prefix match, so state affiliates resolve to their
    national party instead of being dropped. Returns None for anything unrecognised (a
    third party, or a non-party cell), which callers treat as 'not a primary results row'."""
    s = str(cell).strip().lower()
    if s in PARTY_ROW:
        return PARTY_ROW[s]
    for pre, code in _PARTY_PREFIX:
        if s.startswith(pre):
            return code
    return None

def parse_results_tables(html, base_race_id, house=False, at_large=False):
    """[{race_id(+party), party, candidate, votes, pct, table_seq}] for every
    primary-context results table. race_id gets '-<district>' inserted for House.

    at_large=True (added 2026-07-22, historical House scrape): the page has ONE district
    for the whole state (AK/DE/MT/ND/SD/VT/WY, cycle-dependent - some regained/lost a seat
    across redistricting, e.g. SD 2-seat pre-1982, MT 2-seat 2020+) and carries no
    "District N" heading to detect, so seed district=1 up front instead of requiring one to
    appear (matches res_house.csv's own at-large convention, verified: office_seat_name=
    'District 1'). Without this, at-large-year House pages parsed zero rows (the
    `house and district is None` guard below dropped everything)."""
    soup = BeautifulSoup(html, "html.parser")
    rows, seq = [], 0
    stage, district, other_office = "general", (1 if at_large else None), False
    other_level = 9
    OTHER_OFFICE_RX = re.compile(
        r"lieutenant|attorney general|secretary of state|treasurer|auditor|"
        r"comptroller|agriculture|superintendent|land commissioner|railroad",
        re.I)
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name != "table":
            text = el.get_text(" ", strip=True)
            level = int(el.name[1])
            if house:
                m = re.search(r"District\s+(\d+)", text)
                if m:
                    district = int(m.group(1)); stage = "general"
            s, _p = infer_section_context(text)
            if s:
                stage = s
                # a new primary/general section resets the guard - but only one at the guard's
                # own level or above: NM 2026 nests "Primary election" (h4) UNDER "Lieutenant
                # governor" (h3), which reset it and crowned the LG winner Maggie Toulouse
                # Oliver over Deb Haaland (fixed 2026-10-24)
                if level <= other_level:
                    other_office, other_level = False, 9
            # DOWN-TICKET GUARD: gubernatorial pages nest Lieutenant-Governor (etc.)
            # primary results INSIDE the party-primary sections - without this the
            # last-table heuristic crowned running mates (Husted over DeWine, Driscoll
            # over Healey, Fetterman as 2018 PA "governor" winner...). A subsection
            # naming another office poisons tables until the next stage-bearing heading.
            if OTHER_OFFICE_RX.search(text):
                other_office, other_level = True, min(other_level, level)
            continue
        if other_office or stage not in ("primary", "primary runoff"):
            continue
        if "wikitable" not in (el.get("class", []) or []):
            continue
        heads = " ".join(th.get_text(" ", strip=True) for th in el.find_all("th")).lower()
        if "candidate" not in heads or ("votes" not in heads and "%" not in heads):
            continue
        seq += 1
        # RANKED-CHOICE tables (Maine) carry every round in one row - votes, %, transfer, votes,
        # % ... - so the first vote column is only round 1. The nominee is the final-round
        # leader (2026-10-24: Nirav Shah and Joe Baldacci led round 1, Hannah Pingree and
        # Matthew Dunlap won ME-Gov and ME-2 after transfers)
        rcv = "transfer" in heads or "round 2" in heads
        parties, trows = set(), []
        for tr in el.find_all("tr"):
            els = tr.find_all(["td", "th"])
            cells = [c.get_text(" ", strip=True) for c in els]
            if len(cells) < 4:
                continue
            party = party_from_label(cells[1])   # prefix-aware: handles 'Democratic (DFL)'
            if party is None:
                continue
            # TICKET cells: states electing Governor+LG jointly list 'Richard Cordray and
            # Betty Sutton' in one cell -> naive parsing crowned 'sutton r'. Names are
            # individually wikilinked, so the FIRST link is the head of the ticket;
            # fallback splits on ticket separators.
            a = els[2].find("a")
            cand = a.get_text(" ", strip=True) if a else re.split(
                r"\s+and\s+|/", cells[2])[0]
            # strip ALL parenthetical annotations: (incumbent), (write-in), (withdrawn)...
            cand = re.sub(r"\s*\([^)]*\)\s*", " ", cand).strip()
            votes = re.sub(r"[^\d]", "", cells[3])
            pct = re.sub(r"[^\d.]", "", cells[4]) if len(cells) > 4 else ""
            if not cand or not votes:
                continue
            # (round, votes) of the last round this candidate appears in: vote cells sit every
            # third cell from the first (votes, %, transfer); later rounds are blank once out
            final = (1, int(votes))
            if rcv:
                for k, cell in enumerate(cells[3::3], start=1):
                    if re.fullmatch(r"[\d,]+", cell.strip()):
                        final = (k, int(cell.replace(",", "")))
            parties.add(party)
            trows.append((party, cand, int(votes), float(pct) if pct else None, final))
        if not trows or len(parties) != 1:
            continue          # mixed-party table = not a primary results table
        party = parties.pop()
        rid = base_race_id + (f"-{district}" if house and district else "") + "_" + party
        if house and district is None:
            continue
        last_round = max(f[0] for *_x, f in trows)
        for p_, cand, votes, pct, (rnd, fv) in trows:
            # final_votes: the last round's count for the candidates still in it (round 1 =
            # the only round for an ordinary primary), 0 for those eliminated earlier
            rows.append(dict(race_id=rid, party=party, candidate=cand,
                             votes=votes, pct=pct, table_seq=seq,
                             final_votes=fv if rnd == last_round else 0))
    return rows

def page_targets(hist, deep=False):
    """[(year, st, office)] - 2026 from the predictions CSV, hist from the polls scrape.

    deep=True (2026-08-07): every Senate/Governor race the GENERAL model actually trains on,
    back to 1998, taken from polls_long_with_results.csv instead of the primary-polls file.
    The --hist mode derives its targets from primary_polls_wikipedia.csv, which only ever
    covered 2018+, so Senate and Governor primary results were 0.0% populated for EVERY
    pre-2018 cycle while House (scraped by a different script) went back to 1998. That gap
    is why primary_margin covered only 24.8% of the candidate table. Results do not depend
    on the polls file at all - only on the race existing - so this mode asks the training
    data directly.
    """
    if deep:
        import features as _F
        d = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
        d = d[d["office"].isin(["Senate", "Governor"])]
        d = d[pd.to_numeric(d["year"], errors="coerce").between(1998, 2024)]
        out = {(int(y), st, of) for y, st, of in
               zip(d["year"], d["state"], d["office"]) if isinstance(st, str) and len(st) == 2}
        return sorted(out)
    if not hist:
        preds = pd.read_csv(_paths.out("primary_predictions_2026.csv"))
        need = set()
        for rid in preds["race_id"].unique():
            _y, st, of_di, _pty = rid.split("_")
            need.add((2026, st, of_di.split("-")[0]))
        return sorted(need)
    polls = pd.read_csv(os.path.join(HERE, "data", "primary_polls_wikipedia.csv"),
                        low_memory=False)
    out = set()
    for page in polls.loc[polls["stage"] == "primary", "src_page"].unique():
        y, off, st = page.split("-")
        out.add((int(y), st, {"SEN": "Senate", "GOV": "Governor"}[off]))
    return sorted(out)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", action="store_true",
                    help="scrape 2018-2024 results for the historical training pages")
    ap.add_argument("--deep", action="store_true",
                    help="scrape ALL Senate/Governor primaries 1998-2024 from the training "
                         "races (fills the pre-2018 gap --hist never covered)")
    args = ap.parse_args()
    # --deep writes its own file so a partial/failed deep run can never damage the two
    # existing committed archives. build_primary_dataset / load_primary_results read all
    # three, and a later (year,state,office) simply supersedes an earlier one on merge.
    out_path = os.path.join(HERE, "data",
                            "primary_results_deep_hist.csv" if args.deep
                            else "primary_results_hist.csv" if args.hist
                            else "primary_results_2026.csv")

    pages = page_targets(args.hist, deep=args.deep)
    print(f"{len(pages)} pages to scrape")

    allrows = []
    for year, st, office in pages:
        state = STATES.get(st)
        if not state:
            continue
        s = state.replace(" ", "_")
        if office == "Senate":
            url, base, house = (URL_SEN.format(year=year, state=s),
                                f"{year}_{st}_Senate", False)
        elif office == "Governor":
            url, base, house = (URL_GOV.format(year=year, state=s),
                                f"{year}_{st}_Governor", False)
        else:
            url, base, house = (URL_HOUSE.format(year=year, state=s),
                                f"{year}_{st}_House", True)
        # one-seat states have an "...election in <State>" page (singular) with no district
        # headings (2026-10-24: SD/VT/WY House failed to fetch)
        at_large = house and year == 2026 and st in AT_LARGE_2026
        if at_large:
            url = URL_HOUSE.format(year=year, state=s).replace("_elections_in_", "_election_in_")
        html = fetch_page(url)
        if html is None:
            print(f"  {st} {office}: page fetch failed")
            continue
        rows = parse_results_tables(html, base, house=house, at_large=at_large)
        if rows:
            print(f"  {st} {office}: {len(rows)} result rows, "
                  f"{len(set(r['race_id'] for r in rows))} party-races")
        allrows.extend(rows)
        time.sleep(0.8)

    df = pd.DataFrame(allrows)
    if not len(df):
        raise SystemExit("no results parsed - page structure changed?")
    # winner = top votes in the LAST table per (race_id): runoff supersedes round 1; in a
    # ranked-choice table, the top FINAL-round count
    last = df.groupby("race_id")["table_seq"].transform("max")
    fin = df[df["table_seq"] == last].copy()
    fin["is_winner"] = fin.groupby("race_id")["final_votes"].transform("max") == fin["final_votes"]
    fin["cand_key"] = fin["candidate"].map(F.norm_name)

    # ---- drop PARTY CONVENTION / ENDORSEMENT tables masquerading as primary results ----
    # Wikipedia puts a state party's convention or endorsement vote in the same results
    # section as the primary, so the "last table" heuristic above happily grabs it. Found
    # 2026-08-05: CT-Gov-DEM showed Ned Lamont winning with 1,969 total votes and MN-Gov-REP
    # 17,187 - against statewide primary medians of ~375k (Senate) and ~388k (Governor).
    # Worse, all four affected races have election dates in the FUTURE (CT/MN vote Aug 11,
    # MA Sep 1), so the file was asserting outcomes for primaries that had not happened.
    #
    # A statewide primary that small is not a primary. House districts legitimately run
    # small (median ~52k, min ~181 for an uncontested nomination), so the threshold applies
    # to STATEWIDE races only, where the floor is unambiguous.
    STATEWIDE_MIN_VOTES = 25_000
    totals = fin.groupby("race_id")["votes"].sum()
    statewide = [r for r in totals.index
                 if ("_Senate" in str(r) or "_Governor" in str(r))]
    suspect = [r for r in statewide if totals[r] < STATEWIDE_MIN_VOTES]
    if suspect:
        print(f"\nDROPPED {len(suspect)} statewide race(s) whose total vote is below "
              f"{STATEWIDE_MIN_VOTES:,} - convention/endorsement tables, not primaries:")
        for r in suspect:
            print(f"   {r}: {int(totals[r]):,} votes")
        fin = fin[~fin["race_id"].isin(suspect)].copy()

    # ---- DATE GUARD (CONCERNS #37, 2026-10-27): no result for a primary that has not happened ----
    # The vote floor above catches convention tables by size; this catches them, and a pre-filled
    # "results" table for an upcoming primary, by date. Dates come from the calendar the primary
    # model uses (predict_primary.primary_dates: per race, then per state, then the manual file).
    if not args.hist and not args.deep:
        import datetime as _dt
        sys.path.insert(0, os.path.join(HERE, "src"))
        from predict_primary import primary_dates
        per_race, per_state = primary_dates(2026)
        today = pd.Timestamp(_dt.date.today())
        parts = fin["race_id"].str.extract(r"^\d{4}_([A-Z]{2})_(Senate|Governor|House)")
        when = [per_race.get((st, off), per_state.get(st)) for st, off in zip(parts[0], parts[1])]
        fin["_date"] = pd.to_datetime(pd.Series(when, index=fin.index))
        future = sorted(set(fin.loc[fin["_date"] > today, "race_id"]))
        undated = sorted(set(fin.loc[fin["_date"].isna(), "race_id"]))
        if future:
            print(f"\nDROPPED {len(future)} party-race(s) whose primary is still to come: {future[:12]}")
            fin = fin[~fin["race_id"].isin(future)].copy()
        if undated:
            print(f"WARNING: no primary date for {len(undated)} party-race(s), kept unchecked: {undated[:12]}")
        fin = fin.drop(columns="_date")

    # hand-checked ground truth: hard-fail on regression (these are the exact races the
    # Lieutenant-Governor table bug corrupted, plus a runoff case)
    KNOWN = {"2018_OH_Governor_DEM": "cordray r", "2018_OH_Governor_REP": "dewine m",
             "2018_HI_Governor_DEM": "ige d", "2022_MA_Governor_DEM": "healey m",
             "2026_TX_Senate_REP": "paxton k",
             # ranked-choice winners and a nested Lieutenant-Governor section (2026-10-24)
             "2026_ME_Governor_DEM": "pingree h", "2026_ME_House-2_DEM": "dunlap m",
             "2026_ME_Governor_REP": "charles r", "2026_NM_Governor_DEM": "haaland d"}
    wmap = fin[fin["is_winner"]].groupby("race_id")["cand_key"].agg(set).to_dict()
    for rid, want in KNOWN.items():
        if rid in wmap:
            assert want in wmap[rid], f"{rid}: parsed winner {wmap[rid]} != known {want}"

    # the 2026 targets follow the primary model's CURRENT race list, so a re-run must not erase
    # results it no longer scrapes (2026-10-24: a re-run dropped AR/MS/OH-Senate, MD-Governor and
    # the IA/NV House primaries, which would have turned drop_primary_losers off for them).
    # Re-scraped races replace their old rows in place; the others keep theirs.
    if not args.hist and not args.deep and os.path.exists(out_path):
        old = pd.read_csv(out_path)
        fresh = set(fin["race_id"])
        parts = [fin[fin["race_id"] == rid] if rid in fresh else g
                 for rid, g in old.groupby("race_id", sort=False)]
        parts.append(fin[~fin["race_id"].isin(set(old["race_id"]))])
        kept = old["race_id"].nunique() - len(fresh & set(old["race_id"]))
        fin = pd.concat(parts, ignore_index=True)
        print(f"kept {kept} party-races from the previous file that this run did not re-scrape")

    # final_votes only decides is_winner; the file keeps its columns
    fin.drop(columns=["final_votes"], errors="ignore").to_csv(out_path, index=False)
    w = fin[fin["is_winner"]]
    print(f"\nsaved -> {out_path}: {fin['race_id'].nunique()} party-races with results, "
          f"{len(w)} winners (known-winner validation passed)")

if __name__ == "__main__":
    main()
