# -*- coding: utf-8 -*-
"""Scrape candidate BIO descriptors from Wikipedia race pages' primary-section candidate
lists -> office-experience features ("smaller offices" the results archives can't see:
state legislators, mayors, AGs, county officials).

Bullet format on race pages:  "Mallory McMorrow , state senator from the 8th district
(2019-present)" / "Abdul El-Sayed , former Wayne County health director (2023-2025) and
candidate for governor in 2018".

office_level: 4 federal (US sen/rep/cabinet) > 3 statewide (gov/LG/AG/SoS/treasurer) >
2 state legislature > 1 local (mayor/county/city/sheriff/judge) > 0 none-detected.
Plus: bio_in_office (descriptor says 'present'), bio_prior_candidacy ('candidate for' /
'nominee for' - catches runs our results files never tracked, e.g. El-Sayed's 2018
gubernatorial primary bid).

Output: data/candidate_bios.csv (committed). Known honesty caveat (documented in
METHODOLOGY): historical pages are read as they exist TODAY; post-election edits can
close date ranges ('2019-2023'), leaking small amounts of later info into training-era
bios. Office LEVEL - the feature - rarely changes from such edits.

    py -X utf8 fetch_candidate_bios.py
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



# SEPARATE per-office output files (changed 2026-07-24, user instruction after a real
# incident: fetch_house_candidate_bios_hist.py's own resume logic read the OLD
# candidate_bios.csv, but this script's main() had already overwritten it with
# Senate/Governor-only data (no House rows) in between two runs, silently discarding
# ~9,500 already-scraped House rows. Root cause: multiple scripts writing to ONE shared
# file, each with its own "resume from what's there" logic that assumes it's the only
# writer. Fix: each office gets its OWN file, written only by the script(s) that scrape
# that office; pipeline/build/build_office_level_table.py assembles them into candidate_bios.csv
# (NOT combine_candidate_bios.py - that is DEPRECATED as of 2026-07-25 and writes the same
# path with a leaky frozen office_level; running it would silently overwrite the good table)
# (the
# file every consumer - features.py, features_primary.py, check_officeholder.py - reads)
# as an explicit, separate, manual step. No script ever silently overwrites another's
# output again.
OUT_SENATE = os.path.join(HERE, "data", "candidate_bios_senate.csv")
OUT_GOVERNOR = os.path.join(HERE, "data", "candidate_bios_governor.csv")
# (a back-compat `OUT` alias briefly lived here pointing at OUT_SENATE - removed 2026-07-24:
# nothing imports it, and an alias named like the old COMBINED file but pointing at ONE
# office's file is exactly the silent-wrong-file trap this refactor exists to eliminate)
URL_SEN = "https://en.wikipedia.org/wiki/{year}_United_States_Senate_election_in_{state}"
URL_GOV = "https://en.wikipedia.org/wiki/{year}_{state}_gubernatorial_election"
URL_HOUSE = ("https://en.wikipedia.org/wiki/{year}_United_States_House_of_Representatives"
             "_elections_in_{state}")

LEVELS = [
    # US-form variants: 'U.S.', 'US', 'U.S' (missing periods happen: 'U.S representative')
    (4, r"(u\.?s\.?|united states) (senator|representative|secretary)|"
        r"member of (the )?(u\.?s\.?|united states) house|member of congress|"
        # institution phrasing "U.S. House <state>" / "U.S. Senate <state>" (Ballotpedia-style,
        # also appears in some Wikipedia rows) - added 2026-07-29 ("U.S. House Washington" read 0)
        r"(u\.?s\.?|united states)\s+(house|senate)\b|"
        r"white house|congress(wo)?man|"
        # leadership titles imply U.S. House/Senate membership on their own (found
        # 2026-07-23: 'former Majority Leader of the United States House of
        # Representatives' (Eric Cantor) didn't match 'member of ... house' - the phrase
        # doesn't contain 'member of')
        r"(majority|minority) (leader|whip) of the .{0,10}(u\.?s\.?|united states) house|"
        r"speaker of the (u\.?s\.?|united states) house"),
    (3, r"\bgovernor\b|lieutenant governor|attorney general|secretary of state|"
        r"state treasurer|state auditor|state comptroller|commissioner of|"
        r"superintendent of public"),
    (2, r"state senator|state representative|state assembly|state house|"
        r"speaker of the .{0,30}house|state senate|general assembly|state delegate|"
        r"house of delegates|"
        # STATE-NAME legislature phrasing (added 2026-07-29): "Minnesota Senate", "Oregon
        # State Senator", "Ohio House of Representatives" etc. - Wikipedia often names the
        # state, not the literal word "state". Same fix already in classify_ballotpedia; found
        # here via Roger Moe ("Minnesota Senate Majority Leader" was reading 0). Also catch
        # legislative-leadership "(majority|minority) leader ... of the <State> Senate/House".
        r"(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
        r"florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|"
        r"maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|"
        r"nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|"
        r"north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|"
        r"south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|"
        r"wisconsin|wyoming)\s+(?:state\s+)?"
        r"(?:senate|house|assembly|senator|representative|house of representatives|"
        r"house of delegates|general assembly)|"
        r"(?:majority|minority) leader of the .{0,30}(?:senate|house|assembly)"),
    (1, r"\bmayor\b|county (executive|commissioner|supervisor|clerk|judge|attorney|"
        r"treasurer|health director)|city council|county council|sheriff|"
        r"school board|city commissioner|selectman|alderman|district attorney|"
        r"state's attorney|\bjudge\b"),
]
PRIOR_CAND_RX = re.compile(r"candidate for|nominee for", re.I)

# BULLET FORMAT AMBIGUITY (found 2026-07-23, fact-check on the combined bio file: 107 of
# 2848 "incumbent"-descriptor rows misclassified as level 0): Wikipedia race-page bullets
# routinely say bare "incumbent senator" / "incumbent Representative [from X] since Y" with
# NO "U.S."/"member of Congress" qualifier - unambiguous IN CONTEXT (a Senate-page bullet
# saying "incumbent senator" means U.S. Senator; a House-page "incumbent Representative"
# means U.S. Representative) but invisible to a page-blind regex. classify() now takes the
# page's OFFICE so these context-dependent phrases resolve correctly; office=None keeps the
# old page-blind behavior (used nowhere in this repo, kept for safety/back-compat only).
INCUMBENT_BY_OFFICE_RX = {
    "Senate": re.compile(r"incumbent senator\b"),
    "House": re.compile(r"incumbent representative\b"),
}

def classify(desc, office=None):
    d = str(desc).lower()
    # candidacy mentions are NOT offices held: 'candidate for governor in 2018' must not
    # classify as governor (caught by the El-Sayed known-truth check). Broadened 2026-07-29 to
    # also strip "ran for / running for / unsuccessful candidate for / nominee for X" - the
    # institution-phrasing level-4 pattern ("U.S. House/Senate") was matching a "( ran for U.S.
    # House )" parenthetical and mislabeling e.g. "school board member (ran for U.S. House)" as 4.
    d = re.sub(r"(unsuccessful\s+)?(candidate|nominee)\s+for\s+[^,;.()]+", " ", d)
    d = re.sub(r"\b(ran|running)\s+for\s+[^,;.()]+", " ", d)
    # FUTURE offices are not offices held at the time of THIS race (2026-08-01). Wikipedia
    # bios are written after the fact, so a candidate's page can describe an office they only
    # won LATER: "pastor of Ebenezer Baptist Church and future U.S. Senator for this seat"
    # (Warnock, 2016) was classifying as 4, and "Iraq War veteran, former prosecutor and
    # future Florida governor" (DeSantis, 2012) as 3 - both held NO office at the time. That
    # is label leakage: the model would read "this person later became a senator" as a
    # pre-election credential. Strip the future-office phrase only; "later withdrew" /
    # "later ran for governor" describe CAMPAIGN events and must NOT nuke a real credential
    # ("incumbent U.S. representative (ran for governor, later withdrew)" is genuinely 4),
    # which is why this targets 'future X' and 'later became/elected X', not every 'later'.
    d = re.sub(r"\bfuture\s+[^,;.()]+", " ", d)
    d = re.sub(r"\blater\s+(became|elected|won|appointed)\b[^,;.()]*", " ", d)
    d = re.sub(r"\b(subsequently|went\s+on\s+to)\s+(became?|won|elected)\b[^,;.()]*", " ", d)
    if office in INCUMBENT_BY_OFFICE_RX and INCUMBENT_BY_OFFICE_RX[office].search(d):
        return 4
    for lvl, rx in LEVELS:
        if re.search(rx, d):
            return lvl
    return 0

def parse_page(html, house=False):
    """[(district_or_None, party, name, descriptor)] from primary-section <li> bullets."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    stage, party, district = "general", "", None
    in_candidates = False
    cand_section = False    # inside a "Candidates" section (survives a party-only subheading)
    SKIP_SUB_RX = re.compile(r"endorse|polling|results|debate|see also|references|"
                             r"fundrais|predictions|notes", re.I)
    # "Nominee" / "advanced to general" added 2026-07-23 (user pushed back on a "redundant
    # with polls" ablation conclusion for bio_office_level - investigating found the REAL
    # cause was partly coverage: Wikipedia uses subheadings the old regex never matched for
    # a clear front-runner/incumbent - "Nominee" (Sheldon Whitehouse, 2024 RI Senate) and
    # "Advanced to general" (Maria Cantwell, 2024 WA Senate - a top-two/jungle-primary
    # state's terminology) - both verified on live pages. These subheadings matched NONE of
    # the old patterns, so uncontested/dominant incumbents - exactly the highest-
    # office_level candidates - were being silently skipped. This was very likely
    # suppressing real signal in the ablation (see HANDOFF.md 2026-07-23 for the numbers).
    CAND_SUB_RX = re.compile(
        r"candidate|declared|potential|withdr|declined|filed|nominee|advanced to general",
        re.I)
    for el in soup.find_all(["h2", "h3", "h4", "h5", "ul"]):
        if el.name != "ul":
            text = el.get_text(" ", strip=True)
            if house:
                m = re.search(r"District\s+(\d+)", text)
                if m:
                    district = int(m.group(1)); stage, party = "general", ""
            s, p = infer_section_context(text)
            # OVERRIDE (2026-07-23): "Advanced to general" is a top-two/jungle-primary
            # subheading (WA/CA-style) that infer_section_context (the SHARED polling-agg
            # parser) correctly reads as stage='general' in its own sense - but here it's
            # still a bio bullet for a candidate who WAS in that party's primary field, just
            # the one who won it (verified: Maria Cantwell, 2024 WA Senate). Don't let it
            # flip `stage` away from 'primary', or this bio-only parser drops it entirely.
            if re.search(r"advanced to general", text, re.I):
                s = None
            if s:
                stage = s
                in_candidates = False
                cand_section = False          # a new stage ends any old-format Candidates run
            if p:
                party = p
                in_candidates = False
                # NOTE: do NOT clear cand_section here. On pre-2012 flat pages a plain party
                # subheading ("Democrats"/"Republicans") sits UNDER a "Candidates" section
                # and carries only a party (no stage) - clearing the candidate context on it
                # is exactly what dropped those pages to zero. cand_section survives a
                # party-only heading so old_format below can still collect the bullets.
            # subsection tracking: ENDORSEMENT lists live INSIDE primary sections and
            # once scraped Russ Feingold as a Wisconsin "candidate" - only collect
            # bullets under candidate-ish subheadings
            if SKIP_SUB_RX.search(text):
                in_candidates = False
                cand_section = False
            elif CAND_SUB_RX.search(text):
                in_candidates = True
                cand_section = True
            continue
        # TWO collection modes (jungle mode added 2026-07-24):
        # - PARTISAN pages: party comes from a "Democratic/Republican primary" heading;
        #   collect only inside stage=primary + party context (the original path).
        # - JUNGLE/TOP-TWO pages (CA/WA/LA House): there ARE no party-primary headings at
        #   all - structure is District N -> "Candidates" -> bullets with the party inline
        #   as a leading parenthetical ("Lateefah Simon (Democratic), president of...").
        #   These states parsed to ZERO for every cycle and were misdiagnosed twice
        #   (URL bug: wrong; transient fetch failure: wrong) before inspecting the actual
        #   page structure. Gate: house page, a district is set, candidate-ish subheading,
        #   and NO party heading context - then party is read per-bullet instead.
        jungle_mode = (house and district is not None and in_candidates
                       and not party and stage != "primary")
        # OLD-FORMAT mode (added 2026-07-27 for the pre-2012 backfill): Wikipedia's
        # pre-~2012 race pages don't use "Democratic primary"/"Republican primary" stage
        # headings. They list candidates flat under a "Candidates" section split by plain
        # party subheadings ("Democrats", "Republicans"), so infer_section_context returns
        # a PARTY but no stage='primary' - the original gate (which requires stage=='primary')
        # dropped every such page to ZERO rows (verified: 2010 ND, 2008 WA fetched fine but
        # parsed empty; "Earl Pomeroy, incumbent U.S. Representative" was sitting right there
        # under Candidates->Democrats). in_candidates is already guarded by CAND_SUB_RX +
        # SKIP_SUB_RX, so "have a party AND we're under a candidate-ish subheading" is a safe
        # collection signal even without a primary stage. Doesn't affect modern partisan
        # pages: those reach the bullets via stage=='primary' first, and dedup drops repeats.
        old_format = (cand_section and party and stage != "primary" and not jungle_mode)
        if not jungle_mode and not old_format and (
                stage != "primary" or not party or not in_candidates):
            continue
        for li in el.find_all("li", recursive=False):
            t = li.get_text(" ", strip=True)
            t = re.sub(r"\[\s*\d+\s*\]", "", t)          # strip cite brackets
            if not (8 < len(t) < 400) or "," not in t:
                continue
            # NAME-LINK BUG (found 2026-07-23, fact-check on real historical pages):
            # a citation footnote link ('<a href="#cite_note-...">[8]</a>') can be the
            # FIRST <a> tag in a bullet whose candidate name is plain (unlinked) text -
            # li.find("a") then grabbed the footnote marker as the "name" and desc's
            # t[len(name):] slice cut the wrong number of characters off the real text
            # ('Matthew W. Morgan' -> 'ew W. Morgan'). Verified against the ALREADY-
            # COMMITTED candidate_bios.csv: 640 of 4441 rows (14.4%) had a bracket-only or
            # truncated-lowercase name from this exact bug - affects the live 2026 primary
            # model too, not just historical data. Fix: skip cite-note links when hunting
            # for the name link; only a wikilink that ISN'T a footnote counts as the name.
            a = next((x for x in li.find_all("a")
                      if not str(x.get("href", "")).startswith("#cite_note")), None)
            name = (a.get_text(" ", strip=True) if a else t.split(",")[0]).strip()
            if not name or len(name.split()) > 5:
                continue
            # match by content, not by position: 'name' may come from a wikilink whose
            # exact text differs from t's whitespace-normalized form (rare, but the old
            # code's t[len(name):] slice assumed name is a literal prefix of t - true only
            # when name came from t.split(',')[0]; when name came from a link, find it).
            idx = t.find(name)
            desc = (t[idx + len(name):] if idx >= 0 else t[len(name):]).lstrip(" ,").strip()
            if not desc:
                continue
            if jungle_mode:
                # party must come from the bullet itself: "(Democratic), descriptor..."
                # No leading party parenthetical = not a candidate bullet (nav/see-also
                # debris) - skip rather than guess.
                m = re.match(r"^\(\s*([^)]{1,40})\s*\)\s*,?\s*", desc)
                if not m:
                    continue
                bullet_party = F.npar(m.group(1))
                desc = desc[m.end():].strip()
                if not desc:
                    continue
                out.append((district, bullet_party, name, desc))
            else:
                out.append((district if house else None, party, name, desc))
    return out

# Party-name -> our 3-way code, for results-table cells ("Democratic"/"Republican"/etc.)
_TABLE_SKIP_NAME_RX = re.compile(
    r"^(total|majority|turnout|write-?in|others?|blank|void|scattering|n/a|—|-|hold|gain|"
    r"swing|registered|voter)", re.I)

def parse_results_tables(html, house=False, office=None):
    """[(district_or_None, party, name, descriptor)] from election-RESULTS wikitables.

    Added 2026-07-27 for the pre-2012 backfill. Most big-state pre-~2012 House pages (and
    some Senate/Gov) present candidates ONLY in a "Party | Candidate | Votes | %" results
    table with NO descriptor prose - parse_page (which reads candidate bullets) returns zero
    on them (verified: 2008 WA, 2010 FL/CA/TX/NY). This recovers the NAME + PARTY from the
    table. A results table has no descriptor prose, so descriptor is "" (office_level -> 0)
    EXCEPT one free, correct signal it DOES carry: a "(Incumbent)" tag means the candidate
    already held this very seat, so for a Senate/House/Governor page that's office_level 4/4/3
    respectively - we synthesize a descriptor ("incumbent U.S. Representative" etc.) so
    classify() levels them correctly. Non-incumbents keep descriptor "" and are leveled later
    (Ballotpedia / manual). Its job is roster + party completion (+ incumbents for free).

    District: taken from the nearest preceding "District N" heading (house pages). Rows whose
    candidate cell is a summary line (Total votes / Majority / Write-in / etc.) are skipped.
    """
    INC_DESC = {"Senate": "incumbent U.S. Senator",
                "House": "incumbent U.S. Representative",
                "Governor": "incumbent Governor"}
    soup = BeautifulSoup(html, "html.parser")
    out = []
    district = None
    for el in soup.find_all(["h2", "h3", "h4", "h5", "table"]):
        if el.name != "table":
            if house:
                m = re.search(r"District\s+(\d+)", el.get_text(" ", strip=True))
                if m:
                    district = int(m.group(1))
            continue
        if "wikitable" not in (el.get("class") or []):
            continue
        # locate the Party and Candidate columns from the header row
        header = el.find("tr")
        if not header:
            continue
        hcells = [c.get_text(" ", strip=True).lower() for c in header.find_all(["th", "td"])]
        if not ("candidate" in hcells and any("part" in h for h in hcells)):
            continue          # not a candidate-results table (skip predictions/turnout/etc.)
        for tr in el.find_all("tr")[1:]:
            cells = tr.find_all(["th", "td"])
            texts = [c.get_text(" ", strip=True) for c in cells]
            joined = " ".join(texts)
            # results rows carry a color-swatch th (empty) then Party, Candidate, Votes, %.
            # Find the party word + the candidate name among the cells robustly.
            party = ""
            for t in texts:
                p = F.npar(t)
                if p != "OTH" or re.match(r"^(democrat|republic)", t, re.I):
                    party = p
                    break
                if re.match(r"^(independent|libertarian|green|constitution|reform)", t, re.I):
                    party = F.npar(t)
                    break
            # candidate name = first cell that looks like a person (2-4 words, not a summary,
            # not the party word, not a number)
            name = ""
            for t in texts:
                ts = re.sub(r"\[\s*\d+\s*\]", "", t).strip()
                if (2 <= len(ts.split()) <= 4 and not _TABLE_SKIP_NAME_RX.match(ts)
                        and not re.search(r"\d", ts)
                        and not re.match(r"^(democrat|republic|independ|libertar|green|"
                                         r"constitution|reform)", ts, re.I)):
                    name = ts
                    break
            if not name or _TABLE_SKIP_NAME_RX.match(joined):
                continue
            # "(Incumbent)" / "(incumbent)" suffix -> strip from name, but use it: an
            # incumbent held this exact seat, so synthesize the right-level descriptor.
            is_inc = bool(re.search(r"\(\s*incumbent\s*\)", name, re.I))
            name = re.sub(r"\s*\(\s*incumbent\s*\)\s*", "", name, flags=re.I).strip()
            if not name:
                continue
            desc = INC_DESC.get(office, "") if is_inc else ""
            out.append((district if house else None, party, name, desc))
    # dedup within page (a candidate can appear in both a primary and a general table)
    seen, uniq = set(), []
    for row in out:
        k = (row[0], row[1], F.norm_name(row[2]))
        if k not in seen:
            seen.add(k); uniq.append(row)
    return uniq

def _scrape_office(office, targets, out_path):
    """Scrape one office's pages -> its OWN output file (never shared with another
    office's scraper - see the file-header note on why this changed 2026-07-24)."""
    done = set()
    frames = []
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, low_memory=False)
        frames.append(old)
        done = set(zip(old["year"], old["state"]))
        print(f"resuming {office}: {len(done)} pages already in {os.path.basename(out_path)}")

    for i, (year, st) in enumerate(targets):
        if (year, st) in done:
            continue
        state = STATES.get(st)
        if not state:
            continue
        s = state.replace(" ", "_")
        url = (URL_SEN if office == "Senate" else URL_GOV).format(year=year, state=s)
        html = fetch_page(url)
        rows = parse_page(html, house=False) if html else []
        # SUPPLEMENT with results-table rows (pre-2012 pages often have no candidate bullets;
        # 2026-07-27). Bullets carry real descriptors so they WIN - only add a table candidate
        # the bullets didn't already find (matched on party + normalized name).
        if html:
            have = {(F.npar(p), F.norm_name(n)) for _, p, n, _ in rows}
            for tr in parse_results_tables(html, house=False, office=office):
                if (F.npar(tr[1]), F.norm_name(tr[2])) not in have:
                    rows.append(tr)
        df = pd.DataFrame(rows, columns=["district", "party", "name", "descriptor"])
        df["year"], df["office"], df["state"] = year, office, st
        df["cand_key"] = df["name"].map(F.norm_name)
        df["office_level"] = df["descriptor"].map(lambda d: classify(d, office=office))
        df["bio_in_office"] = df["descriptor"].astype(str).str.contains(
            "present", case=False).astype(int)
        df["bio_prior_candidacy"] = df["descriptor"].astype(str).str.contains(
            PRIOR_CAND_RX).astype(int)
        frames.append(df)
        if rows:
            print(f"  {year} {office} {st}: {len(rows)} candidate bios")
        if (i + 1) % 15 == 0:
            pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        time.sleep(0.8)

    allb = pd.concat(frames, ignore_index=True)
    allb = allb.drop_duplicates(subset=["year", "office", "state", "district",
                                        "party", "cand_key"], keep="first")
    allb.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}: {len(allb)} {office} bios")
    print("office_level distribution:", allb["office_level"].value_counts().to_dict())
    return allb

def main():
    # target pages: Senate/Governor 1998-2024 from the RESULTS files (complete, unbiased),
    # + the 2026 predict set (Senate/Governor rows only here - 2026 House is scraped by
    # fetch_house_candidate_bios_hist.py, one script per office family, one output file
    # per office family - see the header note above OUT_SENATE/OUT_GOVERNOR).
    #
    # FIXED 2026-07-23 (user pushed back on an ablation "redundant with polls" conclusion,
    # asked to check what's actually missing in 2024 first): the OLD target list came from
    # primary_polls_wikipedia.csv, which only has a page wherever a primary was interesting
    # enough to POLL. That's a SYSTEMATIC bias, not random sparsity: uncontested/safe-seat
    # incumbent races (exactly where bio_office_level should matter most - a well-known
    # high-office incumbent running essentially unopposed) never got polled and so never
    # got a bio-scrape target either. Measured on 2024 Senate alone: 12 of 33 races (36%)
    # had NO target page under the old list - Sheldon Whitehouse (RI), Maria Cantwell (WA),
    # Amy Klobuchar (MN), Bernie Sanders (VT) and other safe incumbents were structurally
    # unreachable no matter how good parse_page's parsing was. Results files cover EVERY
    # race regardless of how contested the primary was, so they're the right source for a
    # feature meant to describe office-holding history, not primary competitiveness.
    sen_targets, gov_targets = set(), set()
    for fn, office, bucket in [("res_senate.csv", "Senate", sen_targets),
                               ("res_governor.csv", "Governor", gov_targets)]:
        r = pd.read_csv(os.path.join(HERE, "data", fn), low_memory=False)
        r = r[(r["stage"].astype(str).str.lower() == "general")
             & (r["cycle"] >= 1998) & (r["cycle"] % 2 == 0)]   # even-year cycles only
        for cyc, st in zip(r["cycle"], r["state_abbrev"]):
            bucket.add((int(cyc), st))
    preds = pd.read_csv(_paths.out("primary_predictions_2026.csv"))
    for r in preds.drop_duplicates(["state", "office"]).itertuples():
        if r.office == "Senate":
            sen_targets.add((2026, r.state))
        elif r.office == "Governor":
            gov_targets.add((2026, r.state))
    print(f"{len(sen_targets)} Senate pages, {len(gov_targets)} Governor pages to scrape")

    _scrape_office("Senate", sorted(sen_targets), OUT_SENATE)
    _scrape_office("Governor", sorted(gov_targets), OUT_GOVERNOR)

if __name__ == "__main__":
    main()
