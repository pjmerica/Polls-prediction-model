# -*- coding: utf-8 -*-
"""Fill bio_office_level gaps from BALLOTPEDIA for candidates Wikipedia missed
-> data/candidate_bios_ballotpedia.csv.

WHY: after three rounds of Wikipedia scraper fixes, bio_office_level coverage reached
58.1% (67.7% winners), but 1,855 candidates remain uncovered - dominated by pre-2012
(Wikipedia editing depth is thin there) plus scattered modern gaps. Ballotpedia has
structured "office held" data with far better coverage of down-ballot + historical
candidates (confirmed 2026-07-24: John McCain, Jeb Bush, Gray Davis, Ken Calvert all
present; the project already scrapes Ballotpedia in polling-agg, so it's a known-accessible
source, not Cloudflare-walled). This scraper targets ONLY the uncovered set
(data/uncovered_candidates.csv, written by the coverage-measurement step) - it's a gap
filler, not a full re-scrape.

STRUCTURE (verified on real pages): office history lives in the single .infobox element as
clean text, e.g. "U.S. House Michigan District 10 Tenure 2023 - Present". The SAME office-
level classifier as the Wikipedia path (fetch_candidate_bios.classify) runs over that text.

NAME->URL RESOLUTION (the hard part):
  1. try https://ballotpedia.org/First_Last
  2. if that's a disambiguation page ("may refer to"), it lists state-qualified links
     ("John James (Michigan)", "John James (Kentucky)") - pick the one whose parenthetical
     matches the candidate's state.
  3. fallback: try https://ballotpedia.org/First_Last_(state) directly.
A page counts as a real profile only if its infobox mentions the candidate's own state
(guards against landing on a different same-named person).

    py -X utf8 fetch_candidate_bios_ballotpedia.py
Writes data/candidate_bios_ballotpedia.csv (own file - combine_candidate_bios.py merges it,
Wikipedia preferred on conflict). Safe to re-run: resumes from its own output.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

HERE = ROOT   # repo root (paths.py) - this file lives in a subfolder
sys.path.insert(0, AGG)
from bs4 import BeautifulSoup  # noqa: E402

from fetch_candidate_bios import classify, PRIOR_CAND_RX  # noqa: E402
import features as F  # noqa: E402



# Ballotpedia infoboxes name the CHAMBER/institution ("U.S. Senate", "U.S. House",
# "Governor of X", "X State Senate") where Wikipedia prose used the person's TITLE
# ("U.S. Senator", "U.S. Representative", "state senator"). The shared classify() (tuned for
# Wikipedia titles) misses these, so a Ballotpedia-specific classifier runs FIRST and only
# falls back to classify() for descriptor-style text. Same 4/3/2/1/0 scale. Candidacy
# phrasing must be stripped first: Ballotpedia infoboxes lead with "Candidate, <office they
# are RUNNING FOR>" (e.g. "Candidate, Governor of Michigan" for John James) - counting that
# as an office HELD is the exact same trap classify() guards against.
_BP_LEVELS = [
    (4, re.compile(r"u\.?s\.?\s+(senate|house|senator|representative)|"
                   r"united states\s+(senate|house|senator|representative)|"
                   r"member of congress|u\.?s\.?\s+cabinet|"
                   # U.S. cabinet secretaries ONLY. Fixed 2026-07-27: bare "secretary of" was
                   # matching "Indiana Secretary of State" (a STATEWIDE office = level 3) as
                   # level 4 - Evan Bayh's prior office read 4 instead of his real max Governor=3.
                   # A federal cabinet secretary is named either "U.S. Secretary of X" or
                   # "Secretary of <the-Federal-Department>"; "Secretary of State" WITHOUT a U.S.
                   # qualifier is treated as the STATE office (level 3, below) since in this data
                   # state secretaries of state vastly outnumber the lone federal one.
                   r"(u\.?s\.?|united states)\s+secretary of\b|"
                   # Departments that exist ONLY federally keep the bare form...
                   r"secretary of (the treasury|defense|energy|homeland security|"
                   r"veterans affairs|the interior|housing)\b|"
                   # ...but labor/commerce/education/transportation/agriculture also name
                   # STATE cabinet posts ("Maryland Secretary of Commerce" = level 3), so
                   # those require an explicit federal qualifier (fixed 2026-08-01).
                   r"(?:u\.?s\.?|united states|federal)\s+secretary of\s+"
                   r"(?:labor|commerce|education|transportation|agriculture)\b", re.I)),
    (3, re.compile(r"\bgovernor of\b|lieutenant governor|attorney general|"
                   r"secretary of state|state treasurer|state auditor|state comptroller|"
                   r"land commissioner|superintendent of public|insurance commissioner|"
                   # APPOINTED statewide/federal agency heads (added 2026-08-01). Only
                   # "commissioner of <dept>" was matching before, so a state health director
                   # or cabinet secretary read 0 - Nirav Shah (director of the Maine CDC, then
                   # principal deputy director of the federal CDC) classified as "no office
                   # held". These are real executive credentials: name recognition, donor
                   # networks, statewide profile. Requires an agency word so it cannot catch a
                   # corporate "director of marketing".
                   r"(?:(?:deputy|principal deputy|acting)\s+)?"
                   r"(?:director|administrator|commissioner|superintendent|chief)\s+of\s+"
                   r"(?:the\s+)?[a-z' ]{0,40}?"
                   r"(?:department|agency|administration|bureau|division|"
                   r"centers? for disease control|environmental protection)\b|"
                   r"\b(?:state\s+)?(?:health|education|transportation|revenue|labor|"
                   r"corrections|agriculture|commerce|emergency management)\s+"
                   r"(?:department\s+)?(?:director|commissioner|secretary)\b|"
                   r"\bsecretary of (?:commerce|labor|health|education|corrections|"
                   r"administration|revenue|state agencies)\b|"
                   r"\bchief justice\b|\bstate supreme court\b|\bsupreme court of\b|"
                   r"\bunited states ambassador\b|\bu\.?s\.? ambassador\b", re.I)),
    (2, re.compile(r"state senate|state house|state assembly|state legislature|"
                   r"house of delegates|general assembly|state senator|state representative|"
                   r"state delegate|"
                   # STATE-NAME legislature phrasing (added 2026-07-27 for the Stage-3 manual
                   # hardcode + Ballotpedia infoboxes that name the state, not the word "state":
                   # "Idaho House of Representatives", "Connecticut Senate", "Ohio Assembly",
                   # "Maryland House of Delegates"). The 50 state names + DC, then the chamber.
                   r"(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
                   r"delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
                   r"kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
                   r"mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
                   r"new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
                   r"pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
                   r"utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)\s+"
                   r"(?:state\s+)?(?:senate|house|assembly|house of representatives|"
                   r"house of delegates|general assembly)\b", re.I)),
    (1, re.compile(r"\bmayor\b|county (executive|commissioner|clerk|treasurer|sheriff|"
                   r"supervisor|legislature|legislator|prosecutor|attorney)|"
                   r"city council|county council|school board|city commission|"
                   r"district attorney|\bjudge\b|alderman|selectman|"
                   # added 2026-08-01 alongside the level-3 agency-head patterns
                   r"borough president|board of (?:education|supervisors|selectmen)|"
                   r"town council|village trustee|city clerk|\bsheriff\b|"
                   r"circuit court|district court|superior court|county court", re.I)),
]

def classify_ballotpedia(infobox_text, office=None):
    d = re.sub(r"candidate,?\s+[^.]*?(?=(u\.?s\.?|state|governor|mayor|county|city|$))",
               " ", str(infobox_text), count=1, flags=re.I)  # drop leading "Candidate, ..."
    d = re.sub(r"(candidate|nominee) for [^,;.]+", " ", d, flags=re.I)
    for lvl, rx in _BP_LEVELS:
        if rx.search(d):
            return lvl
    return classify(infobox_text, office=office)   # fall back to the Wikipedia classifier

OUT = os.path.join(HERE, "data", "candidate_bios_ballotpedia.csv")
UNCOVERED = os.path.join(HERE, "data", "uncovered_candidates.csv")
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")}
STATE_NAME = {  # abbr -> full, for state-qualified URLs + infobox state matching
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

class RateLimited(Exception):
    """Ballotpedia throttled us (429 / connection reset). Distinct from a real 404 so a
    blocked run doesn't silently record every candidate as 'no profile' (found 2026-07-24:
    the first full run of 1,500 lookups returned 0 profiles - Ballotpedia rate-limited after
    a burst, the old bare-except swallowed it, and everything looked like a clean miss)."""

def _fetch(url, _tries=3):
    """Returns HTML, None (genuine 404 / missing page), or raises RateLimited after retries
    with exponential backoff so the caller can pause instead of poisoning the whole run."""
    for attempt in range(_tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                       # page genuinely doesn't exist
            if e.code in (429, 403, 503):         # throttle / block
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except Exception:                          # timeout, reset, etc. - treat as throttle
            time.sleep(5 * (attempt + 1))
            continue
    raise RateLimited(url)

def _infobox_text(html):
    """Text of the page's OFFICE infobox, or '' if none. A page can have several .infobox
    elements - a 'top_disclaimer infobox' archived-official banner (no office data) and the
    real 'infobox person' (found 2026-07-24: selecting the first .infobox grabbed the
    disclaimer and made Jeb Bush / Gray Davis look like they had no profile). Prefer the
    'person' infobox; skip disclaimer banners; concatenate the rest as a fallback."""
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.find_all(class_=re.compile(r"\binfobox\b"))
    person = [b for b in boxes if "person" in (b.get("class") or [])]
    if person:
        return person[0].get_text(" ", strip=True)
    usable = [b for b in boxes if "top_disclaimer" not in (b.get("class") or [])]
    return " ".join(b.get_text(" ", strip=True) for b in usable)

OFFICE_TITLE_RX = re.compile(
    r"u\.?s\.?\s+(senate|house)|united states\s+(senate|house)|governor of|"
    r"lieutenant governor|attorney general|secretary of state|state (senate|house|assembly)|"
    r"\bmayor\b|county|city council|\bjudge\b|district attorney|state treasurer|"
    r"state auditor|land commissioner|insurance commissioner", re.I)
DATE_RANGE_RX = re.compile(r"(\d{4})\s*[-–]\s*(\d{4}|present)", re.I)
# Broadened date formats (2026-07-27, prompted by Baron Hill's dateless infobox): Ballotpedia
# isn't consistent - besides "2007–2011" it uses "Assumed office January 3, 2019",
# "In office 2007 - 2011", or a lone "Successor:"/"Predecessor:" row with the year only in a
# nearby "Elections" line. We try these IN ORDER; only a real 4-digit START year is ever
# accepted (never guessed). An office with NO recoverable start stays start=None -> the
# builder skips it (leak-safe) and it routes to the Stage-3 manual list with the office known.
DATE_ASSUMED_RX = re.compile(r"assumed office\D*(\d{4})", re.I)      # start only, end=present
DATE_INOFFICE_RX = re.compile(r"in office\D*(\d{4})\s*[-–]\s*(\d{4}|present)", re.I)
DATE_SINGLE_RX = re.compile(r"\b(19|20)\d{2}\b")                     # last-resort lone year

def _parse_office_dates(t):
    """(start, end_or_None) from a fragment, trying several Ballotpedia layouts, or None if
    no real START year is present. end None = present/ongoing."""
    m = DATE_RANGE_RX.search(t) or DATE_INOFFICE_RX.search(t)
    if m:
        end = None if m.group(2).lower() == "present" else int(m.group(2))
        return int(m.group(1)), end
    m = DATE_ASSUMED_RX.search(t)
    if m:
        return int(m.group(1)), None
    return None

def extract_offices(html):
    """[(office_phrase, start_year, end_year_or_None)] for each office in the person infobox,
    for LEAK-FREE as-of-year leveling downstream. Ballotpedia lays each office out as an
    office-title row followed by its date row ('U.S. House Virginia District 7' then 'Years
    in office: 2019 - 2025'), so we pair each office-title text with the NEXT date range that
    appears after it. end_year None = 'Present'. 'Candidate, <office>' (the office they are
    RUNNING FOR) is skipped - not an office held."""
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.find_all(class_=re.compile(r"\binfobox\b"))
    person = [b for b in boxes if "person" in (b.get("class") or [])]
    box = person[0] if person else (boxes[0] if boxes else None)
    if box is None:
        return []
    # ordered stream of short text fragments from the infobox rows
    frags = []
    for el in box.find_all(["tr", "div", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if t and len(t) < 140:
            frags.append(t)
    offices, pending = [], None
    for t in frags:
        low = t.lower()
        if low.startswith("candidate,") or low.startswith("candidate for"):
            pending = None
            continue
        has_dates = _parse_office_dates(t) is not None
        if OFFICE_TITLE_RX.search(t) and not has_dates:
            # a new office title arrives before the previous one's dates were found ->
            # the previous office is genuinely dateless in this infobox: record it with
            # start=None (leak-safe: builder skips it; Stage-3 will supply the year).
            if pending is not None:
                offices.append((pending, None, None))
            pending = t                      # this office title now awaits its date row
            continue
        dr = _parse_office_dates(t)
        if dr and pending:
            offices.append((pending, dr[0], dr[1]))
            pending = None
    if pending is not None:                  # trailing office title with no date row after it
        offices.append((pending, None, None))
    return offices

def _is_disambig(html):
    return "may refer to" in html[:6000].lower()

def _disambig_pick(html, state_full):
    """On a disambiguation page, return the href whose link text names this state."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"}) or soup
    for a in content.find_all("a", href=True):
        t = a.get_text(" ", strip=True)
        if state_full and state_full in t and a["href"].startswith("/"):
            return "https://ballotpedia.org" + a["href"]
    return None

def _name_variants(name):
    """Ballotpedia uses a candidate's COMMON name form, not their full ballot name (found
    2026-07-24: 'William Sam McCann' -> Ballotpedia 'Sam McCann'; 'David B. McKinley' ->
    'David McKinley'; 'Alexander X. Mooney' -> 'Alex Mooney' is a nickname we can't derive,
    but the mechanical variants below recover the initial/suffix cases). Yields URL-slug
    name strings, most-specific first."""
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)   # drop suffixes
    n = re.sub(r",", " ", n)
    parts = [p for p in n.split() if p]
    seen, out = set(), []
    def add(words):
        s = " ".join(words).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    add(parts)                                        # full (minus suffix)
    no_init = [p for p in parts if not re.fullmatch(r"[A-Za-z]\.?", p)]  # drop single initials
    add(no_init)
    if len(no_init) >= 3:                              # 'William Sam McCann' -> 'Sam McCann'
        add([no_init[-2], no_init[-1]])
    if len(no_init) >= 2:                              # first + last only
        add([no_init[0], no_init[-1]])
    return out

def _profile_ok(html, state_full):
    """A fetched page is this candidate's real profile iff it has an office infobox naming
    their state (guards against a different same-named person / a bare stub)."""
    if not html:
        return False
    box = _infobox_text(html)
    return bool(box) and bool(state_full) and state_full in box

def resolve_and_extract(name, state_abbr):
    """-> (infobox_text, resolved_url, offices) or (None, None, None). `offices` is the
    per-office tenure-date list from extract_offices (for leak-free as-of-year leveling)."""
    state_full = STATE_NAME.get(state_abbr, "")
    for variant in _name_variants(name):
        slug = variant.replace(" ", "_")
        url = f"https://ballotpedia.org/{slug}"
        html = _fetch(url)
        if html and _is_disambig(html):
            picked = _disambig_pick(html, state_full)
            if picked:
                h2 = _fetch(picked)
                if _profile_ok(h2, state_full):
                    return _infobox_text(h2), picked, extract_offices(h2)
            continue
        if _profile_ok(html, state_full):
            return _infobox_text(html), url, extract_offices(html)
        alt = f"https://ballotpedia.org/{slug}_({state_full.replace(' ', '_')})"
        h3 = _fetch(alt)
        if _profile_ok(h3, state_full):
            return _infobox_text(h3), alt, extract_offices(h3)
    return None, None, None

def main():
    winners_only = "--winners-only" in sys.argv     # 2012+ winners = the ~102 actionable set
    since = 2012 if winners_only else 0
    unc = pd.read_csv(UNCOVERED, low_memory=False)
    if winners_only:
        unc = unc[(unc["won"] == 1) & (unc["year"] >= since)]
        print(f"WINNERS-ONLY mode: {len(unc)} uncovered 2012+ winner-rows "
              f"(the actionable set, well under Ballotpedia's ~550-request volume block)")
    people = unc.drop_duplicates(["candidate", "state"])[["candidate", "state", "office"]]
    print(f"{len(unc)} uncovered candidate-rows -> {len(people)} distinct (name,state) lookups")

    done = set()
    rows = []
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, low_memory=False)
        rows = old.to_dict("records")
        done = set(zip(old["candidate"], old["state"]))
        print(f"resuming: {len(done)} lookups already done")

    # per-run lookup cap (2026-07-28): do a solid batch then stop cleanly for a cooldown,
    # rather than pushing until a block. Tunable via --max-per-run N (default 100). Keeps each
    # resume cycle productive without hammering Ballotpedia into a longer wall.
    max_per_run = 100
    miss_guard = 20      # consecutive-miss soft-block threshold (original default); tunable
                         # via --miss-guard=N (the non-winner sweep passes a large value because
                         # its absence streaks are genuine no-profile people, not a block)
    for a in sys.argv:
        if a.startswith("--max-per-run="):
            max_per_run = int(a.split("=", 1)[1])
        if a.startswith("--miss-guard="):
            miss_guard = int(a.split("=", 1)[1])

    n_hit = 0
    n_this_run = 0
    consec_miss = 0   # soft-block guard: many consecutive real-fetch misses = a 200-status
    for i, r in enumerate(people.itertuples()):
        if (r.candidate, r.state) in done:
            continue
        if n_this_run >= max_per_run:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  -- reached --max-per-run={max_per_run} this cycle; stopping clean "
                  f"(progress saved, resume to continue)")
            break
        n_this_run += 1
        try:
            info, url, offices = resolve_and_extract(r.candidate, r.state)
        except RateLimited:
            # DO NOT record a miss - back off hard and retry this same candidate. A
            # rate-limited response is NOT a "no profile" answer (the whole point of the
            # 2026-07-24 hardening). Save progress first so nothing already found is lost.
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  ! rate-limited at {r.candidate} ({r.state}) - backing off 60s")
            time.sleep(60)
            try:
                info, url, offices = resolve_and_extract(r.candidate, r.state)
            except RateLimited:
                print("  !! still rate-limited after backoff - stopping cleanly; "
                      "re-run to resume (progress saved, misses NOT poisoned)")
                break
        lvl = classify_ballotpedia(info, office=r.office) if info else None
        if info:
            n_hit += 1
            consec_miss = 0
            rows.append(dict(candidate=r.candidate, state=r.state,
                             office_level=lvl,
                             bio_prior_candidacy=int(bool(PRIOR_CAND_RX.search(info))),
                             offices_json=json.dumps(offices),   # per-office tenure dates
                             src_url=url))
            if lvl and lvl >= 3:
                print(f"  {r.candidate} ({r.state}): level {lvl}")
        else:
            # SOFT-BLOCK GUARD (2026-07-24): Ballotpedia's volume block returns 200-status
            # pages WITHOUT the infobox (not a 404, so RateLimited above never fired) - the
            # first full run silently logged 1,500 of these as "no profile". A long run of
            # consecutive misses among a set that's mostly WINNERS (who nearly all have BP
            # pages) is the block signature, not real absence. Bail out clean rather than
            # poison - a resume will retry these.
            consec_miss += 1
            # SOFT-BLOCK GUARD, now tunable (--miss-guard=N, default 40). Verified 2026-07-28
            # that on the NON-WINNER set a run of 40 consecutive misses was REAL ABSENCE, not a
            # block (Nancy Pelosi's page fetched fine immediately after the stop) - obscure
            # losing challengers genuinely have no Ballotpedia profile. So for that set the guard
            # HURTS (premature stops on honest absence streaks); pass a large --miss-guard and
            # rely on the RateLimited exception (real 429/reset) to catch an actual block instead.
            if consec_miss >= miss_guard:
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f"  !! {consec_miss} consecutive misses - likely a soft volume-block, "
                      f"not real absence. Stopping clean; re-run later to resume.")
                # drop the trailing run of false-miss rows so resume re-tries them
                rows = rows[:-consec_miss]
                break
            rows.append(dict(candidate=r.candidate, state=r.state, office_level=None,
                             bio_prior_candidacy=0, offices_json="[]", src_url=None))
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  ... {i+1}/{len(people)} lookups, {n_hit} profiles found")
        time.sleep(1.5)   # gentler than the 0.8s that got the first run throttled

    pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    found = df[df["office_level"].notna()]
    print(f"\nsaved -> {OUT}: {len(df)} lookups, {len(found)} profiles found "
          f"({len(found)/len(df):.0%} hit rate)")
    print("office_level distribution:", found["office_level"].value_counts().to_dict())

if __name__ == "__main__":
    main()
