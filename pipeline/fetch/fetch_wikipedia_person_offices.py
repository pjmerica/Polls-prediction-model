"""Dated offices from each candidate's OWN Wikipedia article -> data/candidate_bios_wikiperson.csv
(2026-10-13).

Why: a 2026-10-13 random spot check found ~1 in 5 level-0 bio rows wrong. Most were "Stage-4
verified level 0" hand rows whose note says "no office evidence in any source" - i.e. NOT found,
not verified absent. Mark Neumann (U.S. Rep 1995-99), Elaine Bloom / Raj Goyle / Scott Bruun
(state legislators), Ro Khanna and Ben Sasse (sub-cabinet appointees) all read 0. Race-page
descriptors often omit a challenger's past office; the person's own article infobox lists every
office with its term years.

For each target (candidate, state, party, year, office) this searches Wikipedia, accepts an
article only if its title carries the surname and its text names the state, the candidate's
party and the race year, then reads the {{Infobox officeholder}} office / term_start pairs.
Output rows match candidate_bios_manual.csv (candidate, state, offices_json, source_note);
features.load_candidate_bios reads it, and build_candidate_table takes the as-of-year level from
it as a FLOOR (an office counts only if its term started before the election year - leak-free).

    py -X utf8 pipeline/fetch/fetch_wikipedia_person_offices.py --targets data/wikiperson_targets.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT  # noqa: E402,F401

import argparse
import json
import os
import re
import time

import pandas as pd
import requests

API = "https://en.wikipedia.org/w/api.php"
H = {"User-Agent": "polling-prediction-model/1.0 (research; contact via GitHub pjmerica)"}
OUT = "data/candidate_bios_wikiperson.csv"
PARTY_WORD = {"DEM": "Democratic", "REP": "Republican"}
STATE_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}


def _get(params, tries=6):
    """GET with Wikimedia's Retry-After honoured. Unauthenticated clients are throttled hard
    (2026-10-13: six threads drew HTTP 429 with Retry-After ~14 s and crawled to ~3 people/min)."""
    for k in range(tries):
        try:
            r = requests.get(API, params={**params, "format": "json", "formatversion": 2},
                             headers=H, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after") or 10 * (k + 1)) + 1)
                continue
        except (requests.RequestException, ValueError):
            time.sleep(3)
    return {}


def _strip(s):
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", str(s))      # [[a|b]] -> b
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)                               # simple templates
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _year(s):
    m = re.search(r"(1[89]\d\d|20\d\d)", str(s))
    return int(m.group(1)) if m else None


def infobox_offices(wikitext):
    """[[office, start, end], ...] from {{Infobox officeholder}} params, per numbered slot N:
    officeN, or the chamber implied by state_houseN / state_senateN / state_assemblyN /
    state_delegateN, or a U.S. seat implied by jr/sr (Senate) or a {{ushr}} / stateN+districtN
    pair (House). Start/end from term_startN / term_endN, else the years in termN."""
    i = wikitext.lower().find("{{infobox")
    if i < 0:
        return []
    # cut at the infobox's OWN closing braces (2026-10-14): a fixed 12,000-char window ran into
    # later templates whose same-named fields overwrote the real ones
    depth, j = 0, i
    while j < len(wikitext) - 1:
        two = wikitext[j:j + 2]
        if two == "{{":
            depth += 1; j += 2; continue
        if two == "}}":
            depth -= 1; j += 2
            if depth == 0:
                break
            continue
        j += 1
    box = wikitext[i:j]
    params = {}
    # [ \t]*, never \s*: \s crosses the newline, so an EMPTY field ("| caption =") swallowed the
    # next line as its value and that office was lost (Helmke's mayoralty, Kennedy's treasurer)
    for m in re.finditer(r"^[ \t]*\|[ \t]*([a-z_/ ]+?)(\d*)[ \t]*=[ \t]*(.*)$", box, re.M):
        params.setdefault((m.group(1).strip(), m.group(2)), m.group(3))
    idxs = sorted({k[1] for k in params}, key=lambda x: int(x) if x else 0)
    out = []
    for n in idxs:
        g = lambda key: params.get((key, n), "")
        office = _strip(g("office"))
        if not office:
            if _strip(g("state_house")):
                office = f"{_strip(g('state_house'))} House of Representatives"
            elif _strip(g("state_senate")):
                office = f"{_strip(g('state_senate'))} State Senate"
            elif _strip(g("state_assembly")):
                office = f"{_strip(g('state_assembly'))} State Assembly"
            elif _strip(g("state_delegate")):
                office = f"{_strip(g('state_delegate'))} House of Delegates"
            elif _strip(g("jr/sr")) or _strip(g("jr/sr senator")):
                office = "U.S. Senate"
            elif "ushr" in g("district").lower() or (_strip(g("state")) and _strip(g("district"))
                                                     and "congress" in g("district").lower()):
                office = "U.S. House of Representatives"
        if not office:
            continue
        start, end = _year(g("term_start")), _year(g("term_end"))
        if not start and g("term"):
            yrs = re.findall(r"(1[89]\d\d|20\d\d)", g("term"))
            start = int(yrs[0]) if yrs else None
            end = int(yrs[1]) if len(yrs) > 1 else None
        if start:
            out.append([office, start, end])
    return out


# nickname -> legal first name where the two don't share a 3-letter start (2026-10-14 recheck)
_NICKS = {"larry": "lawrence", "jerry": "gerald", "libby": "elizabeth", "quico": "francisco",
          "bill": "william", "billy": "william", "dick": "richard", "ted": "edward",
          "chuck": "charles", "jack": "john", "peggy": "margaret", "betsy": "elizabeth",
          "beth": "elizabeth", "liz": "elizabeth", "hank": "henry", "skip": "william",
          "tom": "thomas", "tommy": "thomas", "jim": "james", "jimmy": "james",
          "bob": "robert", "bobby": "robert", "rob": "robert", "chris": "christopher",
          "steve": "stephen", "kate": "katherine", "kathy": "kathleen", "sue": "susan"}
# verified one-offs where the article uses an unrelated nickname
_SAME_PERSON = {("samuel pearson goddard", "terry goddard")}   # Samuel Pearson Goddard III
# ...and relatives who pass the name rule through a shared middle name
_NOT_SAME = {("robert gerald lorge", "gerald lorge"),   # his father, WI state senator 1955-85
             # PA-7 2026 Dem nominee is a firefighters' union head, not the Murrysville mayor
             ("bob brooks", "robert brooks (pennsylvania politician)"),
             # ME-1 nominee 2024/26 is a retired Army colonel, never in the Maine House
             ("ronald c. russell", "ronald b. russell"), ("ronald russell", "ronald b. russell")}


def title_matches(name, title):
    """The article's first given name must line up with one of the candidate's FULL given
    names - same word, a prefix (Ric/Richard), or a nickname (Tom/Thomas, Bob/Robert).
    Never via an initial (2026-10-13): 'Patrick R. Casey' (PA-10 2002, Bob Casey's brother)
    matched 'Bob Casey Jr.' through the R, and 'Marsha Folsom' matched 'Jim Folsom Jr.'."""
    import features as F
    # surname = last word that is not a generational suffix ("Joe Manchin, III" -> manchin)
    sn = [w for w in (x.strip(",.").lower() for x in name.split())
          if w and w not in ("jr", "sr", "ii", "iii", "iv")][-1]
    if not re.search(r"\b" + re.escape(sn) + r"\b", title.lower()):
        return False          # whole word: 'Rich' is not in 'Heinrich'
    bare = re.sub(r"\(.*?\)", " ", title).lower()            # drop '(politician)'
    # title tokens drop only generational suffixes: a surname particle can BE the first given
    # word ('Van Tran' passed for Jennifer Tran once 'van' joined _NAME_NOISE, 2026-10-15)
    t_toks = [t for t in re.findall(r"[a-z]+", bare)          # in title order
              if t != sn and t not in ("jr", "sr", "ii", "iii", "iv") and len(t) > 1]
    given = [t for t in re.findall(r"[a-z]+", name.lower())
             if t != sn and t not in ("jr", "sr", "ii", "iii", "iv") and len(t) > 1]
    if not t_toks or not given:
        return True
    t0 = t_toks[0]
    if (name.lower(), title.lower()) in _NOT_SAME:
        return False
    if (name.lower(), title.lower()) in _SAME_PERSON:
        return True
    for g in given:
        if _NICKS.get(t0) == g or _NICKS.get(g) == t0:
            return True
        if g == t0 or (len(g) >= 3 and len(t0) >= 3 and (g.startswith(t0) or t0.startswith(g))):
            return True
        if t0 in F._NICK_INITIAL and F._NICK_INITIAL[t0] == g[0]:
            return True
        if g in F._NICK_INITIAL and F._NICK_INITIAL[g] == t0[0]:
            return True
        # short forms, either side, sharing two letters: Mike/Michael, Don/Donald, Joe/Joseph
        # (not just one: 'Mick' Rich passed as 'Martin' Heinrich, 2026-10-14)
        if t0[:2] == g[:2] and min(len(t0), len(g)) <= 4:
            return True
        if len(t0) >= 3 and len(g) >= 3 and t0[:3] == g[:3]:   # Ernie/Ernest, Robin/Robert
            return True
    return False


def plausible(offices):
    """No candidate since 1998 held office before 1940: an older infobox is a namesake
    (Bill Marcy, MS-2 2010, matched William L. Marcy, U.S. Senator 1831-33)."""
    return all(o[1] is None or int(o[1]) >= 1940 for o in offices)


def find_article(name, state, party, year):
    """ONE request per person: the search runs as a generator and returns each hit's wikitext
    in the same call (was a search plus up to four separate page fetches)."""
    # drop suffixes and middle initials (2026-10-15): "Jacob W. Hoogendyk Jr." took "Jr" as
    # the surname, and "Elliott E. Maynard ... politician" never surfaced "Elliott Maynard"
    words = [w for w in re.sub(r",", " ", name).split()
             if w.strip(".").lower() not in ("jr", "sr", "ii", "iii", "iv")]
    sname = words[-1].strip(",.").lower()
    plain = " ".join(w for w in words if len(w.strip(".")) > 1)
    st = STATE_NAME.get(state, state)
    pages = []
    for query in (f"{plain} {st} politician", f"{plain} {st}"):
        q = _get({"action": "query", "generator": "search", "gsrlimit": 4,
                  "gsrsearch": query, "prop": "revisions",
                  "rvprop": "content", "rvslots": "main", "redirects": 1})
        pages += sorted(q.get("query", {}).get("pages", []), key=lambda pg: pg.get("index", 99))
    for pg in pages:
        title = pg.get("title", "")
        if sname not in title.lower() or not title_matches(plain, title):
            continue
        revs = pg.get("revisions") or [{}]
        wt = revs[0].get("slots", {}).get("main", {}).get("content", "")
        if not wt:
            continue
        low = wt.lower()
        if st.lower() not in low or str(year) not in wt:
            continue
        if PARTY_WORD.get(party, "~~").lower() not in low:
            continue
        return title, wt
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recheck", action="store_true",
                    help="drop saved rows whose article title fails title_matches, then exit")
    ap.add_argument("--reparse", action="store_true",
                    help="re-download the saved rows' articles (50 per request) and re-read "
                         "their infoboxes with the current parser, then exit")
    args = ap.parse_args()
    if args.reparse:
        done = pd.read_csv(OUT)
        title = done["source_note"].str.extract(r"infobox: (.*) \(fetched", expand=False)
        titles = sorted(set(title.dropna()))
        wt = {}
        for k in range(0, len(titles), 50):
            q = _get({"action": "query", "titles": "|".join(titles[k:k + 50]),
                      "prop": "revisions", "rvprop": "content", "rvslots": "main"})
            for pg in q.get("query", {}).get("pages", []):
                revs = pg.get("revisions") or [{}]
                wt[pg.get("title")] = revs[0].get("slots", {}).get("main", {}).get("content", "")
        n = 0
        for i, t in title.items():
            if wt.get(t):
                offs = infobox_offices(wt[t])
                if offs and json.dumps(offs) != done.at[i, "offices_json"]:
                    done.at[i, "offices_json"] = json.dumps(offs); n += 1
        done.to_csv(OUT, index=False)
        print(f"reparse: {len(wt)} articles read, {n} people's offices changed")
        return
    if args.recheck:
        done = pd.read_csv(OUT)
        title = done["source_note"].str.extract(r"infobox: (.*) \(fetched", expand=False)
        ok = [title_matches(c, t) and plausible(json.loads(o))
              for c, t, o in zip(done["candidate"], title.fillna(""), done["offices_json"])]
        for c, t, k in zip(done["candidate"], title, ok):
            if not k:
                print(f"  drop: {c} <- {t}")
        done[ok].to_csv(OUT, index=False)
        print(f"recheck: kept {sum(ok)} of {len(done)}")
        return
    t = pd.read_csv(args.targets)
    if args.limit:
        t = t.head(args.limit)
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(
        columns=["candidate", "state", "offices_json", "source_note"])
    have = set(zip(done["candidate"], done["state"]))
    rows = done.to_dict("records")
    todo = [r for r in t.itertuples() if (r.candidate, r.state) not in have]

    def work(r):
        title, wt = find_article(r.candidate, r.state, r.party, int(r.year))
        return r, title, (infobox_offices(wt) if wt else [])

    # 2 concurrent lookups: more only buys HTTP 429 back-off from Wikimedia (2026-10-13)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r, title, offs = f.result()
            if title and offs and plausible(offs):
                rows.append(dict(candidate=r.candidate, state=r.state,
                                 offices_json=json.dumps(offs),
                                 source_note=f"Wikipedia infobox: {title} (fetched 2026-10-13)"))
                print(f"[{i}/{len(todo)}] {r.candidate} ({r.state}) <- {title}: {offs[:3]}", flush=True)
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved -> {OUT} ({len(rows)} people with dated offices)")


if __name__ == "__main__":
    main()
