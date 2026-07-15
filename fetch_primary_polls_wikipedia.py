# -*- coding: utf-8 -*-
"""Fetch HISTORICAL (2018-2024) primary polling tables from Wikipedia election pages.

Why Wikipedia: 538's downloadable poll CSVs NEVER included downballot regular-primary
polls in any era (verified against in-season Wayback captures: Apr-2022 senate file =
generals + jungle only, despite Oz/McCormick being heavily polled) and its historical
exports likewise. Wikipedia's per-race pages carry the 'Republican primary' /
'Democratic primary' polling tables that the polling-agg 2026 scraper already parses -
this script drives that SAME parser (imported cross-repo) over historical page titles,
so historical + 2026 primary polls share one parsing implementation.

Scope: Senate + Governor, 2018-2024 (+ odd-year governors). House historical primary
polling is too sparse for pages to exist reliably - skipped.

Output: data/primary_polls_wikipedia.csv (committed; fetch once - static-data principle).

    py -X utf8 fetch_primary_polls_wikipedia.py
"""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets")
sys.path.insert(0, AGG)
from scrapers.wikipedia_polls import _scrape_state_race, STATES  # noqa: E402

OUT = os.path.join(HERE, "data", "primary_polls_wikipedia.csv")

STATE_NAME = {v: k for k, v in STATES.items()}   # sanity only; STATES: abbrev -> name

URL_SEN = "https://en.wikipedia.org/wiki/{year}_United_States_Senate_election_in_{state}"
URL_GOV = "https://en.wikipedia.org/wiki/{year}_{state}_gubernatorial_election"

def races_by_cycle():
    """(year, office, state_abbrev) for every Senate/Governor general race we have
    results for, 2018-2024 (odd years included for governors)."""
    out = []
    for fn, office, url in [("res_senate.csv", "SEN", URL_SEN),
                            ("res_governor.csv", "GOV", URL_GOV)]:
        r = pd.read_csv(os.path.join(HERE, "data", fn), low_memory=False)
        r = r[r["stage"].astype(str).str.lower() == "general"]
        r = r[(r["cycle"] >= 2018) & (r["cycle"] <= 2024)]
        # specials have their own pages and dual-seat ambiguity - regular races only
        if "special" in r.columns:
            r = r[~r["special"].astype(str).str.lower().isin(["true", "1"])]
        for (cyc, st), _ in r.groupby(["cycle", "state_abbrev"]):
            out.append((int(cyc), office, st, url))
    return sorted(set(out))

def main():
    targets = races_by_cycle()
    print(f"{len(targets)} race pages to scrape")
    done = set()
    frames = []
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, low_memory=False)
        frames.append(old)
        done = set(old["src_page"].unique())
        print(f"resuming: {len(done)} pages already fetched")

    for i, (year, off, st, url_t) in enumerate(targets):
        state = STATES.get(st)
        if not state:
            continue
        url = url_t.format(year=year, state=state.replace(" ", "_"))
        rid = f"{year}-{off}-{st}"
        if rid in done:
            continue
        try:
            rows = _scrape_state_race(url, rid)
        except Exception as e:
            print(f"  {rid}: ERROR {type(e).__name__} {e}")
            rows = []
        prim = [r for r in rows if r.get("stage") in ("primary", "primary runoff")]
        df = pd.DataFrame(prim if prim else [],
                          columns=["race_id", "pollster", "candidate", "party", "stage",
                                   "sample_size", "end_date", "implied_prob", "partisan",
                                   "poll_id", "question_id"])
        df["src_page"] = rid
        df["year"] = year
        frames.append(df)
        if prim:
            print(f"  {rid}: {len(prim)} primary poll rows")
        if (i + 1) % 10 == 0:
            pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)  # checkpoint
        time.sleep(1.0)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    real = out[out["stage"] == "primary"]
    print(f"\nsaved -> {OUT}: {len(out)} rows ({len(real)} stage=primary)")
    if len(real):
        real = real.copy()
        real["y"] = real["year"]
        print("primary rows by cycle:", real.groupby("y").size().to_dict())
        print("distinct races with primary polls:",
              real.groupby(["src_page", "party"]).ngroups)

if __name__ == "__main__":
    main()
