# -*- coding: utf-8 -*-
"""Build data/primary_dates_hist.csv: the primary ELECTION DATE for every historical
(2018-2024) primary race we scraped polls for.

Dates come from a STATIC hand-entered table of statewide primary dates (public record -
these are verifiable calendar facts, not measurements; the provenance concern about
hand-typed *data values* doesn't apply, and three iterations of Wikipedia prose-mining
proved unreliable: filing deadlines, runoff dates and news dates near the word 'primary'
kept winning). NY is entered per-office (its 2018/2022 state vs federal primaries were on
different days).

Safety net: every date is CROSS-CHECKED against the poll record - the race's last
in-season primary poll must fall within [date - 60d, date + 2d]. Disagreements are
printed as warnings for manual review, and (state,cycle) pairs missing from the table
fall back to last-in-season-poll + 4 days, flagged approx=1.

    py -X utf8 fetch_primary_dates.py     (offline - no network)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "primary_dates_hist.csv")

# statewide primary date by (year, state) - public record
DATES = {
 2018: dict(AL="2018-06-05", AK="2018-08-21", AR="2018-05-22", AZ="2018-08-28",
            CA="2018-06-05", CO="2018-06-26", CT="2018-08-14", DE="2018-09-06",
            FL="2018-08-28", GA="2018-05-22", HI="2018-08-11", IA="2018-06-05",
            ID="2018-05-15", IL="2018-03-20", IN="2018-05-08", KS="2018-08-07",
            MA="2018-09-04", MD="2018-06-26", ME="2018-06-12", MI="2018-08-07",
            MN="2018-08-14", MO="2018-08-07", MS="2018-06-05", MT="2018-06-05",
            ND="2018-06-12", NH="2018-09-11", NM="2018-06-05", NV="2018-06-12",
            NY="2018-09-13",   # state-level (Governor); federal override below
            OH="2018-05-08", OK="2018-06-26", OR="2018-05-15", PA="2018-05-15",
            RI="2018-09-12", SC="2018-06-12", SD="2018-06-05", TN="2018-08-02",
            TX="2018-03-06", UT="2018-06-26", VA="2018-06-12", WA="2018-08-07",
            WI="2018-08-14", WV="2018-05-08", WY="2018-08-21"),
 2019: dict(KY="2019-05-21", MS="2019-08-06"),
 2020: dict(AL="2020-03-03", CO="2020-06-30", DE="2020-09-15", GA="2020-06-09",
            KS="2020-08-04", KY="2020-06-23", MA="2020-09-01", ME="2020-07-14",
            MI="2020-08-04", MT="2020-06-02", NC="2020-03-03", NE="2020-05-12",
            NH="2020-09-08", NM="2020-06-02", SC="2020-06-09", TN="2020-08-06",
            TX="2020-03-03", UT="2020-06-30", VA="2020-06-23", WA="2020-08-04",
            WV="2020-06-09", WY="2020-08-18"),
 2021: dict(NJ="2021-06-08", VA="2021-06-08"),
 2022: dict(AK="2022-08-16", AL="2022-05-24", AR="2022-05-24", AZ="2022-08-02",
            CA="2022-06-07", CO="2022-06-28", CT="2022-08-09", FL="2022-08-23",
            GA="2022-05-24", HI="2022-08-13", IA="2022-06-07", ID="2022-05-17",
            IL="2022-06-28", KS="2022-08-02", MA="2022-09-06", MD="2022-07-19",
            ME="2022-06-14", MI="2022-08-02", MO="2022-08-02", NC="2022-05-17",
            NE="2022-05-10", NH="2022-09-13", NM="2022-06-07", NV="2022-06-14",
            NY="2022-06-28",   # state-level (Governor); federal override below
            OH="2022-05-03", OK="2022-06-28", OR="2022-05-17", PA="2022-05-17",
            RI="2022-09-13", SC="2022-06-14", SD="2022-06-07", TN="2022-08-04",
            TX="2022-03-01", UT="2022-06-28", VT="2022-08-09", WA="2022-08-02",
            WI="2022-08-09"),
 2024: dict(AZ="2024-07-30", CA="2024-03-05", DE="2024-09-10", FL="2024-08-20",
            IN="2024-05-07", MA="2024-09-03", MD="2024-05-14", MI="2024-08-06",
            MO="2024-08-06", MT="2024-06-04", NC="2024-03-05", ND="2024-06-11",
            NH="2024-09-10", NJ="2024-06-04", NV="2024-06-11", NY="2024-06-25",
            OH="2024-03-19", PA="2024-04-23", TN="2024-08-01", TX="2024-03-05",
            UT="2024-06-25", VA="2024-06-18", WA="2024-08-06", WI="2024-08-13",
            WV="2024-05-14"),
}
# NY split its primaries: federal (Senate) and state (Governor) on different days
OFFICE_OVERRIDES = {
    (2018, "NY", "SEN"): "2018-06-26",   # federal primary; Governor Sep 13
    (2022, "NY", "SEN"): "2022-08-23",   # court-ordered split; Governor Jun 28
}

def main():
    polls = pd.read_csv(os.path.join(HERE, "data", "primary_polls_wikipedia.csv"),
                        low_memory=False)
    polls = polls[polls["stage"] == "primary"]
    pages = sorted(polls["src_page"].unique())
    print(f"{len(pages)} race pages with primary polls")

    rows, warns = [], 0
    for page in pages:
        year, off, st = page.split("-")
        year = int(year)
        dt = OFFICE_OVERRIDES.get((year, st, off)) or DATES.get(year, {}).get(st)
        approx = 0
        ends = pd.to_datetime(polls.loc[polls["src_page"] == page, "end_date"],
                              errors="coerce", format="mixed")
        in_season = ends[ends <= pd.Timestamp(f"{year}-09-20")]
        last = in_season.max() if len(in_season) else ends.max()
        if dt is None:
            dt = ((last + pd.Timedelta(days=4)).date().isoformat()
                  if pd.notna(last) else None)
            approx = 1
            print(f"  {page}: not in table -> poll-anchored fallback {dt}")
        elif pd.notna(last):
            # cross-check: last in-season primary poll must precede the date (60d window)
            gap = (pd.Timestamp(dt) - last).days
            if not (-2 <= gap <= 60):
                warns += 1
                print(f"  WARN {page}: table date {dt} vs last in-season poll "
                      f"{last.date()} (gap {gap}d) - check for leakage or a wrong date")
        rows.append(dict(page=page, year=year,
                         office={"SEN": "Senate", "GOV": "Governor"}[off],
                         state=st, primary_date=dt, approx=approx))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}: {len(df)} races | approx: {int(df['approx'].sum())} "
          f"| cross-check warnings: {warns}")

if __name__ == "__main__":
    main()
