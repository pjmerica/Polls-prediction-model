"""Wikipedia pages AS THEY WERE on a date - month-end aggregator averages (2026-09-25).

VoteHub's open API was the per-poll source for current-term approval and the generic
ballot. It STALLED after June 2026: approval 56-64 polls/month through June, then 0 in
July, 7 in August, 0 in September; generic ballot 45-55/month, then 1, 5, 2. Months
built from it were missing (July) or rested on a handful of polls.

Wikipedia keeps a table of the major AGGREGATORS' current averages (DDHQ, RCP, Silver
Bulletin, NYT, ...), and its revision history preserves that table as of any date. So a
month's value = the table's mean as of the month's last day (or now, for the current
month) - an average of averages, the same all-pollster basis as the VoteHub months.
Nothing here is used for training cycles; it only fills 2026 months.
"""

import io
import re

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (research; polling-prediction-model)"}
API = "https://en.wikipedia.org/w/api.php"


def html_asof(title, when):
    """HTML of `title` as of timestamp `when` (the latest revision at or before it)."""
    r = requests.get(API, params={
        "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
        "rvstart": pd.Timestamp(when).strftime("%Y-%m-%dT%H:%M:%SZ"), "rvdir": "older",
        "rvprop": "ids|timestamp", "format": "json"}, headers=H, timeout=60)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    revs = page.get("revisions") or []
    if not revs:
        return None, None
    rid, ts = revs[0]["revid"], revs[0]["timestamp"]
    h = requests.get("https://en.wikipedia.org/w/index.php", params={"oldid": rid},
                     headers=H, timeout=60)
    h.raise_for_status()
    return h.text, ts


def _mean_excluding_average(t, col, parse):
    first = t[t.columns[0]].astype(str).str.replace(r"\[\d+\]", "", regex=True).str.strip()
    vals = t.loc[~first.str.lower().str.startswith("average"), col].map(parse)
    vals = vals.dropna()
    return (float(vals.mean()), int(len(vals))) if len(vals) else (None, 0)


def _pct(s):
    m = re.search(r"-?[\d.]+", str(s).replace("−", "-"))
    return float(m.group()) if m else None


def approval_asof(when):
    """Mean presidential APPROVE % across the aggregators as of `when`."""
    html, ts = html_asof("Opinion polling on the second Trump presidency", when)
    if html is None:
        return None, 0, None
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in t.columns]
        if cols and cols[0].startswith("Aggregator") and "Approve" in cols:
            v, n = _mean_excluding_average(t, "Approve", _pct)
            return v, n, ts
    return None, 0, ts


def _margin(s):
    m = re.search(r"(Democrat|Republican)\w*\s*\+\s*([\d.]+)", str(s))
    if not m:
        return None
    v = float(m.group(2))
    return v if m.group(1).startswith("Democrat") else -v


def generic_ballot_asof(when, cycle=2026):
    """Mean generic-ballot D-R margin across the aggregators as of `when`."""
    html, ts = html_asof(f"{cycle} United States House of Representatives elections", when)
    if html is None:
        return None, 0, None
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in (t.columns.get_level_values(-1)
                                 if hasattr(t.columns, "get_level_values") else t.columns)]
        has = lambda w: any(w in c for c in cols)
        if has("Democrat") and has("Republican") and has("Margin") and len(t) <= 25:
            mcol = [c for c in t.columns if "Margin" in str(c)][-1]
            v, n = _mean_excluding_average(t, mcol, _margin)
            return v, n, ts
    return None, 0, ts


def month_ends(start, end=None):
    """[(month_start, as_of)] for each month from `start` through the current month; as_of is
    the month's last second, or now for the current month."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = []
    for ms in pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                            (end or now), freq="MS"):
        me = ms + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
        out.append((ms, min(me, now)))
    return out
