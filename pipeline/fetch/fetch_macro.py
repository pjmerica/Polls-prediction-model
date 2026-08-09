"""One-time fetch of monthly macro series -> data/macro_monthly.csv.

This data is STATIC historical record (past months never change), so pull it ONCE
and commit the CSV. The model reads the CSV; it does NOT re-pull on every run.

    python fetch_macro.py

Source: **DBnomics** (https://db.nomics.world), a free aggregator of official statistics
with no API key. We use it instead of FRED because FRED's CSV host (`fredgraph.csv`) was
unreachable from both the dev sandbox and the user's machine, whereas DBnomics responds.
DBnomics does not carry FRED itself, so we pull from the upstream agencies (BLS, BEA, EIA,
Federal Reserve, U. Michigan, DOL) that FRED republishes.

Each series is resolved to monthly. Any series that fails to fetch is SKIPPED with a note
(the CSV is still produced). Output: long format [date, value, metric].
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from paths import ROOT, AGG  # noqa: E402  (repo-root-relative paths; see paths.py)

import io, os, time, requests
import pandas as pd



# Pull the FULL available history of each series (CPI/unemployment ~1947, gas ~1990, etc.).
# It's static reference data, so we keep everything in case we get older polls later.
# The model only *uses* the cycles it can match to polls (2018+), but the data is all here.
START = None   # set to e.g. "2016-01-01" to truncate; None = each series' full range
OUT = "data/macro_monthly.csv"
H = {"User-Agent": "Mozilla/5.0 (research)"}
BASE = "https://api.db.nomics.world/v22/series"

# metric -> (provider, dataset, series_code, frequency)
#   freq: M monthly/weekly (mean to month), Q quarterly (ffill), A annual (ffill)
# Codes confirmed to return data from DBnomics as of build time; unverified ones will skip.
SERIES = {
    "unemployment": ("BLS", "ln",  "LNS14000000",   "M"),  # unemployment rate, SA %
    "cpi":          ("BLS", "cu",  "CUSR0000SA0",   "M"),  # CPI-U, SA index -> YoY downstream
    "gas":          ("EIA", "PET", "EMM_EPMR_PTE_NUS_DPG.M", "M"),  # regular gas $/gal
    "fed_funds":    ("FED", "H15", "RIFSPFF_N.M",   "M"),  # effective fed funds rate %
    # --- additional series: kept if they resolve, skipped (with a note) if not ---
    "cpi_core":     ("BLS", "cu",  "CUSR0000SA0L1E", "M"), # core CPI (ex food & energy)
    "unemp_u6":     ("BLS", "ln",  "LNS13327709",   "M"),  # U-6 underemployment %
    "sentiment":    ("SCSMICH", "MICS", "ICS",      "M"),  # UMich Index of Consumer Sentiment (1978+)
}

# Generic-ballot monthly D-R margin, produced by `python fetch_generic_ballot.py --monthly`
# (raw_polls House-G-US polls 1998-2022 + VoteHub 2024-12+; the 2024 cycle window has no
# machine-readable per-poll source -> those months are absent = NaN features, documented).
GENERIC_CSV = "data/generic_ballot_monthly.csv"

def fetch(provider, dataset, code, timeout=45):
    u = f"{BASE}/{provider}/{dataset}/{code}?observations=1"
    r = requests.get(u, timeout=timeout, headers=H)
    docs = r.json().get("series", {}).get("docs", [])
    if not docs:
        raise RuntimeError(f"no data ({r.status_code})")
    d = docs[0]
    s = pd.Series(d["value"], index=pd.PeriodIndex(d["period"], freq="M").to_timestamp()
                  if len(d["period"][0]) == 7 else pd.to_datetime(d["period"]))
    return pd.to_numeric(s, errors="coerce").dropna()

# DBnomics' BLS mirror lags (it stopped at 2025-01 as of 2026-07), so BLS series get an
# overlay of the last ~10 years straight from the BLS public API (no key; 25 req/day cap,
# one request covers all our BLS series). API data wins where the two overlap.
def fetch_bls_recent(series_ids, years_back=9, timeout=60):
    end = pd.Timestamp.now().year
    r = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                      json={"seriesid": list(series_ids),
                            "startyear": str(end - years_back), "endyear": str(end)},
                      headers={"Content-Type": "application/json", **H}, timeout=timeout)
    j = r.json()
    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API: {j.get('status')} {j.get('message')}")
    out = {}
    for s in j["Results"]["series"]:
        rows = []
        for d in s["data"]:
            if not d["period"].startswith("M") or d["period"] == "M13":
                continue
            v = pd.to_numeric(d["value"], errors="coerce")   # '-' = BLS missing placeholder
            if pd.notna(v):
                rows.append((pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), float(v)))
        out[s["seriesID"]] = pd.Series(dict(rows)).sort_index()
    return out

# UMich consumer sentiment: same story as BLS, worse. DBnomics' mirror of SCSMICH/MICS/ICS
# stopped advancing at 2025-08 (11 months stale as of 2026-08-08, while every other series in
# this file ran to 2026-06/07), which made `sentiment_last12_delta` 100% NaN at serve time -
# the 12-month window simply could not be filled. See CONCERNS #30.
#
# FRED's UMCSENT is the SAME University of Michigan Index of Consumer Sentiment, current, and
# needs no API key (the fredgraph CSV export is public). Verified 2026-08-08 against the
# existing series: 664 overlapping months, max absolute difference EXACTLY 0.0.
#
# Overlay, not replacement - deliberately. Our file has 302 pre-1978 rows that FRED's monthly
# series does not carry (UMich surveyed quarterly before 1978), so a replace would silently
# drop 25 years of history. combine_first keeps ours and adds FRED's newer months.
FRED_SENTIMENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UMCSENT"

# FRED also mirrors every BLS-backed series we use, which makes it a complete fallback for the
# BLS API - not just a sentiment source. Added 2026-08-08 after the BLS API went "Temporarily
# Down for Maintenance" (HTTP 503) and the DBnomics BLS mirror was ~18 months behind, which
# would have rolled cpi/cpi_core/unemp_u6/unemployment back to 2025-01 at live-scoring time.
#
# Verified 2026-08-08 against the committed values, whole overlapping history:
#   unemployment 942 months, cpi 953, cpi_core 833, unemp_u6 390, fed_funds 865, sentiment 674
#   -> max absolute difference 0.0000 on ALL of them.
# (gas is weekly at FRED, so it is resampled to a monthly mean here; that matches our stored
#  values to 0.0025 - pure rounding - and reaches 3 months FURTHER than the DBnomics series.)
#
# Order of preference per series: BLS API (authoritative, freshest) -> FRED -> DBnomics mirror.
FRED_SERIES = {
    "unemployment": "UNRATE",
    "cpi":          "CPIAUCSL",
    "cpi_core":     "CPILFESL",
    "unemp_u6":     "U6RATE",
    "gas":          "GASREGW",     # WEEKLY -> resampled to monthly mean below
    "fed_funds":    "FEDFUNDS",
}
FRED_WEEKLY = {"gas"}
FRED_CACHE_DIR = "data/fred_cache"

def fetch_fred_series(sid, timeout=60, attempts=4, cache_path=None, pace=1.5):
    """One FRED series as a Series. Retries, paces, and falls back to a committed cache.

    FRED resets the connection on rapid-fire sequential requests (seen 2026-08-08: six
    back-to-back pulls, the second one killed with WinError 10054), so callers must space
    them out - hence `pace`, applied by the caller between series.
    """
    last = None
    for i in range(attempts):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                             timeout=timeout, headers=H)
            r.raise_for_status()
            s = _parse_umcsent(r.text)          # same 2-column observation_date,<ID> shape
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    s.rename(sid).rename_axis("observation_date").to_csv(cache_path)
                except Exception as ce:
                    print(f"  (could not cache {sid}: {ce})")
            return s
        except Exception as e:                  # noqa: PERF203 - retry loop
            last = e
            if i < attempts - 1:
                time.sleep(3 * (i + 1))
    if cache_path and os.path.exists(cache_path):
        s = _parse_umcsent(open(cache_path, encoding="utf-8").read())
        print(f"  FRED {sid} unreachable - using cache (through {s.index.max().date()})")
        return s
    raise RuntimeError(f"FRED {sid} unavailable after {attempts} attempts: {last}")

def fetch_fred_all(series=None, pace=1.5):
    """{metric: Series} for every FRED-mirrored metric. Never raises - a metric that fails is
    simply absent, and the caller falls back to BLS/DBnomics (and then to the no-drop guard)."""
    out = {}
    for metric, sid in (series or FRED_SERIES).items():
        try:
            s = fetch_fred_series(sid, cache_path=os.path.join(FRED_CACHE_DIR, f"{sid}.csv"))
            if metric in FRED_WEEKLY:
                s = s.resample("MS").mean()
            out[metric] = s
            print(f"  FRED {metric:14} {sid:9} {len(s):5} obs, latest {s.index.max().date()}")
        except Exception as e:
            print(f"  FRED {metric:14} {sid:9} unavailable ({e})")
        time.sleep(pace)
    return out

FRED_SENTIMENT_CACHE = "data/umcsent_fred.csv"

def _parse_umcsent(text):
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]                          # observation_date,UMCSENT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # '.' = FRED missing marker
    s = df.dropna().set_index("date")["value"].sort_index()
    if s.empty:
        raise RuntimeError("no usable rows")
    return s

def fetch_fred_sentiment(timeout=60, attempts=3, use_cache=True):
    """UMich Index of Consumer Sentiment from FRED (no API key). Returns a monthly Series.

    Retries, then falls back to a COMMITTED CACHE at data/umcsent_fred.csv.

    Why the cache: FRED is intermittently unreachable from some networks. Measured
    2026-08-08 within a single hour - three read timeouts, then 10/10 successes averaging
    0.7s, then more timeouts. The series itself only changes once a month, so re-fetching a
    73-year history on every run is both wasteful and the least reliable part of this script.
    A successful fetch REFRESHES the cache; a failed one reads it. That also makes the whole
    macro build reproducible offline, which is this repo's stated principle for every other
    static input (see README "All static data is pulled once and committed").
    """
    last = None
    for i in range(attempts):
        try:
            r = requests.get(FRED_SENTIMENT_URL, timeout=timeout, headers=H)
            r.raise_for_status()
            s = _parse_umcsent(r.text)
            try:                                            # refresh the cache on success
                os.makedirs(os.path.dirname(FRED_SENTIMENT_CACHE), exist_ok=True)
                s.rename("UMCSENT").rename_axis("observation_date").to_csv(FRED_SENTIMENT_CACHE)
            except Exception as ce:
                print(f"  (could not update {FRED_SENTIMENT_CACHE}: {ce})")
            return s
        except Exception as e:                              # noqa: PERF203 - retry loop
            last = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    if use_cache and os.path.exists(FRED_SENTIMENT_CACHE):
        s = _parse_umcsent(open(FRED_SENTIMENT_CACHE, encoding="utf-8").read())
        print(f"  FRED unreachable ({type(last).__name__}) - using committed cache "
              f"{FRED_SENTIMENT_CACHE} (through {s.index.max().date()})")
        return s
    raise RuntimeError(f"FRED UMCSENT unavailable after {attempts} attempts and no cache: {last}")

# Presidential approval now comes from data/approval_monthly.csv, produced by
# fetch_approval.py (Gallup via UCSB American Presidency Project, 1993->present).
# Run `python fetch_approval.py` first if that file is missing.
APPROVAL_CSV = "data/approval_monthly.csv"

def build():
    os.makedirs("data", exist_ok=True)
    bls_codes = [code for (prov, ds, code, f) in SERIES.values() if prov == "BLS"]
    try:
        bls_recent = fetch_bls_recent(bls_codes)
        print(f"  BLS API overlay: {len(bls_recent)} series, latest "
              f"{max(s.index.max() for s in bls_recent.values()).date()}")
    except Exception as e:
        bls_recent = {}
        print(f"  BLS API overlay unavailable ({e}) - DBnomics only")

    try:
        fred_sent = fetch_fred_sentiment()
        print(f"  FRED UMCSENT overlay: {len(fred_sent)} months, latest "
              f"{fred_sent.index.max().date()}")
    except Exception as e:
        fred_sent = None
        print(f"  FRED UMCSENT overlay unavailable ({e}) - DBnomics only (likely STALE)")

    # FRED mirrors of the BLS-backed series. These matter most exactly when the BLS API is
    # down: without them, those metrics fall back to the DBnomics mirror, which is ~18 months
    # behind and would silently roll the live-scoring macro block back to 2025-01.
    #
    # Only fetched for metrics BLS did not already supply - BLS is authoritative and usually
    # fresher, and skipping the redundant pulls keeps us well clear of FRED's rate limiting.
    _bls_by_metric = {m: code for m, (prov, ds, code, f) in SERIES.items() if prov == "BLS"}
    _need = {m: sid for m, sid in FRED_SERIES.items()
             if _bls_by_metric.get(m) not in bls_recent}
    fred_all = fetch_fred_all(_need) if _need else {}
    if not _need:
        print("  FRED mirrors not needed - BLS API supplied every series it covers")

    frames, ok, skipped = [], [], []
    for metric, (prov, ds, code, freq) in SERIES.items():
        try:
            s = fetch(prov, ds, code)
            # Precedence: BLS API (authoritative) > FRED mirror > DBnomics. Each combine_first
            # only ADDS months the lower-priority source lacks, so long history is never lost.
            if metric in fred_all:
                s = fred_all[metric].combine_first(s)
            if code in bls_recent:                     # fresh months win over the lagged mirror
                s = bls_recent[code].combine_first(s)
            if metric == "sentiment" and fred_sent is not None:
                # FRED's current months win; our pre-1978 quarterly-era rows are kept
                s = fred_sent.combine_first(s)
            s = s.resample("MS").ffill() if freq in ("Q", "A") else s.resample("MS").mean()
            if START is not None:
                s = s[s.index >= START]
            if s.empty:
                raise RuntimeError("empty after window")
            out = s.reset_index(); out.columns = ["date", "value"]; out["metric"] = metric
            frames.append(out); ok.append(metric)
            print(f"  OK   {metric:14} {len(out)} months  ({s.index.min().date()}..{s.index.max().date()})")
        except Exception as e:
            # sentiment has a SECOND, independent source. If DBnomics fails (its usual
            # failure is a read timeout) but FRED came back, use FRED alone rather than
            # skipping the metric - the two are the identical UMich index.
            # These metrics have an INDEPENDENT source. If DBnomics fails (its usual failure is
            # a read timeout) but FRED came back, use FRED alone rather than skipping - the
            # values are verified identical where the two overlap.
            alt = fred_sent if metric == "sentiment" else fred_all.get(metric)
            if alt is not None:
                s = alt.resample("MS").mean()
                if code in bls_recent:                 # BLS still wins if we have it
                    s = bls_recent[code].combine_first(s)
                if START is not None:
                    s = s[s.index >= START]
                out = s.reset_index(); out.columns = ["date", "value"]; out["metric"] = metric
                frames.append(out); ok.append(metric)
                print(f"  OK   {metric:14} {len(out)} months  ({s.index.min().date()}.."
                      f"{s.index.max().date()})  <- FRED only, DBnomics failed ({e})")
                continue
            skipped.append(metric); print(f"  SKIP {metric:14} {prov}/{ds}/{code}  ({e})")

    if not os.path.exists(APPROVAL_CSV):
        raise FileNotFoundError(f"{APPROVAL_CSV} not found - run `python fetch_approval.py` first")
    ap = pd.read_csv(APPROVAL_CSV, parse_dates=["date"])
    if START is not None:
        ap = ap[ap["date"] >= START]
    frames.append(ap); ok.append("approval")
    print(f"  OK   approval       {len(ap)} months  ({ap['date'].min().date()}..{ap['date'].max().date()})  <- {APPROVAL_CSV}")

    if os.path.exists(GENERIC_CSV):
        gb = pd.read_csv(GENERIC_CSV, parse_dates=["date"])
        if START is not None:
            gb = gb[gb["date"] >= START]
        frames.append(gb); ok.append("generic_ballot")
        print(f"  OK   generic_ballot {len(gb)} months  ({gb['date'].min().date()}..{gb['date'].max().date()})  <- {GENERIC_CSV}")
    else:
        print(f"  SKIP generic_ballot ({GENERIC_CSV} missing - run `python fetch_generic_ballot.py --monthly`)")

    allm = pd.concat(frames, ignore_index=True).sort_values(["metric", "date"])

    # ------------------------------------------------------------------ NO-SILENT-DROP GUARD
    # Added 2026-08-08. Before this, a SKIPPED series was simply absent from `frames` and the
    # file was overwritten anyway - so a transient network blip DELETED that metric's entire
    # history. That is exactly what happened on 2026-08-08: a DBnomics read timeout wiped the
    # `sentiment` metric, taking 16 model features with it. Nothing failed loudly; the only
    # thing that caught it was predict.py's artifact-feature assert refusing to run.
    #
    # A fetch failure must never be able to destroy data we already have. Carry the previous
    # file's rows forward for any metric we could not fetch this run, and say so.
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, parse_dates=["date"])
        lost = sorted(set(prev["metric"].unique()) - set(allm["metric"].unique()))
        if lost:
            carried = prev[prev["metric"].isin(lost)]
            allm = (pd.concat([allm, carried], ignore_index=True)
                      .sort_values(["metric", "date"]))
            print(f"\n  !! CARRIED FORWARD {len(lost)} metric(s) this run could not fetch: "
                  f"{lost}")
            for m_ in lost:
                mx = carried.loc[carried["metric"] == m_, "date"].max()
                print(f"     {m_:14} kept {int((carried['metric'] == m_).sum())} existing rows "
                      f"(through {mx.date()}) - STALE, not refreshed")
            print("     Re-run when the source is back; the data was NOT deleted.")

        # Second guard: a metric that came back but SHRANK is also suspicious (a partial or
        # truncated response). Warn loudly rather than silently accepting fewer rows.
        for m_ in sorted(set(allm["metric"]) & set(prev["metric"])):
            n_new = int((allm["metric"] == m_).sum())
            n_old = int((prev["metric"] == m_).sum())
            if n_new < n_old:
                print(f"  !! {m_} SHRANK {n_old} -> {n_new} rows - check the source before "
                      f"trusting this file")

    allm.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}  ({len(allm)} rows)")
    print(f"metrics OK ({len(ok)}): {ok}")
    if skipped:
        print(f"skipped ({len(skipped)}): {skipped}  (previous values carried forward if the "
              f"file already had them - see the guard above)")

if __name__ == "__main__":
    build()
