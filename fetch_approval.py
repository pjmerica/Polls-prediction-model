"""One-time fetch of presidential job approval (Gallup, via UCSB American Presidency Project)
-> data/approval_monthly.csv  [date, value, metric='approval']

Replaces the hand-typed APPROVAL table that used to live in fetch_macro.py (provenance was
weak and it only covered 2016-2024). This pulls the official Gallup series for every president
from Clinton onward, so it covers the pre-2018 training cycles (1998-2016) AND the current
term (for 2026 predictions).

Static-data policy (same as everything else): run once, commit the CSV, never re-pull on a
model run. Re-run only to EXTEND to new months.

    python fetch_approval.py
"""
import io
import requests
import pandas as pd

H = {"User-Agent": "Mozilla/5.0 (research)"}
BASE = "https://www.presidency.ucsb.edu/statistics/data/"

# UCSB page slugs. Trump's second term is a separate page; we try known slug variants and
# keep whichever responds (site slugs have changed before).
PRESIDENTS = {
    "clinton":  ["william-j-clinton-public-approval"],
    "bush43":   ["george-w-bush-public-approval"],
    "obama":    ["barack-obama-public-approval"],
    "trump45":  ["donald-j-trump-public-approval"],
    "biden":    ["joseph-r-biden-public-approval"],
    "trump47":  ["donald-j-trump-2nd-term-public-approval",
                 "donald-j-trump-second-term-public-approval",
                 "donald-trump-2nd-term-public-approval"],
}

def fetch_president(slugs):
    last = None
    for slug in slugs:
        try:
            r = requests.get(BASE + slug, timeout=60, headers=H)
            r.raise_for_status()
            tables = pd.read_html(io.StringIO(r.text))
            t = tables[0]
            # first block of columns is the overall series: Start Date / End Date / Approving
            t = t[["Start Date", "End Date", "Approving"]].dropna()
            t["end"] = pd.to_datetime(t["End Date"], errors="coerce")
            t["value"] = pd.to_numeric(t["Approving"], errors="coerce")
            t = t.dropna(subset=["end", "value"])
            if len(t) == 0:
                raise RuntimeError("empty table")
            return t[["end", "value"]], slug
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    return None, last

def fetch_votehub():
    """Current-term approval polls from VoteHub's open API (all pollsters, per-poll).

    Continuation source for months AFTER the UCSB/Gallup series ends (UCSB has no
    Trump-2nd-term page yet). Methodology note: this is an all-pollster average rather
    than Gallup-only — a small level shift is possible at the seam; the model consumes
    windowed stats (means/trends), which are robust to that.
    """
    r = requests.get("https://api.votehub.com/polls",
                     params={"poll_type": "approval", "subject": "Trump"},
                     timeout=60, headers=H)
    r.raise_for_status()
    rows = []
    for p in r.json():
        app = next((a["pct"] for a in p.get("answers", [])
                    if str(a.get("choice", "")).lower().startswith("approv")), None)
        end = pd.to_datetime(p.get("end_date"), errors="coerce")
        if app is not None and pd.notna(end):
            rows.append((end, float(app)))
    t = pd.DataFrame(rows, columns=["end", "value"])
    return t[t["end"] >= "2025-01-20"]          # second-term polls only

def build():
    frames = []
    for name, slugs in PRESIDENTS.items():
        t, info = fetch_president(slugs)
        if t is None:
            print(f"  SKIP {name:8} ({info})")
            continue
        print(f"  OK   {name:8} {len(t)} polls  {t['end'].min().date()} .. {t['end'].max().date()}  <- {info}")
        frames.append(t)
    ucsb_end = max(f["end"].max() for f in frames)
    try:
        vh = fetch_votehub()
        vh = vh[vh["end"] > ucsb_end]           # only the months UCSB doesn't cover
        if len(vh):
            print(f"  OK   votehub  {len(vh)} polls  {vh['end'].min().date()} .. {vh['end'].max().date()}  <- api.votehub.com")
            frames.append(vh)
    except Exception as e:
        print(f"  SKIP votehub  ({type(e).__name__}: {e})")
    allp = pd.concat(frames, ignore_index=True).sort_values("end")
    # monthly = mean of all Gallup readings whose END date falls in that month
    monthly = (allp.set_index("end")["value"].resample("MS").mean().dropna().round(1))
    out = monthly.reset_index()
    out.columns = ["date", "value"]
    out["metric"] = "approval"
    out.to_csv("data/approval_monthly.csv", index=False)
    print(f"\nsaved -> data/approval_monthly.csv  ({len(out)} months, "
          f"{out['date'].min().date()} .. {out['date'].max().date()})")

if __name__ == "__main__":
    build()
