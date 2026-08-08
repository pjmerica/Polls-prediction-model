"""ONE command to refresh every model input feed, re-predict 2026, and update the dashboard.

    python refresh_dashboard.py            # full refresh (feeds + predictions + dashboard)
    python refresh_dashboard.py --no-feeds # skip macro/approval re-pull (just re-predict)

What each variable's feed is, and how it refreshes here:
  polls (2026)      polling-agg repo, auto-refreshed by ITS GitHub Actions (2x daily + market
                    refresh) — nothing to do locally; this script just consumes the CSVs.
  markets           same polling-agg Actions; the Actions also re-run model_compare.py so the
                    dashboard's market side refreshes even when predictions don't change.
  economy           fetch_macro.py — BLS public API overlay (current to last month) + DBnomics
                    history. Re-pulled by this script.
  approval          fetch_approval.py — Gallup via UCSB. Re-pulled; the Trump-2nd-term page
                    404s as of 2026-07 (script auto-picks it up whenever UCSB posts it).
  generic ballot    fetched live inside predict*.py (Wikipedia aggregator mean).
  incumbency        frozen data/races.csv (already covers 2026) — static, nothing to refresh.
  prior margins     frozen 2024 results — static until the 2026 results exist (MEDSL, later).
  house effects     recomputed from committed history inside predict*.py — static.

After predicting, the CSVs are copied into the polling-agg repo (data/processed/model_*.csv)
and model_compare.py regenerates docs/model_data.js. COMMIT + PUSH the polling-agg repo to
publish (this script prints the exact commands; it does not push for you).
"""
import argparse
import os
import shutil
import subprocess
import sys

import os as _os, sys as _sys  # noqa: E402  - bootstrap: this file lives in src/,
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ...so the repo ROOT (which holds paths.py) must go on sys.path before importing it.
from paths import ROOT, AGG, SRC   # noqa: F401  (one definition for the whole repo)
import paths as _paths   # module handle: `out` is a very common local variable name here

HERE = ROOT

# Script locations. Fetch steps live in pipeline/fetch/; the predict + explain entrypoints
# moved from the repo root into src/ on 2026-08-08. Paths are built with os.path.join so this
# keeps working on POSIX runners as well as Windows.
#
# This file itself now lives in src/ too. It is still invoked as `python src/refresh_dashboard.py`
# from the repo root - polling-agg's model-refresh.yml runs it with working-directory set to
# the model repo, and that workflow was updated to the src/ path in the same commit. A
# root-level refresh_dashboard.py shim is kept as a fallback so an OLD workflow checkout (or
# anyone's muscle memory) still works instead of failing at 13:15 UTC.
_FETCH = os.path.join(ROOT, "pipeline", "fetch")

STEPS_FEEDS = [
    ([sys.executable, os.path.join(_FETCH, "fetch_approval.py")],
     "approval (Gallup/UCSB + VoteHub)"),
    ([sys.executable, os.path.join(_FETCH, "fetch_generic_ballot.py"), "--monthly"],
     "generic ballot monthly (538 hist + VoteHub)"),
    ([sys.executable, os.path.join(_FETCH, "fetch_macro.py")],
     "economy incl. sentiment (BLS API + DBnomics) + merge"),
]
STEPS_PREDICT = [
    ([sys.executable, os.path.join(SRC, "predict.py")], "win probabilities"),
    ([sys.executable, os.path.join(SRC, "predict_margin.py")], "margins"),
    ([sys.executable, os.path.join(SRC, "predict_primary.py")], "primary nominee probabilities"),
    ([sys.executable, os.path.join(SRC, "predict_primary_margin.py")], "primary margins"),
    ([sys.executable, os.path.join(SRC, "explain_2026.py")],
     "SHAP explanations (writes polling-agg copy itself)"),
    ([sys.executable, os.path.join(SRC, "explain_primary.py")],
     "primary SHAP explanations (writes polling-agg copy itself)"),
]

def run(cmd, label, cwd=HERE):
    print(f"\n=== {label}: {' '.join(os.path.basename(c) for c in cmd)} ===")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {label} (exit {r.returncode})")

def check_feed_freshness():
    """Loud staleness warnings — a dead upstream (VoteHub, DBnomics, BLS) fails SOFTLY in
    the fetch scripts, so a stale feed looks identical to a working one without this."""
    import pandas as pd
    today = pd.Timestamp.now()
    checks = [  # (file, metric filter or None, max acceptable months of lag)
        ("data/approval_monthly.csv", None, 2),
        ("data/generic_ballot_monthly.csv", None, 2),
        ("data/macro_monthly.csv", "unemployment", 2),
        ("data/macro_monthly.csv", "sentiment", 13),   # DBnomics mirror lags ~1yr (known)
    ]
    warned = False
    for path, metric, max_lag in checks:
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            print(f"!! FEED MISSING: {path}"); warned = True; continue
        df = pd.read_csv(p, parse_dates=["date"])
        if metric is not None:
            df = df[df["metric"] == metric]
        lag = (today.year - df["date"].max().year) * 12 + (today.month - df["date"].max().month)
        label = f"{path}" + (f" [{metric}]" if metric else "")
        if lag > max_lag:
            print(f"!! FEED STALE: {label} ends {df['date'].max().date()} "
                  f"({lag} months old, allowed {max_lag}) — upstream may have died silently")
            warned = True
    if not warned:
        print("feed freshness: all OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-feeds", action="store_true",
                    help="skip macro/approval re-pull; just re-predict + refresh dashboard")
    args = ap.parse_args()

    if not args.no_feeds:
        for cmd, label in STEPS_FEEDS:
            run(cmd, label)
    check_feed_freshness()
    for cmd, label in STEPS_PREDICT:
        run(cmd, label)

    for src, dst in [("predictions_2026.csv", "model_predictions_2026.csv"),
                     ("predictions_2026_meta.json", "model_predictions_meta.json"),
                     ("margin_predictions_2026.csv", "model_margin_predictions_2026.csv"),
                     ("primary_predictions_2026.csv", "model_primary_predictions_2026.csv"),
                     ("primary_predictions_2026_meta.json", "model_primary_predictions_meta.json"),
                     ("primary_margin_predictions_2026.csv",
                      "model_primary_margin_predictions_2026.csv")]:
        shutil.copyfile(_paths.out(src),
                        os.path.join(AGG, "data", "processed", dst))
        print(f"copied {src} -> polling-agg/data/processed/{dst}")
    # timestamp sidecar: CI checkouts reset file mtimes, so the dashboard's model-staleness
    # display needs an explicit record of when predictions were actually generated
    import datetime
    with open(os.path.join(AGG, "data", "processed", "model_predictions_as_of.txt"), "w") as f:
        f.write(datetime.datetime.now().isoformat(timespec="seconds"))

    run([sys.executable, os.path.join("analysis", "model_compare.py")],
        "model vs markets page data", cwd=AGG)
    run([sys.executable, os.path.join("analysis", "model_compare_primary.py")],
        "PRIMARY vs markets page data", cwd=AGG)

    print("\nDone. To publish the dashboard:")
    print(f'  cd "{os.path.abspath(AGG)}"')
    print('  git add data/processed/model_*.csv data/processed/model_*.json'
          ' docs/model_data.js docs/primary_model_data.js')
    print('  git commit -m "model predictions refresh" && git push')

if __name__ == "__main__":
    main()
