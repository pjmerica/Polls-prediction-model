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

from paths import ROOT, AGG          # noqa: F401  (one definition for the whole repo)

HERE = ROOT

# Script locations after the 2026-08-02 reorganisation. The fetch steps live in
# pipeline/fetch/; the predict + explain steps stay at the repo ROOT because they are the
# entrypoints CI and humans call by name. Paths are built with os.path.join so this keeps
# working on POSIX runners as well as Windows.
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
    ([sys.executable, os.path.join(ROOT, "predict.py")], "win probabilities"),
    ([sys.executable, os.path.join(ROOT, "predict_margin.py")], "margins"),
    ([sys.executable, os.path.join(ROOT, "predict_primary.py")], "primary nominee probabilities"),
    ([sys.executable, os.path.join(ROOT, "explain_2026.py")],
     "SHAP explanations (writes polling-agg copy itself)"),
    ([sys.executable, os.path.join(ROOT, "explain_primary.py")],
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
                     ("primary_predictions_2026_meta.json", "model_primary_predictions_meta.json")]:
        shutil.copyfile(os.path.join(HERE, src),
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
