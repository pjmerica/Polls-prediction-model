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

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "..", "Polling Agg", "Polling agg and Prediction markets")

STEPS_FEEDS = [
    ([sys.executable, "fetch_approval.py"], "approval (Gallup/UCSB)"),
    ([sys.executable, "fetch_macro.py"], "economy (BLS API + DBnomics)"),
]
STEPS_PREDICT = [
    ([sys.executable, "predict.py"], "win probabilities"),
    ([sys.executable, "predict_margin.py"], "margins"),
]

def run(cmd, label, cwd=HERE):
    print(f"\n=== {label}: {' '.join(os.path.basename(c) for c in cmd)} ===")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {label} (exit {r.returncode})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-feeds", action="store_true",
                    help="skip macro/approval re-pull; just re-predict + refresh dashboard")
    args = ap.parse_args()

    if not args.no_feeds:
        for cmd, label in STEPS_FEEDS:
            run(cmd, label)
    for cmd, label in STEPS_PREDICT:
        run(cmd, label)

    for src, dst in [("predictions_2026.csv", "model_predictions_2026.csv"),
                     ("margin_predictions_2026.csv", "model_margin_predictions_2026.csv")]:
        shutil.copyfile(os.path.join(HERE, src),
                        os.path.join(AGG, "data", "processed", dst))
        print(f"copied {src} -> polling-agg/data/processed/{dst}")

    run([sys.executable, os.path.join("analysis", "model_compare.py")],
        "model vs markets page data", cwd=AGG)

    print("\nDone. To publish the dashboard:")
    print(f'  cd "{os.path.abspath(AGG)}"')
    print('  git add data/processed/model_*.csv docs/model_data.js')
    print('  git commit -m "model predictions refresh" && git push')

if __name__ == "__main__":
    main()
