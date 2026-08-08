# -*- coding: utf-8 -*-
"""COMPATIBILITY SHIM -> src/refresh_dashboard.py

The real orchestrator moved to src/ on 2026-08-08 with the folder reorganisation. This stub
stays at the repo root on purpose.

WHY: polling-agg's `.github/workflows/model-refresh.yml` runs
`python -u refresh_dashboard.py --no-feeds` with `working-directory: 'Polling prediction
model'`. That workflow lives in the OTHER repo and only runs on a schedule (13:15 UTC), so if
the two repos are ever out of step - an old checkout, a revert, a PR that lands one side
first - the run fails in the middle of the night with a FileNotFoundError and the dashboard
silently serves stale predictions until someone notices. A three-line shim removes that whole
failure mode for good, and costs nothing.

Both spellings work and do exactly the same thing:

    py -X utf8 refresh_dashboard.py --no-feeds        # old path (this shim)
    py -X utf8 src/refresh_dashboard.py --no-feeds    # canonical

Do not add logic here. Everything lives in src/refresh_dashboard.py.
"""
import os
import runpy
import sys

_REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "refresh_dashboard.py")

if not os.path.exists(_REAL):
    sys.exit(f"expected the real orchestrator at {_REAL} - did src/ get moved again?")

# run_path executes it as __main__, so its `if __name__ == "__main__"` block fires and
# sys.argv (the --no-feeds flag CI passes) is inherited unchanged.
runpy.run_path(_REAL, run_name="__main__")
