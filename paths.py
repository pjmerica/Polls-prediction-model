# -*- coding: utf-8 -*-
"""THE single source of truth for where things live on disk.

Added 2026-08-02 with the folder reorganisation. Before it, 26 scripts each computed
`HERE = os.path.dirname(os.path.abspath(__file__))` and then resolved `data/` and the sibling
polling-agg repo relative to THEMSELVES. That works only while every script sits in the repo
root - move one into a subfolder and it silently reads from the wrong place (or, worse,
writes there). This module resolves everything from the REPO ROOT instead, so a file can live
at any depth.

Usage (replaces the old HERE idiom):

    from paths import ROOT, DATA, AGG, data, agg
    df = pd.read_csv(data("primary_polls_long.csv"))
    out = agg("data", "processed", "model_predictions_2026.csv")

Scripts in subfolders must be able to `import paths` at all, which means the repo root has to
be on sys.path. Importing this module puts it there as a side effect, so the standard header
for a subfolder script is:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import data          # everything else resolves from here

Root-level scripts and notebooks need no such prelude.
"""
import os
import sys

# This file lives in the repo root, so ROOT is simply its own directory.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

# The sibling polling-agg repo (dashboard + raw poll/market feeds). Kept as ONE definition
# because 11 scripts used to spell this out with their own "..", each of which would break at
# a different depth. Overridable for CI/testing via POLLING_AGG_DIR.
AGG = os.environ.get(
    "POLLING_AGG_DIR",
    os.path.join(os.path.dirname(ROOT), "Polling Agg", "Polling agg and Prediction markets"))

# Ensure `import features` etc. work from any subfolder (see docstring).
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The pipeline folders are also importable BY BARE MODULE NAME. Several scripts import each
# other directly - features.py imports fetch_candidate_bios_ballotpedia, build_office_level_table
# imports fetch_candidate_bios, fetch_house_primary_results_hist imports fetch_primary_results_2026 -
# and rewriting all of those into package-qualified imports would mean turning every folder into
# a package and touching far more code than the reorganisation warrants. Putting the folders on
# the path keeps `import fetch_candidate_bios` working from anywhere, exactly as it did when
# every file sat in the repo root.
for _sub in ("pipeline/fetch", "pipeline/build", "models/poll", "models/fundamentals", "tools"):
    _p = os.path.join(ROOT, *_sub.split("/"))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)


def root(*parts):
    """Path inside the repo root."""
    return os.path.join(ROOT, *parts)


def data(*parts):
    """Path inside data/."""
    return os.path.join(DATA, *parts)


def agg(*parts):
    """Path inside the sibling polling-agg repo."""
    return os.path.join(AGG, *parts)
