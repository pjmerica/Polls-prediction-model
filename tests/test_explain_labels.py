"""Every feature of a served general-election model has a plain-English label and description in
src/explain_2026.py (2026-10-30: primary_margin, opp_primary_margin and primary_margin_diff were
published to the forecast and map explainers as raw column names with no description).
The primary models label their own features in src/explain_primary.py."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from explain_2026 import describe, friendly  # noqa: E402


def test_every_general_feature_is_explained():
    feats = set()
    for p in glob.glob(os.path.join(ROOT, "data", "*features*.json")):
        if "primary" in os.path.basename(p):
            continue
        with open(p) as f:
            d = json.load(f)
        feats |= set(d if isinstance(d, list) else d.get("features", d.get("feature_names", [])))
    assert feats
    missing = sorted(f for f in feats if friendly(f) == f or not describe(f))
    assert not missing, f"add these to FRIENDLY and DESC in src/explain_2026.py: {missing}"
