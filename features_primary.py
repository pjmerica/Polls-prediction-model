# -*- coding: utf-8 -*-
"""Primary-election feature pipeline: long primary-poll rows -> one row per candidate per
primary race. Shared by training (primary_model.py) and prediction (predict_primary.py) -
same never-fork rule as features.py.

Race = (cycle, state, office[, district], party): a WITHIN-PARTY multi-candidate contest.
Design notes vs the general pipeline (features.py):
- All poll aggregates are plain means of raw polls (same no-weighting rule).
- Poll recency is measured against the PRIMARY's own election_date, which every historical
  row carries (verified: real primary dates incl. moved ones; days_to_elec via
  F.prepare_polls on the long frame).
- No natl_env / bias priors / house effects: within-party races have no partisan channel.
- Macro is OPTIONAL (build_macro_asof windows keyed to each primary's date, publication-lag
  guarded). Default OFF: ~200 training races cannot support 144 macro columns; the
  training script runs the ablation and documents it.
- Fundraising: fund_share is recomputed WITHIN the primary field (that party's candidates
  only) - the general pipeline's race-wide share would leak the other party's money into
  the denominator.
- is_defending_party: this party currently holds the seat (races.csv incumbent_party).
  True candidate-level incumbency isn't derivable from committed data (no incumbent name);
  documented gap.
"""
import numpy as np
import pandas as pd

import features as F
from cycles import PRES_PARTY

def build_primary_table(d, fec=None, inc_map=None, macro_asof=None):
    """d: prepared long frame (F.prepare_polls applied) with columns
    race_id, year, state, office, district, party_std, candidate, cand_key, pct, end_date,
    days_to_elec, sample_size, pollster [, won].
    macro_asof: optional callable date -> {feature: value} (macro_features.build_macro_asof);
    evaluated once per unique election_date and merged onto every candidate row.
    """
    lead_changes = {rid: F.count_lead_changes(g) for rid, g in d.groupby("race_id")}
    margin_dyn = {rid: F.margin_dynamics(g) for rid, g in d.groupby("race_id")}
    has_won = "won" in d.columns

    macro_cache = {}
    def macro_for(ed):
        if macro_asof is None or pd.isna(ed):
            return {}
        k = pd.Timestamp(ed).normalize()
        if k not in macro_cache:
            macro_cache[k] = macro_asof(k)
        return macro_cache[k]

    rows = []
    for race_id, g in d.groupby("race_id"):
        yr = int(g["year"].iloc[0]); st = g["state"].iloc[0]
        of = g["office"].iloc[0]; di = F.dist_str(g["district"].iloc[0])
        party = g["party_std"].iloc[0]
        ed = g["election_date"].dropna().max()
        incp = inc_map.get((yr, st, of, di)) if inc_map else None
        dyn = margin_dyn.get(race_id, {})
        for ck, gc in g.groupby("cand_key"):
            gc = gc.sort_values("end_date")
            dated = gc.dropna(subset=["end_date"])
            last30 = gc[gc["days_to_elec"] <= 30]
            last60 = gc[gc["days_to_elec"] <= 60].dropna(subset=["pct", "days_to_elec"])
            slope = np.nan
            if len(last60) >= 3:
                x = -last60["days_to_elec"].values.astype(float)
                y = last60["pct"].values.astype(float)
                if np.ptp(x) > 0:
                    slope = np.polyfit(x, y, 1)[0]
            fe = fec.get((yr, st, of, di, ck)) if fec is not None else None
            rec = fe["receipts"] if fe else np.nan
            md = dyn.get(ck, {})
            rows.append(dict(
                race_id=race_id, year=yr, state=st, office=of, district=di, party=party,
                cand_key=ck, candidate=gc["candidate"].iloc[0],
                election_date=ed,
                won=(int(gc["won"].iloc[0]) if has_won and pd.notna(gc["won"].iloc[0]) else np.nan),
                poll_avg=gc["pct"].mean(),
                poll_last=(dated["pct"].iloc[-1] if len(dated) else gc["pct"].mean()),
                poll_last30=(last30["pct"].mean() if len(last30) else gc["pct"].mean()),
                poll_std=gc["pct"].std(),
                n_polls=len(gc),
                avg_sample=gc["sample_size"].mean(),
                min_days=gc["days_to_elec"].min(),
                poll_momentum=slope,
                n_lead_changes=lead_changes.get(race_id, 0),
                avg_margin_over_time=md.get("avg_margin_over_time", np.nan),
                margin_volatility=md.get("margin_volatility", np.nan),
                min_margin=md.get("min_margin", np.nan),
                margin_trend=md.get("margin_trend", np.nan),
                is_dem_primary=int(party == "DEM"),
                is_senate=int(of == "Senate"), is_gov=int(of == "Governor"),
                # the party defends this seat (races.csv incumbent_party); unknown = NaN
                is_defending_party=((1 if incp == party else 0)
                                    if incp in ("DEM", "REP") else np.nan),
                is_pres_party=int(party == PRES_PARTY.get(yr)),
                _fund_receipts=(rec if fe else np.nan),
                fund_receipts_ln=(np.log1p(rec) if fe and rec and rec > 0 else np.nan),
                **macro_for(ed),
            ))
    c = pd.DataFrame(rows)

    # within-FIELD relatives (the field = this party's candidates = the race group)
    c["field_best"] = c.groupby("race_id")["poll_avg"].transform(
        lambda s: s.nlargest(2).min() if len(s) > 1 else s.max())
    c["poll_lead"] = c["poll_avg"] - c["field_best"]
    c["poll_share"] = c["poll_avg"] / c.groupby("race_id")["poll_avg"].transform("sum")
    c["n_cands"] = c.groupby("race_id")["cand_key"].transform("count")
    c["race_total_polls"] = c.groupby("race_id")["n_polls"].transform("sum")
    c["undecided"] = (100 - c.groupby("race_id")["poll_avg"].transform("sum")).clip(lower=0)
    c["gap_x_recency"] = c["poll_lead"] * (1.0 / (1.0 + c["min_days"].clip(lower=0) / 30.0))
    tot = c.groupby("race_id")["_fund_receipts"].transform("sum")
    c["fund_share"] = np.where(tot > 0, c["_fund_receipts"] / tot, np.nan)
    c = c.drop(columns="_fund_receipts")
    return c

def feature_list_primary(macro_feats=(), fund=False):
    """Model inputs. Everything is computable for a future primary from the raw feed +
    committed statics.

    fund=False is the ARTIFACT default (2026-07-15): FEC receipts are CYCLE-END totals,
    and nominees raise most of their money AFTER winning the primary - so fund_share
    partially encodes the training label (leakage) while a mid-cycle 2026 candidate has
    no such money yet (train/serve skew). Measured: dropping fund leaves race-acc
    IDENTICAL (.895) and costs only Brier .035->.046 - the picks never depended on it.
    Revisit only with as-of-primary-date FEC reports. macro_feats: ablation-only."""
    return [
        "poll_avg", "poll_last", "poll_last30", "poll_std", "n_polls", "avg_sample",
        "min_days", "poll_momentum",
        "poll_lead", "poll_share", "n_cands", "race_total_polls", "undecided",
        "gap_x_recency",
        "n_lead_changes", "avg_margin_over_time", "margin_volatility", "min_margin",
        "margin_trend",
        "is_dem_primary", "is_senate", "is_gov", "is_defending_party", "is_pres_party",
    ] + (["fund_receipts_ln", "fund_share"] if fund else []) + list(macro_feats)
