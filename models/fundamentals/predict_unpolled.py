"""Win probabilities for the 2026 races NO pollster covers -> outputs/fundamentals_predictions_2026.csv.
2026-10-23 (user: "every race should be scored by the model").

The poll model (src/predict.py) scores only polled races: 173 of the 506 on the ballot. This
scores the other 333 with the no-polling model (models/fundamentals/fundamentals_model.py:
incumbency, seat history, national environment, fundraising, candidate office level, macro),
which the 2026-08-02 head-to-head kept OUT of polled races because polls beat it everywhere
they exist (CONCERNS #28). Here there is nothing to beat: it is the model's estimate or none.

Same feature builder as predict.py, run on one synthetic row per nominee with blank poll fields
(the method of analysis/fundamentals_on_unpolled.py); same national environment as the poll
run that just finished (outputs/predictions_2026_meta.json), the same party overrides and
dropped-out list, and the same redistricting patch (new-map Cook PVI for the 2025-26 redraws).
Fragile = the pick flips if the national environment moves 3 points either way.

Roster: data/general_roster_2026.csv (pipeline/fetch/fetch_general_roster_2026.py).

    python models/fundamentals/predict_unpolled.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from paths import ROOT  # noqa: E402

import datetime  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

import features as F  # noqa: E402
from cycles import natl_env as natl_env_hist  # noqa: E402
from macro_features import build_macro  # noqa: E402
from predict import patch_redistricted_priors  # noqa: E402

CYCLE = 2026
OUT = _os.path.join(ROOT, "outputs", "fundamentals_predictions_2026.csv")


def roster(skip):
    r = pd.read_csv(_os.path.join(ROOT, "data", "general_roster_2026.csv"), dtype={"district": str})
    r["district"] = r["district"].fillna("")
    r = r[(r["dropped"] == 0) & (r["write_in"] == 0) & ~r["race_id"].isin(skip)].copy()
    r["party_std"] = r["party"].map(F.npar)
    r["display_party"] = r["party"].where(r["party"].isin(["DEM", "REP", "IND"]), "OTH")
    # the same live overrides predict.py applies (e.g. Osborn: modelled in the DEM slot)
    ov = pd.read_csv(_os.path.join(ROOT, "data", "candidate_party_overrides.csv"))
    for o in ov.itertuples():
        m = (r["race_id"] == o.race_id) & (r["cand_key"] == o.cand_key)
        r.loc[m, "party_std"] = F.npar(o.model_party)
        r.loc[m, "display_party"] = str(o.display_party).upper()[:3]
    do_path = _os.path.join(ROOT, "data", "dropped_out_2026.csv")
    if _os.path.exists(do_path):
        do = pd.read_csv(do_path)
        if {"race_id", "cand_key"} <= set(do.columns):
            gone = set(zip(do["race_id"], do["cand_key"]))
            r = r[[(a, b) not in gone for a, b in zip(r["race_id"], r["cand_key"])]]
    return r


# generic-ballot features that are LEVELS of the D-minus-R margin (a national-mood error moves them;
# std / trend / last12_delta describe its path and do not move)
GB_LEVELS = ("eve", "mean", "max", "min", "avg_3mo", "avg_6mo", "avg_12mo", "max_3mo", "max_6mo",
             "max_12mo")


def macro(gb_shift=0.0):
    m = build_macro(cycles=[CYCLE])
    if gb_shift:
        for k in GB_LEVELS:
            key = f"generic_ballot_{k}"
            if key in m[CYCLE] and m[CYCLE][key] == m[CYCLE][key]:
                m[CYCLE][key] += gb_shift
    return m


def table(r, ne, gb_shift=0.0):
    fake = pd.DataFrame({
        "year": CYCLE, "state": r["state"].values, "office": r["office"].values,
        "district": r["district"].values, "candidate": r["candidate"].values,
        "cand_key": r["cand_key"].values, "party_std": r["party_std"].values,
        "display_party": r["display_party"].values,
        # poll fields blank; the builder groups by poll_id, so give one placeholder per race
        "pct": np.nan, "end_date": pd.NaT, "sample_size": np.nan, "pollster": "none", "won": 0,
        "poll_id": ("nopoll-" + r["race_id"]).values,
        "election_date": pd.Timestamp(f"{CYCLE}-11-03"), "days_to_elec": np.nan,
        "race_id": r["race_id"].values,
    })
    c = F.build_candidate_table(fake, macro(gb_shift), ne, F.load_fundamentals(), house={},
                                fec=F.load_fec(extended=True), bias_priors={},
                                candidate_bios=F.load_candidate_bios(),
                                primary_results=F.load_primary_results())
    # Office level: the model was trained with EVERY candidate coded (0 = no office found in any
    # source, the convention of candidate_bios_manual.csv), so a blank here is an input it never
    # saw. Major-party nominees still blank after every bio source get that same 0. Checked
    # 2026-10-23: the 330 nominees at 0 were re-read from fresh 2026 race pages (49 states), the
    # people's own Wikipedia infoboxes (fetch_wikipedia_person_offices.py, targets = this file's
    # major-party rows at level 0) and Ballotpedia; one real officeholder turned up there (Mike
    # McGuire, CA-1, state senator since 2014) and a handful more via the race pages.
    blank = c["party"].isin(["DEM", "REP"]) & c["bio_office_level"].isna()
    c.loc[blank, "bio_office_level"] = 0.0
    if blank.any():
        print(f"office level: {int(blank.sum())} major-party nominees with no office found -> 0")
    return patch_redistricted_priors(c)


def main():
    meta = json.load(open(_os.path.join(ROOT, "data", "fundamentals_model_general_features.json")))
    FE = meta["features"]
    model = xgb.XGBClassifier()
    model.load_model(_os.path.join(ROOT, "data", "fundamentals_model_general.json"))

    polled = pd.read_csv(_os.path.join(ROOT, "outputs", "predictions_2026.csv"))
    pmeta = json.load(open(_os.path.join(ROOT, "outputs", "predictions_2026_meta.json")))
    ne = dict(natl_env_hist())
    if pmeta.get("natl_env") is not None:
        ne[CYCLE] = float(pmeta["natl_env"])  # the poll run's own number, so both halves agree
    r = roster(set(polled["race_id"]))

    c = table(r, ne)
    missing = [f for f in FE if f not in c.columns]
    assert not missing, f"the model expects features the table lacks: {missing[:8]}"
    c["win_prob"] = model.predict_proba(c.reindex(columns=FE))[:, 1]
    # stress test: the national mood 3 points more Republican / Democratic than measured - the
    # national environment AND every generic-ballot level feature move together (2026-10-24:
    # moving natl_env alone barely registered, since the trees read the mood mostly through the
    # generic-ballot features, so no race ever came out fragile)
    picks = {}
    for label, shift in (("R3", -3.0), ("D3", +3.0)):
        ne2 = dict(ne)
        ne2[CYCLE] = ne.get(CYCLE, 0.0) + shift
        cs = table(r, ne2, gb_shift=shift)
        cs["p"] = model.predict_proba(cs.reindex(columns=FE))[:, 1]
        m = cs.set_index(["race_id", "cand_key"])["p"]
        c[f"win_prob_{label}"] = [m.get((a, b)) for a, b in zip(c["race_id"], c["cand_key"])]
        picks[label] = c.loc[c.groupby("race_id")[f"win_prob_{label}"].idxmax()].set_index("race_id")["cand_key"]
    base = c.loc[c.groupby("race_id")["win_prob"].idxmax()].set_index("race_id")["cand_key"]
    fragile = set()
    for label in ("R3", "D3"):
        fragile |= set(base.index[base != picks[label].reindex(base.index)])
    c["bias_fragile"] = c["race_id"].isin(fragile).astype(int)
    c["win_prob_norm"] = c["win_prob"] / c.groupby("race_id")["win_prob"].transform("sum")
    c["n_polls"] = 0
    c["n_surveys"] = 0
    c["basis"] = "fundamentals"
    for col in ("poll_avg", "poll_lead"):
        if col not in c.columns:
            c[col] = np.nan
    cols = ["race_id", "state", "office", "district", "candidate", "cand_key", "party", "display_party",
            "n_polls", "n_surveys", "poll_avg", "poll_lead", "prior_margin_cand", "is_incumbent",
            "bio_office_level", "fund_share", "win_prob", "win_prob_norm", "win_prob_R3", "win_prob_D3",
            "bias_fragile", "basis"]
    out = c.reindex(columns=cols).sort_values(["race_id", "win_prob"], ascending=[True, False])
    _os.makedirs(_os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    with open(_os.path.splitext(OUT)[0] + "_meta.json", "w") as f:
        json.dump(dict(generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                       natl_env=ne.get(CYCLE), n_races=int(out["race_id"].nunique()),
                       roster=_os.path.join("data", "general_roster_2026.csv"),
                       model="data/fundamentals_model_general.json",
                       # publish gate (tests/test_publish.py): same run as the poll model's files
                       run_id=_os.environ.get("MODEL_RUN_ID"),
                       polled_run_id=pmeta.get("run_id"),
                       feature_fill={f: round(float(c[f].notna().mean()), 4) if f in c else 0.0
                                     for f in FE}), f, indent=1)
    top = out.loc[out.groupby("race_id")["win_prob_norm"].idxmax()]
    print(f"unpolled races scored: {out['race_id'].nunique()} ({dict(top.groupby('office').size())}); "
          f"fragile: {len(fragile)}; natl_env {ne.get(CYCLE):+.1f}")
    print(f"saved -> {OUT}")
    explain(c, model, FE)


def explain(c, model, FE):
    """SHAP top-10 for each race's leading Democrat (or its leader), the same shape as
    src/explain_2026.py writes for the polled races, so the forecast page's "Why?" drawer works
    for every race. Win model only: there is no no-polls margin model."""
    import shap
    from explain_2026 import sigmoid, top_shap
    X = c.reindex(columns=FE)
    sv = shap.TreeExplainer(model)(X)
    races = {}
    for rid, g in c.groupby("race_id"):
        dems = g[g["party"] == "DEM"]
        row = dems.loc[dems["win_prob"].idxmax()] if len(dems) else g.loc[g["win_prob"].idxmax()]
        i = c.index.get_loc(row.name)
        races[rid] = dict(candidate=row["candidate"], party=row.get("display_party", row["party"]),
                          win=top_shap(FE, X.iloc[i].values, sv.values[i], sigmoid(sv.base_values[i]),
                                       row["win_prob"]),
                          margin=None)
    p = _os.path.join(ROOT, "outputs", "fundamentals_explanations_2026.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(cycle=CYCLE, note="No-polls model (models/fundamentals): SHAP top-10 explaining the "
                       "leading Democrat (or the leader if no Democrat), log-odds; base/pred as probabilities.",
                       generated_at=pd.Timestamp.now().isoformat(), races=races), f)
    print(f"explanations: {len(races)} races -> {p}")


if __name__ == "__main__":
    main()
