# -*- coding: utf-8 -*-
"""Does the fundamentals model actually work on UNPOLLED races?

    py -X utf8 analysis/fundamentals_on_unpolled.py

WHY. Every previous evaluation of the fundamentals model scored it on races that HAVE polls -
the only population where a head-to-head against the poll model is possible. On that
population it loses everywhere (analysis/fundamentals_vs_polls_thin.py), which is why we do
not blend.

But that population is ~25% of the field. For 2026: 379 of 506 general races (75%) and 731 of
944 primaries (77%) have NO polling at all, and the poll model produces NOTHING for them. There
the comparison is not "fundamentals .811 vs polls .868" - it is "fundamentals .811 vs no
prediction". So the number that matters for those races is the fundamentals model's accuracy
ON UNPOLLED RACES SPECIFICALLY, and that had never been measured.

The worry that motivates measuring rather than assuming: .811 was measured on POLLED races,
which skew competitive and well-documented. Unpolled races are mostly safe seats. That could
cut either way - safe seats are easier to call from incumbency alone (so .811 understates), or
the model could be leaning on something that only holds where pollsters bother to go.

METHOD. The results archives (data/res_{senate,house,governor}.csv) contain EVERY race,
including ones no one polled; polls_long_with_results.csv contains only polled ones. The
difference is our unpolled test set, with real outcomes the model never trained on. We build
the fundamentals feature table for those races from committed statics only (incumbency, prior
margin, bio, macro - no polls anywhere), train expanding-window on POLLED races strictly
before the test cycle, and score.

Honest limitation stated up front: the model is TRAINED on polled races (there is no other
labelled feature table), so this measures transfer from polled to unpolled, not a model built
for unpolled races. If accuracy holds up, the transfer is fine; if it collapses, it does not.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import ROOT  # noqa: E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

import features as F  # noqa: E402
from cycles import CYCLES, natl_env  # noqa: E402
from macro_features import build_macro  # noqa: E402

OFFICE_FILE = {"Senate": "res_senate.csv", "House": "res_house.csv",
               "Governor": "res_governor.csv"}


def results_universe():
    """Every general-election race in the results archives -> one row per candidate."""
    frames = []
    for office, fn in OFFICE_FILE.items():
        d = pd.read_csv(_os.path.join(ROOT, "data", fn), low_memory=False)
        d = d[d["stage"].astype(str).str.lower() == "general"]
        d = d[pd.to_numeric(d["cycle"], errors="coerce").isin(CYCLES)]
        d["office"] = office
        # district: "District 7" -> "7"; statewide -> ""
        seat = d["office_seat_name"].astype(str)
        d["district"] = (seat.str.extract(r"District\s+(\d+)")[0].fillna("")
                         if office == "House" else "")
        d["district"] = d["district"].map(F.dist_str)
        d["party_std"] = d["ballot_party"].map(F.npar)
        d["cand_key"] = d["candidate_name"].map(F.norm_name)
        d["won"] = d["winner"].astype(str).str.lower().isin(("true", "1", "yes"))
        frames.append(d[["cycle", "state_abbrev", "office", "district", "party_std",
                         "cand_key", "candidate_name", "won", "percent"]])
    r = pd.concat(frames, ignore_index=True)
    r = r.rename(columns={"cycle": "year", "state_abbrev": "state"})
    r = r[r["party_std"].isin(["DEM", "REP"])]
    r = r.dropna(subset=["cand_key"])
    r["race_id"] = (r["year"].astype(str) + "_" + r["state"] + "_" + r["office"]
                    + r["district"].radd("-").where(r["district"] != "", ""))
    # keep two-party races with exactly one winner - the same shape the model is trained on
    ok = r.groupby("race_id")["won"].sum() == 1
    r = r[r["race_id"].isin(ok[ok].index)]
    return r.drop_duplicates(["race_id", "cand_key"])


def polled_table():
    """The normal training table (polled races only), with the fundamentals feature set."""
    d = pd.read_csv(_os.path.join(ROOT, "polls_long_with_results.csv"), low_memory=False)
    d = d[d["has_result"] == 1].copy()
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    d["race_id"] = (d["year"].astype(str) + "_" + d["state"] + "_" + d["office"]
                    + d["district"].radd("-").where(d["district"] != "", ""))
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    return F.build_candidate_table(d, build_macro(), natl_env(), funds,
                                   house_train_years=CYCLES, fec=fec,
                                   bias_priors=F.compute_bias_priors(d),
                                   candidate_bios=F.load_candidate_bios()), d


def unpolled_table(res, polled, polls_long):
    """Fundamentals features for races in the results archive that were never polled.

    Built by running the SAME builder over a synthetic long frame with one row per candidate
    and NaN poll values - so every poll-derived column is NaN (the fundamentals feature set
    ignores them) while incumbency / prior margin / bio / macro fill in normally.
    """
    unpolled_ids = set(res["race_id"]) - set(polled["race_id"])
    u = res[res["race_id"].isin(unpolled_ids)].copy()
    fake = pd.DataFrame({
        "year": u["year"].astype(int), "state": u["state"], "office": u["office"],
        "district": u["district"], "candidate": u["candidate_name"],
        "cand_key": u["cand_key"], "party_std": u["party_std"],
        "pct": np.nan, "end_date": pd.NaT, "sample_size": np.nan,
        "pollster": "none", "won": u["won"].astype(int),
        "election_date": pd.to_datetime(u["year"].astype(str) + "-11-05"),
        "days_to_elec": np.nan, "race_id": u["race_id"],
    })
    funds, fec = F.load_fundamentals(), F.load_fec(extended=True)
    c = F.build_candidate_table(fake, build_macro(), natl_env(), funds,
                                house=({} if True else None), fec=fec,
                                bias_priors=F.compute_bias_priors(polls_long),
                                candidate_bios=F.load_candidate_bios())
    return c


def main():
    meta = json.load(open(_os.path.join(ROOT, "data",
                                        "fundamentals_model_general_features.json")))
    FE = meta["features"]
    P = {k: v for k, v in meta["xgb_params"].items() if k not in ("random_state", "n_jobs")}

    res = results_universe()
    polled, polls_long = polled_table()
    unp = unpolled_table(res, polled, polls_long)
    print(f"results-archive races: {res['race_id'].nunique()}")
    print(f"  polled (model's usual world): {polled['race_id'].nunique()}")
    print(f"  UNPOLLED (never scored before): {unp['race_id'].nunique()}")
    print(f"  unpolled share: {100*unp['race_id'].nunique()/res['race_id'].nunique():.0f}%")

    rows = []
    for ty in (2018, 2020, 2022, 2024):
        tr = polled[polled["year"] < ty]
        if not len(tr):
            continue
        m = xgb.XGBClassifier(**P, random_state=42, n_jobs=-1)
        m.fit(tr[FE], tr["won"].astype(int))
        for tag, te in (("polled", polled[polled["year"] == ty]),
                        ("unpolled", unp[unp["year"] == ty])):
            if not len(te):
                continue
            te = te.copy()
            te["p"] = m.predict_proba(te[FE])[:, 1]
            pick = te.loc[te.groupby("race_id")["p"].idxmax()]
            rows.append(dict(cycle=ty, set=tag, races=te["race_id"].nunique(),
                             race_acc=pick["won"].mean()))
    t = pd.DataFrame(rows).pivot(index="cycle", columns="set",
                                 values=["races", "race_acc"])
    print("\n=== fundamentals model: POLLED vs UNPOLLED races (expanding-window) ===")
    print(t.round(3).to_string())
    means = pd.DataFrame(rows).groupby("set")["race_acc"].mean()
    print("\nMEAN race_acc:", means.round(3).to_dict())
    if "unpolled" in means and "polled" in means:
        d = means["unpolled"] - means["polled"]
        print(f"\nUnpolled races are {'EASIER' if d > 0 else 'HARDER'} for the model "
              f"by {abs(d):.3f} race_acc.")
        print("Baseline to beat on unpolled races (always pick the incumbent party):")
        u = unp.copy()
        inc = u[u["is_incumbent"] == 1]
        if len(inc):
            pick = inc.drop_duplicates("race_id")
            print(f"  incumbent-party pick: {pick['won'].mean():.3f} "
                  f"(on the {pick['race_id'].nunique()} unpolled races that have one)")


if __name__ == "__main__":
    main()
