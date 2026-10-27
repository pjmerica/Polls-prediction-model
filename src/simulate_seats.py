"""Seat-count ranges and control odds for 2026 -> outputs/seat_sim_2026.json (2026-10-27, audit item 4).

    python src/simulate_seats.py [--runs 20000]

Every race keeps its published win probabilities (outputs/predictions_2026.csv for polled races,
outputs/fundamentals_predictions_2026.csv for the rest). Races are tied together by ONE shared
national error per run - a Gaussian copula: race r draws u_r = Phi(sqrt(rho) g + sqrt(1-rho) e_r),
uniform on (0, 1) for any rho, and goes Republican if u_r < P(R), Democratic if u_r > 1 - P(D),
otherwise to the other candidate (an independent or minor party). A positive g moves every race
toward the Democrats at once. rho comes from data/seat_sim_params.json
(analysis/national_error_corr.py: maximum likelihood on 1998-2024 out-of-sample predictions).

Seats not up in 2026 (Senate 31 R / 34 D incl. Sanders and King, who caucus with Democrats;
governors in the 14 states with no race: 8 R / 6 D, after Democrats held New Jersey and won
Virginia in 2025). Senate control: Democrats need 51 with their caucus; Republicans 50 (the Vice
President breaks ties). An independent who wins a seat without caucusing (Osborn) is counted apart;
if neither side reaches its number, "independents decide".
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

_sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _sys_path)
import paths as _paths  # noqa: E402
from paths import ROOT  # noqa: E402

NOT_UP = {"Senate": {"D": 34, "R": 31, "I": 0}, "Governor": {"D": 6, "R": 8, "I": 0},
          "House": {"D": 0, "R": 0, "I": 0}}
TOTAL = {"Senate": 100, "Governor": 50, "House": 435}


def race_sides(d):
    """One row per race: P(D), P(R), P(I/other). The Dem slot counts as D unless the candidate
    is an independent (display_party IND/OTH, e.g. Osborn), who counts as I."""
    d = d.copy()
    disp = d["display_party"].fillna(d["party"]).astype(str).str.upper().str[:3]
    side = np.where((d["party"] == "DEM") & (disp == "DEM"), "D",
                    np.where(d["party"] == "REP", "R", "I"))
    d["side"] = side
    p = d.pivot_table(index=["race_id", "office"], columns="side", values="win_prob_norm",
                      aggfunc="sum", fill_value=0.0)
    for s in ("D", "R", "I"):
        if s not in p.columns:
            p[s] = 0.0
    p = p[["D", "R", "I"]].reset_index()
    tot = p[["D", "R", "I"]].sum(axis=1)
    p[["D", "R", "I"]] = p[["D", "R", "I"]].div(tot, axis=0)
    return p


def simulate(p, rho, runs, seed=2026):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(runs)[:, None]
    e = rng.standard_normal((runs, len(p)))
    u = norm.cdf(np.sqrt(rho) * g + np.sqrt(1 - rho) * e)
    pr, pdm = p["R"].values[None, :], p["D"].values[None, :]
    won_r = u < pr
    won_d = u > 1 - pdm
    won_i = ~(won_r | won_d)
    return won_d, won_r, won_i, g.ravel()


def summarize(office, p, won_d, won_r, won_i):
    m = (p["office"] == office).values
    base = NOT_UP[office]
    D = won_d[:, m].sum(axis=1) + base["D"]
    R = won_r[:, m].sum(axis=1) + base["R"]
    I = won_i[:, m].sum(axis=1) + base["I"]
    if office == "Senate":
        d_ctl, r_ctl = D >= 51, R >= 50
    else:
        half = TOTAL[office] // 2 + 1
        d_ctl, r_ctl = D >= half, R >= half
    q = lambda x, a: int(np.percentile(x, a))
    hist = pd.Series(D).value_counts().sort_index()
    return dict(
        races=int(m.sum()), not_up=base, total=TOTAL[office],
        dem_seats=dict(mean=round(float(D.mean()), 1), median=q(D, 50), p10=q(D, 10), p90=q(D, 90),
                       p05=q(D, 5), p95=q(D, 95)),
        rep_seats=dict(mean=round(float(R.mean()), 1), median=q(R, 50), p10=q(R, 10), p90=q(R, 90)),
        other_seats=dict(mean=round(float(I.mean()), 2), max=int(I.max())),
        p_dem_control=round(float(d_ctl.mean()), 4), p_rep_control=round(float(r_ctl.mean()), 4),
        p_neither=round(float((~d_ctl & ~r_ctl).mean()), 4),
        expected_dem_from_odds=round(float(p.loc[m, "D"].sum() + base["D"]), 1),
        dem_seat_hist={int(k): int(v) for k, v in hist.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20000)
    args = ap.parse_args()
    params = json.load(open(os.path.join(ROOT, "data", "seat_sim_params.json")))
    rho = float(params["rho"])
    frames = [pd.read_csv(_paths.out("predictions_2026.csv"))]
    fp = _paths.out("fundamentals_predictions_2026.csv")
    if os.path.exists(fp):
        f = pd.read_csv(fp)
        frames.append(f[~f["race_id"].isin(set(frames[0]["race_id"]))])
    d = pd.concat(frames, ignore_index=True)
    p = race_sides(d)
    won_d, won_r, won_i, g = simulate(p, rho, args.runs)
    out = dict(generated_at=pd.Timestamp.now().isoformat(timespec="seconds"),
               run_id=os.environ.get("MODEL_RUN_ID"), runs=args.runs, rho=rho,
               rho_ci95=params.get("rho_ci95"),
               chambers={o: summarize(o, p, won_d, won_r, won_i) for o in ("Senate", "House", "Governor")},
               note="Each race keeps its published odds; one shared national error per run "
                    "(Gaussian copula, rho fitted on 1998-2024 out-of-sample predictions).")
    path = _paths.out("seat_sim_2026.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for o, s in out["chambers"].items():
        print(f"{o}: D seats median {s['dem_seats']['median']} (80% range {s['dem_seats']['p10']}-"
              f"{s['dem_seats']['p90']}); D control {s['p_dem_control']:.0%}, R control "
              f"{s['p_rep_control']:.0%}, neither {s['p_neither']:.0%}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
