"""Shared feature pipeline: long poll rows -> one row per candidate per race.

Used by BOTH model.ipynb (training/CV) and predict.py (live 2026+ races), so features are
guaranteed to be computed identically at train and predict time.

Design rules (see CONCERNS.md):
- RAW POLLS ONLY: no poll weighting of any kind (no recency/sample/grade weights). The old
  pipeline weighted by 538's pollster grade, which does not exist for future polls -> that
  was train/serve skew. Averages are plain means now (user decision, 2026-07-05).
- No 538-only columns: numeric_grade / pollscore / partisan-lean are never used.
- Every input must be available in a bare poll feed: state, office, district, candidate,
  party, pollster, end_date, pct, sample_size (+ election_date to compute days_to_elec).
- Missing fundamentals are NaN (XGBoost routes missing natively) - never silently 0.
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

from cycles import PRES_PARTY

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- small parsers

def dist_str(v):
    """Normalize a district value to canonical string form ('1', '23', or '').
    Guards against the CSV float round-trip bug ('1' -> 1.0 -> '1.0')."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s in ("", "nan"):
        return ""
    try:
        return str(int(float(s)))
    except ValueError:
        return s

def pdist(v):
    m = re.search(r"(\d+)", str(v))
    return str(int(m.group(1))) if m else ""

def npar(p):
    p = str(p).upper()
    return "DEM" if p.startswith("DEM") else "REP" if p.startswith("REP") else "OTH"

def norm_name(s):
    """Candidate-name join key: strip accents/suffixes -> 'lastname firstinitial'."""
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    parts = [w for w in s.split() if w]
    if not parts:
        return None
    last = parts[-1]
    fi = parts[0][0] if parts[0] != last else ""
    return f"{last} {fi}".strip()

JUNK_ANSWERS = {
    "generic democrat", "generic republican", "generic ballot", "generic candidate",
    "don't know", "dont know", "undecided", "someone else", "other",
    "would not vote", "neither", "no opinion", "refused", "none of these",
    "skipped", "will not vote", "not sure",
}

def is_junk_answer(name):
    return str(name).strip().lower() in JUNK_ANSWERS

# ---------------------------------------------------------------- fundamentals

def load_fundamentals():
    """Incumbency + prior-margin lookups from the committed static files in data/.

    Returns dict(inc_map=..., margin_map=...). No network, ever.
    """
    rc = pd.read_csv(os.path.join(DATA_DIR, "races.csv"), low_memory=False)
    off = rc["office_name"].astype(str).str.lower()
    rc["office"] = np.select(
        [off.str.contains("senate"), off.str.contains("house"), off.str.contains("governor")],
        ["Senate", "House", "Governor"], default=None)
    rc["state"] = rc["state_abbrev"].str.upper()
    rc["district"] = ""
    hm = rc["office"] == "House"
    rc.loc[hm, "district"] = rc.loc[hm, "office_seat_name"].map(pdist)
    inc_map = {(r.cycle, r.state, r.office, r.district): npar(r.incumbent_party)
               for r in rc[rc["office"].notna()].itertuples()
               if pd.notna(r.incumbent_party)}

    def _load_res(fn, office):
        r = pd.read_csv(os.path.join(DATA_DIR, fn), low_memory=False)
        r = r[r["stage"].astype(str).str.lower().str.contains("general", na=False)]
        r["office"] = office
        r["state"] = r["state_abbrev"].str.upper()
        r["district"] = "" if office != "House" else r["office_seat_name"].map(pdist)
        r["p"] = r["ballot_party"].map(npar)   # 'party' col is null in these files
        r["pct"] = pd.to_numeric(r["percent"], errors="coerce")
        return r[["cycle", "state", "office", "district", "p", "pct"]]

    allres = pd.concat([_load_res("res_senate.csv", "Senate"),
                        _load_res("res_house.csv", "House"),
                        _load_res("res_governor.csv", "Governor")])
    piv = (allres[allres["p"].isin(["DEM", "REP"])]
           .groupby(["cycle", "state", "office", "district", "p"])["pct"].max().unstack("p"))
    for col in ["DEM", "REP"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv["margin"] = piv["DEM"].fillna(0) - piv["REP"].fillna(0)
    margin_map = {idx: row.margin for idx, row in piv.iterrows()}
    return dict(inc_map=inc_map, margin_map=margin_map)

def prior_margin(margin_map, year, state, office, district):
    """Most recent same-office two-party margin strictly BEFORE `year` (leak-free)."""
    for back in range(2, 9, 2):
        v = margin_map.get((year - back, state, office, district))
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return np.nan

# ---------------------------------------------------------------- FEC fundraising

def fec_cand_key(name):
    """'PELTOLA, MARY (ALIAS)' -> norm_name('mary peltola') -> 'peltola m'."""
    s = re.sub(r"\(.*?\)", "", str(name))
    parts = s.split(",", 1)
    s = (parts[1] + " " + parts[0]) if len(parts) == 2 else s
    return norm_name(s)

def load_fec(path=None, extended=False):
    """data/fec_summary.csv -> {(cycle,state,office,district,cand_key): {receipts,...}}.

    Senate district = ''; FEC House at-large '00' -> '1' (matches our race keys).
    NOTE the cutoff caveat in fetch_fec.py: historical totals run through Dec 31, so
    RATIO features (share/composition) are the trustworthy ones; raw totals are secondary.

    extended=True (BATCH 5+ ONLY — changes feature values, so artifacts and predict must
    flip together, then retrain) additionally merges:
      - data/fec_detail.csv (API): itemized individual money -> small-dollar share
        (unitemized = bulk total individual minus API itemized).
      - data/governor_finance.csv (FollowTheMoney): governor receipts -> fund_receipts_ln
        + fund_share finally exist for Governor rows (composition stays NaN).
    """
    path = path or os.path.join(DATA_DIR, "fec_summary.csv")
    f = pd.read_csv(path, dtype={"district": str})
    f["cand_key"] = f["cand_name"].map(fec_cand_key)
    f["district"] = [("" if o == "Senate"
                      else ("1" if str(di) in ("00", "0", "nan") else str(int(float(di)))))
                     for o, di in zip(f["office"], f["district"])]
    f = f.sort_values("receipts", ascending=False).drop_duplicates(
        ["cycle", "state", "office", "district", "cand_key"])
    out, by_id = {}, {}
    for r in f.itertuples():
        k = (r.cycle, r.state, r.office, r.district, r.cand_key)
        out[k] = dict(receipts=r.receipts, indiv=r.indiv_contrib, pac=r.pac_contrib,
                      party=r.party_contrib, self=r.self_fund, small=np.nan)
        by_id[(r.cycle, r.cand_id)] = k
    if not extended:
        return out

    det_path = os.path.join(DATA_DIR, "fec_detail.csv")
    if os.path.exists(det_path):
        det = pd.read_csv(det_path)
        itemized = det.groupby(["cycle", "cand_id"])["indiv_itemized"].max()
        for (cyc, cid), item in itemized.items():
            k = by_id.get((cyc, cid))
            if k and out[k]["indiv"] and out[k]["indiv"] > 0 and pd.notna(item):
                out[k]["small"] = float(np.clip(1 - item / out[k]["indiv"], 0, 1))

    gov_path = os.path.join(DATA_DIR, "governor_finance.csv")
    if os.path.exists(gov_path):
        g = pd.read_csv(gov_path)
        g["cand_key"] = g["cand_name"].map(fec_cand_key)
        g = g.sort_values("receipts", ascending=False).drop_duplicates(
            ["cycle", "state", "cand_key"])
        for r in g.itertuples():
            if pd.notna(r.receipts):
                out.setdefault((r.cycle, r.state, "Governor", "", r.cand_key),
                               dict(receipts=float(r.receipts), indiv=np.nan, pac=np.nan,
                                    party=np.nan, self=np.nan, small=np.nan))
    return out

FUND_FEATS = ["fund_receipts_ln", "fund_share", "fund_indiv_pct", "fund_pac_pct",
              "fund_party_pct", "fund_self_pct"]
FUND_FEATS_EXT = FUND_FEATS + ["fund_smalldollar_pct"]   # batch 5+ (extended FEC)

# ---------------------------------------------------------------- poll prep

def prepare_polls(d):
    """Coerce types on a long poll frame (one row per poll-candidate). Modifies a copy."""
    d = d.copy()
    d["end_date"] = pd.to_datetime(d["end_date"], errors="coerce", format="mixed")
    d["election_date"] = pd.to_datetime(d["election_date"], errors="coerce", format="mixed")
    d["days_to_elec"] = (d["election_date"] - d["end_date"]).dt.days
    for c in ["pct", "sample_size", "days_to_elec"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # harmonize instrument precision: the live 2026 feed carries ~1-decimal pcts while the
    # 538-era training files carry more — round BOTH paths so train and serve match.
    d["pct"] = d["pct"].round(1)
    d["district"] = d["district"].map(dist_str)
    return d

def norm_pollster(p):
    """Normalize pollster names so house effects match across feeds (538 vs NYT/Wikipedia):
    casefold, drop partisan tags, '&'->'and', 'Co.'->'company', strip punctuation."""
    s = str(p).casefold().strip()
    s = re.sub(r"\(([dr]|dem|rep)\)", "", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\bco\b\.?", "company", s)
    s = re.sub(r"\binc\b\.?|,", "", s)
    s = re.sub(r"[^a-z0-9/ ]", "", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- race dynamics

def count_lead_changes(g):
    """How often the running-mean front-runner flipped over the race's poll dates."""
    g = g.dropna(subset=["end_date", "pct"]).sort_values("end_date")
    prev, changes = None, 0
    for dt in g["end_date"].drop_duplicates().sort_values():
        means = g[g["end_date"] <= dt].groupby("cand_key")["pct"].mean()
        if means.empty:
            continue
        leader = means.idxmax()
        if prev is not None and leader != prev:
            changes += 1
        prev = leader
    return changes

def margin_dynamics(g):
    """Per-candidate margin-vs-best-opponent trajectory stats over the campaign."""
    g = g.dropna(subset=["end_date", "pct"]).sort_values("end_date")
    dates = g["end_date"].drop_duplicates().sort_values()
    series = {}
    t0 = dates.min()
    for dt in dates:
        means = g[g["end_date"] <= dt].groupby("cand_key")["pct"].mean()
        if len(means) == 0:
            continue
        elapsed = (dt - t0).days
        for ck, val in means.items():
            others = means.drop(ck)
            best_other = others.max() if len(others) else 0.0
            series.setdefault(ck, []).append((elapsed, val - best_other))
    out = {}
    for ck, pts in series.items():
        m = np.array([p[1] for p in pts], dtype=float)
        x = np.array([p[0] for p in pts], dtype=float)
        trend = np.polyfit(x, m, 1)[0] if (len(m) >= 2 and np.ptp(x) > 0) else 0.0
        out[ck] = dict(avg_margin_over_time=float(np.mean(m)),
                       margin_volatility=float(np.std(m)) if len(m) > 1 else 0.0,
                       min_margin=float(np.min(m)),
                       margin_trend=float(trend))
    return out

# ---------------------------------------------------------------- house effect

def compute_house_effect(d, train_years, shrink_k=5.0):
    """Per-pollster DEM-REP margin deviation vs the race consensus, TRAIN years only.
    Keyed by NORMALIZED pollster name (norm_pollster) so 2026-feed names match history.
    Empirical-Bayes shrunken toward 0 by n/(n+k): a 2-poll pollster's raw 'house effect'
    is mostly noise and used to be applied at full strength."""
    mar = (d[d["party_std"].isin(["DEM", "REP"])]
           .pivot_table(index=["race_id", "poll_id", "pollster", "year"],
                        columns="party_std", values="pct", aggfunc="max").reset_index())
    for col in ["DEM", "REP"]:
        if col not in mar.columns:
            mar[col] = np.nan
    mar["m"] = mar["DEM"] - mar["REP"]
    tm = mar[mar["year"].isin(list(train_years))].copy()
    tm["dev"] = tm["m"] - tm.groupby("race_id")["m"].transform("mean")
    tm["pollster_key"] = tm["pollster"].map(norm_pollster)
    g = tm.groupby("pollster_key")["dev"].agg(["mean", "count"])
    return (g["mean"] * g["count"] / (g["count"] + shrink_k)).to_dict()

def compute_bias_priors(d, shrink_k=8.0):
    """{(cycle, state): shrunken PRIOR-cycles mean signed poll-margin error} + (cycle,'_nat').

    e = polled(D−R) − actual(D−R) per race; positive = polls overstated Democrats there.
    For target cycle Y only cycles < Y contribute (leak-free). State means are shrunk toward
    the national prior mean by n/(n+k). Historically this shifts ±4-7 pts between cycles —
    the single biggest correlated risk (HANDOFF.md). d must carry vote_pct (training frame)."""
    dd = d[d["party_std"].isin(["DEM", "REP"])].dropna(subset=["vote_pct"])
    g = (dd.groupby(["year", "race_id", "party_std"])
           .agg(poll=("pct", "mean"), act=("vote_pct", "first")).reset_index())
    p = g.pivot_table(index=["year", "race_id"], columns="party_std", values=["poll", "act"])
    e = ((p[("poll", "DEM")] - p[("poll", "REP")])
         - (p[("act", "DEM")] - p[("act", "REP")])).dropna()
    err = e.rename("e").reset_index()
    err["state"] = err["race_id"].str.split("_").str[1]
    out = {}
    years = sorted(err["year"].unique())
    for y in years + [years[-1] + 2]:            # +2 covers the next (predict) cycle
        past = err[err["year"] < y]
        if past.empty:
            continue
        nat = float(past["e"].mean())
        out[(y, "_nat")] = nat
        for s, grp in past.groupby("state"):
            n = len(grp)
            out[(y, s)] = float((grp["e"].mean() * n + nat * shrink_k) / (n + shrink_k))
    return out

def candidate_poll_adj(d, house):
    """Per (race_id, cand_key) plain mean of house-effect-adjusted pct.

    Lets CV folds swap in a leak-free poll_adj (house effect from train cycles only)
    without rebuilding the whole candidate table."""
    dd = d[["race_id", "cand_key", "party_std", "pct", "pollster"]].copy()
    dd["sign"] = dd["party_std"].map({"DEM": 1, "REP": -1}).fillna(0)
    dd["adj"] = dd["pct"] - dd["sign"] * dd["pollster"].map(norm_pollster).map(house).fillna(0.0)
    return dd.groupby(["race_id", "cand_key"])["adj"].mean()

# ---------------------------------------------------------------- main builder

def build_candidate_table(d, macro, natl_env_map, funds, house_train_years=None, house=None,
                          fec=None, bias_priors=None):
    """Collapse prepared long polls `d` -> one row per candidate per race, with features.

    d must have: race_id, year, state, office, district, candidate, cand_key, party_std,
    pct, end_date, days_to_elec, sample_size, pollster. `won` optional (NaN at predict time).
    All poll aggregates are PLAIN averages (no weighting).

    House effect: pass `house_train_years` to compute it from those cycles of `d`, or pass a
    precomputed `house` dict directly (predict time: computed from historical polls, applied
    to the new cycle's polls).
    """
    if house is None:
        house = compute_house_effect(d, house_train_years or [])
    lead_change_map = {rid: count_lead_changes(g) for rid, g in d.groupby("race_id")}
    margin_dyn_map = {rid: margin_dynamics(g) for rid, g in d.groupby("race_id")}
    inc_map, margin_map = funds["inc_map"], funds["margin_map"]
    has_won = "won" in d.columns

    rows = []
    for race_id, g in d.groupby("race_id"):
        yr = int(g["year"].iloc[0]); st = g["state"].iloc[0]
        of = g["office"].iloc[0];    di = dist_str(g["district"].iloc[0])
        dyn = margin_dyn_map.get(race_id, {})
        for ck, gc in g.groupby("cand_key"):
            gc = gc.sort_values("end_date")
            dated = gc.dropna(subset=["end_date"])          # NaT polls can't be "most recent"
            last30 = gc[gc["days_to_elec"] <= 30]
            party = gc["party_std"].iloc[0]
            sign = 1 if party == "DEM" else -1 if party == "REP" else 0

            incp = inc_map.get((yr, st, of, di))
            pm = prior_margin(margin_map, yr, st, of, di)

            last60 = gc[gc["days_to_elec"] <= 60].dropna(subset=["pct", "days_to_elec"])
            slope = np.nan
            if len(last60) >= 3:
                x = -last60["days_to_elec"].values.astype(float)
                y = last60["pct"].values.astype(float)
                if np.ptp(x) > 0:
                    slope = np.polyfit(x, y, 1)[0]

            adj = gc["pct"] - gc["pollster"].map(lambda p: sign * house.get(norm_pollster(p), 0.0))
            md = dyn.get(ck, {})

            fe = fec.get((yr, st, of, di, ck)) if fec is not None else None
            rec = fe["receipts"] if fe else np.nan
            fund = dict(
                fund_receipts_ln=(np.log1p(rec) if fe and rec > 0 else np.nan),
                fund_indiv_pct=(fe["indiv"] / rec if fe and rec > 0 else np.nan),
                fund_pac_pct=(fe["pac"] / rec if fe and rec > 0 else np.nan),
                fund_party_pct=(fe["party"] / rec if fe and rec > 0 else np.nan),
                fund_self_pct=(fe["self"] / rec if fe and rec > 0 else np.nan),
                fund_smalldollar_pct=(fe.get("small", np.nan) if fe else np.nan),
                _fund_receipts=(rec if fe else np.nan),
            ) if fec is not None else {}

            rows.append(dict(
                race_id=race_id, year=yr, state=st, office=of, district=di,
                cand_key=ck, candidate=gc["candidate"].iloc[0], party=party,
                won=(int(gc["won"].iloc[0]) if has_won and pd.notna(gc["won"].iloc[0]) else np.nan),
                # actual vote share — LABEL for the margin model, never a feature
                vote_pct=(pd.to_numeric(gc["vote_pct"], errors="coerce").iloc[0]
                          if "vote_pct" in gc.columns else np.nan),
                poll_avg=gc["pct"].mean(),
                poll_last=(dated["pct"].iloc[-1] if len(dated) else gc["pct"].mean()),
                poll_last30=(last30["pct"].mean() if len(last30) else gc["pct"].mean()),
                poll_std=gc["pct"].std(),
                n_polls=len(gc),
                n_polls_over50=int((gc["pct"] > 50).sum()),
                avg_sample=gc["sample_size"].mean(),
                min_days=gc["days_to_elec"].min(),
                prior_margin_cand=(sign * pm if not (isinstance(pm, float) and np.isnan(pm)) else np.nan),
                # unknown incumbency = NaN (missing), never a silent 0
                is_incumbent=((1 if incp == party else 0) if incp in ("DEM", "REP") else np.nan),
                is_inc_party_race=(1 if incp in ("DEM", "REP") else 0),
                natl_env_cand=(sign * natl_env_map.get(yr, np.nan)),
                # prior-cycles poll-bias prior for this state (leak-free), candidate-signed:
                # positive = polls here historically overstated THIS candidate's party
                bias_prior_cand=(sign * _bp if bias_priors is not None and sign != 0
                                 and (_bp := bias_priors.get((yr, st),
                                                             bias_priors.get((yr, "_nat"))))
                                 is not None else np.nan),
                poll_momentum=slope,
                poll_adj=adj.mean(),
                n_lead_changes=lead_change_map.get(race_id, 0),
                lead_changed=int(lead_change_map.get(race_id, 0) > 0),
                avg_margin_over_time=md.get("avg_margin_over_time", np.nan),
                margin_volatility=md.get("margin_volatility", np.nan),
                min_margin=md.get("min_margin", np.nan),
                margin_trend=md.get("margin_trend", np.nan),
                is_president_party=int(party == PRES_PARTY.get(yr)),
                **fund,
                **macro.get(yr, {}),
            ))
    c = pd.DataFrame(rows)

    if fec is not None:
        # share of the race's (matched) money — the ratio feature robust to cutoff dates
        tot = c.groupby("race_id")["_fund_receipts"].transform("sum")
        c["fund_share"] = np.where(tot > 0, c["_fund_receipts"] / tot, np.nan)
        c = c.drop(columns="_fund_receipts")

    # race-relative features (all based on the plain poll average)
    c["field_best"] = c.groupby("race_id")["poll_avg"].transform(
        lambda s: s.nlargest(2).min() if len(s) > 1 else s.max())
    c["poll_lead"] = c["poll_avg"] - c["field_best"]
    c["poll_share"] = c["poll_avg"] / c.groupby("race_id")["poll_avg"].transform("sum")
    c["n_cands"] = c.groupby("race_id")["cand_key"].transform("count")
    c["race_total_polls"] = c.groupby("race_id")["n_polls"].transform("sum")
    c["frac_polls_over50"] = c["n_polls_over50"] / c["n_polls"]
    c["is_dem"] = (c["party"] == "DEM").astype(int)
    c["is_rep"] = (c["party"] == "REP").astype(int)
    c["is_senate"] = (c["office"] == "Senate").astype(int)
    c["is_gov"] = (c["office"] == "Governor").astype(int)

    dem = c[c["party"] == "DEM"].groupby("race_id")["poll_avg"].max()
    rep = c[c["party"] == "REP"].groupby("race_id")["poll_avg"].max()
    tp = dem - rep
    c["twoparty_margin_cand"] = (c["race_id"].map(tp)
                                 * c["party"].map({"DEM": 1, "REP": -1}).fillna(0))
    c["abs_gap"] = c["race_id"].map(tp.abs())
    c["tossup"] = (c["abs_gap"] < 3).astype(int)
    c["undecided"] = (100 - c.groupby("race_id")["poll_avg"].transform("sum")).clip(lower=0)
    c["gap_x_recency"] = c["poll_lead"] * (1.0 / (1.0 + c["min_days"].clip(lower=0) / 30.0))
    return c

def feature_list(macro_feats, fund=False):
    """The model's input columns. Everything here is available for future races.
    fund=True appends the FEC fundraising features (pass fec=load_fec() to the builder)."""
    return ([] if not fund else list(FUND_FEATS)) + [
        "poll_avg", "poll_last", "poll_last30", "poll_std", "n_polls",
        "n_polls_over50", "frac_polls_over50", "race_total_polls",
        "avg_sample", "min_days",
        "poll_lead", "poll_share", "n_cands",
        "is_dem", "is_rep", "is_senate", "is_gov",
        "prior_margin_cand", "is_incumbent", "is_inc_party_race",
        "twoparty_margin_cand", "abs_gap", "tossup", "undecided", "gap_x_recency",
        "natl_env_cand", "bias_prior_cand", "poll_momentum", "poll_adj",
        "n_lead_changes", "lead_changed",
        "avg_margin_over_time", "margin_volatility", "min_margin", "margin_trend",
        "is_president_party",
    ] + list(macro_feats)
