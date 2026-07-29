# -*- coding: utf-8 -*-
"""Measure bio_office_level coverage of the general-model candidate table + regenerate the
uncovered-target roster. ONE authoritative definition of "covered" (2026-07-27):

    A candidate-race is COVERED iff data/candidate_bios.csv has a matching row whose
    office_level > 0, OR whose office_level == 0 but with a REAL descriptor (genuinely held
    no prior office). A level-0 row with a BLANK descriptor is a "table-zero" - it came from
    a results-table scrape that never stated what office the person held, so its level is
    UNKNOWN, not a confirmed 0. Table-zeros count as UNCOVERED.

Why this matters (found via Tom Carper, DE Governor in 2000): the earlier lenient measure
counted any level-0 row as covered, so real prior-officeholders whose only bio row was a
blank-descriptor results-table row (level 0) were never queued for Ballotpedia/manual
resolution. Treating table-zeros as unknown surfaces them (user decision 2026-07-27:
"treat table-zeros as UNCOVERED").

    py -X utf8 measure_office_coverage.py            # print coverage
    py -X utf8 measure_office_coverage.py --write     # + regenerate the target roster CSV

Writes data/office_level_backfill_targets.csv (year, office, state, district, party,
candidate, cand_key, is_winner, reason) - the still-uncovered rows the backfill stages target.
`reason` = "no_bio" (no matching row at all) or "table_zero" (only an uninformative level-0).
"""
import sys, os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from cycles import CYCLES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def _real_descriptor(v):
    return str(v).strip() not in ("", "nan", "None")

def covered_keys_and_tablezeros():
    """Return (covered_keys, tablezero_keys). covered = level>0 or level0-with-descriptor.
    tablezero = level0 with blank descriptor (its own set so target-gen can label the reason)."""
    b = pd.read_csv(os.path.join(DATA, "candidate_bios.csv"), low_memory=False)
    b["district"] = b["district"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    desc_col = b["descriptor"] if "descriptor" in b.columns else pd.Series([""] * len(b))
    src_col = b["src"] if "src" in b.columns else pd.Series([""] * len(b))
    covered, tablezero = set(), set()
    for r, desc, src in zip(b.itertuples(), desc_col, src_col):
        k = (str(int(r.year)), r.office, r.state, r.district, F.npar(r.party), r.cand_key)
        # covered iff: a real office (level>0), OR a level-0 that is VERIFIED - either a
        # Wikipedia descriptor stating no office, or a hand-coded/Ballotpedia-resolved row
        # (src manual/ballotpedia is verified data, so its level-0 is a real "no prior office",
        # not a blank results-table unknown). Only a blank-descriptor Wikipedia level-0 is a
        # table-zero (unknown).
        if (r.office_level > 0 or _real_descriptor(desc)
                or src in ("manual", "ballotpedia", "wiki_xref")):
            covered.add(k)
        else:
            tablezero.add(k)
    tablezero -= covered            # a covered row elsewhere for the same key wins
    return covered, tablezero

def candidate_races():
    d = pd.read_csv(os.path.join(HERE, "polls_long_with_results.csv"), low_memory=False)
    d = F.prepare_polls(d)
    d = d[d["year"].isin(CYCLES)].copy()
    return d.drop_duplicates(subset=["year", "office", "state", "district", "cand_key", "party_std"])

def main():
    write = "--write" in sys.argv
    covered, tablezero = covered_keys_and_tablezeros()
    cr = candidate_races().copy()

    def keyof(r):
        return (str(int(r.year)), r.office, r.state, F.dist_str(r.district),
                F.npar(r.party_std), r.cand_key)
    keys = [keyof(r) for r in cr.itertuples()]
    cr["covered"] = [k in covered for k in keys]
    cr["reason"] = ["" if k in covered else ("table_zero" if k in tablezero else "no_bio")
                    for k in keys]

    w = cr[cr["won"] == 1]
    print("=== bio_office_level coverage (table-zeros = UNCOVERED) ===")
    print(f"OVERALL: {cr['covered'].mean():.1%}  ({int(cr['covered'].sum())} of {len(cr)})")
    print(f"WINNERS: {w['covered'].mean():.1%}  ({int(w['covered'].sum())} of {len(w)})")
    unc = cr[~cr["covered"]]
    print(f"\nuncovered: {len(unc)}  (winners: {int((unc['won'] == 1).sum())})")
    print("  by reason:", unc["reason"].value_counts().to_dict())
    print("  by era: pre-2012 =", int((unc["year"] < 2012).sum()),
          "| 2012+ =", int((unc["year"] >= 2012).sum()))

    if write:
        out = unc[["year", "office", "state", "district", "party_std", "candidate",
                   "cand_key", "won", "reason"]].copy()
        out = out.rename(columns={"party_std": "party", "won": "is_winner"})
        out["is_winner"] = (out["is_winner"] == 1).astype(int)   # NaN (2026, no result) -> 0
        out = out.sort_values(["is_winner", "year", "state", "office"],
                              ascending=[False, True, True, True])
        out.to_csv(os.path.join(DATA, "office_level_backfill_targets.csv"), index=False)
        print(f"\nwrote data/office_level_backfill_targets.csv: {len(out)} uncovered rows "
              f"({int(out['is_winner'].sum())} winners), sorted winners-first")

if __name__ == "__main__":
    main()
