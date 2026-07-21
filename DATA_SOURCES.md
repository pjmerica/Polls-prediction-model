# Data sources — where the results come from and how they were found

This documents every data source the project pulls from, the exact URLs, what each
file contains, and the investigation that led to them. Goal: anyone (including future
me) can reproduce the dataset from scratch or know exactly where to look.

There is **no single downloadable file** of "every poll + who won," so the dataset is
assembled by joining several public sources. Below, each is grouped by what it provides.

---

## 1. Election RESULTS — who won every race

**Source:** FiveThirtyEight `election-results` GitHub repo
**Repo:** https://github.com/fivethirtyeight/election-results (branch `main`)

| File | Download URL | Rows | Cycles | What it is |
|---|---|---|---|---|
| `election_results_senate.csv` | https://raw.githubusercontent.com/fivethirtyeight/election-results/main/election_results_senate.csv | 4,132 | 1976–2024 | Every U.S. Senate race, candidate-level |
| `election_results_house.csv` | https://raw.githubusercontent.com/fivethirtyeight/election-results/main/election_results_house.csv | 24,581 | 1976–2024 | Every U.S. House race, candidate-level |
| `election_results_gubernatorial.csv` | https://raw.githubusercontent.com/fivethirtyeight/election-results/main/election_results_gubernatorial.csv | 2,351 | 1996–2024 | Every Governor race, candidate-level |
| `races.csv` | https://raw.githubusercontent.com/fivethirtyeight/election-results/main/races.csv | — | — | Race-level metadata incl. **`incumbent_party`** |

**This is the authoritative "who won" source and it covers ALL races** (general,
jungle primary, runoff, recall), back to 1976 — far earlier than the polling data.

**Schema (22 columns, shared across the three result files):**
```
id, race_id, state_abbrev, state, office_id, office_name, office_seat_name,
cycle, stage, special, party, politician_id, candidate_id, candidate_name,
ballot_party, ranked_choice_round, votes, percent, unopposed, winner,
alt_result_text, source
```
Key fields the project uses: `state_abbrev`, `office_seat_name` (House district),
`cycle`, `stage` (filter to `general`), `ballot_party` (DEM/REP — **note: the `party`
column is all-null; the real party is in `ballot_party`**), `percent`, and **`winner`**
(the 1/0 outcome label).

A small representative sample of each file is committed under [`data_samples/`](data_samples/)
so the schema and content are visible in the repo without downloading the full ~6 MB.

### Mirrors / fallbacks
If the raw GitHub URL is unavailable, the same files are served via jsDelivr CDN:
`https://cdn.jsdelivr.net/gh/fivethirtyeight/election-results@main/<filename>`

---

## 2. POLLS — the predictor data

FiveThirtyEight was dissolved in **March 2025**; its old poll endpoints
(`projects.fivethirtyeight.com/polls/...`) now redirect to ABC News HTML. So polls
come from two live replacements that share the **same 538 schema**:

### Current cycle — New York Times
NYT publishes 538-schema poll CSVs, updated continuously, CC-BY:
- https://www.nytimes.com/newsgraphics/polls/senate.csv
- https://www.nytimes.com/newsgraphics/polls/house.csv
- https://www.nytimes.com/newsgraphics/polls/governor.csv

### Historical — frozen, COMMITTED to the repo
The historical poll files originally came from a Wayback Machine snapshot of the old 538 files:
```
http://web.archive.org/web/2id_/https://projects.fivethirtyeight.com/polls/data/senate_polls_historical.csv
http://web.archive.org/web/2id_/https://projects.fivethirtyeight.com/polls/data/house_polls_historical.csv
http://web.archive.org/web/2id_/https://projects.fivethirtyeight.com/polls/data/governor_polls_historical.csv
```
**These are now committed** as `data/{senate,house,governor}_historical.csv` (they never change),
so the pipeline reads them locally and **does not hit the Internet Archive**. The Wayback URLs
above are kept only as documentation of origin — the Archive is slow/unreliable (often times out),
which is exactly why we froze the files into the repo.

**Coverage limit:** machine-readable polls effectively begin in **2018** (when 538's
poll database started). Pre-2018 downballot polls lived in HuffPost Pollster (defunct,
not recoverable) and RealClearPolitics (no bulk export), so **2018 is the floor.**

---

## 3. Partisan lean (state + district) — a model feature

**Source:** FiveThirtyEight `data` repo, `partisan-lean/` folder
- https://raw.githubusercontent.com/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_STATES.csv
- https://raw.githubusercontent.com/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_DISTRICTS.csv

A CPVI-style lean per state (51 rows) and per district (435 rows), 2022 vintage.
Negative = Republican-leaning. **Caveat:** single vintage, so in the model feature table
`lean_cand` is ~59% missing (post-2022 redistricting districts don't match).

---

## 4. National environment — a model feature

Per-cycle generic-ballot DEM−REP margin (last ~30 days before each election):
**2018 +7.8, 2020 +7.5, 2022 +0.7, 2024 +0.1** — captures the Democratic wave fading to the
flat 2024 environment.

**Source:** originally the 538 generic-ballot file via Internet Archive
(`generic_ballot_polls_historical.csv`). The Archive is now unreachable (times out for us), so
these four static values are **hardcoded** as `natl_env` in `model.ipynb` — they are fixed
historical record, computed/verified from that snapshot earlier. No live fetch.

---

## 5. Macro / political-climate features — `fetch_macro.py` + `macro_features.py`

National economic & political conditions. **Architecture: pull once, commit, never re-pull.**
Past months never change, so `fetch_macro.py` downloads **monthly** series (2016 → now) from
FRED a single time and saves `data/macro_monthly.csv` (committed as static reference data).
`macro_features.py` then reads that CSV **with no network** and, for each election cycle, uses
**each cycle's own window = prior election eve → this election eve** (2018 → Nov 2016–Nov 2018,
2020 → 2016–Nov 2020, 2022 → 2016–Nov 2022, 2024 → 2016–Nov 2024). Each indicator is condensed
into trajectory stats: `eve / mean / max / min / std / trend / last12_delta`. The model decides
which matter (heavy regularization drops the rest).

**Series pulled** (FRED CSV endpoint `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`, no key):

| metric | FRED id | freq | meaning |
|---|---|---|---|
| `unemployment` | `UNRATE` | monthly | unemployment rate, % |
| `cpi` → `inflation` | `CPIAUCSL` | monthly | CPI index; converted to YoY % inflation downstream |
| `gas` | `GASREGW` | weekly→monthly | regular gas price, $/gal |
| `gdp` | `A191RL1Q225SBEA` | quarterly→monthly | real GDP growth, annualized % |
| `sentiment` | `UMCSENT` | monthly | U. Michigan consumer sentiment ("the vibe" — strong election predictor) |
| `real_income` | `DSPIC96` | monthly | real disposable personal income |
| `sp500` | `SP500` | daily→monthly | S&P 500 index (wealth effect) |
| `mortgage30` | `MORTGAGE30US` | weekly→monthly | 30-yr fixed mortgage rate, % |
| `fed_funds` | `FEDFUNDS` | monthly | effective federal funds rate, % |
| `jobless_claims` | `ICSA` | weekly→monthly | initial unemployment claims |
| `real_wage` | `LES1252881600Q` | quarterly→monthly | real median weekly earnings |
| `med_income` | `MEHOINUSA672N` | annual→monthly | real median household income |

Presidential **approval** is included from a documented monthly table inside `fetch_macro.py`
(FRED has no clean approval series); replace with a live source if you have one.
**President's party** per cycle: `PRES_PARTY` in `macro_features.py` (2018 REP, 2020 REP, 2022
DEM, 2024 DEM). `is_president_party` (candidate's party == president's party) is the interaction
key that lets XGBoost learn each macro effect's *direction* (e.g. low approval hurts the in-party).

**Coverage:** the CSV stores each series' **full available history** (CPI 1947, unemployment
1948, fed funds 1954, core CPI 1957, gas 1990, U-6 1994 → present) — not just the poll era.
We keep it all in case older poll data becomes available later; the model only *uses* cycles it
can match to polls (2018+). Approval is ~2016+ (documented table).

**To (re)generate:** `python fetch_macro.py` (pulls from DBnomics; no key) → writes
`data/macro_monthly.csv`. Set `START` in the script to truncate; `None` = full history.
Only needed once, or to extend to a new election cycle.

> Note: FRED's `fredgraph.csv` host is blocked in some sandboxes. DBnomics (a free aggregator)
> does **not** mirror FRED — only the upstream agencies (BLS/BEA/EIA) — so it's not a drop-in
> substitute. Run `fetch_macro.py` on a normal network.

**Caveat:** these are national values constant within a cycle; with only 4 cycles they carry
little independent signal for *win/lose* and need heavy regularization. They're mainly here for
calibration and for a future *margin* model.

---

## Related docs
- **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** — every column in both the long file and the
  model feature table, with meanings.
- **[MISSINGNESS_REPORT.md](MISSINGNESS_REPORT.md)** — per-column missingness for both datasets.

---

## How the results were found (the investigation)

1. **Tried the obvious first.** Searched for a single "polls + winners" dataset — none
   exists. The two clean halves are 538 (polls) and academic/official result sets.
2. **538's live poll endpoints were dead.** `projects.fivethirtyeight.com/polls/data/*.csv`
   returned an ABC News HTML page (200 OK but `<!doctype html>`), because the site was
   dissolved. Confirmed by inspecting the first bytes of the response.
3. **Found the results repo.** A web search surfaced `fivethirtyeight/election-results`
   — a *separate, still-live* GitHub repo (distinct from the dead poll endpoints) with a
   per-candidate `winner` flag. Verified the raw URLs return real CSV and the schema via
   the jsDelivr CDN.
4. **Found live poll replacements.** The NYT publishes 538-schema poll CSVs (discovered
   via a community scraper that documented the URLs), and the **Internet Archive** has
   frozen captures of the old 538 historical poll files (verified they return CSV, not
   the 429/HTML the live site gives).
5. **Established the 2018 floor.** Checked the Wayback CDX API and the file contents:
   the 538 poll DB only ever contained 2018+. Results go to 1976, but polls don't.
6. **Primary results (FEC).** For a primary model, found the FEC "Federal Elections"
   biennial compilations (`fec.gov/.../federal-elections-<year>/` → `/documents/<id>/federalelections<year>.xlsx`),
   which include candidate-level **primary** vote shares for Senate & House, 1982–present.
   Parsed cleanly (11,113 primary candidate rows for 2018/2020/2022). **However**, the
   project does not use them: 538/NYT collected essentially **no historical partisan-primary
   polls** (only jungle primaries / runoffs), so there's no poll↔result pair to train a
   primary model on. Documented here so the dead end isn't re-explored.

---

## Reproducing the full dataset

Everything is downloaded and cached by `build_dataset.ipynb` (into a gitignored
`data/` folder). The committed [`data_samples/`](data_samples/) files are illustrative
only — run the notebook to fetch the complete, current data.

```bash
pip install pandas numpy requests jupyter openpyxl
jupyter nbconvert --to notebook --execute build_dataset.ipynb
```

---

## Added 2026-07-05 (future-proofing overhaul)

| data | source | file | notes |
|---|---|---|---|
| Pre-2018 polls (1998–2016) | 538 pollster-ratings `raw_polls.csv` — `https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv` | `data/raw_polls_538.csv` (frozen) | 20k polls 1998–2023; we use downballot general, even cycles < 2018. Top-two candidates per poll. |
| Generic ballot 1996–2016 | 538 `congress-generic-ballot/generic_topline_historical.csv` | `data/generic_ballot_hist_538.csv` (frozen) | daily D/R estimates; `cycles.py` averages last 30 days before each election. |
| Presidential approval 1993–present | Gallup via UCSB American Presidency Project (`presidency.ucsb.edu/statistics/data/<president>-public-approval`) 1993–2025-01, **VoteHub API continuation** 2025-01+ | `fetch_approval.py` → `data/approval_monthly.csv` | replaces the hand-typed table. UCSB never posted a Trump-2nd-term page, but the VoteHub handoff (§ "Added 2026-07-06" below) keeps the feed live — verified current through 2026-06 as of 2026-07-21. |
| Future raw polls (2026+) | **polling-agg repo** (`..\Polling Agg\Polling agg and Prediction markets\data\raw\{nyt_polls,wikipedia_polls}.csv`) | consumed by `predict.py` | pct = `implied_prob`×100; race_id `2026-SEN-MI` / `2026-H-AL-01`; has internal+cross-source dupes → predict.py dedups. |
| Future results (label 2026 later) | MIT Election Data + Science Lab (electionlab.mit.edu) | loader TBD after Nov 2026 | convert to the `res_*.csv` schema. |

## Added 2026-07-06 (macro refresh, redistricting, campaign finance)

### Macro (corrects the fictional FRED table in §5 above — we use DBnomics, not FRED)
The real macro pipeline pulls from **DBnomics** (free, no key) with a **BLS public-API overlay**
for the current months, NOT the FRED ids listed in section 5. Series actually used
(`fetch_macro.py`): unemployment (BLS LNS14000000), cpi→inflation (BLS CUSR0000SA0), gas
(EIA), fed_funds (Fed H15), cpi_core, unemp_u6, **sentiment** (UMich via DBnomics
`SCSMICH/MICS/ICS`, 1952+), plus approval and generic_ballot merged in as monthly metrics.

| data | source | file | notes |
|---|---|---|---|
| UMich consumer sentiment | DBnomics `SCSMICH/MICS/ICS` | in `data/macro_monthly.csv` | 1952→present; DBnomics mirror lags ~1yr (watchdog allows 13mo). |
| Generic ballot (monthly, 1995+) | 538 daily hist (1995-2016) + `raw_polls` House-G-US (2017-22) + **VoteHub API** (2024-12+) | `fetch_generic_ballot.py --monthly` → `data/generic_ballot_monthly.csv` | 2024-cycle window is NaN (no surviving source). |
| Approval (current-term continuation) | **VoteHub API** `api.votehub.com/polls?poll_type=approval` | `fetch_approval.py` → `data/approval_monthly.csv` | all-pollster avg after UCSB/Gallup ends 2025-01. |
| Cook PVI (current maps) | Wikipedia Cook PVI tables (435 districts + 50 states) | `fetch_district_pvi.py` → `data/district_pvi_current.csv` | PREDICT-TIME only (historical PVI paywalled); patches prior_margin in redrawn districts. |
| 2025-26 redistricting | Wikipedia `2025-2026 United States redistricting` | `data/redistricted_2026.csv` | 11 states redrew House maps; flags 41% of predicted House races. |

### Campaign finance (the funding data) — Senate, House, AND Governor

| data | source | file | coverage | what it is |
|---|---|---|---|---|
| **FEC candidate totals (Sen + House)** | FEC bulk `webl{yy}.zip` (no API key) | `fetch_fec.py` → `data/fec_summary.csv` | **1998–2026, all 15 cycles, 29,186 candidates** | per-candidate receipts + individual/PAC/party/self-fund breakdown |
| **FEC itemized individual (Sen + House)** | FEC API `/candidates/totals` (`api.data.gov` key in `.env`) | `fetch_fec_detail.py` → `data/fec_detail.csv` | **1998–2026, 46,441 rows** | itemized individual money → **small-dollar share** = 1 − itemized/total-individual (grassroots proxy) |
| **Governor campaign finance** | **FollowTheMoney / NIMSP** API (`FTM_API_KEY` in `.env`) | `fetch_governor_finance.py` → `data/governor_finance.csv` | **1998–2002 so far (557 cands); 2004+ pending FTM daily quota** | governor total receipts (FEC is federal-only, so governors need this state-level source) |

**Features built from these** (`features.py` `FUND_FEATS_EXT`, active since batch 5):
`fund_receipts_ln`, `fund_share` (candidate's % of the race's total money — the ratio feature
robust to filing-date differences), `fund_indiv_pct`, `fund_pac_pct`, `fund_party_pct`,
`fund_self_pct`, `fund_smalldollar_pct`. Governors get receipts + share only (no composition
breakdown from FTM); Sen/House get all seven.

**Leakage note:** historical FEC totals run through Dec 31 (past the Sep-30 info cutoff), so
the RATIO/composition features are the trustworthy ones (both candidates truncate at the same
date); raw `fund_receipts_ln` is the one to distrust. A true *average donation* is impossible
from any FEC aggregate (no donor counts are published) — small-dollar share is the standard
substitute.

**Keys:** both `FEC_API_KEY` (api.data.gov) and `FTM_API_KEY` (followthemoney.org) live in
`.env` (gitignored, never committed). The fetch scripts read them; committed CSVs mean no key
is needed to run the models.
