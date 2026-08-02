# Polls prediction model

> **Layout changed 2026-08-02** — poll-based models now live in `models/poll/`,
> no-polling models in `models/fundamentals/`, scrapers in `pipeline/fetch/`, dataset
> assembly in `pipeline/build/`, one-off repairs in `tools/`. See **STRUCTURE.md** for the
> map and the run commands. Everything is still run from the repo root.

Predicts U.S. downballot elections — Senate, House, and Governor — from polls plus
political/economic context. Three separate models: **win probability**, **margin of
victory**, and (new) **primary nominee**. Trained on **14 cycles, 1998–2024** (~1,970
races); 2026 predictions are published to a companion dashboard and compared against
Kalshi/Polymarket prices.

---

## For a non-technical reader (what is this?)

Every election, pollsters ask voters who they'll support. This project asks: **how well
can you predict the actual winner from those polls, and does adding "context" (the economy,
the president's approval, who's the incumbent, campaign fundraising) help?**

We gathered every downballot poll we could find back to 1998, matched each one to who
actually won, and trained machine-learning models. The honest headline result:

> **For calling the winner, just betting on whoever is ahead in the polls is already about
> as good as it gets** — the model ties that baseline, it doesn't beat it. Where the model
> *does* help is (a) giving a trustworthy **probability** (e.g. "72% chance") instead of a
> flat coin-flip-vs-not read, and (b) **predicting the margin of victory**, where it
> genuinely beats a poll-based baseline.

That's a real, well-known fact about U.S. elections — the polls already contain most of
the win/lose signal. The separate margin model, and a third model that predicts **primary
nominees**, are where the added features earn their keep.

---

## For a data scientist (how it works)

**Three models, all XGBoost, all sharing one feature pipeline (`features.py` /
`features_primary.py`) so training and live prediction can never compute features
differently:**

| model | target | notebook/script | headline (honest, expanding-window) |
|---|---|---|---|
| **Win** | `won ∈ {0,1}` | `model.ipynb` | AUC .966 / AUC-PR .947 / Brier .072 / race-acc .864 (ties the poll baseline) |
| **Margin** | signed vote margin vs. best opponent | `margin_model.ipynb` | MAE ~6.5–7.2 vs. ~7.5 calibrated-poll baseline (**beats polls**) |
| **Primary nominee** | `won ∈ {0,1}` (becomes the party's nominee) | `primary_model.py` | AUC-PR .97 / Brier .02 / race-acc .93 vs. poll-leader .72 baseline |

(Numbers as of the most recent retrain — see `feature_importance.csv` /
`margin_feature_importance.csv` / `data/primary_model_features.json` for the exact current
run, and `HANDOFF.md` for the dated history of every retrain.)

**Validation:** nested, never random. Hyperparameters are tuned by leave-one-cycle-out CV on
**older cycles only** (1998–2016 general models; ≤2020 for the primary model); the headline
numbers come from **expanding-window evaluation** on the modern cycles the tuner never saw
(train strictly on cycles before the test cycle). See `METHODOLOGY.md` for exact windows.

**Why no single dataset exists:** there's no public file of "polls + who won," so several
sources are joined — see `DATA_SOURCES.md` for the full list (results, historical +
current polls, macro/approval/generic-ballot feeds, FEC fundraising, candidate
bios/history for the primary model). The **future-proofing rule**: production inputs are
raw polls + economic data only — nothing that only exists inside a defunct 538 file, since
2026+ can't get it. See `AGENTS.md` rule 6.

📄 **Deep docs:** [AGENTS.md](AGENTS.md) (start here if you're contributing — architecture +
the rules learned the hard way) · [HANDOFF.md](HANDOFF.md) (in-flight state + dated history)
· [CONCERNS.md](CONCERNS.md) (the living risk register + improvement roadmap) ·
[METHODOLOGY.md](METHODOLOGY.md) (**exact time windows for every feature**) ·
[DATA_SOURCES.md](DATA_SOURCES.md) (every URL + how found) ·
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) (every variable) ·
[MISSINGNESS_REPORT.md](MISSINGNESS_REPORT.md).

---

## Pipeline (run order)

```
1. build_dataset.ipynb      → polls (1998-2016 raw_polls_538 + 2018-24 historical, all
                               committed) joined with results → polls_long_with_results.csv
                               (14MB, committed for CI; regenerate via this notebook if missing)
2. fetch_approval.py        → approval feed (Gallup/UCSB 1993-2025 + VoteHub API 2025+)
   fetch_macro.py           → economy: DBnomics history + BLS-API overlay (current)
   fetch_generic_ballot.py  → generic-ballot monthly (re-run to EXTEND; committed, static otherwise)
3. model.ipynb              → WIN model: nested tune/eval, final fit on all 14 cycles
   margin_model.ipynb       → MARGIN model (separate artifact, same scheme)
   primary_model.py         → PRIMARY nominee model (script, runs in minutes)
4. predict.py / predict_margin.py / predict_primary.py
                             → score the live 2026 poll feed for each model
5. refresh_dashboard.py     → one command: feeds → predict all three → copy CSVs to the
                               companion dashboard repo → regenerate its compare pages
```

### All static data is pulled once and committed
Nothing in this pipeline re-downloads on a normal run. Historical polls, results, macro,
approval, generic ballot, FEC, candidate bios — all committed once, re-pulled only to
*extend* to a new month/cycle. The one exception is `predict*.py`'s live generic-ballot
fetch (current-cycle info can't be frozen by definition).

## Run

```bash
pip install -r requirements.txt
```

All commands run from the **repo root** (paths after the 2026-08-02 move — see STRUCTURE.md):

1. **`pipeline/build/build_dataset.ipynb`** — only needed if `polls_long_with_results.csv` is
   missing (it's gitignored at 15MB, so a fresh clone DOES need this step).
2. **`models/poll/model.ipynb`**, then **`models/poll/margin_model.ipynb`**, then
   **`py -X utf8 models/poll/primary_model.py`** — run top to bottom. **Run notebooks one at a
   time**, never concurrently (nbconvert races and overwrites outputs on parallel runs).
3. **`py -X utf8 refresh_dashboard.py`** — re-predict 2026 and refresh the companion dashboard.

Optional: **`py -X utf8 models/fundamentals/fundamentals_model.py`** trains the no-polling
reference models (not shipped to the dashboard; a prior/floor for thin-poll races).

> ⚠️ **Workflow rule:** whenever you change the model's feature set, **re-run the entire
> notebook end-to-end including the grid search** — never reuse old hyperparameters.
> Params tuned for one feature set can make a new set look worse than it is. Let
> regularization drop non-predictive features rather than hand-curating. Applies to all
> three models. (More rules + traps in [AGENTS.md](AGENTS.md).)

## Where to look next

`CONCERNS.md` "Improvement roadmap" is the ranked, actively-maintained backlog. Recurring
themes: a race-level two-party reframe (kills the win/margin split-model ambiguity),
snapshot training (mid-campaign honesty — training races' freshest poll skews much closer
to election day than a July forecast), and redistricting-aware House fundamentals.
