# Handoff: in-flight state, breakdown risks, next steps (2026-07-26)

For the next agent. Read AGENTS.md first (architecture + rules), CONCERNS.md second
(risk register + roadmap). This file: what's mid-flight RIGHT NOW, what's most likely to
break, and what to do next, in order.

## CURRENT STATE 2026-07-29 — office_level coverage = 100% OVERALL and WINNERS.

Every one of the 4,844 candidate-races now has a leak-free, defensible office_level.
check_officeholder passes. Sources: 28,582 wikipedia + 1,741 manual + 194 wiki_xref +
173 ballotpedia. This completes the whole backfill arc (Stages 1-4).
- **The zeros are VERIFIED, not assumed** (user insisted on this). Every level-0 person has
  NO office evidence in ANY source: no race win, no Wikipedia office descriptor (scanned even
  blank-leveled rows for office words outside candidacy phrasing), no Ballotpedia profile.
- **Audit caught 228 people who would have been WRONGLY zeroed** - incl 25 who won federal
  races (Duckworth, Lazio, Bera, Barr, DelBene...). All hand-coded with leak-free as-of-year
  offices in data/candidate_bios_manual.csv instead. Verified: Duckworth 2006->0 (VA roles +
  House all came later), Lazio 2000->4 (sitting US Rep), Denny Heck 2010->0 (not a US Rep till
  2013), Dino Rossi 2008->2 (fwd from 2004 state-senator row), Roger Moe 2002->2.
- **1,166 bulk verified-0** (no evidence anywhere) appended to candidate_bios_manual.csv with
  offices_json []; a descriptor safety-scan confirmed 0 hidden officeholders among them.
- measure_office_coverage.py --write now writes an EMPTY roster (nothing uncovered).
STILL: feature is opt-in / NOT shipped (the whole arc is data completeness). A future
re-ablation now runs on 100%-covered data - the strongest possible test. Re-run the ablation
(model.ipynb harness) before ever shipping bio_office_level.

## CURRENT STATE 2026-07-29 — non-winner backfill in progress + Wikipedia self-cross-ref.

Working toward non-winner coverage after WINNERS hit 100% (Stage 3). Overall now **72.6%**.
- **Wikipedia SELF-CROSS-REFERENCE (user asked "are you cross-referencing Wikipedia too?"):**
  a person's office history is on SOME of their Wikipedia race pages but not others (Dino
  Rossi: "state senator"=2 in 2004/2010/2016 but 0 on his blank 2008/2018 table rows). Added a
  builder step that propagates a person's own informative Wikipedia levels FORWARD ONLY - an
  office held as-of bio-year Y applies to their races in years >= Y (leak-free; offices
  persist). BACKWARD propagation was tried + REMOVED (it gave Denny Heck's 2010 race level 4
  from a 2024 "former US Rep" row, but he wasn't a US Rep until 2013 - a leak). Emitted as
  src="wiki_xref". Resolved ~240 rows from data already in hand, 70.8% -> 72.6% overall.
- **Ballotpedia non-winner sweep:** 826/1315 people looked up, but the TAIL yield is only ~5-8%
  (the remaining are mostly genuine no-profile losing challengers = verified level 0). Key
  finding: the soft-block guard's consecutive-miss stops were REAL ABSENCE, not blocks
  (verified Pelosi fetched fine mid-stop) - so the scraper now takes --miss-guard=N (large value
  for this set) + --max-per-run=N. Default miss_guard restored to 20 (user request). Ballotpedia
  IS now genuinely rate-limiting the IP (real RateLimited, needs multi-hour cooldown) after
  heavy use today.
- **REMAINING to 100% overall:** 1,234 uncovered people - 485 BP-checked-no-profile (safe to
  bulk verified-level-0) + 714 never-BP-reached (block cut off; ~7% would have profiles).
  Endgame: finish/limit the BP sweep, then bulk-append the confirmed-no-profile people to
  data/candidate_bios_manual.csv with offices_json [] (=verified level 0), rebuild, re-measure.
  Feature still opt-in / not shipped - data completeness only.

## CURRENT STATE 2026-07-28 — Stage 3 done: WINNERS office_level coverage = 100%.

Hand-coded office histories for all 248 uncovered winners no scraper could reach ->
**winners 100.0% (1,967/1,967), overall 70.8%** (honest measure, table-zeros=uncovered).
- data/candidate_bios_manual.csv (NEW, committed): per-person offices_json tenure dates
  (same leak-free format as Ballotpedia) + source_note, researched from public bio record.
  Merged into build_office_level_table.py's off_map; manual OVERRIDES Ballotpedia; an empty
  offices_json [] = "verified: no prior office" -> a real level 0 (distinct from unknown).
- THREE classifier/builder bugs found + fixed while wiring this in (all also affected real
  Ballotpedia data, not just manual):
  1. "Secretary of State" (a STATEWIDE office = 3) was matching the level-4 "secretary of"
     pattern -> Evan Bayh read 4 instead of his real max Governor=3. Level-4 now only matches
     U.S./cabinet secretaries.
  2. STATE-NAME legislature phrasing ("Idaho House of Representatives", "Connecticut Senate")
     didn't match the level-2 regex (which wanted the literal word "state") -> 43 hardcoded
     people read 0. Added a 50-state-name alternation to the level-2 pattern.
  3. A VERIFIED level-0 (manual/ballotpedia, candidate truly held no prior office) was
     indistinguishable from a Wikipedia table-zero (unknown) -> Renzi/Ellmers-type first-time
     winners counted as uncovered. Fixed in BOTH places: builder dedup now prefers
     src manual>ballotpedia>wikipedia on equal level (so the verified row survives), and
     measure_office_coverage.py counts any manual/ballotpedia row as covered regardless of level.
- Leak-free verified: Renzi 2002->0, Labrador 2010->2 (state house), Mark Sanford 2002->4
  (ex-US Rep), Bayh 1998->3 (ex-Gov), Klobuchar 2006->1 (county attorney). check_officeholder
  passes. Feature STILL opt-in / not shipped - this whole arc is data completeness; a future
  re-ablation now has winners at 100% coverage to run on.
- REMAINING (optional, lower value): 1,413 uncovered are all NON-winners (1,102 no_bio +
  311 table_zero) - losing challengers, many genuinely level 0. Roster in
  office_level_backfill_targets.csv; regenerate with measure_office_coverage.py --write.

## CURRENT STATE 2026-07-28 — Stage 2 (Ballotpedia) done + coverage DEFINITION corrected.

**Two things landed since Stage 1:**
1. **Stage 2 Ballotpedia backfill (winners+repeats):** drip-scraped 251/267 targeted people
   over ~11 cooldown cycles, 149 hits. Merged 161 leak-free rows. Also: `extract_offices`
   hardened (Assumed-office/In-office/dateless formats; dateless offices emitted with
   start=None so the builder skips them leak-safely instead of dropping them silently -
   found via Baron Hill's dateless infobox). And a builder PRECEDENCE fix: a Ballotpedia
   real-level row now overrides a Wikipedia "table-zero" row (level 0 from a blank
   results-table descriptor) - verified Schumer 1998 -> 4.
2. **COVERAGE DEFINITION CORRECTED (user decision "treat table-zeros as UNCOVERED"):** a
   level-0 bio row with a BLANK descriptor came from a results-table scrape that never said
   what office the person held - its level is UNKNOWN, not a confirmed 0. Counting it as
   "covered" hid real prior-officeholders (Tom Carper, DE Governor 2000, had only a
   blank-descriptor table row -> never queued for Ballotpedia). The authoritative measure is
   now `measure_office_coverage.py` (NEW, committed): covered = level>0 OR level-0-with-real-
   descriptor; table-zeros are uncovered. **Honest coverage: 62.7% overall / 82.7% winners**
   (the earlier 71-74%/87-93% figures counted table-zeros as covered). Regenerates
   data/office_level_backfill_targets.csv (1,808 uncovered: 1,270 no_bio + 538 table_zero;
   340 winners; winners-first). Run `py -X utf8 measure_office_coverage.py --write` after any
   bio change to refresh both the numbers and the roster.

**Stage 2 ROUND 2 (DONE):** re-ran Ballotpedia against the honest winners roster (292 people
incl. table-zero winners like Gray Davis->3, Carper-class). Coverage **62.7%/82.7% ->
64.2%/86.0%**. Also fixed a BUILDER MAPPING BUG: build_office_level_table.py used to map BP
hits against a transient file (uncovered_candidates.csv rewritten winners-only each round, or
the regenerated roster which excludes now-covered people) - both dropped BP hits for anyone
not in that particular file (BP rows swung 161->100->123). Now it maps against the POLL-FEED
GROUND TRUTH (every candidate-race in polls_long_with_results.csv, keyed candidate+state), so
a BP-resolved person's as-of-year level lands on ALL their races regardless of scrape round or
coverage status (BP rows -> 250, stable). Schumer 1998->4 + fact-check still clean.

**NEXT = Stage 3 (winners):** 276 uncovered WINNERS remain (down from 340); the rest of the
Ballotpedia misses increasingly have NO BP profile (60% hit rate = 40% genuinely absent), so
they need manual online lookup + hardcode. User chose (2026-07-28) to hardcode the 276
uncovered winners to reach ~100% winners. The full roster (1,732 uncovered: 1,226 no_bio +
506 table_zero) is in data/office_level_backfill_targets.csv; run
`py -X utf8 measure_office_coverage.py --write` to refresh it. Losing-challenger / no-bio
non-winners are lower-value and deferred. Feature still opt-in / not shipped (data pass only).

## CURRENT STATE 2026-07-27 — pre-2012 coverage backfill (Stage 1 of 3); committed.

**User asked for TRUE 100% office_level coverage (historical, one-time — "hardcoding and
finding answers online is fine").** Three-stage plan; Stage 1 done + committed:
- **Stage 1 (DONE):** root-caused the pre-2012 gap = Wikipedia rewrote its election-page
  template ~2012. Two parser additions in fetch_candidate_bios.py: (1) `parse_page`
  OLD-FORMAT mode (`cand_section` flag survives a party-only subheading → flat
  "Candidates→Democrats" bullets collect, real descriptors → real levels); (2)
  `parse_results_tables` (NEW) pulls NAME+PARTY from results wikitables for table-only big
  states, with "(Incumbent)" → synthesized descriptor → free level 4/4/3. Both wired into
  Senate/Gov + House scrapers (bullet-first, table-supplement). Pre-2012 re-scraped
  (+5,343 House bios; Sen/Gov rebuilt, pre-2012 originals archived per the rule).
  **Coverage 54.5%→71.1% overall, 68.8%→87.3% winners; pre-2012 era 32-48%→69-85%.**
  check_officeholder passes (7/7, 96%). Modern pages regression-checked unchanged.
- **Stage 2 (NEXT — user chose to commit Stage 1 first):** Ballotpedia person-level backfill
  for the 1,398 still-uncovered (data/office_level_backfill_targets.csv, regenerated;
  1,248 people, 249 winners; 1,009 are 2012+ losing challengers, 389 pre-2012 table-only).
  This fills table-only NON-incumbents that read level 0 (Schumer 1998, Carper 2000 — they
  show in the fact-check DISAGREE list, correctly flagged as needing person-level data).
  Ballotpedia hard-blocks ~every 25 reqs → many cooldown cycles. SCOPE STILL TO BE CHOSEN
  (winners+repeats-first vs full drip) — ASK the user before launching a multi-hour scrape.
- **Stage 3:** manual/hardcode the final stragglers with no page anywhere → true 100%.
NOTE: the ablation verdict is UNCHANGED by this — it's a data-completeness pass so a future
re-ablation runs on maximal coverage. Feature still opt-in / not shipped.

## CURRENT STATE 2026-07-26 — office_level table rebuilt LEAK-FREE (as-of-year); committed.

**DO NOT "fix" a candidate whose office_level differs across years — that is BY DESIGN.**
`bio_office_level` is now the highest office held **strictly before that election year**
(no look-ahead), so the same person legitimately reads different levels in different cycles
(Spanberger 2018→0 then 2020+→4; Lankford 2010→0 then 2012+→4). Full rule + verified
spot-checks are in METHODOLOGY.md ("Leak-free as-of-year office-level table"). If a value
looks wrong, check it against the strictly-before rule there BEFORE changing anything.

- **Authoritative builder is now `build_office_level_table.py`** (run after ANY bio-source
  change → rebuilds `data/candidate_bios.csv` fresh). It SUPERSEDES `combine_candidate_bios.py`,
  which is now DEPRECATED (it did a frozen-peak merge = a real look-ahead leak; kept per the
  archive rule, docstring says so). Where item 5 in the older 2026-07-24 entry below calls
  combine_candidate_bios.py "the ONLY writer of candidate_bios.csv" — that is now STALE;
  build_office_level_table.py is the writer.
- Sources: Wikipedia per-office files (contemporaneous, used as-is, PREFERRED on overlap) +
  Ballotpedia gap-filler (`candidate_bios_ballotpedia.csv`, person-level but carries per-office
  TENURE DATES in `offices_json`; builder computes as-of-year level from tenure starts).
- **Coverage: 68.8% of winner-rows** (1,357/1,971), up from 67.7%. VERY uneven by era:
  2012–2024 ≈ 88–100%, 1998–2010 ≈ 32–48% (older Wikipedia race pages are sparse).
- **Ballotpedia scrape is PARTIAL + RESUMABLE:** got 21 of 91 targeted 2012+ uncovered winners,
  then hit a persistent multi-hour IP block (tried 20-min cycles ×2 + a 60-min cooldown; block
  held). **59 winners still uncovered.** Resume later with
  `py -X utf8 fetch_candidate_bios_ballotpedia.py --winners-only` (skips done rows; progress
  saved, misses NOT poisoned). Then re-run build_office_level_table.py + check_officeholder.py.
  NOTE: don't do a probe/CLEAR-test fetch before resuming — it consumes the freshly-reset
  quota and re-triggers the block on the first real lookup.
- Two fixes shipped with this: build_office_level_table.py normalizes the uncovered-list's
  "S" statewide district → "" for Senate/Gov; features.load_candidate_bios uses crash-safe
  `dist_str()` instead of raw `int(district)` (was crashing on non-numeric district values).
- **RE-ABLATED on this corrected table 2026-07-26 — DEFINITIVE: still NOT shipped.** The
  mixed pattern REPLICATED on the clean leak-free data (expanding-window, fixed shared
  hyperparameters): calibration nudges positive (AUC +0.0006, Brier -0.0011) but race-acc
  goes DOWN -0.0028, same 2020-fold regression (-0.0199) as before. So it's not a leak
  artifact — poll features already carry this signal for the general election. bio_office_level
  stays opt-in (`feature_list(candidate_bios=True)`, OFF by default) in BOTH win + margin
  models; earns its keep only in the PRIMARY model. User confirmed "don't ship, just document"
  after seeing the numbers. Full table in METHODOLOGY.md. Do NOT re-open without a materially
  different input (e.g. coverage pushed past the pre-2012 wall). Production is UNCHANGED — no
  retrain/redeploy needed from this work.

## CURRENT STATE 2026-07-24 — bio_office_level fully investigated, FINAL VERDICT: not
## production. Coverage story + save-discipline refactor below; older entries follow.

**The whole bio_office_level arc, condensed (full detail in the dated sections below):**

1. First ablation (32.7% coverage): null on both models. User pushed back — "I think the
   issue is coverage. Let's start with 2024 who you are missing." **User was right.**
2. Investigating the missing-2024-winners list found the old scrape target list came from
   primary-POLL pages, systematically excluding uncontested/safe-seat incumbent races
   (36% of 2024 Senate races had NO page target; Whitehouse/Cantwell/Klobuchar/Sanders all
   unreachable). Also found + fixed two subsection-heading parser gaps ("Nominee",
   "Advanced to general") that dropped dominant incumbents even on scraped pages.
3. Rebuilt the target list from the RESULTS files (complete, unbiased): coverage
   **32.7% → 58.1% overall, 67.7% among winners** (57.3%/66.4% before a further
   jungle-primary parser fix; see the 2026-07-24 jungle-mode commit). `check_officeholder.py` still passes
   (7/7, 99% consistency, 20,969 total bios).
4. **Re-ablation with fixed coverage — the honest final result is MIXED, not binary:**
   - WIN model: calibration metrics ALL flipped positive (AUC +0.0003, AUC-PR +0.0002,
     KS +0.0019, Brier −0.0005) but race-acc still −0.0048, driven by the 2020 (−0.0132)
     and 2024 (−0.0149) folds. A matched-races-only split confirmed this is NOT a
     NaN-dilution artifact — the mixed result persists on races with real bio data.
   - MARGIN model: uniformly worse, MAE +0.0296 with regressions in ALL 4 eval cycles.
   - **VERDICT: not wired into production** (feature_list(candidate_bios=True) exists,
     opt-in, off by default). Better calibration doesn't outweigh worse pick-accuracy in
     the two most 2026-relevant folds + a uniformly worse margin model.
5. **Save-discipline refactor (user instruction, after a real data-loss incident):** the
   two bio scrapers both wrote to ONE shared candidate_bios.csv with independent resume
   logic; a re-run sequence silently discarded ~9,500 House rows once (recovered by
   re-scraping). Now: candidate_bios_senate.csv / _governor.csv / _house.csv are each
   written ONLY by their own scraper; **combine_candidate_bios.py is the ONLY writer of
   the merged candidate_bios.csv** every consumer reads. Old files archived (never
   deleted) under archive/ with timestamps — that's the standing rule for ALL data files.
6. **Verification sweep (user asked to re-check the session's work):** found + fixed —
   .gitignore missing the three per-office files (they'd have been uncommittable),
   a dead misleading `OUT` alias in fetch_candidate_bios.py, a resume-skip that was
   printed but never implemented in the House bio scraper (now real, which also makes
   re-runs fetch ONLY missing pages), stale coverage figures in features.py docstrings.
   Also self-caught and reverted TWO wrong diagnoses of the CA/WA/LA-House-empty problem:
   a "Washington needs URL disambiguation" theory (plain "Washington" fetches fine) and a
   "transient fetch failure" theory. **Real root cause (fixed 2026-07-24 jungle-mode
   commit):** CA/WA/LA are JUNGLE/top-two states — their House pages have NO party-primary
   headings; candidates sit under one "Candidates" list with party inline as "(Democratic),
   ...". parse_page required a party heading, so it collected nothing. Jungle-mode branch
   added (reads party per-bullet); CA now covered all cycles 2012-2024. Coverage nudged
   57.3%→58.1% total, 66.4%→67.7% winners (modest — most CA candidates already matched via
   other paths; the fix filled genuinely-absent ones). Lesson: inspect the page DOM before
   theorizing about the failure — two wrong guesses cost real time here.

**OPEN ITEMS (next agent, in order):**
- **Third ablation round — PROBABLY NOT WORTH IT, but the call is open**: coverage is now
  58.1%/67.7%-winners (was 66.4% at the last ablation). +1.3pt on winners is small enough
  that it's unlikely to flip the mixed verdict (win-model race-acc −0.0048, margin
  uniformly worse). Re-run the two ablation scripts in
  scratchpad if you want certainty, but don't expect a different answer.
- **Independents section gap**: Bernie Sanders (and any independent) is filed under an
  "Independents" heading that infer_section_context doesn't classify as primary-stage, so
  parse_page drops the bullet. Root-cause fix belongs in this repo's parse_page (a local
  stage override like the "Advanced to general" one), NOT in the shared polling-agg
  infer_section_context. User explicitly chose root-cause fix over a manual override file.
- Small classifier tail (documented, low priority): "Kansas Insurance Commissioner",
  "Land Commissioner" (4 rows), bare "representative from X's Nth congressional district"
  without a U.S. qualifier, ~200 comma-parse garbage names (1.4%).
- The same OFFICE-scoped question for fetch_house_primary_results_hist.py: it shares the
  transient-miss pattern (WA/LA/CA gaps) — re-run it too if primary-result features are
  ever revisited; its data is committed but has the same holes.

## OLDER: IN-FLIGHT STATE 2026-07-23 (superseded by the 2026-07-24 section above)

**RESOLVED since first written**: the House candidate-bio re-scrape finished cleanly
(14,309 total bios, 9,496 net new House rows 1998-2024) and is committed + pushed
(`2ae97a9`). `check_officeholder.py` re-run on the full combined file: 7/7 known-truth
checks pass, cross-source consistency **improved 96% → 99%** (the incumbent-context fix
resolved the Whitehouse/Cantwell/Cantor misclassifications — none of them appear in the
disagreement list anymore). Citation-link corruption down to 1.4% (was 14.4%).

### What happened this session, in order

**1. Primary-result features for the general model (user request) — DONE, ABLATED, DROPPED.**
Built `fetch_house_primary_results_hist.py` (new: House primary RESULTS 1998-2024, one
Wikipedia page per state per cycle, "X United States House of Representatives elections in
Y") to complement the existing Senate/Governor `fetch_primary_results_2026.py --hist`.
Combined: 4,900 House party-races + Senate/Governor, fact-checked (0/4900 races have other
than exactly one winner; 5 hard historical spot-checks pass incl. two sitting-incumbent
LOSSES — Cantor 2014, Meijer 2022). Found + fixed a real bug in the process: 11 Texas-2012
races had first-round + runoff results merged into one table (no separate "Runoff" heading
for the parser to key off) — pct columns summed to ~200%; two-pass guard added (see that
script's docstring for the exact logic and why a naive "top-2 sum to 100%" version would
have corrupted 69 OTHER, legitimate 3+ candidate races).

Added `features.load_primary_results()` + `primary_margin`/`primary_uncontested` columns to
`build_candidate_table` + `feature_list(primary_results=True)` (opt-in, same pattern as
`fund=True`). **Ablated on BOTH win and margin models (fixed hyperparameters, expanding-
window eval) — dropped from production**: every metric moved flat-to-worse in both models
(win: AUC -0.0001, race-acc -0.0031; margin: MAE +0.019, race-acc -0.0046),
`primary_uncontested` had near-zero feature importance in both. Matches the `poll_adj`
precedent exactly (real-looking feature, no honest out-of-sample value — polls already
price in primary-contest weakness by general-election time). **The loader/columns stay in
features.py** (harmless, committed data, may earn its keep on a future feature) but
`primary_results=True` is NOT the production default — do not flip it on without a fresh
ablation if this comes up again.

Files from this: `fetch_house_primary_results_hist.py`, `data/house_primary_results_hist.csv`
(11,218 rows), `data/primary_results_hist.csv` (found this was NEVER actually committed
despite being static-data-principle output — fixed the `.gitignore` gap same day), both
committed + pushed already (commit `0ddc88f`). The `features.py` load_primary_results()
addition is NOT yet committed (see below).

**2. General-election office-level feature (user request) — SCRAPING IN PROGRESS, TWO
REAL BUGS FOUND AND FIXED, NOT YET FACT-CHECK-CONFIRMED ON THE FULL FILE, NOT WIRED IN.**

The existing `candidate_bios.csv` (built for the PRIMARY model) only covers Senate/
Governor for 2018-2024 + House for 2026 only — because its historical target-page list
derives from `primary_polls_wikipedia.csv`, which was deliberately scoped Senate/Governor-
only (same root cause as the primary-results gap above). Built
`fetch_house_candidate_bios_hist.py` reusing the proven House-page-target machinery from
step 1, driving the EXISTING `fetch_candidate_bios.py` parser (`parse_page`/`classify`,
imported not reimplemented) over 1998-2024 House pages.

**Two real parser bugs found while fact-checking, BOTH ALREADY AFFECTING THE LIVE PRIMARY
MODEL, not just this new work — fixed in `fetch_candidate_bios.py` this session:**
  a. **Citation-link name mangling** (`li.find("a")` grabbed a footnote link `<a href=
     "#cite_note-...">[8]</a>` instead of skipping it when the candidate's real name was
     plain unlinked text, then the description-slicing logic cut the wrong number of
     characters — "Matthew W. Morgan" became "ew W. Morgan"). Affected **640 of 4,441 rows
     (14.4%) of the already-committed, already-in-production bio data.** Fixed by skipping
     `#cite_note` links when hunting for the name link, and finding the name's position in
     the text by content instead of assuming it's a literal prefix.
  b. **Incumbent-context ambiguity**: bare "incumbent senator" / "incumbent Representative
     [from X] since Y" (no "U.S." qualifier — Wikipedia relies on the reader knowing which
     page they're on) matched NONE of the office_level regexes, misclassifying 107 of 2,848
     "incumbent"-descriptor rows as level 0. Also missed "Majority Leader of the United
     States House of Representatives" (Eric Cantor) — a leadership-title phrase with no
     "member of" in it. Fixed: `classify()` now takes an `office` parameter and resolves
     these context-dependent phrases correctly per page (with a verified safety check that
     a Senate-page phrase does NOT get credited on a House page).

**Both fixes required RE-SCRAPING the already-committed Senate/Governor/2026 data** (the
old `candidate_bios.csv` was archived, not deleted, per user instruction — see
`archive/candidate_bios_20260723_102524_pre-namefix.csv` and
`archive/candidate_bios_20260723_191417_pre-incumbent-context-fix.csv`). The Senate/
Governor/2026 re-scrape with BOTH fixes is done (`py -X utf8 fetch_candidate_bios.py`,
log: `candidate_bios_rescrape_log2.txt`) — verified the 3 "incumbent senator" rows in that
slice now correctly classify as level 4. **The House historical re-scrape with both fixes
was STILL RUNNING when this session ended** (see the check command at the top of this
section).

### Next steps, in order

1. **Confirm/finish the House bio re-scrape** (see check command above).
2. **Run `check_officeholder.py`** (existing fact-check battery, no changes needed) on the
   full combined `data/candidate_bios.csv`. Expect it to pass (7/7 known-truth, consistency
   ≥85%) but LOOK AT THE DISAGREEMENTS PRINTED, not just the pass/fail line — that's how
   both bugs above were found even though the battery "passed" both times. Known small
   residual NOT yet fixed: ~88 rows (1.8%) where a bullet with no wikilinked name and an
   unusual comma structure produces an obviously-garbage name ("prison officer" instead of
   a person) — low priority, each is clearly wrong not silently wrong, but worth a look if
   there's time.
3. **Measure real coverage** against the general model's full candidate table (all offices,
   all 14 cycles) the same way it was measured for primary_results in step 1 — expect
   something in the 30-50% range based on the pre-House-fix Senate/Governor-only number
   (39.1% within Senate/Governor alone once the name-mangling bug was accounted for).
4. **Wire `bio_office_level` into `features.py`** (a `build_candidate_table` parameter +
   `feature_list()` flag, same opt-in pattern as `primary_results`/`fund` — do NOT make it
   default-on without ablating first).
5. **Ablate on the WIN model AND the margin model separately** (fixed hyperparameters,
   expanding-window eval) — do not skip either; primary_results' ablation this session only
   became trustworthy once BOTH were checked (win-only looked like a clean "drop it," but
   the user correctly asked to check margin too before finalizing). Use the ablation script
   pattern in this session's transcript (or reconstruct from `model.ipynb` cells 11/13/29 +
   `margin_model.ipynb`'s equivalent) — do NOT edit the notebooks directly until the
   ablation result is known.
6. **If it earns its keep**: also run a per-cycle overfitting check (seed sweep) before
   trusting it — this is the exact lesson from the PRIMARY model's own `bio_office_level`
   feature (METHODOLOGY.md: a 10-feature set looked good in aggregate but was "one-cycle
   luck," a per-cycle 6-seed sweep found only ONE of the ten features — office level itself
   — actually generalized). Don't skip this step even if the aggregate ablation looks good.
7. **Retrain both notebooks for real** (model.ipynb then margin_model.ipynb, never
   parallel) only after 4-6 confirm the feature is worth keeping.
8. **Commit + push.** Nothing from this session is pushed except the primary-results work
   (commit `0ddc88f`). Still uncommitted as of session end: `features.py`'s
   `load_primary_results()` addition, `fetch_candidate_bios.py`'s two bug fixes, the
   re-scraped `data/candidate_bios.csv`, `fetch_house_candidate_bios_hist.py`, and all the
   scrape log files. Write the commit message covering BOTH bugs found (cite the exact
   numbers above — 14.4% and 107/2848) since they affect the ALREADY-LIVE primary model,
   not just new work — this alone is worth a standalone commit even before the office-level
   feature is fully wired in and ablated, so the bugfix benefit reaches production sooner.
9. **Update METHODOLOGY.md's "PRIMARY nominee model" section** to note the bio-scraper
   bugfixes (it documents the ORIGINAL 3 fact-check iterations for this exact scraper —
   this session's 2 more bugs are a direct continuation of that story and belong in the
   same place, not a new section).

## Retrain "run 2/3" baselines (2026-07-06, STALE — superseded many times since; kept for
## historical reference only, see the 2026-07-21/22 entries below the "Found in the
## 2026-07-06 late audit" section for anything current)
Baselines to compare against from that era — run 1 (vintage fixes, 112 features):
WIN AUC .969 / Brier .069 / race-acc .859 (poll baseline .868);
MARGIN MAE 6.47 vs calibrated-poll 7.52 / raw 7.90.

## Breakdowns I can see happening (ranked by likelihood × damage)

1. **Silent feature mismatch between artifact and predict path.** predict.py does
   `X = cand.reindex(columns=meta['features'])` — if the saved model expects fund_* but
   predict didn't pass `fec=`, reindex fills them ALL NaN and predicts garbage *without
   erroring*. After any wiring change, assert: every feature in model_features.json exists
   in the built cand table (add this assert — it doesn't exist yet).
2. **Editing a notebook while nbconvert is executing it** — nbconvert rewrites the file at
   completion and your edits vanish. Also: on a cell error, nbconvert can leave the
   PREVIOUS run's outputs in place (looks like "no change"). Always check the printed
   feature counts/dates in outputs match expectations.
3. **polling-agg feed schema drift.** predict assumes: implied_prob = pct/100, race_id like
   `2026-SEN-ME` / `2026-H-AL-01`, stage contains 'general', end_date parseable. A scraper
   change upstream silently shifts all of these. Schema guards are STILL NOT IMPLEMENTED
   (roadmap) — predictions would be confidently wrong.
4. **VoteHub is now a triple single-point-of-failure** (approval 2025+, generic-ballot
   monthly 2024-12+, candidate for natl_env). If the API dies or renames poll_type values,
   fetch_approval soft-skips (approval goes stale) and fetch_generic_ballot loses its
   current segment. No alerting exists — a stale feed looks like a working one.
5. **races.csv incumbency vintage vs redistricting.** The frozen races.csv (July 2025
   vintage from 538's unmaintained repo) assigns 2026 incumbents to districts that ELEVEN
   states then redrew (data/redistricted_2026.csv). In redrawn states, is_incumbent /
   prior_margin_cand can be wrong-district. The dashboard flags these rows (REDRAWN badge)
   but THE MODEL DOES NOT KNOW. 41% of predicted House races are affected.
6. **CI failures are swallowed.** Both polling-agg workflows run model_compare.py with
   `|| echo skipped` — if it starts crashing (e.g. schema drift), the dashboard quietly
   serves stale model_data.js forever. Consider making it fail loudly or adding a
   freshness check to the page (predictions_as_of is displayed — watch it).
7. **GitHub Pages transient deploy failures** — site serves a stale build while the repo
   looks fine. `gh run list --repo pjmerica/polling-agg-2026` → `gh run rerun <id>`.
8. **The 2024 generic-ballot hole.** generic_ballot_* features are NaN for the 2024 cycle
   (no surviving source). Do NOT "fix" this with hand-typed numbers — that provenance
   pattern is what we spent a day eliminating. Needs a real archived source or stays NaN.
9. **Windows papercuts**: use `py -X utf8` for scripts that print (cp1252 chokes on unicode);
   `python` alias doesn't exist (use `py`); rm -rf __pycache__ after editing helper modules;
   CRLF warnings are noise.
10. **polls_long_with_results.csv is gitignored** (12MB). Fresh clone → run
    build_dataset.ipynb FIRST (fully offline) or everything downstream fails.

## 2026-07-16: model predictions now AUTO-REFRESH daily (GitHub Action)

polling-agg .github/workflows/model-refresh.yml (daily 13:15 UTC + manual dispatch):
clones THIS repo (public - no secrets), mirrors the local sibling directory layout,
scrapes both market venues (market CSVs are gitignored - absent on fresh checkouts),
runs refresh_dashboard.py --no-feeds (predict win/margin/primary + SHAP + compares),
commits to polling-agg. Verified end-to-end (run 29529497733 -> commit c9b3652).
CONSEQUENCES:
- Local runs of refresh_dashboard.py are now only needed after RETRAINING (new
  artifacts must be committed+pushed to THIS repo before 13:15 UTC to reach the site,
  or run the refresh locally / dispatch the Action manually).
- polls_long_with_results.csv is now COMMITTED (14MB; CI needs it for house
  effects/bias priors) - build_dataset.ipynb is no longer a fresh-clone prerequisite
  for predicting, only for regenerating that file.
- The model tabs' "model as-of" should never trail the poll feed by more than ~1 day;
  if it does, check the Model refresh workflow runs first.

## Next steps, in order

1. Finish the in-flight sequence above (verify run 2 → wire FEC → run 3 → publish).
2. **Add the missing asserts**: artifact-features ⊆ cand columns (predict), schema guards
   on the polling-agg loaders (pct∈[0,100], expected race count, required columns).
3. **Ask the user for two free API keys** (they were told; follow up):
   - api.data.gov key → FEC per-report as-of-Sep-30 totals (kills the Dec-31 caveat) +
     small-dollar share + true average-donation metric.
   - FollowTheMoney key → governor campaign finance (state-level; FEC is federal-only).
4. Redistricting-aware fundamentals: add `is_redistricted` feature and/or NaN the
   prior_margin/incumbency for redrawn districts (feature change ⇒ full retrain; consider
   bundling with the next feature batch instead of a standalone run).
5. After Nov 2026: MEDSL results loader → label 2026 → add cycle to cycles.py → retrain →
   **backtest the logged market edges** (polling-agg git history holds hourly market
   snapshots) before trusting edges for 2028 sizing.

## Model improvement possibilities (beyond CONCERNS.md's roadmap)

- **Race-level two-party reframe** and **distributional margin (quantile) model** — still
  the top two; they merge the split-model problem away and price Kalshi MOV ladders.
- **Snapshot training** (features as-of T days out) — biggest honesty gain for mid-campaign
  prediction, and it makes the FEC quarterly filings line up naturally.
- **Monotonic constraints** in XGBoost (win prob non-decreasing in poll_lead / fund_share)
  — cheap guard against weird extrapolation in thin-poll races.
- **Overperformance target** (actual margin MINUS polled margin) — removes the dominant
  poll signal from the target so fundamentals/money get full credit; often better-behaved
  than raw margin.
- **Isotonic calibration per office** on out-of-fold predictions; ship the α≈0.5
  model/poll-softmax blend that already wins the blend sweep.
- **Uncertainty-aware market comparison**: scale edge by prediction uncertainty (poll count,
  model disagreement) → a proper Kelly-ish sizing signal instead of raw edge sorting.

## Found in the 2026-07-06 late audit (numbers verified)
- **Dual-seat race collisions.** Training: 9 races (43 candidate rows) merge two same-state
  contests into one race_id (dual Senate seats: MN-2018, OK-2014/2022, NY-2010, MS/WY-2008,
  SC-2014, NE-2024; House special NY-19-2022) — two `won=1` rows per "race", race-relative
  features computed across candidates who never faced each other. Fix: add seat/special
  disambiguation to the race key in build_dataset.ipynb (poll files carry seat_name; results
  carry `special`), then retrain. Predict-time version FIXED same day (FL/OH 2026 specials
  now keyed `..._Senate-S`).
- **Cycle-correlated poll bias is large and unhedged.** Mean signed poll-margin error (D−R)
  by cycle swings from −3.9 (1998, 2012) to +6.7 (2020): within a cycle, errors share a
  common component of ±4-7 points. Per-race MAE (~6.5) understates PORTFOLIO risk for
  betting: if 2026 polls share a bias, every model edge moves together. Mitigations to
  build: cycle-bias prior feature (prior cycles' signed error by state/party, shrunken),
  and report edge-portfolio exposure by party on the dashboard.

## Retrain batch 4 (staged 2026-07-06, run AFTER run 3 publishes)
Feature-changing items implemented-or-designed but NOT yet in the training path:
1. **Dual-seat fix** — build_dataset.ipynb already patched (source-only): specials get
   district 'S' from results `special` flag + seat class (polls) + '-GS_' race marker
   (raw_polls); House special NY-19-2022 dropped. RE-RUN build_dataset first (regenerates
   polls_long_with_results.csv), THEN both model notebooks.
2. **Cycle-bias prior feature** (to build in features.py): mean signed poll-margin error of
   the state's (or region's) races in PRIOR cycles, shrunken toward the national prior-cycle
   mean. Strictly prior-cycle info = leak-free.
3. **Shrunken house effects**: multiply each pollster's dev by n/(n+K), K≈5 polls.
4. **is_redistricted feature** from data/redistricted_2026.csv (0 for all training cycles,
   1 for 2026 redrawn-state House rows) + consider NaNing prior_margin_cand there.
5. **Tune the poll-softmax baseline temperature** on TUNE_CYCLES before comparing (currently
   hardcoded 3.0 — potentially understates the baseline).
Then: run build_dataset → model.ipynb → margin_model.ipynb → refresh_dashboard → publish.

## Bias-robustness tooling (shipped 2026-07-06, active at predict time — no retrain needed)
- predict.py / predict_margin.py now emit win_prob_R3 / win_prob_D3 (pred_margin_R3/D3):
  re-predictions under a uniform ±3-point national poll shift; races whose PICK flips are
  marked `bias_fragile` and badged ≈ FRAGILE on the dashboard (treat as no-edge).
- Model tab meta line shows EDGE EXPOSURE (n D-lean vs R-lean edges): if your open edges
  are lopsided toward one party, you're making ONE correlated bet on 2026 poll bias.
- Tab shows a DATA STALE banner when model_data.js is >30h old (CI swallows compare
  failures; this makes them visible).
- refresh_dashboard.py runs a feed-freshness watchdog (approval/GB/unemployment ≤2mo lag,
  sentiment ≤13mo) — soft upstream deaths now print loud FEED STALE warnings.
- predict loader schema guards: required columns, pct∈[0,100], date parse rate, min races.

## Incident 2026-07-10: dashboard race collapse (41 -> 10) — resolved
Breakdown risk #6 ("CI failures are swallowed") materialized, chained with a new one:
data/raw/primaries.json is GITIGNORED, so CI market-refresh runs crashed the compare step
(FileNotFoundError, silenced by `|| echo`); Daily-refresh runs scraped a fresh Ballotpedia
calendar that only lists UPCOMING elections, so past primaries vanished and the
primaries-decided filter collapsed (23 states -> 8 -> 10 races, no governors). Local runs
looked fine (older, fuller primaries.json on disk) — classic works-on-my-machine.
FIX (polling-agg commit 47431f1): primary dates accumulate in committed
data/processed/primary_calendar_2026.json (max-date-per-state, merge-only, seeded with the
June calendar); compare tolerates a missing raw file. Plus model_predictions_as_of.txt
sidecar (CI checkouts reset mtimes — the staleness display was lying).
LESSON: any CI-consumed input must be committed or the consumer must tolerate its absence;
scraped "calendar" pages forget the past — accumulate, never replace.

## 2026-07-12: poll-data audit, population labels, party overrides
Full audit of the poll feeds + display + party handling. Details in the polling-agg repo's
**POLL_DATA_AUDIT.md**; the model-repo pieces:
- **Population labels (LV/RV/A/V)**: scrapers now capture surveyed population; the historical
  `polls_long_with_results.csv` already has a `population` column, so making it a MODEL
  feature later is a features.py change + retrain (not done yet — display-only for now).
- **`data/candidate_party_overrides.csv`** (NEW, committed): `model_party` + `display_party`
  columns. Osborn (NE-Sen) = model DEM / display IND — the "effective-party slot" pattern
  (see AGENTS.md rule 9). Loeffler = REP/REP (plain correction). Applied in predict.py; flows
  through features.py (`display_party` column) to predict/margin/explain output + the tab.
- **`data/dropped_out_2026.csv`** (NEW, committed): Duggan (MI-Gov withdrew) + NE-Sen fringe
  Dems (Burbank/Forbes) who diluted the Osborn two-way. predict.py drops their poll rows.
- **explain_2026.py** now reports `display_party`; the Model-vs-Markets tab marks the Dem-slot
  candidate's real affiliation, e.g. "Dan Osborn (I)".
MAINTENANCE: both CSVs are hand-maintained 2026 lists. When a candidate drops out or a party
label is wrong/independent-challenger, add a row and re-run refresh_dashboard.py. `cand_key`
= features.norm_name output ("lastname firstinitial", e.g. `osborn d`).
OPEN: population as a model feature (adult-poll downweight); the SHAP "leading Democrat"
pick can still land on a fringe candidate if the real challenger is an unfixed independent.

## 2026-07-12 (later): dropped poll_adj feature after ablation
User asked how the model looks with/without the pollster house-effect adjustment. Ran an
honest LOCO ablation (2018-2024). Findings (verified):
- **poll_adj** ranked #4/187 importance in BOTH models but removing it: win AUC .9683->.9679
  and race-acc .8623->.8620 (both noise), margin MAE 6.347->6.313 (slightly BETTER). It's
  ~redundant with poll_avg (poll_adj = poll_avg minus a per-pollster house-effect shift), so
  XGB re-routes the signal. Plus a train/serve hazard: house-effect table matches ~67% of
  2026-feed pollster names, so poll_adj is computed on a different basis for a third of future
  polls. **Decision (user): drop it.**
- Also measured for reference: Group A minus {poll_adj,bias_prior_cand,avg_sample} and the
  recency/dynamics block — see git history if you want fuller numbers. Only poll_adj was cut.
IMPLEMENTATION: removed from `feature_list()` in features.py (the column is still built in
build_candidate_table, harmless). Full re-tune of both models. compute_house_effect /
candidate_poll_adj machinery left in place (unused by features, cheap to keep).
LESSON: high XGB importance != marginal value when a feature is collinear with a kept one;
always ablate before trusting importance, and prefer cutting features that behave differently
on future data.

## 2026-07-14: full pipeline audit → Model-vs-Markets fixes + macro silent-zero retrain

Full audit of model + polling-agg Model-vs-Markets chain. Training/eval side came back
CLEAN (no label leakage: bias priors strictly past-cycle, Sep-30 macro windows, no
post-election polls, dual-seat specials keyed, win_prob_norm sums to 1). The page side had
two classes of FAKE edges, both fixed the same day:

1. **model_dem summed ALTERNATIVE primary candidates.** model_compare.py summed
   win_prob_norm over all Dem rows; races carrying leftover hypothetical-matchup polls
   (6 Dems in ME-Sen) published nonsense (ME-Sen "87% Dem" vs Kalshi 51.5 = fake 35-pt
   edge; NE-H-02 and two CA races showed 100%). FIX: model_dem = LEADING Dem's win_prob
   renormalized vs LEADING Rep's (matches the market's D/(D+R) vig normalization), plus an
   `unresolved_field` flag (page badge "? FIELD") when >1 same-party candidate survives the
   stale filter. ME-Sen honest number: Jackson 78.7 vs market ~52 — a real disagreement now,
   flagged. (User decision: leading matchup = Troy Jackson.)
2. **Independent-slot races compared different events.** NE-Sen: model P(Osborn)=.221 was
   compared to "Will the DEMOCRATIC party win" (.007) = fake 21-pt edge. Both venues list
   independent-win markets (Kalshi .32 / Poly .295); model_compare now matches those when
   dem_display != DEM (`slot_market='IND'`, page badge "I MKT"), normalized over the full
   D+R+I book. Honest edge: ~-9 pts (market likes Osborn MORE than the model).

Other fixes shipped with it:
- **macro_features.py silent zeros**: _trend (<2 obs) and _last12_delta (<13 obs) returned
  0.0, not NaN. 15 TRAINING values were wrong (2018/2020/2022 generic_ballot trends claimed
  "flat" from single-point windows) and 2026 sentiment_last12_delta silently asserted "no
  change" from year-old data. Feature-VALUE change ⇒ full re-tune+retrain of both models.
- **Expanding-window eval cells** added to both notebooks (train strictly < test cycle;
  the LOCO-minus-expanding gap measures LOCO's optimism from training on future cycles).
- **predict.py meta sidecar** (predictions_2026_meta.json): newest poll end_date consumed +
  natl_env used; dashboard meta line shows "newest poll used" so stale-poll edges are visible.
- **market-refresh.yml never committed docs/model_data.js** (regenerated every 2h, thrown
  away — model tab's market side actually updated only 2x/day). Fixed; both workflows also
  now fail LOUDLY on a compare crash (after the data commit) instead of `|| echo` swallowing.
- **Page bugs**: mv-meta line was rendering raw JS source (string concat inside the template
  literal); vig-normalize single-sided/ladder anchors; margin symmetrization uses each
  party's leading candidate.
- **natl_env train/serve mismatch documented** (538 model avg @ eve vs Wikipedia aggregator
  mean @ today) — METHODOLOGY.md section C.

UPDATE (same day): the measured LOCO-vs-expanding gap came back ~zero for the WIN model
(AUC +0.0015, AUC-PR +0.0003), so per user ("if it's negligible, let's do it the correct
way") **EXPANDING-WINDOW IS NOW THE PRIMARY HONEST EVAL** in both notebooks; LOCO is
demoted to a companion table that monitors the gap each retrain; the poll-baseline
benchmark + blend sweep train expanding-window too. Tuning stays LOCO-inside-1998-2016
(internal selection, no honesty issue, better use of the small old-cycle set — see
METHODOLOGY.md). Headline win-model numbers under the new scheme: AUC .966 / AUC-PR .947 /
KS .812 / Brier .072 / race-acc .864.
CAVEAT found when the margin run landed: the MARGIN model's gap is NOT negligible —
LOCO 6.23 vs expanding 7.24 MAE, entirely the 2018 fold (10.2 with only 10 train cycles;
2020/2022/2024 = 6.7/6.2/5.9 converge). The honest headline is now "MAE 7.24 over
2018-2024 as-they-happened", with the 2026-relevant fold (2024: 13 train cycles) at 5.87
and gap-free. Quote the margin model accordingly — do NOT keep citing 6.23/6.24.

## 2026-07-15: PRIMARY nominee model + "Primary vs Markets" page (built same-day)

User asked how fast a primary model could be built; answer: same day. Third separate model
(P(candidate becomes party nominee)); full design in METHODOLOGY.md "PRIMARY nominee model".

DATA STORY (the hard part):
- **NEGATIVE FINDING (do not re-attempt): 538's downloadable poll CSVs NEVER carried
  downballot regular-primary rows in any era.** Verified against in-season Wayback captures
  (Apr-2022 = generals+jungle only despite Oz/McCormick raging; May/Aug-2020 same) and
  post-season ones. The committed *_historical.csv were never filtered — there was nothing
  to filter.
- Recovery: **Wikipedia race pages' primary polling tables**, scraped by driving the
  polling-agg 2026 wikipedia scraper's OWN parser over 2018-2024 page titles
  (fetch_primary_polls_wikipedia.py) → 11,466 primary poll rows, 324 party-races.
  One parser for train AND serve.
- **Primary DATES = hand-entered static table** (fetch_primary_dates.py) after THREE failed
  iterations of Wikipedia prose-mining (filing deadlines/runoffs/news dates near 'primary'
  kept winning; poll-anchored windows got dragged by contamination). Dates are verifiable
  calendar facts — the hand-typed-data ban is about measurements. NY is per-office (2018/
  2022 split primaries). Every date cross-checked against the poll record; warnings printed.
- **Contamination guard**: some pages leak GENERAL polls into primary sections (NC-2022).
  build_primary_dataset drops rows dated after their primary (+2d): 3,062 rows dropped.
- Labels: nominee = the party's general-election candidate (res_*.csv join) — 96% of races
  matched. Kept: 214 contested labeled races / 6,666 poll rows / 1,076 candidate rows.

MODEL (primary_model.py — a SCRIPT, runs in minutes, not a notebook):
- Tune LOCO on ≤2020, honest eval EXPANDING-WINDOW on 2022+2024 vs poll-leader baseline.
- **Headline: AUC .971 / AUC-PR .918 / KS .865 / Brier .046 / race-acc .895 vs poll-leader
  .723** (2022: .924/66 races; 2024: .865/37). The +17pt pick edge over the naive baseline
  comes from recency features (poll-leader uses stale means; primaries break late).
- **FUND FEATURES EXCLUDED from the artifact (leakage)**: FEC receipts are cycle-END totals
  and nominees raise most money AFTER winning the primary → fund_share partially encodes
  the training label, while a mid-2026 candidate has no such money (train/serve skew).
  Measured: identical picks without fund (.895 = .895); only Brier differs (.035 vs .046 =
  the leak-suspect juice). Revisit only with as-of-primary-date FEC reports.
- Macro: available per-primary-date via build_macro_asof() (also the machinery snapshot
  training needs) but ~noise on 214 races — artifact ships NO-macro; both ablations
  re-measured and printed on every training run.

SERVE: predict_primary.py (feed stage='primary', per-race dates from *_current.csv →
primaries.json → calendar; same dedup/stale/junk filters; jungle states CA/WA/LA/AK
excluded) → 183 races, 62 upcoming. polling-agg analysis/model_compare_primary.py prices
vs POLYMARKET candidate primary markets (Kalshi has none downballot with usable ids;
its 'nominee' markets are 2028-VP trivia) → docs/primary_model_data.js → new
"Primary vs Markets" tab (upcoming primaries only — inverse of the general tab's filter).
Thin-poll races (<3 polls) sort LAST regardless of edge size: a 90-pt "edge" off one stale
survey (CT-Gov REP: 1 poll vs a 97% market favorite = Stewart presumably out) is the market
knowing something the polls don't. Both CI workflows run + commit the primary compare.

SAME-DAY UPGRADES (2026-07-15, after user review):
- **Explain modal** on the Primary tab (explain_primary.py -> SHAP top-10 per predicted
  nominee; reuses explain_2026's label machinery).
- **RESULTS-BASED LABELS** (user: "the primary winner might not be the one in the general"
  — Platner): fetch_primary_results_2026.py scrapes actual primary-results tables (2026 +
  --hist 2018-24). THREE parser bugs caught by known-winner validation before they poisoned
  anything: (1) Lt-Gov primary tables nested in gubernatorial primary sections crowned
  running mates (Husted over DeWine, "Fetterman" as 2018-PA-gov); (2) joint-ticket cells
  ("Richard Cordray and Betty Sutton" -> 'sutton r') — take the first wikilink; (3)
  '(withdrawn)' annotations leaking into name keys. Also: NEVER trust `cmd | tail` exit
  codes — a masked crash shipped an empty hist file once.
- **Nickname-alias merging** (features_primary.merge_nickname_aliases, train+serve): same
  person split across name variants ('Bobby'/'Robert Charles') diluted their own polls —
  36 merges in the 2026 feed alone. Same-last-name + nickname-equivalence, within-race only
  (Mayra vs Eric Flores stay distinct). The GENERAL model's feed likely has the same issue
  — queued as roadmap item #21.
- **Population splits** (user request): poll_avg/last/last30/std/n_polls/lead per LV/RV/A.
  Ablation: identical picks, Brier slightly better -> kept.
- **New headline** (corrected labels): race-acc .910 / AUC-PR .923 / Brier .046 vs
  poll-leader .723. **2026 backtest vs actual winners: 84 contested decided primaries,
  picks 81.0% vs poll-leader 67.9%, AUC .962.** (The pre-correction v1 artifact scores
  85.7% on the same set; the 4-race delta is binomial noise at n=84 and was isolated to
  the label/tune changes, NOT the population features — v2 stands because its labels are
  verifiably correct. Both numbers recorded here deliberately.) The value-add stat: in
  the 19 races where model and poll-leader disagreed, model right 14 / polls 3 / both
  wrong 2. Full reproducible scorecard: **analysis/primary_backtest_2026.ipynb**
  (executed outputs committed; bump AS_OF + re-run the results scraper to extend after
  the August primaries). Corrected labels also cleared two fake
  high-confidence "misses" (NM-Gov "Haaland lost" was an LG-table artifact — she won;
  ME-Gov was the Bobby/Robert duplicate). Remaining high-confidence misses are ALL House
  races — the office the training set lacks; treat House primary probabilities as softer.

AUDIT ITEMS DEFERRED (user: "address after"): snapshot training (top priority — training
races' freshest poll is median 6d pre-election vs ~112d for 2026 predictions, so honest-eval
numbers do NOT describe July forecasts), race-level two-party reframe, shipping the α≈0.5
blend, has_result selection bias, is_incumbent still wrong-district in redrawn states,
grid-search selects on AUC not Brier, edge/uncertainty weighting, backtest logging schema.
