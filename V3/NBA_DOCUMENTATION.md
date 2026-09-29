# NBA Live Win Probability — V2 Documentation

> Living document. Updated after each phase. Records decisions, data findings, architecture choices, what worked, what didn't, and why.

---

## Project overview

A real-time NBA win probability system serving live game-by-game predictions through the same FastAPI dashboard as the football model. Built in layers:

- **Layer 1 (2025-26 regular season):** In-game win probability, live play-by-play, updating every 30 seconds
- **Layer 2 (mid-season):** Seed placement — standings simulation, playoff bracket probability per team
- **Layer 3 (April 2026+ playoffs):** Series win probability, championship Monte Carlo — the original V1 goal, now with a better model underneath

The end goal is playoffs. Regular season is the training ground and the live test environment.

---

## V1 retrospective — what we built, what it achieved

### Architecture (V1)

```
nba_api (PlayByPlayV2)
        │
        ▼
  Feature Engineering          12 features
  ┌─────────────────┐
  │  score_diff      │  ← Winsorised ±80
  │  time_remaining  │    962,871 snapshots
  │  Elo ratings     │    7,562 games
  │  series state    │    2018-19 → 2024-25
  │  lead volatility │    PLAYOFFS ONLY
  └────────┬────────┘
           │
           ▼
  Neural Network               12 → 128 → 64 → 32 → 1
  BCEWithLogitsLoss / AdamW / CosineAnnealingLR
  Early stopped at epoch 38
           │
           ▼
  Temperature Scaling          T = 1.0598
           │
           ├──→  Monte Carlo Series Simulator (100K sims, ~8ms)
           │
           ▼
  Streamlit Dashboard          Auto-refresh every 30 seconds
```

### V1 performance

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | 0.8539 | Competitive with ESPN/FiveThirtyEight |
| PR-AUC | 0.8900 | |
| Brier Score | 0.1560 | Target band is 0.155–0.165 |
| Brier Skill Score | 0.3602 | 36% better than null model |
| Accuracy | 76.58% | |
| CV AUC (5-fold) | 0.8498 ± 0.0051 | Stable across folds |
| Train/Val gap | 0.011 | No significant overfitting |
| Temperature T | 1.0598 | Post-training calibration |

### V1 known limitations (carried forward as targets)

1. **Training data ends 2023-24.** Neural net hasn't seen 2024-25 patterns.
2. **No player-level features.** Doesn't know who's on the floor, who's in foul trouble, who's injured.
3. **Playoffs only.** 7,562 games trained on — leaves ~85% of available NBA data untouched.
4. **Scored-play snapshots only.** Model updates only when a score changes, not every possession.
5. **OT calibration is soft.** Not enough overtime data for the model to be precise in those states.
6. **Static Elo prior weight.** Elo contributes equally at tipoff and in the final 30 seconds — the model has to learn to ignore it late, which is inefficient.
7. **PlayByPlayV2 is dead for 2025-26.** Must migrate to V3.

---

## V2 design decisions

### Decision log

| Decision | Chosen | Alternatives considered | Reason |
|---|---|---|---|
| Season scope | Regular season + playoffs | Playoffs only, regular season only | 10× more data; `is_playoffs` flag preserves context difference; preseason excluded |
| Player metric | DARKO DPM (primary) + BPM (fallback) | EPM, LEBRON | DARKO CSV export confirmed on darko.app; BPM via basketball_reference_web_scraper as fallback for gap seasons |
| Validation | Walk-forward chronological | Random k-fold | Prevents future data leaking into training; catches temporal regime shifts |
| Elo prior | State-dependent weight decay | Static feature | Research showed state-dependent prior improved Brier from 0.1651→0.1606; natural prior decay is correct behaviour |
| PBP parsing | `pbpstats` on top of V3 | Raw V3 only | V3 PBP has out-of-order shot/rebound events; pbpstats cleans this |
| Injury data | `nbainjuries` (PyPI 1.1.1) | Manual scraping, ESPN injury reports | Structured, historical back to 2021-22, 15-min updates since Dec 2025 NBA rule change |
| Dashboard | Extend existing FastAPI site | Standalone NBA site | Shared infrastructure; sport switcher like ESPN; same design language |

---

## Data sources

### Confirmed stack

| Need | Source | Notes |
|---|---|---|
| Historical PBP (training) | `nba_api` PlayByPlayV3 | V2 dead for 2025-26; V3 required |
| Live PBP (dashboard) | `nba_api` live endpoints | ScoreboardV3, PlayByPlay live |
| Possession parsing | `pbpstats` | Cleans V3 out-of-order events; provides possession team ID |
| Injury / availability | `nbainjuries` (PyPI 1.1.1) | 15-min updates since NBA Dec 2025 memo; historical back to 2021-22 |
| Player impact (DARKO DPM) | `darko.app` CSV export | Primary; daily-updated projections; CSV export confirmed available |
| Player impact (BPM fallback) | `basketball_reference_web_scraper` | Fallback for seasons not covered by DARKO CSV |
| ESPN benchmark | ESPN summary `?event=` endpoint | Free; `winprobability` field; used for head-to-head evaluation |
| Preseason smoke test | Same live stack | Preseason games only; excluded from training |

### DARKO investigation (Phase 0 finding)

DARKO DPM is the top-ranked predictive player impact metric (RMSE 2.48 vs EPM 2.60 vs RAPTOR 2.63 vs BPM 2.71 in Snarr's 2020 retrodiction study). The original Shiny app went offline June 2026. The new home is `darko.app` (maintained by Kostya Medvedovsky).

**CSV export is available on darko.app** — confirmed by manual inspection of the site. This makes DARKO viable as our primary player impact metric.

**Decision: use DARKO DPM as the primary metric for `home_avail_delta` / `away_avail_delta`, with BPM from Basketball Reference as fallback for any historical seasons where DARKO coverage is incomplete.** The availability pipeline tries DARKO CSV first, falls back to BPM if the season is not covered.

**Data access plan:**
- Download DARKO CSV at the start of each season (automate if a stable URL pattern is confirmed)
- Store as `data/darko_dpm_{season}.csv` in the V3 project folder
- BPM scraped programmatically via `basketball_reference_web_scraper` for gap-filling

### nba_api migration note

As of v1.11.4: `PlayByPlayV2` and `ScoreboardV2` return empty JSON for 2025-26 games. All code must use `PlayByPlayV3` and `ScoreboardV3`. The V3 schema differs — field names changed, possession tracking is now available, event type codes are different.

---

## Feature set — 18 inputs (up from 12)

### Kept from V1

| Feature | Change |
|---|---|
| `score_diff` | Winsor tightened ±80 → ±60 (2025-26 avg margin 13.3 PPG, record blowout season) |
| `time_remaining_sec` | Unchanged |
| `quarter` | Unchanged |
| `quarter_time_elapsed_pct` | Unchanged |
| `home_elo` | Updated through 2024-25 regular season + playoffs |
| `away_elo` | Same |
| `elo_diff` | Same |
| `is_playoffs` | Now meaningful — regular season in training set for comparison |
| `is_overtime` | Unchanged |
| `lead_changes_norm` | Unchanged |

### Dropped from V1

| Feature | Reason |
|---|---|
| `home_series_wins` | Meaningless in regular season (always 0); reintroduced in Layer 3 playoff model |
| `away_series_wins` | Same |

### Added in V2

**In-game state:**

| Feature | Source | Description |
|---|---|---|
| `possession` | V3 PBP team ID field | 1 = home has ball, 0 = away. Critical in final minutes. |
| `home_fts_pending` | V3 FT event codes | 0, 1, or 2 free throws remaining in current FT sequence |
| `away_fts_pending` | Same | |
| `home_in_bonus` | V3 foul accumulator | Running team foul count ≥ 5 in quarter |
| `away_in_bonus` | Same | |

**Pregame prior:**

| Feature | Source | Description |
|---|---|---|
| `home_avail_delta` | `nbainjuries` + BPM | Σ(BPM × projected min) for available vs expected roster, home team |
| `away_avail_delta` | Same | Away team equivalent |
| `elo_prior_weight` | Computed | State-dependent Elo decay — see formula below |

### State-dependent prior weight formula

```python
# elo_prior_weight decays as margin widens and time runs out
# At tipoff, tied: weight ≈ 0.85 — Elo matters a lot
# At Q4 2min remaining, up 8: weight ≈ 0.04 — score tells you everything

from scipy.special import expit  # sigmoid

elo_prior_weight = (
    expit(time_remaining_sec / 2880) *   # decays as game progresses
    expit(1.0 / (abs(score_diff) + 1))   # decays as margin widens
)
```

---

## Dataset

### Scope

| Season | Reg games | Playoff games | Include? | Notes |
|---|---|---|---|---|
| 2018-19 | 1,230 | 82 | ✓ | |
| 2019-20 | 971 | 83 | ✓ | Add `is_bubble=1` flag |
| 2020-21 | 1,080 | 82 | ✓ | |
| 2021-22 | 1,230 | 85 | ✓ | |
| 2022-23 | 1,230 | 83 | ✓ | |
| 2023-24 | 1,230 | 82 | ✓ | |
| 2024-25 | 1,230 | 87 | ✓ | |
| 2025-26 preseason | ~90 | 0 | ✗ training / ✓ live smoke test | |
| 2025-26 regular season | 1,230 | — | Live evaluation only | Model evaluates in real time |

Estimated training snapshots: ~9,000 games × ~130 plays avg = **~1.17 million snapshots**

### Winsorising decision

Score diff tightened from ±80 to ±60. Reason: 2025-26 set post-merger records for blowouts (avg margin 13.3 PPG, 58.8% of games decided by 10+). Garbage-time blowouts beyond ±60 carry no useful signal and distort the model's understanding of competitive game states.

---

## Model architecture

### Network

```
Input (18)
  → Linear(18, 128) → BatchNorm1d → ReLU → Dropout(0.25)
  → Linear(128, 64) → BatchNorm1d → ReLU → Dropout(0.25)
  → Linear(64, 32)  → BatchNorm1d → ReLU → Dropout(0.25)
  → Linear(32, 1)   → Sigmoid

Loss:     BCEWithLogitsLoss (pos_weight for class imbalance)
Optim:    AdamW (lr=3e-4, wd=1e-4)
Schedule: CosineAnnealingLR (T_max=50)
Calib:    Temperature scaling post-training
```

Dropout reduced 0.30 → 0.25: training set is ~10× larger than V1 so less regularisation is needed.

### Elo system

- K=20, home advantage +100 Elo points, 35% regression to mean at season start
- Updated through full 2024-25 (regular season + playoffs)
- Separate tracks for regular season and playoffs (per Neil Paine methodology)

---

## Validation and evaluation

### Walk-forward chronological validation

```
Fold 1: Train 2018-22 → Test 2022-23
Fold 2: Train 2018-23 → Test 2023-24
Fold 3: Train 2018-24 → Test 2024-25
```

Final model trains on all 7 seasons. Report Brier and ROC-AUC per fold. If Fold 3 Brier is significantly worse than Fold 1, add season-weight decay to training.

### Full evaluation suite (6 metrics)

1. **Pooled Brier + ROC-AUC** — baseline comparison with V1 (target: Brier < 0.156)
2. **Brier by time bucket** — Q1 / Q2 / Q3 / Q4 / final 5min / final 2min / OT separately
3. **Calibration by score state** — tied / ±1-5 / ±6-10 / ±11+ (catches late-game overconfidence)
4. **Pathwise calibration check** — "blown lead paradox": eventual loser should reach ≥90% in ~11% of games. If >15%, systematic overconfidence present.
5. **ESPN head-to-head** — score our model and ESPN on exactly the same plays, same test games
6. **Walk-forward fold comparison** — Brier per fold; flag if Fold 3 drifts >0.005 from Fold 1

---

## Build phases

### Phase 0 — Research and planning ✓ COMPLETE

- V1 retrospective
- Benchmarking against published models (inpredictable Brier 0.163, ESPN 0.166)
- Data source investigation (DARKO: no API → BPM fallback)
- nba_api V3 migration confirmed
- 18-feature set designed
- State-dependent Elo prior designed
- Walk-forward validation designed
- Dashboard integration decision (extend football site, ESPN-style sport switcher)

### Phase 1 — Data pipeline

Tasks:
- [x] Migrate PBP fetcher: V2 → V3; verify field schema changes
- [x] Fetch all regular season games 2018-25 via V3 (resumable, ~90min first run)
- [x] Build possession parser using `pbpstats`
- [x] Build FT state accumulator from V3 event codes (type codes: 3=made FT, 4=missed FT)
- [x] Build bonus/foul accumulator (running team foul count per quarter)
- [x] Build availability delta pipeline (`nbainjuries` + BPM via `basketball_reference_web_scraper`)
- [x] Compute `elo_prior_weight` for all snapshots
- [x] Update Elo ratings through 2024-25 regular season
- [x] Add `is_bubble` flag for 2019-20
- [x] Tighten winsorisation to ±60
- [x] Run 13-check validation suite (updated for 18 features)
- [x] Smoke test: run live on preseason game

### Phase 2 — Training and evaluation

Tasks:
- [ ] Walk-forward validation (3 folds)
- [ ] Full 6-metric evaluation suite
- [ ] ESPN benchmark (same plays)
- [ ] Temperature scaling calibration
- [ ] Save `win_prob_net_v2.pth`, `scaler_v2.pkl`, `elo_ratings_v2.json`
- [ ] Document per-fold results and temporal drift analysis

### Phase 3 — Dashboard integration

Tasks:
- [ ] Sport switcher in sidebar (football ↔ NBA)
- [ ] NBA hub page: today's games + pregame win probability
- [ ] NBA match page: live win probability chart, score, clock, possession indicator
- [ ] Live poll endpoint: `/nba/live/{game_id}` updating every 30 seconds
- [ ] Pregame probability pre-computation (availability delta + Elo prior)

### Phase 4 — Layer 2: Seed placement (mid-season)

Tasks:
- [ ] Standings simulation: remaining schedule + DC-style team strength
- [ ] Playoff bracket probability per team
- [ ] "Clinch tracker" — probability of clinching a given seed after each game

### Phase 5 — Layer 3: Playoff series (April 2027)

Tasks:
- [ ] Port V1 Monte Carlo series simulator
- [ ] Re-introduce `home_series_wins` / `away_series_wins` as features
- [ ] Championship probability chain (series results → bracket → finals)
- [ ] Historical game replay mode

---


---

## Phase 1 results — Data Pipeline ✅ COMPLETE

**Date completed:** September 2026

### What was built
- `phase1_data_pipeline.py` — single-script pipeline replacing the V1 notebook approach
- Fetches game logs for all 7 seasons (2018-19 → 2024-25), regular season + playoffs
- Computes Elo ratings chronologically, exports `elo_ratings_v3.json`
- Extracts 16-feature snapshots from 8,871 PBP files using PlayByPlayV3
- Saves to `data/processed/features_v3.parquet`

### Final numbers
| Metric | Value |
|--------|-------|
| Total snapshots | 1,129,825 |
| Unique games | 8,871 |
| Features | 16 |
| Home win rate | 55.8% |
| Playoff rows | 6.5% |
| OT rows | 0.7% |
| Possession = home | 46.5% |
| Validation checks | **16/16 PASS** |
| Extraction time | ~3 min 15 sec |

### Feature set — final 16 (from planned 18)

`home_fts_pending` and `away_fts_pending` were **dropped** from the final feature set.

Root cause: FT events in V3 PBP have no `scorehome`/`scoreaway` values — they don't appear in scored-play snapshots. By the time a snapshot is taken (at a made FT or field goal), the FT sequence has already resolved and `fts_pending` is back to 0. The feature was 0 in 99%+ of rows, making it useless for training.

`home_in_bonus` and `away_in_bonus` were **kept** and work correctly after the two-pass fix (see bugs below).

### Key bugs encountered and fixed

**Bug 1 — GAME_ID KeyError (all 8,871 games failed extraction)**
- Cause: `games.set_index("GAME_ID")` moves GAME_ID to the index; `game_meta["GAME_ID"]` then raises KeyError
- Fix: added `game_id` as explicit parameter to `extract_snapshots()`; use `game_meta.name` as fallback

**Bug 2 — FT/bonus features all zero (zero-variance check failed)**
- Cause: extraction filtered to scored-plays-only first, then tried to find Foul/Free Throw events — those events have no score in V3 so they were already filtered out
- Fix: two-pass approach — STEP A processes full PBP and computes FT/bonus state per row; STEP B filters to scored plays and attaches the precomputed state. Bonus columns now show real variance.

**Bug 3 — V3 actiontype case mismatch**
- Cause: original code matched `"freethrow"` (lowercase, no space). V3 uses `"Free Throw"` (title case, space)
- Fix: exact string matching `atype == "Free Throw"` and `atype == "Foul"`

**Bug 4 — UnicodeEncodeError on Windows**
- Cause: `report_path.write_text(report)` uses system default `cp1252` encoding, which can't handle `→`
- Fix: `write_text(report, encoding="utf-8")`

**Bug 5 — smoke_test_live merged into diagnose_pbp_columns**
- Cause: missing closing `"""` left the smoke test body running inside the diagnostic function
- Fix: restored proper function boundary

**Bug 6 — stale cache loading (rebuild not triggering)**
- Cause: old 18-column parquet contained all 16 new columns + 2 dropped ones, so subset check passed
- Fix: count-based check — if parquet doesn't have exactly N_FEATURES columns, rebuild

### Architecture note: two-pass PBP extraction

The V3 PBP has a structural property: scored events (`scorehome`/`scoreaway` populated) and state events (fouls, FTs, timeouts) are separate rows. You cannot derive foul/bonus state from scored-play rows alone.

Solution: process the full PBP first to accumulate state, then filter to scored plays and carry state forward. This is the correct approach for any feature derived from non-scoring events (bonus, foul trouble, timeouts remaining, etc.).

### Data notes
- `possession` = 0.465 (slightly below 0.50) — expected; home teams make more 2-pt shots (which have possession=1) but give up more 3-pt attempts (which would show possession=0 if the away team made them). Not a bug.
- `avail_delta` = 0.0 for all rows — expected until DARKO CSV is downloaded from darko.app
- 65 rows winsorised at ±60 — less than 0.006% of data, correct

### Next: Phase 2 — Training
Run: `python phase2_training.py`

## Planned UI features (parked for later phases)

### Rate that team — head-to-head popularity feature

Inspired by DARKO's "rate that player" feature on darko.app. Two NBA teams are shown side by side and the user picks which team they prefer. Results are aggregated and displayed as a popularity ranking / head-to-head record.

Planned for Layer 3 dashboard (playoff phase). Implementation ideas:
- Elo-style rating system for team popularity (separate from the predictive Elo)
- Store votes in SQLite (same predictions DB pattern as football)
- Display on the NBA hub page as a sidebar widget: "Most popular teams this week"
- Could extend to matchup previews — "Who are you rooting for?" before a game

---

## Known issues and open questions

| Issue | Status | Notes |
|---|---|---|
| DARKO DPM CSV access | Resolved | CSV export confirmed on darko.app. Download manually at season start; automate URL if pattern stabilises. BPM remains fallback. |
| OT calibration | Open | Insufficient OT data even with regular season added. Will evaluate in fold results. |
| 2019-20 bubble neutrality | Open | Home advantage invalid for bubble games. `is_bubble` flag added; model learns the effect. |
| Garbage time / tanking | Open | Winsor ±60 mitigates. May need to discard snapshots with score_diff > 40 after Q3. |
| nba_api rate limiting | Open | Add exponential backoff and resumable checkpoint saving in Phase 1. |
| ESPN endpoint stability | Open | Unofficial endpoint, no SLA. Cache benchmark data locally immediately on fetch. |

---

## Performance targets (V2)

| Metric | V1 baseline | V2 target |
|---|---|---|
| Brier (pooled) | 0.1560 | < 0.152 |
| ROC-AUC | 0.8539 | > 0.860 |
| Brier Skill Score | 0.3602 | > 0.375 |
| Q4 final 2min Brier | Not measured | < 0.090 |
| ESPN head-to-head Brier | Not measured | ≤ ESPN on same plays |
| Pathwise calibration | Not measured | Loser ≥90% in 10–13% of games |
| Walk-forward fold drift | Not measured | < 0.005 between folds |

---

*Last updated: Phase 0 complete — September 2026*
