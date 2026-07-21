#  NBA Live Win Probability

A real-time NBA in-game win probability system built with a PyTorch neural network, live play-by-play data from `nba_api`, and a Streamlit dashboard that updates every 30 seconds during live games.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

##  Dashboard
> **Live dashboard →** *([Click to see model]([https://fifa-worldcup-prediction-kc3azpj8tc2ynw4ylnjqv8.streamlit.app/](https://nba-live-win-probability.streamlit.app/)))*  
> **Related project →** [FIFA Worldcup Model]([https://github.com/rhkhabure/NBA-Live-Win-Probability](https://github.com/rhkhabure/FIFA-WORLDCUP-PREDICTION))
> Live during the 2025 NBA Playoffs — Conference Finals & NBA Finals

The dashboard shows real-time win probability updating play-by-play, series win probability via Monte Carlo simulation, and Finals championship probability for all remaining teams.

---

##  Architecture

```
nba_api (PlayByPlayV3)
        │
        ▼
  Feature Engineering          Phase 1 — Data Pipeline
  ┌─────────────────┐
  │  score_diff      │  ← 962,871 snapshots across
  │  time_remaining  │    7,562 games (2018-19 → 2024-25)
  │  Elo ratings     │
  │  series state    │
  │  lead volatility │
  └────────┬────────┘
           │
           ▼
  Neural Network               Phase 2 — Training
  ┌─────────────────┐
  │  12 → 128       │  ← BatchNorm + ReLU + Dropout
  │  128 → 64       │    BCEWithLogitsLoss
  │  64  → 32       │    AdamW + CosineAnnealingLR
  │  32  → 1 (σ)    │    Early stopping @ epoch 38
  └────────┬────────┘
           │
           ▼
  Temperature Scaling          Phase 3 — Calibration
  P_cal = σ(logit / 1.0598)
           │
           ├──→  Monte Carlo Series Simulator (100K sims)
           │
           ▼
  Streamlit Dashboard          Phase 3 — Live System
  Auto-refresh every 30s
```

---

##  Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8539** |
| PR-AUC | 0.8900 |
| Brier Score | 0.1560 |
| Brier Skill Score | **0.3602** (36% better than null model) |
| Log Loss | 0.4662 (vs 0.6808 null) |
| Accuracy | 76.58% |
| CV AUC (5-fold) | 0.8498 ± 0.0051 |
| CV Brier (5-fold) | 0.1579 ± 0.0026 |
| Temperature T | 1.0598 |
| Training epochs | 38 (early stopped at 53) |

> ROC-AUC of 0.854 is competitive with published ESPN / FiveThirtyEight in-game models. Brier Skill Score of 0.36 confirms the model is meaningfully better than predicting the historical home win rate.

---

##  Project Structure

```
NBA LIVE PREDICTIONS/
│
├── nba_win_prob/
│   ├── raw/
│   │   ├── game_logs_all.parquet      # 6 seasons of game logs
│   │   └── pbp/                       # ~7,500 per-game PBP parquets
│   │
│   ├── processed/
│   │   ├── features_raw.parquet       # 962,871 game-state snapshots
│   │   └── tensors.pt                 # GPU-ready train/val/test tensors
│   │
│   ├── model/
│   │   ├── win_prob_net.pth           # Trained weights + config + metrics
│   │   ├── scaler.pkl                 # StandardScaler (fit on train only)
│   │   ├── temperature.json           # T=1.0598 calibration parameter
│   │   ├── elo_ratings.json           # Current Elo for all 30 teams
│   │   ├── training_history.json      # Loss/AUC per epoch
│   │   ├── hosmer_lemeshow.csv        # Calibration group table
│   │   └── cv_results.csv             # 5-fold cross-validation results
│   │
│   └── plots/
│       ├── training_curves.png
│       ├── calibration_analysis.png
│       ├── calibration_before_after.png
│       └── win_prob_by_game_state.png
│
├── phase1_data_pipeline.ipynb         # Data collection & feature engineering
├── phase1_fixes.ipynb                 # Clock parser & score column fixes
├── phase2_training.ipynb              # Neural network training & validation
├── phase3_setup.ipynb                 # Temperature scaling & Elo export
└── app.py                             # Streamlit live dashboard
```

---

##  Features

### 12 Input Features per Play-by-Play Snapshot

| Feature | Description |
|---------|-------------|
| `score_diff` | Home − Away score (winsorised ±80) |
| `time_remaining_sec` | Seconds left in regulation (0 in OT) |
| `quarter` | Period number (5+ = OT) |
| `quarter_time_elapsed_pct` | Fraction of current period elapsed |
| `home_elo` | Pre-game Elo rating — home team |
| `away_elo` | Pre-game Elo rating — away team |
| `elo_diff` | `home_elo − away_elo` |
| `home_series_wins` | Wins in current playoff series |
| `away_series_wins` | Wins in current playoff series |
| `is_playoffs` | Binary flag |
| `is_overtime` | Binary flag |
| `lead_changes_norm` | Lead changes ÷ plays so far (game volatility) |

### Neural Network — `WinProbNet`

```
Input (12)
  → Linear(12, 128) → BatchNorm1d → ReLU → Dropout(0.30)
  → Linear(128, 64) → BatchNorm1d → ReLU → Dropout(0.30)
  → Linear(64, 32)  → BatchNorm1d → ReLU → Dropout(0.30)
  → Linear(32, 1)   → Sigmoid
```

- **Loss:** `BCEWithLogitsLoss` with `pos_weight` for class imbalance  
- **Optimiser:** AdamW (`lr=3e-4`, `wd=1e-4`)  
- **Scheduler:** CosineAnnealingLR (`T_max=50`)  
- **Regularisation:** BatchNorm + Dropout + gradient clipping (max norm 1.0)  
- **Init:** Kaiming normal (ReLU-optimal)

### Elo Rating System

- Start: 1500 per team, K=20, home advantage +100 Elo points
- Season-start regression to mean: 35% toward 1500
- Updated chronologically through 2024-25 playoffs
- Top 5 heading into 2025 playoffs: OKC 1766 · IND 1684 · BOS 1673 · CLE 1652 · MIN 1647

### Monte Carlo Series Simulator

Fully-vectorised NumPy simulation — 100K trials in **~8ms** on CPU.

```python
simulate_series(p_home=0.65, home_wins=2, away_wins=1)
# → P(home wins series) = 0.784
```

Chains series outcomes to compute P(each team wins NBA Finals).

---

##  Setup & Usage

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install nba_api pandas numpy scipy scikit-learn tqdm matplotlib seaborn pyarrow
pip install streamlit plotly
```

### 2. Phase 1 — Data Pipeline

```bash
jupyter notebook phase1_data_pipeline.ipynb
```

Pulls 6 seasons of game logs + ~7,500 play-by-play files from `nba_api` (resumable). Takes ~90 minutes on first run; subsequent runs load from cache. Runs all 13 data quality checks and saves GPU-ready tensors.

>  If you see a `quarter_time_elapsed_pct` NaN correlation, run `phase1_fixes.ipynb` — it patches the ISO 8601 clock parser.

### 3. Phase 2 — Training

```bash
jupyter notebook phase2_training.ipynb
```

Trains on RTX 4060 (~4.5s/epoch). Runs full statistical validation suite and saves `win_prob_net.pth`.

### 4. Phase 3 — Setup (run once)

```bash
jupyter notebook phase3_setup.ipynb
```

Fits temperature scaling calibration, exports team Elo ratings through current 2024-25 playoffs, and verifies the live `nba_api` connection.

### 5. Launch Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Auto-refreshes every 30 seconds during live games.

---

##  Statistical Validation

### Phase 1 Data Quality (13/13 PASS)

- No missing values, no infinite values
- Target class balance: 56.1% home win rate (well within 40–65% range)
- Score diff validated in realistic NBA range [−67, +78] (winsorised to ±80 for training)
- Zero-variance feature check: all 12 features have meaningful variance
- Group-aware train/val/test split — entire games in one split, no leakage verified

### Phase 2 Model Validation (9/10 PASS)

| Check | Threshold | Result |
|-------|-----------|--------|
| ROC-AUC | > 0.80 |  0.8539 |
| Brier Score | < 0.20 |  0.1560 |
| Brier Skill | > 0.10 |  0.3602 |
| Log Loss | < 0.60 |  0.4662 |
| Accuracy | > 0.70 |  76.6% |
| Hosmer-Lemeshow | p > 0.05 |  0.0000* |
| CV AUC std | < 0.02 |  0.0051 |
| CV Brier std | < 0.005 |  0.0026 |
| Calibration slope | 0.85–1.15 |  0.999 |
| Train/Val gap | < 0.05 |  0.011 |

> *HL test fails due to large sample size (n=192K). At this scale even sub-1% calibration gaps produce massive chi-squared statistics. The calibration slope of 0.999 and reliability diagram confirm the model is well-calibrated in practice. Temperature scaling (T=1.0598) applied post-training.

### Inference Smoke Tests

| Scenario | OKC Home Prob | Expected |
|----------|--------------|----------|
| Q4 2min, tied | 67.1% | ~65–70% ✅ |
| Q4 2min, up 5 | 92.3% | ~88–95% ✅ |
| Q4 2min, down 10 | 4.8% | ~3–8% ✅ |
| Q1 start, tied | 60.1% | ~58–64% ✅ |
| OT 1min, up 3 | 67.0% | ~60–75% ✅ |

---

##  Hardware

Trained and runs on:

```
GPU    : NVIDIA GeForce RTX 4060 Laptop GPU
VRAM   : 8.6 GB
CUDA   : 12.4
cuDNN  : 90100
PyTorch: 2.6.0+cu124
Python : 3.13.13
```

Training time: ~4.5 seconds/epoch · 38 epochs · ~3 minutes total

---

##  Live Data

Uses `nba_api` live endpoints — no API key required:

```python
from nba_api.live.nba.endpoints import scoreboard, playbyplay

# Today's games
board  = scoreboard.ScoreBoard()
games  = board.games.get_dict()

# Live play-by-play
pbp     = playbyplay.PlayByPlay(game_id)
actions = pbp.actions.get_dict()
```

The dashboard polls every 30 seconds during live games and pauses auto-refresh when the game is in pre/post-game state.

---

##  Known Limitations

- **Training data ends 2023-24.** Elo ratings are updated through 2024-25, but the neural net has not seen 2024-25 game patterns. Performance on players/teams with dramatically different 2024-25 form (injuries, trades, rookie breakouts) may be slightly off.
- **No player-level features.** The model doesn't know who's on the floor, who's in foul trouble, or who's on a hot streak. Score differential and Elo capture team quality; individual matchup effects are not modelled.
- **OT calibration is softer.** End-of-regulation tied games (OT openers) have slightly underconfident probabilities — the model hasn't seen enough overtime to be precise in those states.
- **Home court advantage is static.** The Elo home advantage (+100 points ≈ +2.9% win probability) is a league-wide average. Specific arena effects (e.g. altitude in Denver) are not captured.

---

##  Roadmap

- [ ] Add player-level features (on-court lineup Elo, individual PER)
- [ ] Retrain on 2024-25 data after playoffs conclude
- [ ] Add possession-by-possession granularity (currently scored-play snapshots only)
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add historical game replay mode

---

##  License

MIT
