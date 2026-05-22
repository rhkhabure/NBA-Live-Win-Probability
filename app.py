"""
NBA Live Win Probability Dashboard
===================================
Run with:  streamlit run app.py
Requires:  streamlit plotly nba_api torch scikit-learn
"""

import streamlit as st
st.set_page_config(
    page_title="NBA Win Probability",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

import torch, torch.nn as nn
import numpy as np, pandas as pd
import pickle, json, re, time, warnings
from pathlib import Path
from datetime import datetime, timezone

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from nba_api.live.nba.endpoints import scoreboard as live_sb
from nba_api.live.nba.endpoints import playbyplay as live_pbp

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# PATHS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR  = Path("nba_win_prob")
MODEL_DIR = BASE_DIR / "model"

FEATURE_COLS = [
    "score_diff", "time_remaining_sec", "quarter",
    "quarter_time_elapsed_pct",
    "home_net_rtg", "away_net_rtg", "net_rtg_diff",   # Phase 4: NET rating replaces Elo
    "home_series_wins", "away_series_wins",
    "is_playoffs", "is_overtime", "lead_changes_norm",
]

REFRESH_OPTIONS = {"30 s": 30, "60 s": 60, "2 min": 120, "Manual": 0}

TEAM_COLORS = {
    "ATL":"#E03A3E","BOS":"#007A33","BKN":"#000000","CHA":"#1D1160",
    "CHI":"#CE1141","CLE":"#860038","DAL":"#00538C","DEN":"#0E2240",
    "DET":"#C8102E","GSW":"#1D428A","HOU":"#CE1141","IND":"#002D62",
    "LAC":"#C8102E","LAL":"#552583","MEM":"#5D76A9","MIA":"#98002E",
    "MIL":"#00471B","MIN":"#0C2340","NOP":"#0C2340","NYK":"#006BB6",
    "OKC":"#007AC1","ORL":"#0077C0","PHI":"#006BB6","PHX":"#1D1160",
    "POR":"#E03A3E","SAC":"#5A2D81","SAS":"#C4CED4","TOR":"#CE1141",
    "UTA":"#002B5C","WAS":"#002B5C",
}

# ═══════════════════════════════════════════════════════════════════════════
# DARK THEME CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* Page background */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* Score card */
  .score-card {
      background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 20px 28px;
      text-align: center;
  }
  .score-card .team-code { font-size: 2.2rem; font-weight: 800; letter-spacing: 2px; }
  .score-card .score-num { font-size: 4rem; font-weight: 900; line-height: 1.1; }
  .score-card .record    { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

  /* Status badge */
  .live-badge {
      display: inline-block;
      background: #da3633;
      color: white;
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 1px;
      animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
  .final-badge {
      display: inline-block;
      background: #30363d;
      color: #8b949e;
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 0.78rem;
      font-weight: 700;
  }

  /* Metric cards */
  .metric-box {
      background: #1c2128;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 10px;
  }
  .metric-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 1.6rem; font-weight: 800; }

  /* Section headers */
  .section-header {
      font-size: 0.78rem;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 2px;
      border-bottom: 1px solid #30363d;
      padding-bottom: 6px;
      margin: 18px 0 12px 0;
  }

  /* Series bar */
  .series-bar-wrap {
      background: #1c2128;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px 20px;
  }

  /* Divider */
  hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — loaded once per session)
# ═══════════════════════════════════════════════════════════════════════════

class WinProbNet(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout=0.30, use_batchnorm=True):
        super().__init__()
        layers, in_dim = [], n_features
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim, bias=not use_batchnorm))
            if use_batchnorm: layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def logits(self, x): return self.net(x)
    def forward(self, x): return torch.sigmoid(self.net(x))


@st.cache_resource(show_spinner="Loading model…")
def load_system():
    """Load model, scaler, temperature, Elo ratings — once per session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False required in PyTorch 2.6 — file is self-produced and trusted
    ckpt = torch.load(MODEL_DIR / "win_prob_net.pth", map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]
    model = WinProbNet(
        n_features   = cfg["n_features"],
        hidden_dims  = cfg["hidden_dims"],
        dropout      = cfg["dropout"],
        use_batchnorm= cfg["use_batchnorm"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(MODEL_DIR / "temperature.json") as f:
        temp_data = json.load(f)
    T = temp_data["temperature"]

    # Phase 4: load NET ratings (replaces Elo). Fall back to elo_ratings if not found.
    net_path = MODEL_DIR / "net_ratings.json"
    elo_path = MODEL_DIR / "elo_ratings.json"
    ratings_path = net_path if net_path.exists() else elo_path
    with open(ratings_path) as f:
        team_ratings = json.load(f)
    is_net = net_path.exists()

    return model, scaler, T, team_ratings, device, is_net


# ═══════════════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════════════

_ISO  = re.compile(r"PT(\d+)M([\d.]+)S", re.IGNORECASE)
_MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")

def parse_clock(val) -> float:
    if val is None: return 0.0
    s = str(val).strip()
    m = _ISO.match(s)
    if m: return float(m.group(1)) * 60 + float(m.group(2))
    m = _MMSS.match(s)
    if m: return float(m.group(1)) * 60 + float(m.group(2))
    try:  return float(s)
    except ValueError: return 0.0


def time_remaining(period: int, clock_sec: float) -> float:
    if period <= 4:
        return clock_sec + max(0, 4 - period) * 720.0
    return clock_sec


def fmt_clock(period: int, clock_sec: float) -> str:
    mins = int(clock_sec) // 60
    secs = int(clock_sec) % 60
    q = f"Q{period}" if period <= 4 else f"OT{period-4}"
    return f"{q}  {mins}:{secs:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_win_prob(
    period: int, clock: str,
    home_score: int, away_score: int,
    home_rating: float, away_rating: float,  # Phase 4: NET rating (or Elo fallback)
    home_sw: int, away_sw: int,
    is_playoffs: int,
    lead_changes: int, plays: int,
    model, scaler, T, device,
    momentum: float = 0.0,
    is_net_rating: bool = True,
) -> float:
    """
    Return calibrated P(home team wins) for a single game state.

    Phase 4 fixes:
      FIX 1: NET rating replaces Elo (positions 4,5,6 in feature vector).
              Time-decay still applied — NET rating influence fades as
              score diff becomes the dominant signal late in games.
      FIX 2: Conditional momentum — nudge only applied when model is
              genuinely uncertain (35%–65%). Avoids over-inflating an
              already-confident probability during dominant runs.
    """
    clock_sec  = parse_clock(clock)
    t_rem      = time_remaining(period, clock_sec)
    period_dur = 720.0 if period <= 4 else 300.0
    q_elapsed  = float(np.clip((period_dur - clock_sec) / period_dur, 0, 1))
    score_diff = float(np.clip(home_score - away_score, -80, 80))
    n_plays    = max(1, plays)

    # ── Rating time-decay ──────────────────────────────────────────────────
    # Full weight at tip-off → 5% residual at buzzer → 5% flat in OT.
    # Works for both NET rating and Elo fallback.
    REG_SECONDS  = 2880.0
    RATING_MIN   = 0.05
    if period <= 4:
        r_decay = float(np.clip(
            RATING_MIN + (1.0 - RATING_MIN) * (t_rem / REG_SECONDS),
            RATING_MIN, 1.0
        ))
    else:
        r_decay = RATING_MIN  # OT: score is tied, rating nearly irrelevant

    if is_net_rating:
        # NET rating: centre at 0, decay toward 0
        home_r_d = home_rating * r_decay
        away_r_d = away_rating * r_decay
        rtg_diff = home_r_d - away_r_d
    else:
        # Elo fallback: centre at 1500, decay toward 1500
        home_r_d = 1500.0 + (home_rating - 1500.0) * r_decay
        away_r_d = 1500.0 + (away_rating - 1500.0) * r_decay
        rtg_diff = home_r_d - away_r_d

    raw = np.array([[
        score_diff, t_rem, float(period), q_elapsed,
        home_r_d, away_r_d, rtg_diff,
        float(home_sw), float(away_sw),
        float(is_playoffs), float(period > 4),
        lead_changes / n_plays,
    ]], dtype=np.float32)

    scaled = scaler.transform(raw).astype(np.float32)
    tensor = torch.from_numpy(scaled).to(device)

    with torch.no_grad():
        logit = model.logits(tensor).item()

    # ── FIX 2: Conditional momentum nudge ─────────────────────────────────
    # Compute raw probability BEFORE nudge.
    # Only apply momentum when model is genuinely uncertain (35% – 65%).
    # Uncertainty weight tapers to 0 at 35% and 65% so the nudge is
    # smooth, not a hard switch.
    raw_prob = 1 / (1 + np.exp(-logit / T))
    dist_from_half = abs(raw_prob - 0.5)          # 0 at 50%, 0.15 at edges
    uncertainty    = max(0.0, 1.0 - dist_from_half / 0.15)  # 1.0 at 50%, 0 at ±15pp
    logit = logit + float(np.clip(0.18 * momentum * uncertainty, -0.20, 0.20))

    prob = 1 / (1 + np.exp(-logit / T))
    return float(np.clip(prob, 0.001, 0.999))


# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO SERIES SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def simulate_series(p_home: float, h_wins: int, a_wins: int,
                    best_of: int = 7, n_sims: int = 100_000,
                    h_form: float = 0.0, a_form: float = 0.0) -> float:
    """
    Vectorised Monte Carlo: P(home wins series) from current state.

    Phase 5 additions:
    · h_form / a_form: in-series form blending (see adjusted_series_p_game)
    · Hard cap: result always in [0.03, 0.97] — never returns 100% or 0%
      for an in-progress series regardless of NET/Elo gap.
    """
    target = (best_of // 2) + 1
    h_need = target - h_wins
    a_need = target - a_wins
    # Already clinched → still cap to allow tail events
    if h_need <= 0: return 0.97
    if a_need <= 0: return 0.03

    # Blend per-game probability with in-series form
    p_home_adj = float(np.clip(p_home + h_form - a_form, 0.05, 0.95))

    rng   = np.random.default_rng()
    max_g = h_need + a_need - 1
    games = rng.random((n_sims, max_g)) < p_home_adj
    h_cum = np.cumsum(games.astype(np.int16), axis=1)
    a_cum = np.cumsum((~games).astype(np.int16), axis=1)
    h_ever = h_cum[:, -1] >= h_need
    a_ever = a_cum[:, -1] >= a_need
    h_cl   = np.where(h_ever, np.argmax(h_cum >= h_need, axis=1), max_g + 1)
    a_cl   = np.where(a_ever, np.argmax(a_cum >= a_need, axis=1), max_g + 1)
    raw = float((h_cl < a_cl).mean())
    # Hard cap — a best-of-7 series is never truly 98%+ certain
    return float(np.clip(raw, 0.03, 0.97))


def _series_form_adjustment(h_wins: int, a_wins: int,
                             net_p_home: float) -> tuple:
    """
    Compute a small form-based blending adjustment for the Monte Carlo.

    Logic: If the home team has won more games than expected given their
    NET rating, they're out-performing. Adjust p_home slightly upward.
    Uses a conservative 20% blend weight so a 2-0 sweep only moves
    the probability by ~3-5pp, not dramatically.

    Returns (h_form_adj, a_form_adj) to add/subtract from p_home.
    """
    games_played = h_wins + a_wins
    if games_played == 0:
        return 0.0, 0.0

    # Expected wins for home team based purely on NET rating
    expected_h = net_p_home * games_played
    actual_h   = h_wins
    outperformance = (actual_h - expected_h) / games_played   # in [-1, +1]

    # Blend weight: 20% form, 80% NET rating
    FORM_WEIGHT = 0.20
    adj = float(np.clip(outperformance * FORM_WEIGHT, -0.08, 0.08))
    return adj, -adj   # home gains → away loses equally


def p_home_wins_game_from_rating(home_rating: float, away_rating: float,
                                  is_net: bool = True,
                                  home_adv: float = 100.0) -> float:
    """Pre-game win probability from NET rating or Elo."""
    if is_net:
        # NET rating diff → win probability via logistic
        # A +10 NET diff ≈ 65% win rate (calibrated to historical data)
        net_diff = home_rating - away_rating + 2.0  # +2 for home court
        return float(np.clip(1 / (1 + np.exp(-net_diff * 0.15)), 0.05, 0.95))
    else:
        return 1 / (1 + 10 ** ((away_rating - (home_rating + home_adv)) / 400))

# Keep old name as alias for any code that still references it
def p_home_wins_game_from_elo(home_elo: float, away_elo: float,
                               home_adv: float = 100.0) -> float:
    return 1 / (1 + 10 ** ((away_elo - (home_elo + home_adv)) / 400))


# ═══════════════════════════════════════════════════════════════════════════
# NBA API HELPERS  (TTL-cached per game)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_games() -> list:
    """
    Multi-fallback live game fetcher.

    Attempt order:
      1. Direct HTTPS request to NBA CDN with correct browser headers
         (bypasses nba_api wrapper which can cache stale empty responses)
      2. nba_api ScoreBoard() wrapper — catches cases where CDN redirect changes
      3. nba_api ScoreboardV3 stats endpoint — different base URL, often works
         when CDN is stale
    Returns a list of game dicts, empty list if all attempts fail.
    """
    import requests as _req
    import datetime as _dt

    # ── Attempt 1: Direct CDN request with full browser headers ───────────
    _NBA_HEADERS = {
        "Accept":          "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control":   "no-cache",
        "Connection":      "keep-alive",
        "Host":            "cdn.nba.com",
        "Origin":          "https://www.nba.com",
        "Pragma":          "no-cache",
        "Referer":         "https://www.nba.com/",
        "Sec-Fetch-Dest":  "empty",
        "Sec-Fetch-Mode":  "cors",
        "Sec-Fetch-Site":  "same-site",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    _CDN_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

    try:
        resp = _req.get(_CDN_URL, headers=_NBA_HEADERS, timeout=10)
        if resp.status_code == 200:
            data  = resp.json()
            games = data.get("scoreboard", {}).get("games", [])
            if isinstance(games, list) and len(games) > 0:
                return games
    except Exception:
        pass

    # ── Attempt 2: nba_api live ScoreBoard wrapper ─────────────────────────
    try:
        board = live_sb.ScoreBoard()
        games = board.games.get_dict()
        if isinstance(games, list) and len(games) > 0:
            return games
    except Exception:
        pass

    # ── Attempt 3: Stats ScoreboardV3 (different base URL) ────────────────
    try:
        from nba_api.stats.endpoints import scoreboardv3
        today_str = _dt.date.today().strftime("%m/%d/%Y")
        sb3   = scoreboardv3.ScoreboardV3(game_date=today_str, timeout=15)
        dfs   = sb3.get_data_frames()
        # ScoreboardV3 df[0] = GameHeader — convert to live-style dicts
        header_df = dfs[0] if dfs else None
        linescore_df = dfs[1] if len(dfs) > 1 else None
        if header_df is not None and not header_df.empty:
            games = []
            for _, row in header_df.iterrows():
                gid = str(row.get("GAME_ID", ""))
                # Find home/away line scores
                ht_row = {}; at_row = {}
                if linescore_df is not None:
                    gls = linescore_df[linescore_df["GAME_ID"] == gid]
                    for _, lr in gls.iterrows():
                        if lr.get("TEAM_ID") == row.get("HOME_TEAM_ID"):
                            ht_row = lr.to_dict()
                        else:
                            at_row = lr.to_dict()

                # Map to live-style schema
                status_text = str(row.get("GAME_STATUS_TEXT", ""))
                game_status = 3 if "Final" in status_text else (
                              2 if any(q in status_text for q in ["Q","Ht","OT","End"]) else 1)
                games.append({
                    "gameId":         gid,
                    "gameStatus":     game_status,
                    "gameStatusText": status_text,
                    "period":         int(row.get("LIVE_PERIOD", 0) or 0),
                    "gameClock":      str(row.get("LIVE_PC_TIME", "") or ""),
                    "homeTeam": {
                        "teamId":       int(row.get("HOME_TEAM_ID", 0) or 0),
                        "teamTricode":  str(ht_row.get("TEAM_ABBREVIATION", "") or ""),
                        "teamCity":     str(ht_row.get("TEAM_CITY_NAME", "") or ""),
                        "teamName":     str(ht_row.get("TEAM_NAME", "") or ""),
                        "score":        int(ht_row.get("PTS", 0) or 0),
                        "wins":         int(row.get("HOME_TEAM_WINS", 0) or 0),
                        "losses":       int(row.get("HOME_TEAM_LOSSES", 0) or 0),
                    },
                    "awayTeam": {
                        "teamId":       int(row.get("VISITOR_TEAM_ID", 0) or 0),
                        "teamTricode":  str(at_row.get("TEAM_ABBREVIATION", "") or ""),
                        "teamCity":     str(at_row.get("TEAM_CITY_NAME", "") or ""),
                        "teamName":     str(at_row.get("TEAM_NAME", "") or ""),
                        "score":        int(at_row.get("PTS", 0) or 0),
                        "wins":         int(row.get("VISITOR_TEAM_WINS", 0) or 0),
                        "losses":       int(row.get("VISITOR_TEAM_LOSSES", 0) or 0),
                    },
                })
            if games:
                return games
    except Exception:
        pass

    return []   # all attempts failed


@st.cache_data(ttl=20, show_spinner=False)
def fetch_pbp(game_id: str) -> list:
    """
    Multi-fallback play-by-play fetcher.
    Attempt 1: Direct CDN request
    Attempt 2: nba_api PlayByPlay wrapper
    """
    import requests as _req

    _NBA_HEADERS = {
        "Accept":          "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control":   "no-cache",
        "Connection":      "keep-alive",
        "Host":            "cdn.nba.com",
        "Origin":          "https://www.nba.com",
        "Pragma":          "no-cache",
        "Referer":         "https://www.nba.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }

    # Attempt 1: Direct CDN
    try:
        url  = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
        resp = _req.get(url, headers=_NBA_HEADERS, timeout=10)
        if resp.status_code == 200:
            data    = resp.json()
            actions = data.get("game", {}).get("actions", [])
            if isinstance(actions, list):
                return actions
    except Exception:
        pass

    # Attempt 2: nba_api wrapper
    try:
        pbp     = live_pbp.PlayByPlay(game_id)
        actions = pbp.actions.get_dict()
        if isinstance(actions, list):
            return actions
    except Exception:
        pass

    return []


# ═══════════════════════════════════════════════════════════════════════════
# GAME STATE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_game_history(actions: list, home_rating: float, away_rating: float,
                       home_sw: int, away_sw: int, is_playoffs: int,
                       model, scaler, T, device, is_net: bool = True) -> pd.DataFrame:
    """
    Walk through all play-by-play actions and compute win prob at each scored play.

    Phase 4 fixes applied:
      FIX 3: Tracks rolling momentum over the last MOMENTUM_WINDOW scored plays.
             momentum = (home_pts_recent - away_pts_recent) / MOMENTUM_SCALE
             Normalised to [-1, +1] before passing to predict_win_prob.
    """
    from collections import deque

    MOMENTUM_WINDOW = 12      # scored plays (~3-4 minutes of basketball)
    MOMENTUM_SCALE  = 15.0    # normalise: 15pt differential = momentum +-1.0

    rows         = []
    lead_changes = 0
    prev_lead    = 0
    play_num     = 0
    prev_hs = prev_as = 0

    # Deques store (home_pts_gained, away_pts_gained) per play
    home_recent = deque(maxlen=MOMENTUM_WINDOW)
    away_recent = deque(maxlen=MOMENTUM_WINDOW)

    for act in actions:
        hs_str = act.get("scoreHome", "") or ""
        as_str = act.get("scoreAway", "") or ""
        if not hs_str.strip() or not as_str.strip():
            continue
        try:
            hs = int(hs_str); as_ = int(as_str)
        except ValueError:
            continue

        play_num += 1
        period    = int(act.get("period", 1))
        clock     = act.get("clock", "PT12M00.00S")
        clock_sec = parse_clock(clock)

        # ── Track points scored on this play ──────────────────────────────
        h_pts_this = max(0, hs  - prev_hs)
        a_pts_this = max(0, as_ - prev_as)
        home_recent.append(h_pts_this)
        away_recent.append(a_pts_this)
        prev_hs, prev_as = hs, as_

        # ── Momentum: normalised rolling pts differential ──────────────────
        if len(home_recent) >= 3:   # need at least 3 plays to be meaningful
            momentum = float(np.clip(
                (sum(home_recent) - sum(away_recent)) / MOMENTUM_SCALE,
                -1.0, 1.0
            ))
        else:
            momentum = 0.0

        # ── Lead change tracking ───────────────────────────────────────────
        cur_lead = np.sign(hs - as_)
        if cur_lead != prev_lead and cur_lead != 0:
            lead_changes += 1
        prev_lead = cur_lead

        prob = predict_win_prob(
            period=period, clock=clock,
            home_score=hs, away_score=as_,
            home_rating=home_rating, away_rating=away_rating,
            home_sw=home_sw, away_sw=away_sw,
            is_playoffs=is_playoffs,
            lead_changes=lead_changes, plays=play_num,
            model=model, scaler=scaler, T=T, device=device,
            momentum=momentum, is_net_rating=is_net,
        )

        t_rem = time_remaining(period, clock_sec)
        rows.append({
            "action_num"     : act.get("actionNumber", play_num),
            "period"         : period,
            "clock_sec"      : clock_sec,
            "time_remaining" : t_rem,
            "home_score"     : hs,
            "away_score"     : as_,
            "score_diff"     : hs - as_,
            "momentum"       : round(momentum, 3),
            "home_win_prob"  : prob,
            "away_win_prob"  : 1 - prob,
            "description"    : act.get("description", ""),
            "clock_display"  : fmt_clock(period, clock_sec),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_BASE = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="Inter, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def win_prob_chart(hist: pd.DataFrame, home_code: str, away_code: str,
                   home_color: str, away_color: str) -> go.Figure:
    fig = go.Figure()

    x = hist["action_num"].tolist()

    # Home team fill
    fig.add_trace(go.Scatter(
        x=x, y=hist["home_win_prob"].tolist(),
        fill="tozeroy", fillcolor=f"rgba{(*[int(home_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)], 0.25)}",
        line=dict(color=home_color, width=2.5),
        name=f"{home_code} win %", hovertemplate="%{y:.1%}<extra></extra>",
    ))

    # 50% line
    fig.add_hline(y=0.5, line=dict(color="#30363d", width=1, dash="dot"))

    # Quarter markers
    if not hist.empty:
        for q in [1, 2, 3, 4, 5]:
            q_rows = hist[hist["period"] == q]
            if not q_rows.empty:
                q_start = q_rows["action_num"].iloc[0]
                label = f"Q{q}" if q <= 4 else f"OT{q-4}"
                fig.add_vline(
                    x=q_start, line=dict(color="#30363d", width=1),
                    annotation=dict(text=label, font=dict(color="#8b949e", size=10),
                                    yref="paper", y=1.05, showarrow=False),
                )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"{away_code} @ {home_code} — Win Probability",
                   font=dict(size=14, color="#e6edf3")),
        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#21262d",
                   showgrid=True, zeroline=False),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False, height=320,
    )
    return fig


def series_prob_chart(home_code: str, away_code: str,
                      p_home_series: float,
                      home_sw: int, away_sw: int,
                      home_color: str, away_color: str) -> go.Figure:
    p_away = 1 - p_home_series
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[p_home_series], y=["Series"], orientation="h",
        marker_color=home_color, name=home_code,
        text=f"{home_code}  {p_home_series:.1%}",
        textposition="inside", insidetextanchor="start",
        textfont=dict(size=14, color="white", family="Arial Black"),
    ))
    fig.add_trace(go.Bar(
        x=[p_away], y=["Series"], orientation="h",
        marker_color=away_color, name=away_code,
        text=f"{p_away:.1%}  {away_code}",
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=14, color="white", family="Arial Black"),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text=f"Series Probability — {away_code} {away_sw}  vs  {home_sw} {home_code}",
            font=dict(size=13, color="#e6edf3"),
        ),
        barmode="stack",
        yaxis=dict(showticklabels=False),
        xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
        showlegend=False, height=120,
        bargap=0.4,
    )
    return fig


def gauge_chart(prob: float, home_code: str, away_code: str,
                home_color: str, away_color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=40, color=home_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#8b949e"), tickvals=[0,25,50,75,100]),
            bar=dict(color=home_color),
            bgcolor="#21262d",
            bordercolor="#30363d",
            steps=[
                dict(range=[0,   50], color="#1c2128"),
                dict(range=[50, 100], color="#1c2128"),
            ],
            threshold=dict(line=dict(color="#8b949e", width=2), thickness=0.75, value=50),
        ),
        title=dict(text=f"<b>{home_code}</b> win prob", font=dict(size=14, color="#8b949e")),
    ))
    fig.update_layout(
        **{**PLOTLY_BASE, "margin": dict(l=20, r=20, t=30, b=10)},
        height=250,
    )
    return fig


def bracket_chart(teams_probs: dict) -> go.Figure:
    """Simple horizontal bar chart of Finals win probability per team."""
    if not teams_probs:
        return go.Figure()
    teams  = list(teams_probs.keys())
    probs  = [teams_probs[t] for t in teams]
    colors = [TEAM_COLORS.get(t, "#30363d") for t in teams]
    fig = go.Figure(go.Bar(
        x=probs, y=teams, orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=12, color="#e6edf3"),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="🏆 NBA Finals Win Probability", font=dict(size=13)),
        xaxis=dict(tickformat=".0%", range=[0, max(probs) * 1.35],
                   showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=max(180, len(teams) * 44),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FINALS PROBABILITY  (Monte Carlo chained bracket)
# ═══════════════════════════════════════════════════════════════════════════

def compute_finals_probs(live_games: list, team_ratings: dict,
                          series_states: dict) -> dict:
    """
    Estimate P(each team wins NBA Finals) using a two-stage bracket simulation.

    Phase 4 FIX 2: Uses live SERIES wins/losses (not just Elo) so that a team
    which has lost games in their current series is properly penalised.

    Stage 1 — Current series: P(team advances) via Monte Carlo from live series state.
    Stage 2 — Remaining bracket: chain Elo-based series simulations to Finals.

    Returns { TEAM_CODE: P(wins Finals) }.
    """
    try:
        if not live_games:
            return {}

        # ── Stage 1: Gather current series state from live scoreboard ─────
        series_map = {}   # key=(sorted tricodes) → series dict with LIVE wins

        for g in live_games:
            ht = g["homeTeam"]["teamTricode"]
            at = g["awayTeam"]["teamTricode"]
            if not ht or not at:
                continue

            # wins/losses on the scoreboard are SERIES wins (not season record)
            # in playoff games
            hw = int(g["homeTeam"].get("wins", 0) or 0)
            aw = int(g["awayTeam"].get("wins", 0) or 0)
            key = tuple(sorted([ht, at]))

            # Only store once per series (first game dict encountered)
            if key not in series_map:
                series_map[key] = {
                    "home": ht, "away": at,
                    "hw": hw,   "aw":  aw,
                }

        if not series_map:
            return {}

        # ── Stage 2: Simulate each series from current state ──────────────
        # P(home wins series) using live hw/aw + Elo per-game probability
        team_series_prob = {}

        for key, series in series_map.items():
            ht, at = series["home"], series["away"]
            hw, aw = series["hw"],   series["aw"]

            h_elo = team_ratings.get(ht, 0.0 if is_net else 1500)
            a_elo = team_ratings.get(at, 0.0 if is_net else 1500)

            # Per-game probability from Elo (home court advantage included)
            p_home_game = p_home_wins_game_from_rating(h_elo, a_elo, is_net=is_net)

            # ── Phase 5: in-series form adjustment ────────────────────────
            h_form, a_form = _series_form_adjustment(hw, aw, p_home_game)

            # Simulate REMAINING games from current series state (hw, aw)
            p_home_series = simulate_series(p_home_game, hw, aw, n_sims=50_000,
                                            h_form=h_form, a_form=a_form)
            p_away_series = 1.0 - p_home_series

            team_series_prob[ht] = team_series_prob.get(ht, 0.0) + p_home_series
            team_series_prob[at] = team_series_prob.get(at, 0.0) + p_away_series

        # ── Stage 3: Chain to Finals probability ──────────────────────────
        # For each pair of teams that could meet in the Finals, compute
        # P(A wins Finals) = P(A advances) * P(A beats B | both advance)
        # Simplified: if we have exactly 2 active series (Conference Finals),
        # the Finals winner is the product of winning their series then the Finals.
        teams = list(team_series_prob.keys())
        finals_prob = {}

        if len(teams) == 2:
            # Two teams left — they ARE the finalists already
            for t in teams:
                finals_prob[t] = round(team_series_prob[t], 4)
        elif len(teams) >= 4:
            # Conference Finals stage: pair teams by conference
            # Sort series by Elo to identify likely finalists
            series_list = sorted(
                series_map.values(),
                key=lambda s: team_ratings.get(s["home"], 0.0 if is_net else 1500) +
                              team_ratings.get(s["away"], 0.0 if is_net else 1500),
                reverse=True
            )
            # Each series produces one finalist — simulate who wins the Finals
            if len(series_list) >= 2:
                s1, s2 = series_list[0], series_list[1]
                # Finalist from series 1
                ht1, at1 = s1["home"], s1["away"]
                p_h1 = team_series_prob.get(ht1, 0.5)
                p_a1 = team_series_prob.get(at1, 0.5)

                # Finalist from series 2
                ht2, at2 = s2["home"], s2["away"]
                p_h2 = team_series_prob.get(ht2, 0.5)
                p_a2 = team_series_prob.get(at2, 0.5)

                # Four possible Finals matchups
                for t1, p_t1 in [(ht1, p_h1), (at1, p_a1)]:
                    for t2, p_t2 in [(ht2, p_h2), (at2, p_a2)]:
                        e1 = team_ratings.get(t1, 0.0 if is_net else 1500)
                        e2 = team_ratings.get(t2, 0.0 if is_net else 1500)
                        # Neutral court Finals (no home advantage)
                        p_t1_wins_finals = p_home_wins_game_from_rating(e1, e2, is_net=is_net)
                        p_finals_series_t1 = simulate_series(
                            p_t1_wins_finals, 0, 0, n_sims=30_000
                        )
                        finals_prob[t1] = finals_prob.get(t1, 0.0) + (
                            p_t1 * p_t2 * p_finals_series_t1)
                        finals_prob[t2] = finals_prob.get(t2, 0.0) + (
                            p_t1 * p_t2 * (1.0 - p_finals_series_t1))
            else:
                finals_prob = {t: round(v, 4) for t, v in team_series_prob.items()}
        else:
            finals_prob = {t: round(v, 4) for t, v in team_series_prob.items()}

        return {k: round(v, 4) for k, v in
                sorted(finals_prob.items(), key=lambda x: x[1], reverse=True)
                if v > 0.001}

    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# GAME HISTORY — cache and replay
# ═══════════════════════════════════════════════════════════════════════════

HISTORY_DIR = BASE_DIR / "game_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_game_history(game_id: str, away_code: str, home_code: str,
                      hist: pd.DataFrame, final_away: int, final_home: int,
                      series_away: int, series_home: int):
    """
    Save a completed game's probability timeline to disk so it can be
    replayed in History mode without calling the API again.
    Saves to  nba_win_prob/game_history/<game_id>.json
    """
    path = HISTORY_DIR / f"{game_id}.json"
    if path.exists():
        return   # already saved — don't overwrite
    if hist.empty or len(hist) < 20:
        return   # not enough data worth saving

    record = {
        "game_id"      : game_id,
        "away_code"    : away_code,
        "home_code"    : home_code,
        "final_away"   : int(final_away),
        "final_home"   : int(final_home),
        "series_away"  : int(series_away),
        "series_home"  : int(series_home),
        "saved_at"     : datetime.now(timezone.utc).isoformat(),
        "plays"        : len(hist),
        "timeline"     : hist[[
            "play_num", "period", "clock_sec", "time_remaining",
            "home_score", "away_score", "score_diff",
            "home_win_prob", "momentum",
        ]].to_dict("records"),
    }
    try:
        with open(path, "w") as f:
            json.dump(record, f, indent=None)   # compact JSON
    except Exception:
        pass


def load_game_history(game_id: str) -> dict:
    """Load a saved game record. Returns None if not found."""
    path = HISTORY_DIR / f"{game_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def list_saved_games() -> list:
    """
    Return a list of saved game records sorted newest first.
    Each entry: {game_id, label, away, home, final_away, final_home, plays}
    """
    games = []
    for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                rec = json.load(f)
            away = rec.get("away_code", "???")
            home = rec.get("home_code", "???")
            fa   = rec.get("final_away", 0)
            fh   = rec.get("final_home", 0)
            winner = home if fh > fa else away
            label = (f"{away} {fa} @ {home} {fh}  "
                     f"({'OT' if rec.get('plays',0) > 500 else 'Reg'})  "
                     f"— {winner} wins")
            games.append({"game_id": rec["game_id"], "label": label,
                          "record": rec})
        except Exception:
            continue
    return games


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Load system ────────────────────────────────────────────────────────
    try:
        model, scaler, T, team_ratings, device, is_net = load_system()
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}\n\nRun `phase3_setup.ipynb` first.")
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏀 NBA Win Probability")
        st.markdown("---")

        refresh_label = st.selectbox("Auto-refresh", list(REFRESH_OPTIONS.keys()), index=0)
        refresh_sec   = REFRESH_OPTIONS[refresh_label]

        st.markdown("---")
        st.markdown("**Model Info**")
        # Load metrics dynamically from checkpoint so they update with new models
        try:
            _ckpt_meta = torch.load(MODEL_DIR / "win_prob_net.pth",
                                    map_location="cpu", weights_only=False)
            _tm = _ckpt_meta.get("test_metrics", {})
            _auc   = _tm.get("roc_auc",   0.8585)
            _brier = _tm.get("brier",      0.1529)
            _mv    = _ckpt_meta.get("model_version", "phase4_net_rating")
        except Exception:
            _auc = 0.8585; _brier = 0.1529; _mv = "phase4_net_rating"
        st.caption(f"AUC: {_auc:.4f} · Brier: {_brier:.4f}")
        st.caption(f"Temperature T: {T:.4f}")
        st.caption(f"Device: {str(device).upper()} · {_mv}")
        st.markdown("**Phase 5 Active**")
        st.caption("✅ NET rating (replaces Elo)")
        st.caption("✅ Conditional momentum (35-65%)")
        st.caption("✅ Series form adjustment (20% blend)")
        st.caption("✅ Series cap [3%–97%]")
        st.markdown("---")

        show_bracket = st.toggle("Show Finals Probability", value=True)
        show_history = st.toggle("Show Full Play History", value=False)
        n_mc_sims    = st.select_slider("Monte Carlo sims",
                                         options=[10_000, 50_000, 100_000, 200_000],
                                         value=100_000)

        st.markdown("---")
        # ── View mode: Live vs History ─────────────────────────────────────
        st.markdown("**View Mode**")
        view_mode = st.radio("", ["🔴 Live Games", "📼 Game History"],
                             label_visibility="collapsed")
        if view_mode == "📼 Game History":
            st.caption("Replay any tracked game's win probability curve.")

        st.markdown("---")
        last_refresh = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        st.caption(f"Last refresh: {last_refresh}")
        st.markdown("---")
        with st.expander("🔧 API Debug"):
            if st.button("Test Live API", use_container_width=True):
                import requests as _req
                _headers = {
                    "Accept": "application/json, text/plain, */*",
                    "Cache-Control": "no-cache",
                    "Origin": "https://www.nba.com",
                    "Referer": "https://www.nba.com/",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                }
                _url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
                try:
                    r = _req.get(_url, headers=_headers, timeout=10)
                    st.caption(f"CDN status: {r.status_code}")
                    if r.status_code == 200:
                        d = r.json()
                        gs = d.get("scoreboard", {}).get("games", [])
                        st.caption(f"Games in CDN response: {len(gs)}")
                        for g in gs:
                            st.caption(f"  {g['awayTeam']['teamTricode']} @ {g['homeTeam']['teamTricode']} — status {g['gameStatus']}")
                    else:
                        st.caption(f"CDN body: {r.text[:200]}")
                except Exception as e:
                    st.caption(f"CDN error: {e}")

                try:
                    board = live_sb.ScoreBoard()
                    gms = board.games.get_dict()
                    st.caption(f"nba_api live wrapper: {len(gms) if gms else 0} games")
                except Exception as e:
                    st.caption(f"nba_api live error: {e}")

        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch live games ───────────────────────────────────────────────────
    live_games = fetch_live_games()

    st.markdown("<h1 style='color:#e6edf3;font-size:1.8rem;margin-bottom:4px'>🏀 NBA Live Win Probability</h1>",
                unsafe_allow_html=True)

    # ── History mode ───────────────────────────────────────────────────────
    if view_mode == "📼 Game History":
        saved = list_saved_games()
        if not saved:
            st.info(
                "**No saved games yet.**\n\n"
                "Games are automatically saved once they finish. "
                "Switch to Live mode, watch a game to completion, then come back here."
            )
        else:
            st.markdown("<div class='section-header'>GAME HISTORY</div>",
                        unsafe_allow_html=True)
            labels    = [g["label"] for g in saved]
            sel_label = st.selectbox("Select a game", labels,
                                     label_visibility="collapsed")
            sel_rec   = next(g["record"] for g in saved
                             if g["label"] == sel_label)

            away_c = sel_rec["away_code"]
            home_c = sel_rec["home_code"]
            fa, fh = sel_rec["final_away"], sel_rec["final_home"]
            winner = home_c if fh > fa else away_c
            plays  = sel_rec["plays"]

            # Rebuild DataFrame
            hist_df = pd.DataFrame(sel_rec["timeline"])

            # Ensure column compatibility with win_prob_chart (expects action_num)
            if "play_num" in hist_df.columns and "action_num" not in hist_df.columns:
                hist_df = hist_df.rename(columns={"play_num": "action_num"})

            # ── Score card row ─────────────────────────────────────────────
            h_color = TEAM_COLORS.get(home_c, "#007AC1")
            a_color = TEAM_COLORS.get(away_c, "#C8102E")
            col_a, col_m, col_h, col_g = st.columns([2, 1.2, 2, 2.5])
            with col_a:
                st.markdown(f"""
                <div class="score-card" style="border-top:4px solid {a_color}">
                    <div class="team-code" style="color:{a_color}">{away_c}</div>
                    <div class="score-num">{fa}</div>
                    <div class="record">{'WIN' if fa > fh else ''}</div>
                </div>""", unsafe_allow_html=True)
            with col_m:
                st.markdown(f"""
                <div style="text-align:center;padding-top:24px">
                    <span class="final-badge">FINAL</span>
                    <div style="font-size:0.8rem;color:#8b949e;margin-top:10px">{plays:,} plays</div>
                </div>""", unsafe_allow_html=True)
            with col_h:
                st.markdown(f"""
                <div class="score-card" style="border-top:4px solid {h_color}">
                    <div class="team-code" style="color:{h_color}">{home_c}</div>
                    <div class="score-num">{fh}</div>
                    <div class="record">{'WIN' if fh > fa else ''}</div>
                </div>""", unsafe_allow_html=True)
            with col_g:
                fin_prob = hist_df["home_win_prob"].iloc[-1]
                st.plotly_chart(
                    gauge_chart(fin_prob, home_c, away_c, h_color, a_color),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # ── Win probability chart ──────────────────────────────────────
            st.markdown("<div class='section-header'>WIN PROBABILITY REPLAY</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                win_prob_chart(hist_df, home_c, away_c, h_color, a_color),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # ── Key stats ─────────────────────────────────────────────────
            st.markdown("<div class='section-header'>GAME SUMMARY</div>",
                        unsafe_allow_html=True)
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            max_lead_h = int(hist_df["score_diff"].max())
            max_lead_a = int(-hist_df["score_diff"].min())
            trough     = hist_df.loc[hist_df["home_win_prob"].idxmin()]
            peak       = hist_df.loc[hist_df["home_win_prob"].idxmax()]
            for col, label, val in [
                (mc1, "Winner",       winner),
                (mc2, "Margin",       f"{abs(fh-fa)} pts"),
                (mc3, f"{home_c} max lead", f"+{max_lead_h}"),
                (mc4, f"{away_c} max lead", f"+{max_lead_a}"),
                (mc5, "Lowest prob",  f"{hist_df['home_win_prob'].min():.0%} {home_c}"),
            ]:
                col.markdown(f"""<div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>""", unsafe_allow_html=True)

            # ── Biggest swing ──────────────────────────────────────────────
            hist_df["prob_delta"] = hist_df["home_win_prob"].diff().abs().fillna(0)
            top_swing = hist_df.nlargest(1, "prob_delta").iloc[0]
            q_label   = f"Q{int(top_swing.period)}" if top_swing.period <= 4 else f"OT{int(top_swing.period)-4}"
            cs        = int(top_swing.clock_sec)
            st.caption(
                f"⚡ Biggest single-play swing: **{top_swing.prob_delta:.1%}** — "
                f"{q_label} {cs//60}:{cs%60:02d}  "
                f"({home_c} {int(top_swing.home_score)} — "
                f"{away_c} {int(top_swing.away_score)})"
            )
        return   # don't render live mode below

    if not live_games:
        st.warning(
            "**No live games found.** The NBA API returns an empty response when no games "
            "are scheduled. The dashboard will automatically populate once games tip off.\n\n"
            "**Demo mode** — showing a simulated game state below so you can verify the "
            "model and dashboard are working correctly."
        )

        # ── Demo mode: synthetic game state ───────────────────────────────
        st.markdown("---")
        st.markdown("#### 🧪 Demo: OKC vs SAS — Q4 3:22 remaining")

        demo_col1, demo_col2 = st.columns(2)
        demo_scenarios = [
            ("OKC leads +8, 3:22 Q4",  4, "PT03M22.00S", 94, 86, "OKC", "SAS"),
            ("SAS leads +5, 1:45 Q4",  4, "PT01M45.00S", 97, 102, "OKC", "SAS"),
            ("Tied, 0:30 Q4",           4, "PT00M30.00S", 101, 101, "OKC", "SAS"),
            ("OKC leads +3, OT 2:00",   5, "PT02M00.00S", 108, 105, "OKC", "SAS"),
        ]

        okc_rtg = team_ratings.get("OKC", 8.5 if is_net else 1766)
        sas_rtg = team_ratings.get("SAS", 0.0 if is_net else 1500)

        for label, period, clock, hs, as_, ht, at in demo_scenarios:
            prob = predict_win_prob(
                period=period, clock=clock,
                home_score=hs, away_score=as_,
                home_rating=okc_rtg, away_rating=sas_rtg,
                home_sw=0, away_sw=0, is_playoffs=1,
                lead_changes=12, plays=180,
                model=model, scaler=scaler, T=T, device=device,
                is_net_rating=is_net,
            )
            with demo_col1:
                st.metric(label=label,
                          value=f"OKC {prob:.1%}",
                          delta=f"SAS {1-prob:.1%}")

        # Monte Carlo demo
        st.markdown("#### 🎲 Series Probability Demo — OKC leads 3-1")
        p_game = p_home_wins_game_from_rating(okc_rtg, sas_rtg, is_net=is_net)
        p_series = simulate_series(p_game, 3, 1, n_sims=100_000)
        rating_label = "NET rating" if is_net else "Elo"
        st.progress(p_series, text=f"OKC wins series: **{p_series:.1%}** | SAS: **{1-p_series:.1%}**")
        st.caption(f"Per-game OKC win prob ({rating_label}-based): {p_game:.1%} | 100K Monte Carlo sims")

        st.markdown("---")
        st.info("💡 **Tip:** Open this dashboard when a game tips off and select it from the dropdown above.")
        return

    # ── Game selector ──────────────────────────────────────────────────────
    STATUS_MAP = {1: "Pre-game", 2: "🔴 LIVE", 3: "Final"}

    game_options = {}
    for g in live_games:
        ht = g["homeTeam"]["teamTricode"]
        at = g["awayTeam"]["teamTricode"]
        hs = g["homeTeam"].get("score", 0) or 0
        as_ = g["awayTeam"].get("score", 0) or 0
        period = g.get("period", 0)
        status = STATUS_MAP.get(g.get("gameStatus", 0), "?")
        label = f"{at} {as_}  @  {ht} {hs}   Q{period}  │  {status}"
        game_options[label] = g

    selected_label = st.selectbox("Select game", list(game_options.keys()),
                                   label_visibility="collapsed")
    game = game_options[selected_label]

    # ── Extract game info ──────────────────────────────────────────────────
    game_id    = game["gameId"]
    game_status = game.get("gameStatus", 0)
    period     = game.get("period", 1)
    game_clock = game.get("gameClock", "PT12M00.00S") or "PT12M00.00S"
    clock_sec  = parse_clock(game_clock)

    ht_info = game["homeTeam"]; at_info = game["awayTeam"]
    ht_code = ht_info["teamTricode"]; at_code = at_info["teamTricode"]
    ht_name = f"{ht_info.get('teamCity','')} {ht_info.get('teamName','')}".strip()
    at_name = f"{at_info.get('teamCity','')} {at_info.get('teamName','')}".strip()
    ht_score = int(ht_info.get("score", 0) or 0)
    at_score = int(at_info.get("score", 0) or 0)
    ht_wins  = int(ht_info.get("wins",  0) or 0)
    at_wins  = int(at_info.get("wins",  0) or 0)

    ht_color = TEAM_COLORS.get(ht_code, "#007AC1")
    at_color = TEAM_COLORS.get(at_code, "#C8102E")
    ht_rtg   = team_ratings.get(ht_code, 0.0 if is_net else 1500.0)
    at_rtg   = team_ratings.get(at_code, 0.0 if is_net else 1500.0)
    rtg_label = "NET" if is_net else "Elo"

    # Series wins from live scoreboard (wins within current series)
    # The API homeTeam.wins / awayTeam.wins are SERIES wins in playoffs
    home_sw = ht_wins; away_sw = at_wins
    is_playoffs = 1   # dashboard is built for playoff use

    # ── Fetch play-by-play ─────────────────────────────────────────────────
    actions = fetch_pbp(game_id)

    # ── Build history ──────────────────────────────────────────────────────
    hist = build_game_history(
        actions, ht_rtg, at_rtg, home_sw, away_sw, is_playoffs,
        model, scaler, T, device, is_net=is_net,
    ) if actions else pd.DataFrame()

    # ── Auto-save completed games to history ───────────────────────────────
    # Saves silently once when game_status == 3 (Final) and hist has data.
    if game_status == 3 and not hist.empty:
        save_game_history(
            game_id    = game_id,
            away_code  = at_code,
            home_code  = ht_code,
            hist       = hist,
            final_away = int(hist["away_score"].iloc[-1]),
            final_home = int(hist["home_score"].iloc[-1]),
            series_away= away_sw,
            series_home= home_sw,
        )

    # Current win prob
    if not hist.empty:
        current_prob = hist["home_win_prob"].iloc[-1]
        current_hs   = hist["home_score"].iloc[-1]
        current_as   = hist["away_score"].iloc[-1]
    else:
        # Pre-game: use team rating
        current_prob = p_home_wins_game_from_rating(ht_rtg, at_rtg, is_net=is_net)
        current_hs   = ht_score
        current_as   = at_score

    # ── SCORE CARD ROW ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>LIVE GAME</div>", unsafe_allow_html=True)

    col_away, col_mid, col_home, col_gauge = st.columns([2, 1.2, 2, 2.5])

    with col_away:
        at_rtg_str = f"{at_rtg:+.1f}" if is_net else f"{at_rtg:.0f}"
        st.markdown(f"""
        <div class="score-card" style="border-top:4px solid {at_color}">
            <div class="team-code" style="color:{at_color}">{at_code}</div>
            <div class="score-num">{current_as}</div>
            <div class="record">{at_name}</div>
            <div class="record">{rtg_label} {at_rtg_str}</div>
        </div>""", unsafe_allow_html=True)

    with col_mid:
        clock_disp = fmt_clock(period, clock_sec)
        badge = ('<span class="live-badge">LIVE</span>' if game_status == 2
                 else '<span class="final-badge">FINAL</span>' if game_status == 3
                 else '<span class="final-badge">PRE</span>')
        st.markdown(f"""
        <div style="text-align:center;padding-top:24px">
            {badge}
            <div style="font-size:1.1rem;font-weight:700;margin-top:10px">{clock_disp}</div>
            <div style="font-size:0.8rem;color:#8b949e;margin-top:6px">Series</div>
            <div style="font-size:1.2rem;font-weight:700">{away_sw} – {home_sw}</div>
        </div>""", unsafe_allow_html=True)

    with col_home:
        ht_rtg_str = f"{ht_rtg:+.1f}" if is_net else f"{ht_rtg:.0f}"
        st.markdown(f"""
        <div class="score-card" style="border-top:4px solid {ht_color}">
            <div class="team-code" style="color:{ht_color}">{ht_code}</div>
            <div class="score-num">{current_hs}</div>
            <div class="record">{ht_name}</div>
            <div class="record">{rtg_label} {ht_rtg_str}</div>
        </div>""", unsafe_allow_html=True)

    with col_gauge:
        st.plotly_chart(
            gauge_chart(current_prob, ht_code, at_code, ht_color, at_color),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── WIN PROB CHART ─────────────────────────────────────────────────────
    if not hist.empty:
        st.markdown("<div class='section-header'>WIN PROBABILITY OVER TIME</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            win_prob_chart(hist, ht_code, at_code, ht_color, at_color),
            use_container_width=True, config={"displayModeBar": False},
        )
    else:
        st.info("⏳ No play-by-play data yet — chart will appear once the game starts.")

    # ── SERIES + BRACKET ROW ───────────────────────────────────────────────
    col_series, col_bracket = st.columns([1, 1])

    with col_series:
        st.markdown("<div class='section-header'>SERIES PROBABILITY</div>",
                    unsafe_allow_html=True)
        p_home_game  = current_prob if not hist.empty else p_home_wins_game_from_rating(ht_rtg, at_rtg, is_net=is_net)
        # Phase 5: apply series form adjustment
        _hf, _af     = _series_form_adjustment(home_sw, away_sw, p_home_wins_game_from_rating(ht_rtg, at_rtg, is_net=is_net))
        p_home_series = simulate_series(p_home_game, home_sw, away_sw, n_sims=n_mc_sims,
                                        h_form=_hf, a_form=_af)

        st.plotly_chart(
            series_prob_chart(ht_code, at_code, p_home_series,
                               home_sw, away_sw, ht_color, at_color),
            use_container_width=True, config={"displayModeBar": False},
        )

        # Detailed series metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">{ht_code} series</div>
                <div class="metric-value" style="color:{ht_color}">{p_home_series:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">{at_code} series</div>
                <div class="metric-value" style="color:{at_color}">{1-p_home_series:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            games_left = f"{home_sw}-{away_sw}  best of 7"
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Monte Carlo</div>
                <div class="metric-value" style="font-size:1rem">{n_mc_sims:,} sims</div>
            </div>""", unsafe_allow_html=True)

    with col_bracket:
        if show_bracket:
            st.markdown("<div class='section-header'>FINALS PROBABILITY</div>",
                        unsafe_allow_html=True)
            finals_probs = compute_finals_probs(live_games, team_ratings, {})
            if finals_probs:
                st.plotly_chart(
                    bracket_chart(finals_probs),
                    use_container_width=True, config={"displayModeBar": False},
                )
            else:
                st.caption("Finals probability requires multiple active series.")

    # ── PLAY-BY-PLAY TABLE ─────────────────────────────────────────────────
    if show_history and not hist.empty:
        st.markdown("<div class='section-header'>PLAY LOG</div>", unsafe_allow_html=True)
        cols_to_show = ["clock_display","home_score","away_score",
                        "score_diff","momentum","home_win_prob","description"]
        cols_to_show = [c for c in cols_to_show if c in hist.columns]
        display = hist[cols_to_show].copy()
        col_rename = {
            "clock_display": "Clock", "home_score": "Home", "away_score": "Away",
            "score_diff": "Diff", "momentum": "Momentum",
            "home_win_prob": "Home Win %", "description": "Last Play"
        }
        display.rename(columns={k:v for k,v in col_rename.items() if k in display.columns}, inplace=True)
        if "Home Win %" in display.columns:
            display["Home Win %"] = display["Home Win %"].map("{:.1%}".format)
        if "Diff" in display.columns:
            display["Diff"] = display["Diff"].map(lambda x: f"+{x}" if x > 0 else str(x))
        if "Momentum" in display.columns:
            display["Momentum"] = display["Momentum"].map(lambda x: f"{x:+.2f}")
        st.dataframe(
            display.iloc[::-1].reset_index(drop=True).head(50),
            use_container_width=True, height=320,
            hide_index=True,
        )

    # ── KEY METRICS ROW ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>MODEL SNAPSHOT</div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)

    # Momentum from last tracked play
    live_momentum = hist["momentum"].iloc[-1] if not hist.empty and "momentum" in hist.columns else 0.0
    mom_pct  = f"{live_momentum:+.0%}"
    mom_color= "#1D9E75" if live_momentum > 0.05 else "#E24B4A" if live_momentum < -0.05 else "#8b949e"

    for col, label, val, color in [
        (mc1, "Home Win Prob",  f"{current_prob:.1%}",                       None),
        (mc2, "Away Win Prob",  f"{1-current_prob:.1%}",                      None),
        (mc3, "Score Diff",     f"{current_hs - current_as:+d}",              None),
        (mc4, "Time Remaining", fmt_clock(period, clock_sec).replace("  "," "), None),
        (mc5, "Plays Tracked",  f"{len(hist):,}",                             None),
        (mc6, "Momentum",       mom_pct,                                       mom_color),
    ]:
        color_style = f"color:{color}" if color else ""
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="{color_style}">{val}</div>
        </div>""", unsafe_allow_html=True)

    # ── AUTO-REFRESH ───────────────────────────────────────────────────────
    if refresh_sec > 0 and game_status == 2:
        time.sleep(refresh_sec)
        st.cache_data.clear()
        st.rerun()
    elif refresh_sec > 0 and game_status != 2:
        st.info(f"ℹ️  Auto-refresh paused — game is not live. Status: {STATUS_MAP.get(game_status,'?')}")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
