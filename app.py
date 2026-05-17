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
    "quarter_time_elapsed_pct", "home_elo", "away_elo", "elo_diff",
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

    with open(MODEL_DIR / "elo_ratings.json") as f:
        elo_ratings = json.load(f)

    return model, scaler, T, elo_ratings, device


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
    home_elo: float, away_elo: float,
    home_sw: int, away_sw: int,
    is_playoffs: int,
    lead_changes: int, plays: int,
    model, scaler, T, device,
) -> float:
    """Return calibrated P(home team wins) for a single game state."""
    clock_sec  = parse_clock(clock)
    t_rem      = time_remaining(period, clock_sec)
    period_dur = 720.0 if period <= 4 else 300.0
    q_elapsed  = float(np.clip((period_dur - clock_sec) / period_dur, 0, 1))
    score_diff = float(np.clip(home_score - away_score, -80, 80))
    n_plays    = max(1, plays)

    raw = np.array([[
        score_diff, t_rem, float(period), q_elapsed,
        home_elo, away_elo, home_elo - away_elo,
        float(home_sw), float(away_sw),
        float(is_playoffs), float(period > 4),
        lead_changes / n_plays,
    ]], dtype=np.float32)

    scaled = scaler.transform(raw).astype(np.float32)
    tensor = torch.from_numpy(scaled).to(device)

    with torch.no_grad():
        logit = model.logits(tensor).item()
    prob = 1 / (1 + np.exp(-logit / T))   # temperature-scaled
    return float(np.clip(prob, 0.001, 0.999))


# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO SERIES SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def simulate_series(p_home: float, h_wins: int, a_wins: int,
                    best_of: int = 7, n_sims: int = 100_000) -> float:
    """Vectorised Monte Carlo: P(home wins series) from current state."""
    target = (best_of // 2) + 1
    h_need = target - h_wins
    a_need = target - a_wins
    if h_need <= 0: return 1.0
    if a_need <= 0: return 0.0

    rng   = np.random.default_rng()
    max_g = h_need + a_need - 1
    games = rng.random((n_sims, max_g)) < p_home
    h_cum = np.cumsum(games.astype(np.int16), axis=1)
    a_cum = np.cumsum((~games).astype(np.int16), axis=1)
    h_ever = h_cum[:, -1] >= h_need
    a_ever = a_cum[:, -1] >= a_need
    h_cl   = np.where(h_ever, np.argmax(h_cum >= h_need, axis=1), max_g + 1)
    a_cl   = np.where(a_ever, np.argmax(a_cum >= a_need, axis=1), max_g + 1)
    return float((h_cl < a_cl).mean())


def p_home_wins_game_from_elo(home_elo: float, away_elo: float,
                               home_adv: float = 100.0) -> float:
    """Elo-based pre-game win probability (no in-game state)."""
    return 1 / (1 + 10 ** ((away_elo - (home_elo + home_adv)) / 400))


# ═══════════════════════════════════════════════════════════════════════════
# NBA API HELPERS  (TTL-cached per game)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_games() -> list:
    try:
        board = live_sb.ScoreBoard()
        games = board.games.get_dict()
        return games if isinstance(games, list) else []
    except Exception as e:
        # nba_api returns empty body when no games today ("Expecting value: line 1")
        # or when the NBA CDN is momentarily unavailable — both are safe to swallow
        return []


@st.cache_data(ttl=20, show_spinner=False)
def fetch_pbp(game_id: str) -> list:
    try:
        pbp = live_pbp.PlayByPlay(game_id)
        actions = pbp.actions.get_dict()
        return actions if isinstance(actions, list) else []
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# GAME STATE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_game_history(actions: list, home_elo: float, away_elo: float,
                       home_sw: int, away_sw: int, is_playoffs: int,
                       model, scaler, T, device) -> pd.DataFrame:
    """
    Walk through all play-by-play actions and compute win prob at each scored play.
    Returns DataFrame with columns: action_num, period, clock_sec, time_remaining,
    home_score, away_score, score_diff, home_win_prob, description.
    """
    rows = []
    lead_changes = 0
    prev_lead    = 0
    play_num     = 0
    prev_hs = prev_as = 0

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
        period = int(act.get("period", 1))
        clock  = act.get("clock", "PT12M00.00S")
        clock_sec = parse_clock(clock)

        cur_lead = np.sign(hs - as_)
        if cur_lead != prev_lead and cur_lead != 0:
            lead_changes += 1
        prev_lead = cur_lead

        prob = predict_win_prob(
            period=period, clock=clock,
            home_score=hs, away_score=as_,
            home_elo=home_elo, away_elo=away_elo,
            home_sw=home_sw, away_sw=away_sw,
            is_playoffs=is_playoffs,
            lead_changes=lead_changes, plays=play_num,
            model=model, scaler=scaler, T=T, device=device,
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
        **PLOTLY_BASE, height=250,
        margin=dict(l=20, r=20, t=30, b=10),
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

def compute_finals_probs(live_games: list, elo_ratings: dict,
                          series_states: dict) -> dict:
    """
    Estimate P(each team wins NBA Finals) by simulating all remaining series.

    series_states : { 'CONF-SERIES-KEY': {'home':CODE,'away':CODE,'hw':int,'aw':int} }
    Returns        : { TEAM_CODE: probability }
    """
    try:
        # Gather all active playoff series from live game data
        conf_finals = {}
        finals_data = {}

        for g in live_games:
            ht = g["homeTeam"]["teamTricode"]
            at = g["awayTeam"]["teamTricode"]
            hw = g["homeTeam"].get("wins", 0)
            aw = g["awayTeam"].get("wins", 0)
            gid = g.get("gameId", "")

            # Identify round by wins (rough heuristic: if game 1 of series, check series wins)
            # For simplicity, track all playoff series
            key = tuple(sorted([ht, at]))
            if key not in conf_finals:
                conf_finals[key] = {"home": ht, "away": at, "hw": hw, "aw": aw}

        if not conf_finals:
            return {}

        # Simulate each remaining series
        team_finals_prob = {}
        series_list = list(conf_finals.values())

        for series in series_list:
            ht = series["home"]; at = series["away"]
            h_elo = elo_ratings.get(ht, 1500)
            a_elo = elo_ratings.get(at, 1500)
            p_home_game = p_home_wins_game_from_elo(h_elo, a_elo)
            p_home_series = simulate_series(p_home_game, series["hw"], series["aw"],
                                             n_sims=50_000)
            team_finals_prob[ht] = team_finals_prob.get(ht, 0) + p_home_series
            team_finals_prob[at] = team_finals_prob.get(at, 0) + (1 - p_home_series)

        # Normalise so values represent rough Finals win probability
        # (simplified: just P of winning their current series, not full bracket)
        return {k: round(v, 4) for k, v in
                sorted(team_finals_prob.items(), key=lambda x: x[1], reverse=True)}

    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Load system ────────────────────────────────────────────────────────
    try:
        model, scaler, T, elo_ratings, device = load_system()
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
        st.caption(f"AUC: {0.8539:.4f} · Brier: {0.1560:.4f}")
        st.caption(f"Temperature T: {T:.4f}")
        st.caption(f"Device: {str(device).upper()}")
        st.markdown("---")

        show_bracket = st.toggle("Show Finals Probability", value=True)
        show_history = st.toggle("Show Full Play History", value=False)
        n_mc_sims    = st.select_slider("Monte Carlo sims",
                                         options=[10_000, 50_000, 100_000, 200_000],
                                         value=100_000)

        st.markdown("---")
        last_refresh = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        st.caption(f"Last refresh: {last_refresh}")

        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch live games ───────────────────────────────────────────────────
    live_games = fetch_live_games()

    st.markdown("<h1 style='color:#e6edf3;font-size:1.8rem;margin-bottom:4px'>🏀 NBA Live Win Probability</h1>",
                unsafe_allow_html=True)

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

        okc_elo = elo_ratings.get("OKC", 1766)
        sas_elo = elo_ratings.get("SAS", 1500)

        for label, period, clock, hs, as_, ht, at in demo_scenarios:
            prob = predict_win_prob(
                period=period, clock=clock,
                home_score=hs, away_score=as_,
                home_elo=okc_elo, away_elo=sas_elo,
                home_sw=0, away_sw=0, is_playoffs=1,
                lead_changes=12, plays=180,
                model=model, scaler=scaler, T=T, device=device,
            )
            with demo_col1:
                st.metric(label=label,
                          value=f"OKC {prob:.1%}",
                          delta=f"SAS {1-prob:.1%}")

        # Monte Carlo demo
        st.markdown("#### 🎲 Series Probability Demo — OKC leads 3-1")
        p_game = p_home_wins_game_from_elo(okc_elo, sas_elo)
        p_series = simulate_series(p_game, 3, 1, n_sims=100_000)
        st.progress(p_series, text=f"OKC wins series: **{p_series:.1%}** | SAS: **{1-p_series:.1%}**")
        st.caption(f"Per-game OKC win prob (Elo-based): {p_game:.1%} | 100K Monte Carlo sims")

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
    ht_elo   = elo_ratings.get(ht_code, 1500.0)
    at_elo   = elo_ratings.get(at_code, 1500.0)

    # Series wins from live scoreboard (wins within current series)
    # The API homeTeam.wins / awayTeam.wins are SERIES wins in playoffs
    home_sw = ht_wins; away_sw = at_wins
    is_playoffs = 1   # dashboard is built for playoff use

    # ── Fetch play-by-play ─────────────────────────────────────────────────
    actions = fetch_pbp(game_id)

    # ── Build history ──────────────────────────────────────────────────────
    hist = build_game_history(
        actions, ht_elo, at_elo, home_sw, away_sw, is_playoffs,
        model, scaler, T, device,
    ) if actions else pd.DataFrame()

    # Current win prob
    if not hist.empty:
        current_prob = hist["home_win_prob"].iloc[-1]
        current_hs   = hist["home_score"].iloc[-1]
        current_as   = hist["away_score"].iloc[-1]
    else:
        # Pre-game: use Elo
        current_prob = p_home_wins_game_from_elo(ht_elo, at_elo)
        current_hs   = ht_score
        current_as   = at_score

    # ── SCORE CARD ROW ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>LIVE GAME</div>", unsafe_allow_html=True)

    col_away, col_mid, col_home, col_gauge = st.columns([2, 1.2, 2, 2.5])

    with col_away:
        st.markdown(f"""
        <div class="score-card" style="border-top:4px solid {at_color}">
            <div class="team-code" style="color:{at_color}">{at_code}</div>
            <div class="score-num">{current_as}</div>
            <div class="record">{at_name}</div>
            <div class="record">Elo {at_elo:.0f}</div>
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
        st.markdown(f"""
        <div class="score-card" style="border-top:4px solid {ht_color}">
            <div class="team-code" style="color:{ht_color}">{ht_code}</div>
            <div class="score-num">{current_hs}</div>
            <div class="record">{ht_name}</div>
            <div class="record">Elo {ht_elo:.0f}</div>
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
        p_home_game  = current_prob if not hist.empty else p_home_wins_game_from_elo(ht_elo, at_elo)
        p_home_series = simulate_series(p_home_game, home_sw, away_sw, n_sims=n_mc_sims)

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
            finals_probs = compute_finals_probs(live_games, elo_ratings, {})
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
        display = hist[["clock_display","home_score","away_score",
                         "score_diff","home_win_prob","description"]].copy()
        display.columns = ["Clock","Home","Away","Diff","Home Win %","Last Play"]
        display["Home Win %"] = display["Home Win %"].map("{:.1%}".format)
        display["Diff"]       = display["Diff"].map(lambda x: f"+{x}" if x > 0 else str(x))
        st.dataframe(
            display.iloc[::-1].reset_index(drop=True).head(50),
            use_container_width=True, height=320,
            hide_index=True,
        )

    # ── KEY METRICS ROW ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>MODEL SNAPSHOT</div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, label, val in [
        (mc1, "Home Win Prob",   f"{current_prob:.1%}"),
        (mc2, "Away Win Prob",   f"{1-current_prob:.1%}"),
        (mc3, "Score Diff",      f"{current_hs - current_as:+d}"),
        (mc4, "Time Remaining",  fmt_clock(period, clock_sec).replace("  "," ")),
        (mc5, "Plays Tracked",   f"{len(hist):,}"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
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
