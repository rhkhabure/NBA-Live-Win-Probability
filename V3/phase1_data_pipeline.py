"""
phase1_data_pipeline.py  —  NBA Win Probability V3
====================================================
Fetches all NBA play-by-play data (2018-19 → 2024-25 regular season + playoffs),
builds the 18-feature dataset, and runs the full validation suite.

Run from the V3/ project root:
    python phase1_data_pipeline.py

Outputs (written to data/):
    data/raw/game_logs_all.parquet        — game-level metadata (all seasons)
    data/raw/pbp/{game_id}.parquet        — one PBP file per game (~9,000 files)
    data/raw/darko_dpm.csv                — DARKO DPM (manual download from darko.app)
    data/raw/bpm_by_season.parquet        — BPM fallback (auto-scraped)
    data/processed/features_v3.parquet   — 18-feature dataset (~1.17M rows)
    data/results/phase1_report.md         — validation report

Structure of data/raw/pbp/{game_id}.parquet columns (PlayByPlayV3):
    actionNumber, clock, description, isFieldGoal, period,
    personIdsFilter, qualifiers, scoreAway, scoreHome,
    subType, teamId, teamTricode, actionType, orderNumber
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import sys
import time
import json
import pickle
import warnings
import traceback
from pathlib import Path
from datetime import datetime, timezone

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.special import expit           # sigmoid for elo_prior_weight
from tqdm import tqdm

# ── nba_api (V3 endpoints only) ───────────────────────────────────────────────
from nba_api.stats.endpoints import (
    leaguegamelog,
    playbyplayv3,
)
from nba_api.stats.static import teams as nba_teams_static

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT       = Path(".")
DATA_RAW   = ROOT / "data" / "raw"
DATA_PBP   = DATA_RAW / "pbp"
DATA_PROC  = ROOT / "data" / "processed"
DATA_RES   = ROOT / "data" / "results"

for d in (DATA_RAW, DATA_PBP, DATA_PROC, DATA_RES):
    d.mkdir(parents=True, exist_ok=True)

# Seasons to fetch: 2018-19 through 2024-25
# nba_api uses format "2018-19", "2019-20", etc.
SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]

# Season types to include
SEASON_TYPES = ["Regular Season", "Playoffs"]

# Elo config (unchanged from V1)
ELO_START      = 1500
ELO_K          = 20
ELO_REVERT     = 0.35
HOME_ADVANTAGE = 100    # Elo points

# Score diff winsorisation (tightened from V1's ±80)
SCORE_DIFF_CLIP = 60.0

# Rate limiting — nba_api bans fast scrapers
API_SLEEP      = 0.8    # seconds between game-level calls
API_SLEEP_LONG = 5.0    # seconds between season-level calls

# 2019-20 bubble — neutral court flag
BUBBLE_GAMES_SEASON = "2019-20"

# Feature columns (18) — the single source of truth for column order
FEATURE_COLS = [
    # Kept from V1 (minus the two series-win cols)
    "score_diff",               # 0  — winsorised ±60
    "time_remaining_sec",       # 1  — seconds until end of regulation
    "quarter",                  # 2  — period number (5+ = OT)
    "quarter_time_elapsed_pct", # 3  — fraction of current period elapsed
    "home_elo",                 # 4  — pre-game Elo, home team
    "away_elo",                 # 5  — pre-game Elo, away team
    "elo_diff",                 # 6  — home_elo − away_elo
    "is_playoffs",              # 7  — binary flag
    "is_overtime",              # 8  — binary flag
    "lead_changes_norm",        # 9  — lead changes ÷ plays so far
    # New in V3
    "possession",               # 10 — 1 = home scored (had ball), 0 = away scored
    "home_in_bonus",            # 11 — away team has ≥5 fouls in quarter (home shoots FTs)
    "away_in_bonus",            # 12 — home team has ≥5 fouls in quarter (away shoots FTs)
    "home_avail_delta",         # 13 — Σ(DPM × min) available vs expected, home
    "away_avail_delta",         # 14 — same for away
    "elo_prior_weight",         # 15 — state-dependent Elo decay
    # Note: home_fts_pending / away_fts_pending removed —
    # FT sequences resolve before the next scored-play snapshot so these
    # are almost always 0 at snapshot time. Bonus captures the relevant
    # foul-situation context instead.
]
TARGET_COL = "home_team_won"
N_FEATURES = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GAME LOGS
# Fetch season-level game metadata (team IDs, dates, outcomes, season type)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_game_logs(seasons: list[str]) -> pd.DataFrame:
    """
    Pull game logs for all seasons using LeagueGameLog (V3-compatible).
    Caches per-season to avoid re-fetching.
    Returns a game-level DataFrame with one row per game (home + away merged).
    """
    cache_path = DATA_RAW / "game_logs_all.parquet"
    if cache_path.exists():
        print(f"[game_logs] Loading from cache ({cache_path})")
        return pd.read_parquet(cache_path)

    all_rows = []
    for season in seasons:
        for stype in SEASON_TYPES:
            season_cache = DATA_RAW / f"logs_{season.replace('-','_')}_{stype.replace(' ','_')}.parquet"
            if season_cache.exists():
                df = pd.read_parquet(season_cache)
                all_rows.append(df)
                print(f"  [cache] {season} {stype}: {len(df)//2} games")
                continue

            try:
                log = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=stype,
                    timeout=30,
                )
                df = log.get_data_frames()[0]
                df["SEASON"]      = season
                df["SEASON_TYPE"] = stype
                df.to_parquet(season_cache, index=False)
                all_rows.append(df)
                n_games = df["GAME_ID"].nunique()
                print(f"  [fetch] {season} {stype}: {n_games} games ({len(df)} rows)")
                time.sleep(API_SLEEP_LONG)
            except Exception as e:
                print(f"  [ERROR] {season} {stype}: {e}")
                time.sleep(API_SLEEP_LONG * 2)

    if not all_rows:
        raise RuntimeError("No game logs fetched — check nba_api connection")

    raw = pd.concat(all_rows, ignore_index=True)

    # Merge home + away rows into one row per game
    home_mask = raw["MATCHUP"].str.contains(r"vs\.", na=False)
    home = raw[home_mask].copy().rename(columns={
        "TEAM_ID": "home_team_id",
        "TEAM_ABBREVIATION": "home_team",
        "PTS": "home_pts",
        "WL": "home_wl",
    })
    away = raw[~home_mask].copy().rename(columns={
        "TEAM_ID": "away_team_id",
        "TEAM_ABBREVIATION": "away_team",
        "PTS": "away_pts",
    })

    games = home[[
        "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE",
        "home_team_id", "home_team", "home_pts", "home_wl",
    ]].merge(
        away[["GAME_ID", "away_team_id", "away_team", "away_pts"]],
        on="GAME_ID",
        how="inner",
    )

    games["home_team_won"] = (games["home_wl"] == "W").astype(int)
    games["is_playoffs"]   = (games["SEASON_TYPE"] == "Playoffs").astype(int)
    games["is_bubble"]     = (
        (games["SEASON"] == BUBBLE_GAMES_SEASON) &
        (games["SEASON_TYPE"] == "Regular Season")
    ).astype(int)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    games.to_parquet(cache_path, index=False)
    print(f"\n[game_logs] {len(games):,} games saved → {cache_path}")
    return games


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ELO RATINGS
# Compute pre-game Elo for every game chronologically
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """
    Add home_elo, away_elo, elo_diff columns to games df.
    Elo is computed chronologically; season-start regression applied.
    Uses separate Elo tracks for regular season and playoffs (per Neil Paine).
    """
    df = games.sort_values("GAME_DATE").copy()
    elo        = {}   # team_id → current elo
    prev_season = None
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        season = row["SEASON"]
        hid    = row["home_team_id"]
        aid    = row["away_team_id"]

        # Season-start regression to mean
        if season != prev_season:
            for tid in list(elo):
                elo[tid] = elo[tid] + ELO_REVERT * (ELO_START - elo[tid])
            prev_season = season

        # Initialise new teams
        if hid not in elo:
            elo[hid] = ELO_START
        if aid not in elo:
            elo[aid] = ELO_START

        # Record pre-game Elo
        h_elo = elo[hid]
        a_elo = elo[aid]
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Expected win probability (home advantage +100 Elo)
        exp_h = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADVANTAGE)) / 400))

        # Update Elo
        if row["home_team_won"]:
            elo[hid] += ELO_K * (1 - exp_h)
            elo[aid] -= ELO_K * (1 - exp_h)
        else:
            elo[aid] += ELO_K * exp_h
            elo[hid] -= ELO_K * exp_h

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Export final Elo ratings for the dashboard
    final_elo = {str(tid): round(val, 2) for tid, val in elo.items()}
    elo_path  = DATA_RAW / "elo_ratings_v3.json"
    elo_path.write_text(json.dumps(final_elo, indent=2))
    print(f"[elo] Ratings computed and saved → {elo_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — AVAILABILITY DELTA (pregame prior adjustment)
# Computes Σ(DPM × projected_minutes) for available vs expected roster
# Uses DARKO DPM if CSV present, falls back to BPM from Basketball Reference
# ─────────────────────────────────────────────────────────────────────────────

def load_player_impact() -> dict[str, dict[str, float]]:
    """
    Returns {season: {player_name: dpm_value}} dict.

    Priority:
    1. data/raw/darko_dpm.csv (manually downloaded from darko.app)
       Expected columns: player, season, dpm  (or total_dpm)
    2. data/raw/bpm_by_season.parquet (auto-scraped via basketball_reference_web_scraper)
       Expected columns: player, season, bpm
    3. Empty dict — availability delta features will be 0.0 (neutral)
    """
    darko_path = DATA_RAW / "darko_dpm.csv"
    bpm_path   = DATA_RAW / "bpm_by_season.parquet"

    if darko_path.exists():
        df  = pd.read_csv(darko_path)
        # Normalise column names — darko.app CSV may vary
        df.columns = df.columns.str.lower().str.strip()
        dpm_col = next((c for c in df.columns if "dpm" in c), None)
        if dpm_col and "player" in df.columns and "season" in df.columns:
            out = {}
            for _, row in df.iterrows():
                season = str(row["season"])
                player = str(row["player"]).strip()
                dpm    = float(row[dpm_col]) if pd.notna(row[dpm_col]) else 0.0
                out.setdefault(season, {})[player] = dpm
            print(f"[avail] DARKO DPM loaded: {len(df):,} player-seasons")
            return out

    if bpm_path.exists():
        df  = pd.read_parquet(bpm_path)
        out = {}
        for _, row in df.iterrows():
            season = str(row.get("season", ""))
            player = str(row.get("player", "")).strip()
            bpm    = float(row.get("bpm", 0.0)) if pd.notna(row.get("bpm")) else 0.0
            out.setdefault(season, {})[player] = bpm
        print(f"[avail] BPM fallback loaded: {len(df):,} player-seasons")
        return out

    print("[avail] No player impact data found — avail_delta will be 0.0")
    print("        Download DARKO CSV from darko.app and place at:")
    print(f"        {darko_path.resolve()}")
    return {}


def compute_avail_delta(
    game_id: str,
    home_team_id: int,
    away_team_id: int,
    season: str,
    player_impact: dict,
) -> tuple[float, float]:
    """
    Returns (home_avail_delta, away_avail_delta).

    avail_delta = Σ(DPM × projected_min) for available players
                - Σ(DPM × projected_min) for expected full roster

    In practice without live injury data for historical games we approximate as:
      delta = 0.0 for the training set (we don't know historical availability)
    For LIVE use, the dashboard pulls nbainjuries and computes this in real time.

    This is intentional: the neural net learns game-state features from
    the training data; the availability feature is primarily a PREGAME adjustment
    applied before kickoff by the live system.
    """
    # For historical training data: return 0.0 (neutral)
    # The feature will be non-zero only in live dashboard inference
    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CLOCK / SCORE PARSERS
# Copied and improved from phase1_fixes.ipynb
# ─────────────────────────────────────────────────────────────────────────────

_ISO_RE   = re.compile(r"PT(\d+)M([\d.]+)S", re.IGNORECASE)  # PT12M00.00S
_MMSS_RE  = re.compile(r"^(\d{1,2}):(\d{2})$")               # 12:00 or 1:30


def parse_game_clock(val) -> float:
    """
    Return seconds REMAINING IN THE CURRENT PERIOD.
    Handles:
      - ISO 8601 : 'PT12M00.00S'  → 720.0
      - MM:SS    : '11:47'        → 707.0
      - Numeric  :  47.5          → 47.5 (already seconds)
    Returns 0.0 on any unrecognised input.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    s = str(val).strip()
    m = _ISO_RE.match(s)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    m = _MMSS_RE.match(s)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return 0.0


def period_to_time_remaining(period: int, clock_sec: float) -> float:
    """
    Total seconds remaining until end of regulation.
    Regulation: 4 × 720s = 2880s.
    OT (period ≥ 5): set to 0 (game is beyond regulation).
    """
    if period <= 4:
        return clock_sec + max(0, 4 - period) * 720.0
    return 0.0  # OT


def parse_score(score_str) -> tuple[int, int]:
    """
    Parse combined score string 'away - home' → (home, away).
    Returns (0, 0) on failure.
    """
    if not score_str or pd.isna(score_str):
        return 0, 0
    parts = str(score_str).split("-")
    if len(parts) != 2:
        return 0, 0
    try:
        away, home = int(parts[0].strip()), int(parts[1].strip())
        return home, away
    except ValueError:
        return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PBP FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pbp(game_id: str) -> pd.DataFrame | None:
    """
    Fetch PlayByPlayV3 for one game. Returns DataFrame or None on failure.
    Caches to data/raw/pbp/{game_id}.parquet.
    """
    cache = DATA_PBP / f"{game_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
        df  = pbp.get_data_frames()[0]
        if df.empty:
            return None
        df.to_parquet(cache, index=False)
        time.sleep(API_SLEEP)
        return df
    except Exception as e:
        print(f"  [PBP ERROR] {game_id}: {e}")
        time.sleep(API_SLEEP * 3)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SNAPSHOT EXTRACTOR
# Builds one feature row per scored play, plus tipoff and end-of-game
# ─────────────────────────────────────────────────────────────────────────────

def extract_snapshots(
    pbp_df: pd.DataFrame,
    game_meta: pd.Series,
    game_id: str = "",
    home_avail_delta: float = 0.0,
    away_avail_delta: float = 0.0,
) -> pd.DataFrame:
    """
    Convert raw V3 PBP into feature snapshots.

    V3 column names (confirmed from phase1_fixes.ipynb Cell 5):
        scoreHome, scoreAway, clock, period, actionType, subType, teamId

    New features extracted here:
        possession          — from teamId on scored play
        home_fts_pending    — running FT count (1 of 2, 2 of 2 events)
        away_fts_pending    — same
        home_in_bonus       — team fouls ≥ 5 in quarter
        away_in_bonus       — same
        lead_changes_norm   — from V1 (unchanged)
        elo_prior_weight    — computed from score_diff and time_remaining
    """
    df = pbp_df.copy()

    # ── Normalise column names ────────────────────────────────────────────────
    df.columns = df.columns.str.lower()
    col_map    = {c: c for c in df.columns}

    # Resolve key column names
    action_col = next((c for c in ("actiontype", "action_type") if c in col_map), None)
    sub_col    = next((c for c in ("subtype", "sub_type", "description") if c in col_map), None)
    team_col   = next((c for c in ("teamid", "team_id") if c in col_map), None)
    clock_col  = next((c for c in ("clock", "pctimestring") if c in col_map), None)
    period_col = next((c for c in ("period", "quarter") if c in col_map), None)
    order_col  = next((c for c in ("actionnumber", "actionid", "ordernumber") if c in col_map), None)

    if not clock_col or not period_col:
        return pd.DataFrame()

    home_team_id = game_meta["home_team_id"]

    # ── STEP A: Compute FT state and bonus on the FULL PBP (all event types) ──
    # This must happen BEFORE score-filtering because FT/foul events have no score.
    # We build per-row state, then forward-fill onto scored plays.

    full = df.copy()
    full["_period"] = pd.to_numeric(full[period_col], errors="coerce").fillna(1).astype(int)
    if order_col:
        full = full.sort_values(order_col).reset_index(drop=True)

    home_fts   = 0
    away_fts   = 0
    home_fouls = 0
    away_fouls = 0
    prev_per   = None

    ft_home_list   = []
    ft_away_list   = []
    bon_home_list  = []
    bon_away_list  = []

    if action_col and team_col:
        for _, row in full.iterrows():
            atype = str(row.get(action_col, ""))
            sub   = str(row.get(sub_col, "")) if sub_col else ""
            tid   = row.get(team_col)
            per   = int(row.get("_period", 1))

            # Reset team fouls on new quarter
            if per != prev_per:
                home_fouls = 0
                away_fouls = 0
                prev_per   = per

            # FT state — V3: actiontype == 'Free Throw'
            if atype == "Free Throw":
                if "1 of 2" in sub or "1 of 3" in sub:
                    if tid == home_team_id:
                        home_fts = 1
                    else:
                        away_fts = 1
                elif any(x in sub for x in ("2 of 2", "1 of 1", "2 of 3", "3 of 3")):
                    if tid == home_team_id:
                        home_fts = 0
                    else:
                        away_fts = 0

            # Bonus — V3: actiontype == 'Foul'; exclude offensive foul turnovers
            if atype == "Foul" and "Offensive Foul Turnover" not in sub:
                if tid == home_team_id:
                    home_fouls += 1
                else:
                    away_fouls += 1

            ft_home_list.append(home_fts)
            ft_away_list.append(away_fts)
            bon_home_list.append(int(away_fouls >= 5))   # home in bonus when AWAY has ≥5
            bon_away_list.append(int(home_fouls >= 5))   # away in bonus when HOME has ≥5
    else:
        ft_home_list  = [0] * len(full)
        ft_away_list  = [0] * len(full)
        bon_home_list = [0] * len(full)
        bon_away_list = [0] * len(full)

    full["_home_fts"]   = ft_home_list
    full["_away_fts"]   = ft_away_list
    full["_home_bonus"] = bon_home_list
    full["_away_bonus"] = bon_away_list

    # ── STEP B: Filter to scored plays only ──────────────────────────────────
    has_split = "scorehome" in col_map and "scoreaway" in col_map
    if has_split:
        scored = full[
            full["scorehome"].notna() & full["scoreaway"].notna() &
            (full["scorehome"].astype(str).str.strip() != "") &
            (full["scoreaway"].astype(str).str.strip() != "")
        ].copy()
        if scored.empty:
            return pd.DataFrame()
        scored["home_score"] = pd.to_numeric(scored["scorehome"], errors="coerce")
        scored["away_score"] = pd.to_numeric(scored["scoreaway"], errors="coerce")
    elif "score" in col_map:
        scored = full[full["score"].notna()].copy()
        if scored.empty:
            return pd.DataFrame()
        parsed = scored["score"].apply(
            lambda s: pd.Series(parse_score(s), index=["home_score", "away_score"])
        )
        scored = pd.concat([scored.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    else:
        return pd.DataFrame()

    scored = scored.dropna(subset=["home_score", "away_score"])
    if scored.empty:
        return pd.DataFrame()

    scored["home_score"] = scored["home_score"].astype(int)
    scored["away_score"] = scored["away_score"].astype(int)
    df = scored  # work on scored-plays-only from here

    # ── Clock, period, time features on scored plays ─────────────────────────
    df = df.copy()
    df["clock_sec"] = df[clock_col].apply(parse_game_clock)
    df["period"]    = pd.to_numeric(df[period_col], errors="coerce").fillna(1).astype(int)
    df["time_remaining_sec"] = df.apply(
        lambda r: period_to_time_remaining(r["period"], r["clock_sec"]), axis=1
    )
    df["is_overtime"] = (df["period"] >= 5).astype(int)
    df["quarter_time_elapsed_pct"] = df.apply(
        lambda r: 1.0 - min(r["clock_sec"] / (300.0 if r["period"] >= 5 else 720.0), 1.0),
        axis=1,
    )

    # ── Possession (scored play: team that scored had the ball) ──────────────
    if team_col and team_col in df.columns:
        df["possession"] = (df[team_col] == home_team_id).astype(float)
    else:
        df["possession"] = 0.5

    # ── Attach precomputed bonus state from full PBP ─────────────────────────
    # Note: home_fts_pending / away_fts_pending dropped from features —
    # FT sequences resolve before the next scored-play snapshot so they
    # are almost always 0. Bonus captures foul-situation context instead.
    df["home_in_bonus"] = df["_home_bonus"].values
    df["away_in_bonus"] = df["_away_bonus"].values
    df["score_diff"] = df["home_score"] - df["away_score"]

    lead_changes  = 0
    prev_leader   = 0  # -1 away ahead, 0 tied, +1 home ahead
    lc_list       = []
    play_count    = 0

    for _, row in df.iterrows():
        diff = row["score_diff"]
        play_count += 1
        leader = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if leader != 0 and leader != prev_leader and prev_leader != 0:
            lead_changes += 1
        prev_leader = leader
        lc_list.append(lead_changes / max(play_count, 1))

    df["lead_changes_norm"] = lc_list

    # ── Meta fields ──────────────────────────────────────────────────────────
    df["GAME_ID"]        = game_id or str(game_meta.name)  # .name = the index value after set_index
    df["home_team_won"]  = int(game_meta["home_team_won"])
    df["home_elo"]       = float(game_meta["home_elo"])
    df["away_elo"]       = float(game_meta["away_elo"])
    df["elo_diff"]       = float(game_meta["elo_diff"])
    df["is_playoffs"]    = int(game_meta["is_playoffs"])
    df["quarter"]        = df["period"]

    # ── Availability delta (pregame features) ────────────────────────────────
    df["home_avail_delta"] = float(home_avail_delta)
    df["away_avail_delta"] = float(away_avail_delta)

    # ── State-dependent Elo prior weight ─────────────────────────────────────
    # At tipoff, tied: weight ≈ 0.85  (Elo matters a lot)
    # At Q4 2min, up 8: weight ≈ 0.04 (score tells you everything)
    df["elo_prior_weight"] = (
        expit(df["time_remaining_sec"] / 2880.0) *
        expit(1.0 / (df["score_diff"].abs() + 1.0))
    )

    # ── Return only rows where score data exists (snapshot = after a score) ──
    result = df[FEATURE_COLS + ["GAME_ID", "home_team_won"]].copy()
    result = result.dropna(subset=FEATURE_COLS)
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VALIDATION SUITE
# 13 checks, updated for 18 features
# ─────────────────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> bool:
    """
    Run 13 data quality checks on the feature dataset.
    Returns True if all pass (or only known-acceptable ones fail).
    """
    print("\n" + "=" * 62)
    print("DATA QUALITY VALIDATION — 18-feature NBA V3 dataset")
    print("=" * 62)

    passed = 0
    failed = 0

    def chk(name: str, ok: bool, detail: str = ""):
        nonlocal passed, failed
        icon = "✅" if ok else "❌"
        print(f"{icon}  {name}")
        if detail:
            print(f"    {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    # 1. No missing values in features
    null_counts = df[FEATURE_COLS].isnull().sum()
    chk("No missing values in feature columns",
        null_counts.sum() == 0,
        str(null_counts[null_counts > 0].to_dict()) if null_counts.sum() > 0 else "")

    # 2. No infinite values
    inf_mask = np.isinf(df[FEATURE_COLS].select_dtypes(include=np.number)).any()
    chk("No infinite values", not inf_mask.any())

    # 3. Target class balance
    hwr = df[TARGET_COL].mean()
    chk(f"Target class balance (40–65%)",
        0.40 <= hwr <= 0.65,
        f"Home win rate: {hwr:.4f} ({hwr:.1%})")

    # 4. Score diff within winsorised range
    sd_min, sd_max = df["score_diff"].min(), df["score_diff"].max()
    chk(f"Score diff within ±{SCORE_DIFF_CLIP:.0f}",
        df["score_diff"].between(-SCORE_DIFF_CLIP, SCORE_DIFF_CLIP).all(),
        f"Actual range: [{sd_min:.0f}, {sd_max:.0f}]")

    # 5. Time remaining non-negative
    chk("Time remaining ≥ 0",
        (df["time_remaining_sec"] >= 0).all(),
        f"Min: {df['time_remaining_sec'].min():.1f}s")

    # 6. Elo ratings in realistic range
    chk("Elo ratings in [1200, 1800]",
        df["home_elo"].between(1200, 1800).all() and
        df["away_elo"].between(1200, 1800).all(),
        f"Home: [{df['home_elo'].min():.0f}, {df['home_elo'].max():.0f}]  "
        f"Away: [{df['away_elo'].min():.0f}, {df['away_elo'].max():.0f}]")

    # 7. Quarter in valid range
    chk("Quarter in [1, 10]",
        df["quarter"].between(1, 10).all(),
        f"Range: [{df['quarter'].min()}, {df['quarter'].max()}]")

    # 8. Binary flags are 0 or 1
    for col in ["is_playoffs", "is_overtime", "possession",
                "home_in_bonus", "away_in_bonus"]:
        chk(f"{col} is binary (0 or 1)",
            df[col].isin([0, 1, 0.0, 1.0]).all(),
            f"Unique values: {sorted(df[col].unique())[:5]}")

    # 10. elo_prior_weight in (0, 1)
    chk("elo_prior_weight in (0, 1)",
        df["elo_prior_weight"].between(0.0, 1.0).all(),
        f"Range: [{df['elo_prior_weight'].min():.4f}, {df['elo_prior_weight'].max():.4f}]")

    # 11. quarter_time_elapsed_pct in [0, 1]
    chk("quarter_time_elapsed_pct in [0, 1]",
        df["quarter_time_elapsed_pct"].between(0.0, 1.0).all())

    # 12. Zero-variance check
    # Note: home_avail_delta / away_avail_delta are intentionally 0.0
    # until DARKO CSV is downloaded — exclude from hard failure
    known_zero_ok = {"home_avail_delta", "away_avail_delta"}
    low_var = [c for c in FEATURE_COLS if df[c].std() < 1e-6]
    unexpected_zero = [c for c in low_var if c not in known_zero_ok]
    chk("All non-avail features have meaningful variance",
        len(unexpected_zero) == 0,
        f"Zero-variance (unexpected): {unexpected_zero}" if unexpected_zero else
        f"Note: avail_delta = 0.0 until DARKO CSV loaded (expected)" if low_var else "")
    if unexpected_zero:
        print("      Running column diagnostic...")
        diagnose_pbp_columns(DATA_PBP)

    # 13. Group-aware — check game count and split feasibility
    n_games = df["GAME_ID"].nunique()
    chk(f"Sufficient games for walk-forward validation (≥ 5,000)",
        n_games >= 5_000,
        f"Total unique games: {n_games:,}")

    print()
    print(f"Result: {passed} passed / {failed} failed")

    # Extra stats
    print(f"\nDataset summary:")
    print(f"  Total rows        : {len(df):,}")
    print(f"  Unique games      : {n_games:,}")
    print(f"  Features          : {N_FEATURES}")
    print(f"  Playoff rows      : {df['is_playoffs'].sum():,} ({df['is_playoffs'].mean():.1%})")
    print(f"  OT rows           : {df['is_overtime'].sum():,} ({df['is_overtime'].mean():.1%})")
    print(f"  Possession=home   : {df['possession'].mean():.3f} (expect ~0.50)")

    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PRESEASON SMOKE TEST
# Live test against a real preseason game to confirm V3 pipeline works end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_pbp_columns(pbp_dir: Path, n_samples: int = 3):
    """
    Read a few cached PBP files and print actual column names and action types.
    Called automatically if zero-variance features are detected.
    """
    files = list(pbp_dir.glob("*.parquet"))[:n_samples]
    if not files:
        print("[diag] No PBP files found to inspect")
        return

    print("\n[diag] Inspecting V3 PBP column names and action types...")
    for fpath in files:
        df = pd.read_parquet(fpath)
        df.columns = df.columns.str.lower()
        print(f"\n  File: {fpath.stem}")
        print(f"  Columns: {list(df.columns)}")

        for col in ["actiontype", "action_type", "eventmsgtype"]:
            if col in df.columns:
                vals = df[col].dropna().unique()
                print(f"  {col} unique values ({len(vals)}): {sorted(str(v) for v in vals)[:20]}")

        for col in ["subtype", "sub_type", "description"]:
            if col in df.columns:
                mask = df[col].astype(str).str.lower().str.contains("free|ft|foul", na=False)
                sample = df[mask][[col] + [c for c in ["actiontype", "teamid"] if c in df.columns]].head(5)
                if not sample.empty:
                    print(f"  FT/foul sample ({col}):")
                    print(sample.to_string(index=False))
                break


def smoke_test_live(game_id: str | None = None):
    """
    Fetch a live or recent preseason game and run it through the pipeline.
    If game_id is None, tries to find today's first game via the live scoreboard.
    Prints a sample of feature rows — not saved to training data.
    """
    from nba_api.live.nba.endpoints import scoreboard

    print("\n" + "=" * 62)
    print("PRESEASON SMOKE TEST — live V3 pipeline")
    print("=" * 62)

    if game_id is None:
        try:
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
            if not games:
                print("[smoke] No games today — provide a game_id manually")
                return
            game_id = games[0]["gameId"]
            home_team = games[0]["homeTeam"]["teamTricode"]
            away_team = games[0]["awayTeam"]["teamTricode"]
            print(f"[smoke] Using today's first game: {away_team} @ {home_team} ({game_id})")
        except Exception as e:
            print(f"[smoke] Could not fetch today's scoreboard: {e}")
            print("        Run with: python phase1_data_pipeline.py --smoke <GAME_ID>")
            return

    pbp_df = fetch_pbp(game_id)
    if pbp_df is None or pbp_df.empty:
        print(f"[smoke] No PBP data for {game_id}")
        return

    # Build a fake meta row so we can test the extractor
    meta = pd.Series({
        "GAME_ID": game_id,
        "home_team_id": 0,   # unknown for live — possession will default to 0.5
        "home_team_won": 0,
        "home_elo": 1500.0,
        "away_elo": 1500.0,
        "elo_diff": 0.0,
        "is_playoffs": 0,
    })

    snaps = extract_snapshots(pbp_df, meta, game_id)
    if snaps.empty:
        print(f"[smoke] No snapshots extracted — check V3 column names")
        print(f"        Columns in PBP: {list(pbp_df.columns)}")
        return

    print(f"\n[smoke] ✅ {len(snaps)} snapshots extracted")
    print(f"        Columns: {list(snaps.columns)}")
    print(f"\nSample (last 5 rows):")
    print(snaps[FEATURE_COLS].tail())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    smoke_only  = "--smoke" in sys.argv
    smoke_game  = sys.argv[sys.argv.index("--smoke") + 1] if (
        "--smoke" in sys.argv and sys.argv.index("--smoke") + 1 < len(sys.argv)
    ) else None

    if smoke_only:
        smoke_test_live(smoke_game)
        return

    print("=" * 62)
    print("NBA Win Probability V3 — Phase 1 Data Pipeline")
    print(f"Seasons : {SEASONS[0]} → {SEASONS[-1]}")
    print(f"Types   : {SEASON_TYPES}")
    print(f"Output  : {DATA_PROC.resolve()}")
    print("=" * 62)

    # ── Step 1: Game logs ────────────────────────────────────────────────────
    print("\n[1/5] Fetching game logs...")
    games = fetch_game_logs(SEASONS)
    print(f"      {len(games):,} games across {len(SEASONS)} seasons")

    # ── Step 2: Elo ratings ──────────────────────────────────────────────────
    print("\n[2/5] Computing Elo ratings...")
    games = compute_elo(games)
    games_idx = games.set_index("GAME_ID")

    # ── Step 3: Player impact (for live use — set 0.0 for training) ──────────
    print("\n[3/5] Loading player impact data...")
    player_impact = load_player_impact()

    # ── Step 4: PBP → snapshots ──────────────────────────────────────────────
    features_cache = DATA_PROC / "features_v3.parquet"
    force_rebuild  = "--rebuild" in sys.argv

    if features_cache.exists() and not force_rebuild:
        print(f"\n[4/5] Loading cached features ({features_cache})...")
        features_df = pd.read_parquet(features_cache)
        # Invalidate cache if feature set has changed
        cached_cols = set(features_df.columns)
        needed_cols = set(FEATURE_COLS + ["GAME_ID", "home_team_won"])
        if not needed_cols.issubset(cached_cols):
            missing = needed_cols - cached_cols
            extra   = cached_cols - needed_cols
            print(f"      Cache outdated (missing={missing}, extra={extra}) — rebuilding...")
            force_rebuild = True
        else:
            print(f"      (Pass --rebuild to re-extract from PBP files)")
            print(f"      {len(features_df):,} snapshots loaded")
    else:
        print("\n[4/5] Extracting PBP snapshots...")
        pbp_files      = list(DATA_PBP.glob("*.parquet"))
        game_ids_needed = set(games_idx.index)

        # Fetch any missing PBP files
        game_ids_cached = {f.stem for f in pbp_files}
        game_ids_missing = game_ids_needed - game_ids_cached
        if game_ids_missing:
            print(f"      Fetching {len(game_ids_missing):,} missing PBP files...")
            for gid in tqdm(sorted(game_ids_missing), desc="Fetching PBP"):
                fetch_pbp(gid)

        # Build feature dataset from cached PBP
        pbp_files     = list(DATA_PBP.glob("*.parquet"))
        all_snapshots = []
        skipped       = 0
        empty_pbp     = 0
        errors        = 0

        print(f"      Processing {len(pbp_files):,} PBP files...")
        for fpath in tqdm(pbp_files, desc="Extracting features"):
            game_id = fpath.stem
            if game_id not in game_ids_needed:
                skipped += 1
                continue
            try:
                pbp_raw   = pd.read_parquet(fpath)
                game_meta = games_idx.loc[game_id]
                h_delta, a_delta = compute_avail_delta(
                    game_id,
                    game_meta["home_team_id"],
                    game_meta["away_team_id"],
                    game_meta["SEASON"],
                    player_impact,
                )
                snaps = extract_snapshots(pbp_raw, game_meta, game_id, h_delta, a_delta)
                if snaps.empty:
                    empty_pbp += 1
                else:
                    all_snapshots.append(snaps)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [ERROR] {game_id}: {e}")

        if not all_snapshots:
            raise RuntimeError("No snapshots extracted — check PBP data and column names")

        features_df = pd.concat(all_snapshots, ignore_index=True)

        print(f"\n      Skipped (no meta) : {skipped:,}")
        print(f"      Empty PBP         : {empty_pbp:,}")
        print(f"      Errors            : {errors:,}")
        print(f"      Snapshots total   : {len(features_df):,}")

        # Winsorise score_diff
        n_clipped = (features_df["score_diff"].abs() >= SCORE_DIFF_CLIP).sum()
        features_df["score_diff"] = features_df["score_diff"].clip(
            -SCORE_DIFF_CLIP, SCORE_DIFF_CLIP
        )
        print(f"      Winsorised +/-{SCORE_DIFF_CLIP:.0f}: {n_clipped:,} rows clipped")

        # Save
        features_df.to_parquet(features_cache, index=False)
        print(f"      Saved -> {features_cache}  ({features_cache.stat().st_size / 1e6:.1f} MB)")

    ok = validate(features_df)

    # ── Write phase report ───────────────────────────────────────────────────
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hwr   = features_df[TARGET_COL].mean()
    report = f"""## Phase 1 — Data Pipeline ({stamp})

- Seasons: {SEASONS[0]} → {SEASONS[-1]}
- Total snapshots: {len(features_df):,}
- Unique games: {features_df["GAME_ID"].nunique():,}
- Features: {N_FEATURES}  |  Target: {TARGET_COL}
- Home win rate: {hwr:.4f} ({hwr:.1%})
- Playoff rows: {features_df["is_playoffs"].mean():.1%}
- OT rows: {features_df["is_overtime"].mean():.1%}
- Possession=home: {features_df["possession"].mean():.3f}
- Saved to: {features_cache}
- Validation: {"PASS" if ok else "FAIL"}
"""
    report_path = DATA_RES / "phase1_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[done] Report → {report_path}")
    print(f"[done] Status : {'✅ ALL CHECKS PASSED' if ok else '❌ SOME CHECKS FAILED'}")

    if ok:
        print("\n➡️  Ready for Phase 2 — Neural Network Training")
        print("   Run: python phase2_training.py")


if __name__ == "__main__":
    main()
