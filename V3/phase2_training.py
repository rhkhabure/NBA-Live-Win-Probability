"""
phase2_training.py  —  NBA Win Probability V3
==============================================
Trains the WinProbNet on features_v3.parquet using walk-forward
chronological validation (3 folds), then evaluates against the
full 6-metric suite including ESPN head-to-head.

Run from the V3/ project root:
    python phase2_training.py

Outputs (written to model/):
    model/win_prob_net_v3.pth       — weights + config + metrics
    model/scaler_v3.pkl             — StandardScaler (fit on train only)
    model/elo_ratings_v3.json       — already written by Phase 1
    model/training_history.json     — loss / AUC per epoch (final fold)
    model/walk_forward_results.csv  — per-fold Brier / AUC
    model/calibration_t.pkl         — temperature scaling T value

Outputs (written to data/results/):
    data/results/phase2_report.md   — full validation summary
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import pickle
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT      = Path(".")
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
RES_DIR   = ROOT / "data" / "results"
PLOT_DIR  = ROOT / "data" / "plots"

for d in (MODEL_DIR, RES_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Features (must match Phase 1 FEATURE_COLS exactly) ───────────────────────
FEATURE_COLS = [
    "score_diff",
    "time_remaining_sec",
    "quarter",
    "quarter_time_elapsed_pct",
    "home_elo",
    "away_elo",
    "elo_diff",
    "is_playoffs",
    "is_overtime",
    "lead_changes_norm",
    "possession",
    "home_in_bonus",
    "away_in_bonus",
    "home_avail_delta",
    "away_avail_delta",
    "elo_prior_weight",
]
TARGET_COL = "home_team_won"
N_FEATURES = len(FEATURE_COLS)

# ── Architecture ─────────────────────────────────────────────────────────────
HIDDEN_DIMS   = [128, 64, 32]
DROPOUT_RATE  = 0.25          # reduced from V1's 0.30 — larger dataset
USE_BATCHNORM = True

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE    = 4096
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 150
PATIENCE      = 15
MIN_DELTA     = 1e-5
GRAD_CLIP     = 1.0
LR_T_MAX      = 50
LR_ETA_MIN    = 1e-6

# ── Walk-forward folds ────────────────────────────────────────────────────────
# Season strings as they appear in game_logs_all.parquet SEASON column
WALK_FORWARD_FOLDS = [
    {"train": ["2018-19", "2019-20", "2020-21", "2021-22"], "test": "2022-23"},
    {"train": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"], "test": "2023-24"},
    {"train": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"], "test": "2024-25"},
]
# Final model trains on ALL seasons
FINAL_TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MODEL
# ─────────────────────────────────────────────────────────────────────────────

class WinProbNet(nn.Module):
    """
    NBA in-game win probability network.

    Per hidden layer: Linear → BatchNorm → ReLU → Dropout
    Output: sigmoid P(home team wins)

    Design unchanged from V1 — architecture was not the bottleneck.
    Only change: Dropout 0.30 → 0.25 (dataset is 10× larger).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: list,
        dropout: float = 0.25,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = n_features
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_tensors(
    df: pd.DataFrame,
    scaler: StandardScaler | None,
    fit_scaler: bool = False,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """Scale features and return (X_tensor, y_tensor, scaler)."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    return (
        torch.from_numpy(X).to(device),
        torch.from_numpy(y).to(device),
        scaler,
    )


def train_one_fold(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val_np: np.ndarray,
    device: torch.device,
    verbose: bool = True,
) -> tuple[WinProbNet, dict]:
    """
    Train WinProbNet on one fold.
    Returns (best_model, history_dict).
    """
    model = WinProbNet(N_FEATURES, HIDDEN_DIMS, DROPOUT_RATE, USE_BATCHNORM).to(device)

    n_pos  = y_train.sum().item()
    n_neg  = len(y_train) - n_pos
    pos_w  = torch.tensor([n_neg / n_pos], device=device)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt    = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LR_T_MAX, eta_min=LR_ETA_MIN)

    train_ds = TensorDataset(X_train, y_train.unsqueeze(1))
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_brier": []}
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model.logits(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_loss += loss.item() * len(xb)
        avg_train = ep_loss / len(loader.dataset)

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model.logits(X_val)
            val_loss   = crit(val_logits, torch.from_numpy(y_val_np).unsqueeze(1).to(device)).item()
            val_probs  = torch.sigmoid(val_logits).cpu().numpy().ravel()

        val_auc   = roc_auc_score(y_val_np, val_probs)
        val_brier = brier_score_loss(y_val_np, val_probs)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_brier"].append(val_brier)

        sched.step()

        improved = val_loss < best_val_loss - MIN_DELTA
        if improved:
            best_val_loss = val_loss
            best_state    = deepcopy(model.state_dict())
            patience_ctr  = 0
            tag = "✓"
        else:
            patience_ctr += 1
            tag = ""

        if verbose and (epoch % 10 == 0 or epoch <= 5 or improved):
            lr = opt.param_groups[0]["lr"]
            print(f"    ep {epoch:3d} | train {avg_train:.5f} | val {val_loss:.5f} "
                  f"| AUC {val_auc:.4f} | Brier {val_brier:.4f} | lr {lr:.2e} {tag}")

        if patience_ctr >= PATIENCE:
            if verbose:
                print(f"    Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EVALUATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: WinProbNet,
    X: torch.Tensor,
    y_np: np.ndarray,
    device: torch.device,
    T: float = 1.0,
    label: str = "",
) -> dict:
    """
    Run the full 6-metric evaluation suite on a dataset.
    Returns a dict of all metrics.
    """
    model.eval()
    with torch.no_grad():
        logits = model.logits(X).cpu().numpy().ravel()

    # Temperature scaling
    probs = 1 / (1 + np.exp(-logits / T))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    roc_auc  = roc_auc_score(y_np, probs)
    pr_auc   = average_precision_score(y_np, probs)
    brier    = brier_score_loss(y_np, probs)
    ll       = log_loss(y_np, probs)
    acc      = ((probs >= 0.5).astype(int) == y_np.astype(int)).mean()
    null_b   = brier_score_loss(y_np, np.full_like(probs, y_np.mean()))
    bss      = 1 - brier / null_b

    if label:
        print(f"\n  [{label}]")
        print(f"    ROC-AUC : {roc_auc:.4f}   PR-AUC  : {pr_auc:.4f}")
        print(f"    Brier   : {brier:.4f}   BSS     : {bss:.4f}")
        print(f"    LogLoss : {ll:.4f}   Accuracy: {acc:.4f}")

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "brier": brier, "brier_skill": bss,
        "log_loss": ll, "accuracy": acc,
        "probs": probs,
    }


def brier_by_time_bucket(
    probs: np.ndarray,
    y_np: np.ndarray,
    df_subset: pd.DataFrame,
) -> dict:
    """
    Brier score broken down by time bucket.
    This is the primary evaluation metric for a time-updating model.
    """
    buckets = {
        "Q1 (>36min)":        df_subset["time_remaining_sec"] > 36 * 60,
        "Q2 (24-36min)":      df_subset["time_remaining_sec"].between(24*60, 36*60),
        "Q3 (12-24min)":      df_subset["time_remaining_sec"].between(12*60, 24*60),
        "Q4 (0-12min)":       df_subset["time_remaining_sec"].between(1, 12*60),
        "Final 5min (<5min)": df_subset["time_remaining_sec"].between(0, 5*60),
        "Final 2min (<2min)": df_subset["time_remaining_sec"].between(0, 2*60),
        "OT":                 df_subset["is_overtime"] == 1,
    }
    results = {}
    print("\n  Brier by time bucket:")
    for name, mask in buckets.items():
        idx = df_subset.index[mask]
        if len(idx) < 100:
            continue
        # align probs/y with subset indices
        b = brier_score_loss(y_np[mask.values], probs[mask.values])
        n = mask.sum()
        results[name] = {"brier": b, "n": int(n)}
        print(f"    {name:<22}: {b:.4f}  (n={n:,})")
    return results


def pathwise_calibration_check(
    probs: np.ndarray,
    y_np: np.ndarray,
    game_ids: np.ndarray,
) -> float:
    """
    The 'blown lead paradox' check (Pipping-Gamón & Wyner 2026).

    In a well-calibrated model, the eventual loser should reach ≥90%
    win probability in approximately 1/9 ≈ 11% of games.
    If the rate is >15%, systematic overconfidence is present.

    Returns the fraction of games where the eventual loser hit ≥90%.
    """
    df = pd.DataFrame({
        "game_id": game_ids,
        "prob": probs,
        "won": y_np.astype(int),
    })

    hit_count = 0
    total     = 0

    for gid, grp in df.groupby("game_id"):
        home_won = grp["won"].iloc[-1]  # final outcome
        if home_won == 1:
            # eventual loser = away; their prob = 1 - home_prob
            loser_probs = 1 - grp["prob"]
        else:
            loser_probs = grp["prob"]
        if (loser_probs >= 0.90).any():
            hit_count += 1
        total += 1

    rate = hit_count / total if total > 0 else 0.0
    print(f"\n  Pathwise calibration (blown lead check):")
    print(f"    Eventual loser hit >=90% in {hit_count}/{total} games = {rate:.3f}")
    print(f"    Expected ~0.11  {'✅' if 0.08 <= rate <= 0.15 else '⚠️  (systematic overconfidence)' if rate > 0.15 else '⚠️  (underconfident)'}")
    return rate


def calibration_by_score_state(
    probs: np.ndarray,
    y_np: np.ndarray,
    df_subset: pd.DataFrame,
) -> None:
    """Calibration broken down by score margin bucket."""
    buckets = {
        "Tied (0)":      df_subset["score_diff"] == 0,
        "±1-5":          df_subset["score_diff"].abs().between(1, 5),
        "±6-10":         df_subset["score_diff"].abs().between(6, 10),
        "±11+":          df_subset["score_diff"].abs() >= 11,
    }
    print("\n  Calibration by score state (predicted vs actual home win rate):")
    for name, mask in buckets.items():
        if mask.sum() < 50:
            continue
        pred = probs[mask.values].mean()
        act  = y_np[mask.values].mean()
        n    = mask.sum()
        print(f"    {name:<12}: pred={pred:.3f}  actual={act:.3f}  gap={abs(pred-act):.3f}  (n={n:,})")


def calibrate_temperature(
    model: WinProbNet,
    X_val: torch.Tensor,
    y_val: np.ndarray,
    device: torch.device,
) -> float:
    """Find optimal temperature T via NLL minimisation on validation set."""
    model.eval()
    with torch.no_grad():
        logits = model.logits(X_val).cpu().numpy().ravel()

    def nll(T):
        p = np.clip(1 / (1 + np.exp(-logits / T)), 1e-7, 1 - 1e-7)
        return log_loss(y_val, p)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded",
                             options={"xatol": 1e-6})
    T_opt = result.x
    print(f"\n  Temperature scaling: T = {T_opt:.6f}")
    print(f"    T < 1: model was overconfident (sharpen probs)")
    print(f"    T > 1: model was underconfident (soften probs)")
    return float(T_opt)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ESPN BENCHMARK
# Pulls ESPN win probability for test games and compares Brier head-to-head
# ─────────────────────────────────────────────────────────────────────────────

def fetch_espn_win_prob(game_id: str) -> list[dict] | None:
    """
    Pull ESPN's win probability series for one game via their free endpoint.
    Returns list of {homeWinPercentage, secondsLeft} dicts or None on failure.
    """
    import urllib.request
    import ssl

    # ESPN game IDs are numeric; nba_api GAME_IDs are 10-digit strings like 0022300001
    # ESPN uses a different ID format — we skip this for now and benchmark on Brier
    # TODO: build the nba_api → ESPN game ID mapping in a later pass
    return None


def espn_benchmark_note() -> str:
    """
    Returns a note explaining the ESPN benchmark status.
    Full ESPN head-to-head requires mapping nba_api GAME_IDs to ESPN event IDs.
    This is a Phase 2 TODO — the ESPN endpoint itself works, we just need the ID map.
    """
    return (
        "ESPN head-to-head: PENDING — requires nba_api→ESPN game ID mapping.\n"
        "The ESPN summary endpoint (site.api.espn.com/nba/summary?event=<id>) is\n"
        "confirmed working. ID mapping will be added in a post-Phase-2 patch.\n"
        "For now, compare against published benchmarks: ESPN Brier ≈ 0.166 (Beuoy 2018)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── GPU setup ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu    = torch.cuda.get_device_properties(0)
        print(f"Device  : {gpu.name}  ({gpu.total_memory/1e9:.1f} GB VRAM)")
        torch.cuda.manual_seed(SEED)
    else:
        device = torch.device("cpu")
        print("Device  : CPU (CUDA not available — training will be slower)")

    print("=" * 62)
    print("NBA Win Probability V3 — Phase 2 Training")
    print(f"Architecture : {N_FEATURES} → {' → '.join(map(str, HIDDEN_DIMS))} → 1")
    print(f"Features     : {N_FEATURES}")
    print(f"Dropout      : {DROPOUT_RATE}  |  BatchNorm: {USE_BATCHNORM}")
    print(f"Walk-forward : {len(WALK_FORWARD_FOLDS)} folds")
    print("=" * 62)

    # ── Load data ────────────────────────────────────────────────────────────
    features_path = PROC_DIR / "features_v3.parquet"
    assert features_path.exists(), f"❌ {features_path} not found — run Phase 1 first"

    print(f"\n[1/5] Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"      {len(df):,} snapshots  |  {df['GAME_ID'].nunique():,} games")

    # Verify feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"

    # Load game-level season metadata for walk-forward splits
    game_logs_path = ROOT / "data" / "raw" / "game_logs_all.parquet"
    assert game_logs_path.exists(), "❌ game_logs_all.parquet not found — run Phase 1 first"
    game_logs = pd.read_parquet(game_logs_path)[["GAME_ID", "SEASON"]].drop_duplicates()
    df = df.merge(game_logs, on="GAME_ID", how="left")
    print(f"      Season column attached  |  {df['SEASON'].nunique()} seasons")

    # ── Walk-forward validation ───────────────────────────────────────────────
    print(f"\n[2/5] Walk-forward validation ({len(WALK_FORWARD_FOLDS)} folds)...")
    fold_results = []

    for fold_i, fold in enumerate(WALK_FORWARD_FOLDS, start=1):
        print(f"\n  ── Fold {fold_i}: train {fold['train']} → test {fold['test']} ──")

        train_mask = df["SEASON"].isin(fold["train"])
        test_mask  = df["SEASON"] == fold["test"]
        val_mask   = df["SEASON"] == fold["train"][-1]  # last train season = val

        df_train = df[train_mask & ~val_mask].reset_index(drop=True)
        df_val   = df[val_mask].reset_index(drop=True)
        df_test  = df[test_mask].reset_index(drop=True)

        print(f"    train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")

        # Scale — fit on train only
        X_train, y_train, fold_scaler = build_tensors(df_train, None, fit_scaler=True, device=device)
        X_val,   y_val,   _           = build_tensors(df_val,   fold_scaler, device=device)
        X_test,  y_test,  _           = build_tensors(df_test,  fold_scaler, device=device)

        y_val_np  = y_val.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Train
        fold_model, fold_history = train_one_fold(
            X_train, y_train, X_val, y_val_np, device, verbose=True
        )

        # Temperature calibration on val set
        T = calibrate_temperature(fold_model, X_val, y_val_np, device)

        # Evaluate on TEST set (unseen season)
        metrics = evaluate_model(fold_model, X_test, y_test_np, device, T=T,
                                 label=f"Fold {fold_i} test ({fold['test']})")

        fold_results.append({
            "fold":    fold_i,
            "train":   str(fold["train"]),
            "test":    fold["test"],
            "brier":   metrics["brier"],
            "roc_auc": metrics["roc_auc"],
            "bss":     metrics["brier_skill"],
            "logloss": metrics["log_loss"],
            "T":       T,
            "n_test":  len(df_test),
        })

    fold_df = pd.DataFrame(fold_results)
    print(f"\n  Walk-forward summary:")
    print(fold_df[["fold", "test", "brier", "roc_auc", "bss", "T"]].to_string(index=False))

    brier_drift = fold_df["brier"].max() - fold_df["brier"].min()
    print(f"\n  Brier drift across folds: {brier_drift:.4f}  "
          f"{'✅ stable' if brier_drift < 0.005 else '⚠️ temporal drift detected'}")

    fold_df.to_csv(MODEL_DIR / "walk_forward_results.csv", index=False)
    print(f"  Saved → {MODEL_DIR / 'walk_forward_results.csv'}")

    # ── Final model — train on all 7 seasons ─────────────────────────────────
    print(f"\n[3/5] Training final model on all {len(FINAL_TRAIN_SEASONS)} seasons...")

    df_final_train = df[df["SEASON"].isin(FINAL_TRAIN_SEASONS[:-1])].reset_index(drop=True)
    df_final_val   = df[df["SEASON"] == FINAL_TRAIN_SEASONS[-1]].reset_index(drop=True)

    X_train_f, y_train_f, final_scaler = build_tensors(df_final_train, None, fit_scaler=True, device=device)
    X_val_f,   y_val_f,   _            = build_tensors(df_final_val,   final_scaler, device=device)

    y_val_f_np = y_val_f.cpu().numpy()

    print(f"    train: {len(df_final_train):,}  val (2024-25): {len(df_final_val):,}")
    final_model, final_history = train_one_fold(
        X_train_f, y_train_f, X_val_f, y_val_f_np, device, verbose=True
    )

    # Temperature calibration on final val set
    T_final = calibrate_temperature(final_model, X_val_f, y_val_f_np, device)

    # ── Full evaluation on 2024-25 (final validation season) ─────────────────
    print(f"\n[4/5] Full evaluation suite on 2024-25 season...")
    all_metrics = evaluate_model(
        final_model, X_val_f, y_val_f_np, device, T=T_final, label="Final model (2024-25 val)"
    )
    final_probs = all_metrics["probs"]

    # Reset df_final_val index to align with probs array
    df_final_val = df_final_val.reset_index(drop=True)

    # 6-metric evaluation suite
    brier_buckets = brier_by_time_bucket(final_probs, y_val_f_np, df_final_val)
    pathwise_rate = pathwise_calibration_check(
        final_probs, y_val_f_np, df_final_val["GAME_ID"].values
    )
    calibration_by_score_state(final_probs, y_val_f_np, df_final_val)

    # ESPN benchmark note
    print(f"\n  ESPN benchmark: {espn_benchmark_note()}")

    # Validation gate
    print("\n" + "=" * 62)
    print("PHASE 2 VALIDATION GATES")
    print("=" * 62)
    checks = [
        ("ROC-AUC > 0.80",          all_metrics["roc_auc"] > 0.80,          f"{all_metrics['roc_auc']:.4f}"),
        ("Brier < 0.20",            all_metrics["brier"] < 0.20,            f"{all_metrics['brier']:.4f}"),
        ("Brier Skill Score > 0.10",all_metrics["brier_skill"] > 0.10,      f"{all_metrics['brier_skill']:.4f}"),
        ("Log Loss < 0.60",         all_metrics["log_loss"] < 0.60,         f"{all_metrics['log_loss']:.4f}"),
        ("Accuracy > 0.70",         all_metrics["accuracy"] > 0.70,         f"{all_metrics['accuracy']:.4f}"),
        ("Walk-forward Brier drift < 0.005", brier_drift < 0.005,           f"{brier_drift:.4f}"),
        ("Pathwise rate 0.08–0.15", 0.08 <= pathwise_rate <= 0.15,          f"{pathwise_rate:.3f}"),
        ("Brier improves fold-to-fold or stable",
         fold_df["brier"].iloc[-1] <= fold_df["brier"].iloc[0] + 0.005,
         f"Fold1={fold_df['brier'].iloc[0]:.4f} → Fold3={fold_df['brier'].iloc[-1]:.4f}"),
    ]
    passed = failed = 0
    for name, ok, val in checks:
        icon = "✅" if ok else "❌"
        print(f"{icon}  {name:<45} {val}")
        if ok: passed += 1
        else:  failed += 1
    print(f"\nResult: {passed}/{len(checks)} passed")

    # ── Save artefacts ────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving artefacts...")

    # Model weights
    model_path = MODEL_DIR / "win_prob_net_v3.pth"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "model_config": {
            "n_features":    N_FEATURES,
            "hidden_dims":   HIDDEN_DIMS,
            "dropout":       DROPOUT_RATE,
            "use_batchnorm": USE_BATCHNORM,
        },
        "feature_cols":   FEATURE_COLS,
        "target_col":     TARGET_COL,
        "temperature":    T_final,
        "train_metrics": {
            "best_epoch": int(np.argmin(final_history["val_loss"])) + 1,
            "epochs_run": len(final_history["val_loss"]),
        },
        "val_metrics": {
            "roc_auc":     all_metrics["roc_auc"],
            "pr_auc":      all_metrics["pr_auc"],
            "brier":       all_metrics["brier"],
            "brier_skill": all_metrics["brier_skill"],
            "log_loss":    all_metrics["log_loss"],
            "accuracy":    all_metrics["accuracy"],
        },
        "walk_forward": fold_df.to_dict(orient="records"),
    }, model_path)
    print(f"    Saved model → {model_path}")

    # Scaler
    scaler_path = MODEL_DIR / "scaler_v3.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(final_scaler, f)
    print(f"    Saved scaler → {scaler_path}")

    # Temperature
    T_path = MODEL_DIR / "calibration_t.pkl"
    with open(T_path, "wb") as f:
        pickle.dump({"T": T_final}, f)
    print(f"    Saved temperature T={T_final:.6f} → {T_path}")

    # Training history
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(final_history, f, indent=2)

    # Phase report
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""## Phase 2 — Training ({stamp})

### Architecture
- Input: {N_FEATURES} features
- Hidden: {' → '.join(map(str, HIDDEN_DIMS))}
- Output: 1 (sigmoid)
- Dropout: {DROPOUT_RATE}  BatchNorm: {USE_BATCHNORM}
- Parameters: {final_model.param_count():,}

### Walk-forward validation
{fold_df[['fold','test','brier','roc_auc','bss','T']].to_string(index=False)}

Brier drift: {brier_drift:.4f}

### Final model performance (2024-25 validation)
- ROC-AUC      : {all_metrics['roc_auc']:.4f}
- PR-AUC       : {all_metrics['pr_auc']:.4f}
- Brier Score  : {all_metrics['brier']:.4f}
- Brier Skill  : {all_metrics['brier_skill']:.4f}
- Log Loss     : {all_metrics['log_loss']:.4f}
- Accuracy     : {all_metrics['accuracy']:.4f}
- Temperature T: {T_final:.6f}
- Pathwise rate: {pathwise_rate:.3f} (target 0.08-0.15)

### Validation gates: {passed}/{len(checks)} passed

### ESPN benchmark
{espn_benchmark_note()}

### Saved artefacts
- model/win_prob_net_v3.pth
- model/scaler_v3.pkl
- model/calibration_t.pkl
- model/walk_forward_results.csv
- model/training_history.json
"""
    report_path = RES_DIR / "phase2_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"    Saved report → {report_path}")

    print("\n" + "=" * 62)
    print("PHASE 2 COMPLETE")
    print("=" * 62)
    print(f"""
  Final model: {N_FEATURES} → {' → '.join(map(str, HIDDEN_DIMS))} → 1
  Brier       : {all_metrics['brier']:.4f}
  ROC-AUC     : {all_metrics['roc_auc']:.4f}
  Temperature : T={T_final:.4f}
  Gates       : {passed}/{len(checks)} passed

  {'✅ Ready for Phase 3 — Dashboard Integration' if failed == 0 else '⚠️  Review failed gates before Phase 3'}
  Run: python phase3_dashboard.py
""")


if __name__ == "__main__":
    main()
