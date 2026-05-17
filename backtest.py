"""
backtest.py — DOM Trade
========================
Layer E: Backtesting + Latency Benchmarking

Replays test set tick by tick through the full pipeline:
  feature → regime → FTRL+Hoeff+PA → ensemble → prediction

Reports:
  - Overall accuracy vs naive baseline
  - Per-regime accuracy
  - Ensemble vs each single model
  - Return asymmetry (correct vs wrong predictions)
  - Latency: avg, p50, p99, max (nanoseconds)

Run:
  python backtest.py \
    --parquet reliance_features_regime.parquet \
    --regime_model regime_model.pkl
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, ".")
from regime.regime_detector_v2 import RegimeDetector
from regime.forecasters import Forecasters
from regime.ensemble import Ensemble

REGIME_NAMES = {0: "Bullish", 1: "Bearish", 2: "Volatile"}


def run_backtest(parquet_path: str, regime_model_path: str):

    # ── LOAD ─────────────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"[load] {len(df):,} rows | UP={df['target'].sum():,} | DOWN={(df['target']==0).sum():,}")

    # ── TRAIN / TEST SPLIT ───────────────────────────────────────────────────
    cut   = int(len(df) * 0.8)
    train = df.iloc[:cut].reset_index(drop=True)
    test  = df.iloc[cut:].reset_index(drop=True)
    print(f"[split] train={len(train):,} | test={len(test):,}")

    # ── INIT PIPELINE ────────────────────────────────────────────────────────
    detector = RegimeDetector.load(regime_model_path)
    fc       = Forecasters()
    ens      = Ensemble(fc, detector)

    # ── WARM UP on train set (models learn but we don't evaluate) ────────────
    print(f"\n[warmup] training on {len(train):,} ticks...")
    for _, row in train.iterrows():
        features = row.to_dict()
        label    = int(row["target"])
        ens.fc.learn_all(features, label)
        detector.learn_one(features, int(row["regime"]) if "regime" in row else 0)
    print("[warmup] done")

    # ── REPLAY TEST SET ──────────────────────────────────────────────────────
    print(f"\n[backtest] replaying {len(test):,} ticks...")

    y_true, y_pred         = [], []
    y_ftrl, y_hoeff, y_pa  = [], [], []
    regimes                = []
    mid_prices             = test["mid_price"].values

    for _, row in test.iterrows():
        features   = row.to_dict()
        true_label = int(row["target"])

        result = ens.step(features, true_label)

        y_true.append(true_label)
        y_pred.append(result["final_pred"])
        y_ftrl.append(result["pred_ftrl"])
        y_hoeff.append(result["pred_hoeff"])
        y_pa.append(result["pred_pa"])
        regimes.append(result["regime"])

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_ftrl  = np.array(y_ftrl)
    y_hoeff = np.array(y_hoeff)
    y_pa    = np.array(y_pa)
    regimes = np.array(regimes)

    # ── REPORT ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("BACKTEST RESULTS")
    print("="*55)

    # 1. Naive baseline
    naive_pred = np.ones(len(y_true), dtype=int)   # always predict UP
    naive_acc  = (naive_pred == y_true).mean()
    model_acc  = (y_pred     == y_true).mean()
    print(f"\n[1] Baseline Comparison")
    print(f"  Naive (always UP) : {naive_acc:.4f}")
    print(f"  Ensemble          : {model_acc:.4f}")
    print(f"  Improvement       : {model_acc - naive_acc:+.4f}")

    # 2. Classification report
    print(f"\n[2] Classification Report (Ensemble)")
    print(classification_report(y_true, y_pred, target_names=["DOWN","UP"], digits=4))

    # 3. Per-regime accuracy
    print(f"[3] Per-Regime Accuracy")
    for r, name in REGIME_NAMES.items():
        mask = regimes == r
        if mask.sum() > 0:
            acc_r = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {name:10s} ({mask.sum():,} ticks): {acc_r:.4f}")

    # 4. Ensemble vs single models
    print(f"\n[4] Ensemble vs Single Models")
    print(f"  FTRL alone    : {(y_ftrl  == y_true).mean():.4f}")
    print(f"  Hoeffding     : {(y_hoeff == y_true).mean():.4f}")
    print(f"  PA alone      : {(y_pa    == y_true).mean():.4f}")
    print(f"  Ensemble      : {model_acc:.4f}")

    # 5. Return asymmetry
    print(f"\n[5] Return Asymmetry")
    if len(mid_prices) > 1:
        returns = np.diff(mid_prices) / mid_prices[:-1]
        n       = min(len(returns), len(y_pred) - 1)
        correct_mask = (y_pred[:n] == y_true[:n])
        wrong_mask   = ~correct_mask
        if correct_mask.sum() > 0:
            print(f"  Avg return when correct : {returns[:n][correct_mask].mean():.6f}")
        if wrong_mask.sum() > 0:
            print(f"  Avg return when wrong   : {returns[:n][wrong_mask].mean():.6f}")

    # 6. Latency
    lats = np.array(ens.latencies_ns)
    print(f"\n[6] Latency (nanoseconds, {len(lats):,} predictions)")
    print(f"  mean : {lats.mean():.0f} ns")
    print(f"  p50  : {np.percentile(lats, 50):.0f} ns")
    print(f"  p99  : {np.percentile(lats, 99):.0f} ns")
    print(f"  max  : {lats.max():.0f} ns")

    # 7. Drift
    print(f"\n[7] Drift Events: {ens.drift_count}")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",       default="reliance_features_regime.parquet")
    parser.add_argument("--regime_model",  default="regime_model.pkl")
    args = parser.parse_args()
    run_backtest(args.parquet, args.regime_model)