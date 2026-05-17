"""
test_new_data.py — DOM Trade
==============================
Test trained models on a new day's data (xlsx or parquet).
No retraining — uses existing regime_model.pkl as-is.
Models continue to learn online during replay (as they would live).

Run:
    python test_new_data.py --input data/new_day.xlsx
    python test_new_data.py --input data/new_day.xlsx --no_learn
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, ".")
from feature_engine import run as run_features
from regime.regime_detector_v2 import RegimeDetector
from regime.forecasters import Forecasters
from regime.ensemble import Ensemble

REGIME_NAMES  = {0: "Bullish", 1: "Bearish", 2: "Volatile"}
REGIME_MODEL  = "regime/regime_detector.pkl"

# FIX 1: Check multiple candidate paths for the training features parquet
WARMUP_CANDIDATES = [
    "reliance_features.parquet",
    "reliance_features_regime.parquet",
    "data/reliance_features.parquet",
    "tmp_features.parquet",
]

ROLLING_WARMUP_ROWS = 60   # must be >= largest rolling window (w=60)


def find_warmup_file():
    for path in WARMUP_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def test(input_path: str, online_learn: bool = True):

    # ── Feature engineer the new file ────────────────────────────────────────
    print("\n[step 1] feature engineering new data...")
    tmp_parquet = "tmp_new_features.parquet"

    # FIX 2: Prepend last ROLLING_WARMUP_ROWS from training data BEFORE
    # feature engineering so rolling windows are not cold at tick 1.
    warmup_path = find_warmup_file()
    prepended_rows = 0

    if input_path.endswith(".xlsx") or input_path.endswith(".csv"):

        if warmup_path is not None:
            # Load raw new-day file and training features tail
            new_raw = pd.read_excel(input_path) if input_path.endswith(".xlsx") \
                      else pd.read_csv(input_path)
            df_train_feat = pd.read_parquet(warmup_path)

            # We only need the last ROLLING_WARMUP_ROWS of the RAW columns
            # (pre-feature columns) from training data.
            # Identify raw columns that exist in both dataframes.
            raw_cols = [c for c in new_raw.columns if c in df_train_feat.columns]

            if len(raw_cols) >= 5:
                tail_raw = df_train_feat[raw_cols].tail(ROLLING_WARMUP_ROWS)
                combined_raw = pd.concat([tail_raw, new_raw], ignore_index=True)
                prepended_rows = ROLLING_WARMUP_ROWS
                print(f"[warmup] prepended {prepended_rows} raw rows for rolling window warmup")

                # Write combined to a temp file and run feature engineering on it
                tmp_combined = "tmp_combined_input.xlsx"
                combined_raw.to_excel(tmp_combined, index=False)
                run_features(tmp_combined, tmp_parquet)
                os.remove(tmp_combined)
            else:
                # Raw columns not available in training features — fall back
                print(f"[warmup] raw columns not alignable — running feature engineering on new data only")
                run_features(input_path, tmp_parquet)
        else:
            print(f"[warmup] no training parquet found — running feature engineering on new data only")
            run_features(input_path, tmp_parquet)

        df = pd.read_parquet(tmp_parquet)
    else:
        df = pd.read_parquet(input_path)

    # Drop the prepended warmup rows BEFORE evaluation
    # (they served their purpose for rolling window initialization)
    if prepended_rows > 0:
        df = df.iloc[prepended_rows:].reset_index(drop=True)
        print(f"[warmup] dropped {prepended_rows} warmup rows — evaluation starts on clean new-day data")

    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"[load] {len(df):,} rows | UP={df['target'].sum():,} | DOWN={(df['target']==0).sum():,}")

    # ── Load trained models ───────────────────────────────────────────────────
    print("\n[step 2] loading trained models...")
    detector = RegimeDetector.load(REGIME_MODEL)
    fc       = Forecasters()
    ens      = Ensemble(fc, detector)

    # ── Warm up forecasters on original training data ─────────────────────────
    # FIX 3: Use the same find_warmup_file() helper so path errors don't silently
    # skip warmup, which causes all models to predict the same default class.
    if warmup_path is not None:
        df_warmup = pd.read_parquet(warmup_path)
        print(f"[warmup] training forecasters on {len(df_warmup):,} original ticks...")
        for _, row in df_warmup.iterrows():
            feat  = row.to_dict()
            label = int(row["target"]) if pd.notna(row.get("target")) else 0
            fc.learn_all(feat, label)
        print("[warmup] forecaster warmup done")
    else:
        print("[warmup] WARNING: no training parquet found anywhere — forecasters start cold")
        print(f"         Searched: {WARMUP_CANDIDATES}")
        print("         Run the full pipeline first: python main.py --input data/orderbook.csv.xlsx")

    # ── Replay new data ───────────────────────────────────────────────────────
    print(f"\n[step 3] replaying {len(df):,} new ticks...")
    learn_str = "with online learning" if online_learn else "inference only"
    print(f"         mode: {learn_str}")

    y_true, y_pred         = [], []
    y_ftrl, y_hoeff, y_pa  = [], [], []
    regimes                = []

    for _, row in df.iterrows():
        features   = row.to_dict()
        true_label = int(row["target"])

        if online_learn:
            result = ens.step(features, true_label)
        else:
            regime     = ens.detector.predict_one(features)
            regime     = regime if regime is not None else 2
            preds_raw  = ens.fc.predict_all(features)
            pred_f     = int(preds_raw["ftrl"])  if preds_raw["ftrl"]  is not None else 0
            pred_h     = int(preds_raw["hoeff"]) if preds_raw["hoeff"] is not None else 0
            pred_p     = int(preds_raw["pa"])    if preds_raw["pa"]    is not None else 0
            weights    = ens._compute_weights(regime)
            score      = weights["ftrl"]*pred_f + weights["hoeff"]*pred_h + weights["pa"]*pred_p
            final      = 1 if score >= 0.5 else 0
            result     = {"final_pred": final, "pred_ftrl": pred_f,
                          "pred_hoeff": pred_h, "pred_pa": pred_p,
                          "regime": regime, "latency_ns": 0}

        y_true.append(true_label)
        y_pred.append(result["final_pred"])
        y_ftrl.append(result["pred_ftrl"])
        y_hoeff.append(result["pred_hoeff"])
        y_pa.append(result["pred_pa"])
        regimes.append(result["regime"])

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    regimes = np.array(regimes)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("NEW DATA TEST RESULTS")
    print("="*55)

    naive_acc = (np.ones_like(y_true) == y_true).mean()
    model_acc = (y_pred == y_true).mean()
    print(f"\n[1] Baseline Comparison")
    print(f"  Naive (always UP) : {naive_acc:.4f}")
    print(f"  Ensemble          : {model_acc:.4f}")
    print(f"  Improvement       : {model_acc - naive_acc:+.4f}")

    print(f"\n[2] Classification Report")
    print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"], digits=4))

    # FIX 4: Sanity check — warn explicitly if all models agree (collapsed predictions)
    unique_preds = len(np.unique(y_pred))
    if unique_preds == 1:
        print("  !! WARNING: ensemble predicted only one class for all ticks.")
        print("     This usually means forecasters were not warmed up correctly.")
        print("     Check that reliance_features.parquet exists before running.\n")

    print(f"[3] Per-Regime Accuracy")
    for r, name in REGIME_NAMES.items():
        mask = regimes == r
        if mask.sum() > 0:
            acc_r = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {name:10s} ({mask.sum():,} ticks): {acc_r:.4f}")

    print(f"\n[4] Ensemble vs Single Models")
    print(f"  FTRL alone : {(np.array(y_ftrl)  == y_true).mean():.4f}")
    print(f"  Hoeffding  : {(np.array(y_hoeff) == y_true).mean():.4f}")
    print(f"  PA alone   : {(np.array(y_pa)    == y_true).mean():.4f}")
    print(f"  Ensemble   : {model_acc:.4f}")

    if online_learn:
        lats = np.array(ens.latencies_ns)
        print(f"\n[5] Latency (ns)")
        print(f"  mean={lats.mean():.0f}  p50={np.percentile(lats,50):.0f}  p99={np.percentile(lats,99):.0f}")
        print(f"\n[6] Drift Events: {ens.drift_count}")

    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,       help="New data: .xlsx or .parquet")
    parser.add_argument("--no_learn",  action="store_true", help="Inference only, no online updates")
    args = parser.parse_args()
    test(args.input, online_learn=not args.no_learn)