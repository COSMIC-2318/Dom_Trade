"""
main.py — DOM Trade
====================
Single entry point. Runs the full pipeline end to end.

Usage:
    python main.py --input data/orderbook.csv.xlsx

Steps:
    1. Feature engineering
    2. Regime detection (GMM + ARF)
    3. Backtest (train/test split, full evaluation)
"""

import argparse
import sys
import os

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(description="DOM Trade — Full Pipeline")
    parser.add_argument("--input",        default="data/orderbook.csv.xlsx",          help="Raw xlsx file")
    parser.add_argument("--features",     default="reliance_features.parquet",         help="Feature parquet output")
    parser.add_argument("--regime_data",  default="reliance_features_regime.parquet",  help="Regime-labeled parquet")
    parser.add_argument("--regime_model", default="regime/regime_model.pkl",           help="Regime model output")
    args = parser.parse_args()

    os.makedirs("regime", exist_ok=True)

    # ── STEP 1: Feature Engineering ──────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1 — FEATURE ENGINEERING")
    print("="*55)
    from feature_engine import run as run_features
    run_features(args.input, args.features)

    # ── STEP 2: Regime Detection ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 2 — REGIME DETECTION (GMM + ARF)")
    print("="*55)
    import pandas as pd
    from regime.regime_detector_v2 import RegimeDetector

    df         = pd.read_parquet(args.features)
    detector   = RegimeDetector(n_regimes=3)
    df_labeled = detector.fit(df)

    df_labeled.to_parquet(args.regime_data, index=False)
    print(f"[save] labeled data → {args.regime_data}")
    detector.save(args.regime_model)

    # ── STEP 3: Backtest ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 3 — BACKTEST + LATENCY")
    print("="*55)
    from backtest import run_backtest
    run_backtest(args.regime_data, args.regime_model)

    print("\n[DONE] Full pipeline complete.")


if __name__ == "__main__":
    main()