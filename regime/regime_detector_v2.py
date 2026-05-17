"""
regime/regime_detector_v2.py — DOM Trade
==========================================
Layer B: Regime Detection

Two-step process (this order is mandatory):
  Step 1 — GMM  : unsupervised, runs ONCE on historical parquet → generates regime labels
  Step 2 — ARF  : supervised, trains on GMM labels → detects regime on new ticks in real time

Regimes:
  0 = Bullish   (positive return, moderate vol)
  1 = Bearish   (negative return, elevated vol)
  2 = Low-Vol   (near-zero return, low vol)

Usage:
  # offline — label your parquet once
  python regime_detector_v2.py --input reliance_features.parquet --output reliance_regimes.parquet

  # online — import and use in live loop
  from regime_detector_v2 import RegimeDetector
  rd = RegimeDetector()
  rd.load("regime_detector.pkl")
  regime = rd.predict_one(features_dict)
  rd.learn_one(features_dict, true_regime)
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from river.ensemble import SRPClassifier

warnings.filterwarnings("ignore")

# ── GMM input features (offline regime labeling) ──────────────────────
GMM_FEATURES = [
    "realized_vol_60t",
    "rolling_return_60t",
]

# ── ARF input features (real-time regime detection) ───────────────────
ARF_FEATURES = [
    "realized_vol_60t",
    "realized_vol_15t",
    "rolling_return_60t",
    "rolling_return_15t",
    "obi_mean_60t",
    "spread_vol_60t",
    "depth_ratio",
]

REGIME_NAMES = {0: "Bullish", 1: "Bearish", 2: "Low-Vol"}
N_REGIMES    = 3


# ─────────────────────────────────────────────
#  STEP 1 — GMM  (offline, run once)
# ─────────────────────────────────────────────

def fit_gmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits GMM on [realized_vol_60t, rolling_return_60t].
    Returns df with new column 'regime' (0/1/2).
    Also prints regime characteristics so you can verify labels make sense.
    """
    X = df[GMM_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        random_state=42,
        max_iter=200,
    )
    gmm.fit(X_scaled)
    raw_labels = gmm.predict(X_scaled)

    # ── interpret and rename regimes by their characteristics ─────────
    # Sort clusters: by mean rolling_return DESC → 0=Bullish,1=Bearish,2=Low-Vol
    df_tmp = df.copy()
    df_tmp["_raw_regime"] = raw_labels

    cluster_stats = {}
    for r in range(N_REGIMES):
        mask = df_tmp["_raw_regime"] == r
        cluster_stats[r] = {
            "mean_return": df_tmp.loc[mask, "rolling_return_60t"].mean(),
            "mean_vol":    df_tmp.loc[mask, "realized_vol_60t"].mean(),
            "count":       mask.sum(),
        }

    # Assign names: highest return → Bullish(0), lowest return → Bearish(1), rest → Low-Vol(2)
    sorted_by_return = sorted(cluster_stats.keys(), key=lambda k: cluster_stats[k]["mean_return"], reverse=True)
    remap = {
        sorted_by_return[0]: 0,   # Bullish
        sorted_by_return[1]: 2,   # Low-Vol  (middle return)
        sorted_by_return[2]: 1,   # Bearish
    }
    df["regime"] = np.array([remap[l] for l in raw_labels])

    # ── print regime summary ──────────────────────────────────────────
    print("\n[GMM] Regime Summary:")
    print(f"  {'Regime':<12} {'Count':>7}  {'Avg Return':>12}  {'Avg Vol':>10}")
    print(f"  {'-'*47}")
    for r in range(N_REGIMES):
        mask = df["regime"] == r
        avg_ret = df.loc[mask, "rolling_return_60t"].mean()
        avg_vol = df.loc[mask, "realized_vol_60t"].mean()
        cnt     = mask.sum()
        print(f"  {REGIME_NAMES[r]:<12} {cnt:>7,}  {avg_ret:>12.6f}  {avg_vol:>10.4f}")

    return df, gmm, scaler


# ─────────────────────────────────────────────
#  STEP 2 — ARF  (online, real-time)
# ─────────────────────────────────────────────

class RegimeDetector:
    """
    Online regime detector backed by AdaptiveRandomForest.
    - Trained on GMM labels from historical data
    - Updates itself on every new tick via learn_one()
    - Replaces weakest tree automatically when drift detected (built into ARF)
    """

    def __init__(self, n_models: int = 10):
        self.arf = SRPClassifier(
            n_models=n_models,
            model=None,
            seed=42,
        )
        self._trained = False

    def train_on_history(self, df: pd.DataFrame):
        """
        Warm-start the ARF on historical (GMM-labeled) data.
        df must have columns ARF_FEATURES + 'regime'.
        """
        print(f"[ARF] Training on {len(df):,} historical rows...")
        for _, row in df.iterrows():
            x = {f: row[f] for f in ARF_FEATURES}
            y = int(row["regime"])
            self.arf.learn_one(x, y)
        self._trained = True
        print("[ARF] Training complete.")

    def predict_one(self, features: dict) -> int:
        """
        Predict regime for a single tick.
        features: dict with keys matching ARF_FEATURES
        Returns: 0=Bullish, 1=Bearish, 2=Low-Vol
        """
        x = {f: features[f] for f in ARF_FEATURES if f in features}
        return self.arf.predict_one(x) or 2   # default Low-Vol if no prediction yet

    def predict_proba_one(self, features: dict) -> dict:
        """
        Returns probability dict: {0: p_bull, 1: p_bear, 2: p_lowvol}
        """
        x = {f: features[f] for f in ARF_FEATURES if f in features}
        return self.arf.predict_proba_one(x) or {0: 0.33, 1: 0.33, 2: 0.34}

    def learn_one(self, features: dict, true_regime: int):
        """Update ARF with ground truth on the current tick."""
        x = {f: features[f] for f in ARF_FEATURES if f in features}
        self.arf.learn_one(x, true_regime)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[ARF] Saved → {path}")

    @staticmethod
    def load(path: str) -> "RegimeDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[ARF] Loaded from {path}")
        return obj


# ─────────────────────────────────────────────
#  VALIDATION HELPER
# ─────────────────────────────────────────────

def validate_regimes(df: pd.DataFrame):
    """Print per-regime price behavior to verify labels are meaningful."""
    print("\n[Validation] Per-regime price behavior:")
    print(f"  {'Regime':<12} {'Count':>7}  {'Avg LTP':>10}  {'Avg Spread':>12}  {'Avg OBI':>10}")
    print(f"  {'-'*57}")
    for r in range(N_REGIMES):
        mask = df["regime"] == r
        if mask.sum() == 0:
            continue
        print(
            f"  {REGIME_NAMES[r]:<12} {mask.sum():>7,}"
            f"  {df.loc[mask,'ltp'].mean():>10.2f}"
            f"  {df.loc[mask,'spread'].mean():>12.4f}"
            f"  {df.loc[mask,'obi'].mean():>10.4f}"
        )


# ─────────────────────────────────────────────
#  OFFLINE PIPELINE  (run once on parquet)
# ─────────────────────────────────────────────

def run_offline(input_path: str, output_path: str, model_path: str):
    print(f"[load] {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[load] {len(df):,} rows")

    # Step 1 — GMM labels
    print("\n[Step 1] Fitting GMM...")
    df, gmm, scaler = fit_gmm(df)

    # Step 2 — ARF trains on GMM labels
    print("\n[Step 2] Training ARF on GMM labels...")
    rd = RegimeDetector(n_models=10)
    rd.train_on_history(df)

    # Validate
    validate_regimes(df)

    # Save labeled parquet
    df.to_parquet(output_path, index=False)
    print(f"\n[DONE] Labeled parquet → {output_path}")
    print(f"[DONE] Regime distribution:\n{df['regime'].value_counts().rename(REGIME_NAMES).to_string()}")

    # Save detector
    rd.save(model_path)

    return df, rd


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="reliance_features.parquet",  help="Feature parquet from feature_engine.py")
    parser.add_argument("--output", default="reliance_regimes.parquet",   help="Output parquet with regime column added")
    parser.add_argument("--model",  default="regime_detector.pkl",        help="Where to save the trained ARF detector")
    args = parser.parse_args()
    run_offline(args.input, args.output, args.model)