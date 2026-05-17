"""
regime/forecasters.py — DOM Trade
===================================
Layer C: Three parallel online forecasters.

  1. FTRL      — linear, adaptive per-feature learning rate, best in trending
  2. Hoeffding — non-linear tree, handles regime boundary conditions
  3. PA        — passive-aggressive, conservative in low-vol, updates on mistakes only

All three:
  - update themselves every tick via learn_one()
  - predict every tick via predict_one()
  - run in parallel, feed into ensemble.py

Usage:
  from regime.forecasters import Forecasters
  fc = Forecasters()
  preds = fc.predict_all(features)      # {'ftrl': 1, 'hoeff': 0, 'pa': 1}
  fc.learn_all(features, true_label)    # update all three
"""

from river.linear_model import LogisticRegression, PAClassifier
from river.tree import HoeffdingTreeClassifier
from river.optim import FTRLProximal
from river.preprocessing import StandardScaler
from river.compose import Pipeline
import pickle

# ── Input features for all three models ──────────────────────────────────────
FORECAST_FEATURES = [
    "obi_1t",
    "obi_5t",
    "obi_15t",
    "spread_5t",
    "spread_15t",
    "ltp_mid_delta",
    "vwmp",
    "rolling_return_5t",
    "rolling_return_15t",
    "dist_to_support",
    "dist_to_resistance",
    "support_strength",
    "resistance_strength",
]

# NOTE: obi_1t = obi_mean_1t, obi_5t = obi_mean_5t etc from feature_engine
# spread_5t = spread_vol_5t, spread_15t = spread_vol_15t
FEATURE_MAP = {
    "obi_1t":             "obi_mean_1t",
    "obi_5t":             "obi_mean_5t",
    "obi_15t":            "obi_mean_15t",
    "spread_5t":          "spread_vol_5t",
    "spread_15t":         "spread_vol_15t",
    "ltp_mid_delta":      "ltp_mid_delta",
    "vwmp":               "vwmp",
    "rolling_return_5t":  "rolling_return_5t",
    "rolling_return_15t": "rolling_return_15t",
    "dist_to_support":    "dist_to_support",
    "dist_to_resistance": "dist_to_resistance",
    "support_strength":   "support_strength",
    "resistance_strength":"resistance_strength",
}


def extract_features(row: dict) -> dict:
    """Map parquet column names → forecaster feature names."""
    return {k: row[v] for k, v in FEATURE_MAP.items()}


class Forecasters:
    """Three parallel online models. All update and predict per tick."""

    def __init__(self):
        # FTRL — adaptive learning rate per feature
        self.ftrl = Pipeline(
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(optimizer=FTRLProximal(alpha=0.1, beta=1.0, l1=0.0, l2=0.0)))
        )

        # Hoeffding Tree — non-linear, no scaling needed
        self.hoeff = HoeffdingTreeClassifier(
            grace_period=50,
            delta=1e-5,
            leaf_prediction="nba",
        )

        # PA Classifier — updates only on mistakes
        self.pa = PAClassifier(C=0.1, mode=1)

        # rolling accuracy trackers (last 100 ticks)
        self._window   = 100
        self._history  = {"ftrl": [], "hoeff": [], "pa": []}

    # ── PREDICT ──────────────────────────────────────────────────────────────

    def predict_one(self, model_name: str, features: dict) -> int:
        x = extract_features(features)
        if model_name == "ftrl":
            return self.ftrl.predict_one(x)
        elif model_name == "hoeff":
            return self.hoeff.predict_one(x)
        elif model_name == "pa":
            return self.pa.predict_one(x)

    def predict_all(self, features: dict) -> dict:
        """Returns predictions from all three models."""
        x = extract_features(features)
        return {
            "ftrl":  self.ftrl.predict_one(x),
            "hoeff": self.hoeff.predict_one(x),
            "pa":    self.pa.predict_one(x),
        }

    def predict_proba_all(self, features: dict) -> dict:
        """Returns P(UP) from each model. Used for soft ensemble weighting."""
        x = extract_features(features)
        def _prob(model):
            p = model.predict_proba_one(x)
            return p.get(1, 0.5) if p else 0.5

        return {
            "ftrl":  _prob(self.ftrl),
            "hoeff": _prob(self.hoeff),
            "pa":    _prob(self.pa),
        }

    # ── LEARN ────────────────────────────────────────────────────────────────

    def learn_one(self, model_name: str, features: dict, label: int):
        x = extract_features(features)
        if model_name == "ftrl":
            self.ftrl.learn_one(x, label)
        elif model_name == "hoeff":
            self.hoeff.learn_one(x, label)
        elif model_name == "pa":
            self.pa.learn_one(x, label)

    def learn_all(self, features: dict, label: int):
        """Update all three models with ground truth label."""
        x = extract_features(features)
        self.ftrl.learn_one(x, label)
        self.hoeff.learn_one(x, label)
        self.pa.learn_one(x, label)

    # ── ROLLING ACCURACY ─────────────────────────────────────────────────────

    def update_accuracy(self, preds: dict, true_label: int):
        """Call after predict_all() + learn_all() every tick."""
        for name, pred in preds.items():
            self._history[name].append(int(pred == true_label))
            if len(self._history[name]) > self._window:
                self._history[name].pop(0)

    def get_accuracies(self) -> dict:
        """Returns rolling accuracy (last 100 ticks) for each model."""
        out = {}
        for name, hist in self._history.items():
            out[name] = sum(hist) / len(hist) if hist else 1/3
        return out

    # ── LEARNING RATE BOOST (called by ensemble on drift) ────────────────────

    def boost_learning_rates(self):
        """Increase learning rates temporarily on drift detection."""
        self.ftrl = Pipeline(
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(optimizer=FTRLProximal(alpha=0.3, beta=1.0, l1=0.0, l2=0.0)))
        )
        self.pa = PAClassifier(C=0.5, mode=1)
        print("[forecasters] learning rates boosted after drift")

    def reset_learning_rates(self):
        """Reset to normal learning rates after recovery."""
        self.ftrl = Pipeline(
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(optimizer=FTRLProximal(alpha=0.1, beta=1.0, l1=0.0, l2=0.0)))
        )
        self.pa = PAClassifier(C=0.1, mode=1)
        print("[forecasters] learning rates reset to normal")

    # ── SAVE / LOAD ──────────────────────────────────────────────────────────

    def save(self, path: str = "forecasters.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[save] forecasters → {path}")

    @staticmethod
    def load(path: str = "forecasters.pkl") -> "Forecasters":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[load] forecasters ← {path}")
        return obj