"""
regime/ensemble.py — DOM Trade
================================
Layer D: Dynamic Weighted Ensemble + ADWIN Drift Detection

Flow per tick:
  1. Regime detector → regime (0/1/2)
  2. Regime sets initial weights (FTRL/Hoeff/PA)
  3. Rolling accuracy adjusts weights every tick
  4. Weighted vote → final prediction (0 or 1)
  5. ADWIN monitors error rate → boosts LR on drift

Usage:
  from regime.ensemble import Ensemble
  from regime.forecasters import Forecasters
  from regime.regime_detector_v2 import RegimeDetector

  detector = RegimeDetector.load("regime_model.pkl")
  fc       = Forecasters()
  ens      = Ensemble(fc, detector)

  # per tick:
  result = ens.step(features, true_label)
  print(result['final_pred'], result['regime'], result['latency_ns'])
"""

import time
import pickle
from river.drift import ADWIN
from regime.forecasters import Forecasters
from regime.regime_detector_v2 import RegimeDetector

# ── Regime-based initial weights ─────────────────────────────────────────────
# These are starting points — rolling accuracy overrides them every tick
INITIAL_WEIGHTS = {
    0: {"ftrl": 0.6, "hoeff": 0.3, "pa": 0.1},   # Bullish  — FTRL leads
    1: {"ftrl": 0.3, "hoeff": 0.5, "pa": 0.2},   # Bearish  — Hoeff leads
    2: {"ftrl": 0.2, "hoeff": 0.3, "pa": 0.5},   # Volatile — PA leads
}

REGIME_NAMES = {0: "Bullish", 1: "Bearish", 2: "Volatile"}


class Ensemble:

    def __init__(self, forecasters: Forecasters, detector: RegimeDetector):
        self.fc          = forecasters
        self.detector    = detector
        self.drift       = ADWIN(delta=0.002)
        self.drift_count = 0

        # stats
        self.total        = 0
        self.correct      = 0
        self.per_regime   = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # [correct, total]
        self.latencies_ns = []

    # ── SINGLE TICK ──────────────────────────────────────────────────────────

    def step(self, features: dict, true_label: int) -> dict:
        """
        Full pipeline for one tick.
        features : dict with all feature_engine columns
        true_label: 0 or 1 (known after the tick for online learning)

        Returns dict with:
          final_pred, regime, regime_name,
          pred_ftrl, pred_hoeff, pred_pa,
          weights, latency_ns
        """
        t_start = time.perf_counter_ns()

        # 1. Regime
        regime      = self.detector.predict_one(features)
        regime      = regime if regime is not None else 2   # default volatile

        # 2. Predictions from all three models
        preds_raw   = self.fc.predict_all(features)

        # cast to int, handle None (Hoeffding warmup)
        pred_ftrl   = int(preds_raw["ftrl"])  if preds_raw["ftrl"]  is not None else 0
        pred_hoeff  = int(preds_raw["hoeff"]) if preds_raw["hoeff"] is not None else 0
        pred_pa     = int(preds_raw["pa"])    if preds_raw["pa"]    is not None else 0

        # 3. Weights — start from regime, adjust by rolling accuracy
        weights = self._compute_weights(regime)

        # 4. Weighted vote
        score = (
            weights["ftrl"]  * pred_ftrl  +
            weights["hoeff"] * pred_hoeff +
            weights["pa"]    * pred_pa
        )
        final_pred = 1 if score >= 0.5 else 0

        t_end      = time.perf_counter_ns()
        latency_ns = t_end - t_start

        # 5. Learn — update all models with true label
        self.fc.learn_all(features, true_label)
        self.fc.update_accuracy(
            {"ftrl": pred_ftrl, "hoeff": pred_hoeff, "pa": pred_pa},
            true_label
        )

        # 6. ADWIN drift detection
        error = int(final_pred != true_label)
        self.drift.update(error)
        if self.drift.drift_detected:
            self.drift_count += 1
            self.fc.boost_learning_rates()

        # 7. Stats
        self.total       += 1
        self.correct     += int(final_pred == true_label)
        self.per_regime[regime][0] += int(final_pred == true_label)
        self.per_regime[regime][1] += 1
        self.latencies_ns.append(latency_ns)

        return {
            "final_pred":  final_pred,
            "regime":      regime,
            "regime_name": REGIME_NAMES.get(regime, "Unknown"),
            "pred_ftrl":   pred_ftrl,
            "pred_hoeff":  pred_hoeff,
            "pred_pa":     pred_pa,
            "weights":     weights,
            "latency_ns":  latency_ns,
            "drift":       self.drift.drift_detected,
        }

    # ── WEIGHTS ──────────────────────────────────────────────────────────────

    def _compute_weights(self, regime: int) -> dict:
        """
        Blend regime-based initial weights with rolling accuracy weights.
        50% regime-prior + 50% rolling accuracy = final weights.
        """
        regime_w   = INITIAL_WEIGHTS.get(regime, INITIAL_WEIGHTS[2])
        acc        = self.fc.get_accuracies()
        total_acc  = sum(acc.values())

        if total_acc == 0:
            acc_w = {"ftrl": 1/3, "hoeff": 1/3, "pa": 1/3}
        else:
            acc_w = {k: v / total_acc for k, v in acc.items()}

        # blend 50/50
        final_w = {
            k: 0.5 * regime_w[k] + 0.5 * acc_w[k]
            for k in ["ftrl", "hoeff", "pa"]
        }

        # normalize to sum=1
        total = sum(final_w.values())
        return {k: v / total for k, v in final_w.items()}

    # ── REPORT ───────────────────────────────────────────────────────────────

    def report(self):
        import numpy as np
        print("\n" + "="*50)
        print("ENSEMBLE RESULTS")
        print("="*50)

        acc = self.correct / self.total if self.total else 0
        print(f"Overall accuracy : {acc:.2%}  ({self.correct:,}/{self.total:,})")
        print(f"Drift detections : {self.drift_count}")

        print("\nPer-regime accuracy:")
        for r, name in REGIME_NAMES.items():
            c, t = self.per_regime[r]
            if t > 0:
                print(f"  {name:10s}: {c/t:.2%}  ({c:,}/{t:,})")

        lats = np.array(self.latencies_ns)
        print(f"\nLatency (ns):")
        print(f"  mean : {lats.mean():.0f}")
        print(f"  p50  : {np.percentile(lats, 50):.0f}")
        print(f"  p99  : {np.percentile(lats, 99):.0f}")
        print(f"  max  : {lats.max():.0f}")
        print("="*50)

    # ── SAVE / LOAD ──────────────────────────────────────────────────────────

    def save(self, path: str = "ensemble.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[save] ensemble → {path}")

    @staticmethod
    def load(path: str = "ensemble.pkl") -> "Ensemble":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[load] ensemble ← {path}")
        return obj