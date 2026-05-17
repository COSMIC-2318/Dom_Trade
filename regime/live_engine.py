"""
regime/live_engine.py — DOM Trade
===================================
Live trading pipeline. Connects to Angel One WebSocket,
runs feature engineering on every tick, feeds into
trained regime detector + ensemble, outputs predictions.

Prerequisites:
    - regime_model.pkl         (from regime_detector_v2.py)
    - reliance_features.parquet (for model warmup)
    - .env with Angel One credentials

Run:
    python regime/live_engine.py
"""

import sys
import os
import time
import threading
import collections
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from regime.regime_detector_v2 import RegimeDetector
from regime.forecasters import Forecasters
from regime.ensemble import Ensemble

# ── Constants ────────────────────────────────────────────────────────────────
SYMBOL        = "RELIANCE-EQ"
TOKEN         = "2885"
FILL_SHARES   = 500
WINDOWS       = [1, 5, 15, 30, 60]
WARMUP_TICKS  = 60     # ticks before predictions start
REGIME_MODEL  = "regime/regime_detector.pkl"
WARMUP_DATA   = "reliance_features.parquet"

REGIME_NAMES  = {0: "Bullish", 1: "Bearish", 2: "Volatile"}


# ── Rolling buffer for multi-scale features ──────────────────────────────────
class TickBuffer:
    """Stores last N ticks to compute rolling features."""

    def __init__(self, maxlen: int = 120):
        self.mid_prices = collections.deque(maxlen=maxlen)
        self.spreads    = collections.deque(maxlen=maxlen)
        self.volumes    = collections.deque(maxlen=maxlen)
        self.obis       = collections.deque(maxlen=maxlen)

    def push(self, mid: float, spread: float, volume: float, obi: float):
        self.mid_prices.append(mid)
        self.spreads.append(spread)
        self.volumes.append(volume)
        self.obis.append(obi)

    def rolling_std(self, series, w):
        arr = np.array(list(series))
        if len(arr) < max(2, w):
            return 0.0
        return float(np.std(arr[-w:]))

    def rolling_mean(self, series, w):
        arr = np.array(list(series))
        if len(arr) < w:
            return float(np.mean(arr)) if len(arr) > 0 else 0.0
        return float(np.mean(arr[-w:]))

    def rolling_return(self, w):
        arr = np.array(list(self.mid_prices))
        if len(arr) <= w:
            return 0.0
        return (arr[-1] - arr[-w-1]) / arr[-w-1] if arr[-w-1] != 0 else 0.0


# ── Feature engineering on a single raw tick ────────────────────────────────
def engineer_tick(raw: dict, buf: TickBuffer) -> dict:
    """Convert one raw Angel One tick dict into full feature dict."""

    bp1 = raw.get("bid_price_1", 0) or 0
    ap1 = raw.get("ask_price_1", 0) or 0
    mid = (ap1 + bp1) / 2.0 if (ap1 + bp1) > 0 else 0.0

    spread     = ap1 - bp1
    ltp        = raw.get("ltp", mid) or mid
    volume     = raw.get("volume", 0) or 0

    # bid/ask quantities
    bq = [raw.get(f"bid_qty_{i}", 0) or 0 for i in range(1, 6)]
    aq = [raw.get(f"ask_qty_{i}", 0) or 0 for i in range(1, 6)]
    bp = [raw.get(f"bid_price_{i}", 0) or 0 for i in range(1, 6)]
    ap = [raw.get(f"ask_price_{i}", 0) or 0 for i in range(1, 6)]

    bid_sum = sum(bq)
    ask_sum = sum(aq)

    # OBI
    obi_l1 = (bq[0] - aq[0]) / (bq[0] + aq[0]) if (bq[0] + aq[0]) > 0 else 0.0
    obi    = (bid_sum - ask_sum) / (bid_sum + ask_sum) if (bid_sum + ask_sum) > 0 else 0.0

    # VWMP
    num  = bp[0] * aq[0] + ap[0] * bq[0]
    den  = bq[0] + aq[0]
    vwmp = num / den if den > 0 else mid

    # depth ratio
    depth_ratio = bid_sum / ask_sum if ask_sum > 0 else 1.0

    # ask/bid depth
    ask_depth = (ap[4] - ap[0]) if ap[4] and ap[0] else 0.0
    bid_depth = (bp[0] - bp[4]) if bp[0] and bp[4] else 0.0

    # vwap pressure
    ask_vwap = sum(ap[i]*aq[i] for i in range(5)) / ask_sum if ask_sum > 0 else ap[0]
    bid_vwap = sum(bp[i]*bq[i] for i in range(5)) / bid_sum if bid_sum > 0 else bp[0]
    vwap_pressure = (ask_vwap - bid_vwap) / mid if mid > 0 else 0.0

    # concentrations
    mean_bq = sum(bq) / 5 if sum(bq) > 0 else 1
    mean_aq = sum(aq) / 5 if sum(aq) > 0 else 1
    bid_conc = bq[0] / mean_bq
    ask_conc = aq[0] / mean_aq

    # fill prices
    def fill(prices, qtys):
        cost, rem = 0.0, FILL_SHARES
        for p, q in zip(prices, qtys):
            f = min(q, rem); cost += f * p; rem -= f
        cost += rem * prices[-1]
        return cost / FILL_SHARES

    fill_buy  = fill(ap, aq)
    fill_sell = fill(bp[::-1], bq[::-1])

    # support / resistance
    si = int(np.argmax(bq))
    ri = int(np.argmax(aq))
    support_price    = bp[si]
    resistance_price = ap[ri]
    dist_support     = ltp - support_price
    dist_resistance  = resistance_price - ltp

    # push to buffer
    buf.push(mid, spread, volume, obi)

    # multi-scale rolling features
    feats = {
        "mid_price": mid, "spread": spread, "spread_pct": (spread/ap1*100) if ap1 else 0,
        "ltp_mid_delta": ltp - mid, "vwmp": vwmp,
        "obi_l1": obi_l1, "obi": obi, "depth_ratio": depth_ratio,
        "ask_depth": ask_depth, "bid_depth": bid_depth,
        "vwap_pressure": vwap_pressure,
        "bid_concentration": bid_conc, "ask_concentration": ask_conc,
        "fill_price_buy": fill_buy, "fill_price_sell": fill_sell,
        "support_price": support_price, "resistance_price": resistance_price,
        "dist_to_support": dist_support, "dist_to_resistance": dist_resistance,
        "support_strength": float(bq[si]), "resistance_strength": float(aq[ri]),
    }

    for w in WINDOWS:
        s = f"_{w}t"
        feats[f"realized_vol{s}"]   = buf.rolling_std(buf.mid_prices, w)
        feats[f"rolling_return{s}"] = buf.rolling_return(w)
        feats[f"obi_mean{s}"]       = buf.rolling_mean(buf.obis, w)
        feats[f"spread_vol{s}"]     = buf.rolling_std(buf.spreads, w)
        vol_arr = np.array(list(buf.volumes)[-w:])
        feats[f"volume_spike{s}"]   = float(vol_arr.sum() - vol_arr.mean()) if len(vol_arr) > 0 else 0.0

    return feats


# ── Live Engine ──────────────────────────────────────────────────────────────
class LiveEngine:

    def __init__(self):
        print("[init] loading models...")
        self.detector   = RegimeDetector.load(REGIME_MODEL)
        self.fc         = Forecasters()
        self.ens        = Ensemble(self.fc, self.detector)
        self.buf        = TickBuffer()
        self.tick_count = 0
        self.lock       = threading.Lock()
        
        # ── Continuous Learning Setup ──
        self.learning_queue = collections.deque()
        self.target_horizon = 30  # Ticks required to wait before a label matures
        
        self._warmup()

    def _warmup(self):
        """Warm up models on historical features before going live."""
        if not os.path.exists(WARMUP_DATA):
            print(f"[warmup] {WARMUP_DATA} not found — skipping warmup")
            return
        df = pd.read_parquet(WARMUP_DATA)
        print(f"[warmup] training on {len(df):,} historical ticks...")
        for _, row in df.iterrows():
            feat  = row.to_dict()
            label = int(row["target"]) if "target" in row and pd.notna(row["target"]) else 0
            self.fc.learn_all(feat, label)
        print("[warmup] done — models ready")

    def on_tick(self, raw: dict):
        """
        Called by Angel One WebSocket on every tick.
        """
        with self.lock:
            self.tick_count += 1

            # 1. Engineer Features
            features = engineer_tick(raw, self.buf)
            current_mid = features["mid_price"]

            # 2. Continuous Online Learning Pipeline (The Resolution Queue)
            # Push the current tick data to the queue to wait for its future outcome
            self.learning_queue.append({
                "tick": self.tick_count,
                "features": features.copy(),
                "mid": current_mid
            })

            # Check if any historical ticks have matured to generate a true label
            while self.learning_queue:
                oldest_data = self.learning_queue[0]
                
                if (self.tick_count - oldest_data["tick"]) >= self.target_horizon:
                    # The prediction from exactly 30 ticks ago has now matured
                    past_data     = self.learning_queue.popleft()
                    past_mid      = past_data["mid"]
                    past_features = past_data["features"]
                    
                    # Compute the true target label
                    if current_mid > 0 and past_mid > 0:
                        true_label = 1 if current_mid > past_mid else 0
                        
                        # Feed the mature label back into the forecasters
                        self.fc.learn_all(past_features, true_label)
                else:
                    # The oldest tick hasn't matured yet; exit loop
                    break

            # 3. Safe Warmup Guard (Prevents inference before rolling windows fill)
            if self.tick_count < WARMUP_TICKS:
                return

            # 4. Regime Detection
            regime      = self.detector.predict_one(features)
            regime      = regime if regime is not None else 2
            regime_name = REGIME_NAMES.get(regime, "Unknown")

            # 5. Ensemble Inference (Predicting the CURRENT tick)
            preds  = self.fc.predict_all(features)
            pred_f = int(preds["ftrl"])  if preds["ftrl"]  is not None else 0
            pred_h = int(preds["hoeff"]) if preds["hoeff"] is not None else 0
            pred_p = int(preds["pa"])    if preds["pa"]    is not None else 0

            weights = self.ens._compute_weights(regime)
            score   = weights["ftrl"]*pred_f + weights["hoeff"]*pred_h + weights["pa"]*pred_p
            final   = 1 if score >= 0.5 else 0

            direction = "▲ UP" if final == 1 else "▼ DOWN"
            
            print(
                f"[tick {self.tick_count:>6}] "
                f"LTP={raw.get('ltp', 0):>8.2f}  "
                f"OBI={features['obi']:>+.3f}  "
                f"Regime={regime_name:8s}  "
                f"Pred={direction}  "
                f"w=[F:{weights['ftrl']:.2f} H:{weights['hoeff']:.2f} P:{weights['pa']:.2f}]"
            )

    def run_with_api(self):
        """Connect to Angel One and stream live ticks."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from smartapi.smartapi_client import SmartAPIClient

            with SmartAPIClient() as client:
                client.register_price_callback(self.on_tick)
                client.start_websocket(on_tick=self.on_tick)
                client.wait_for_websocket()
                client.subscribe_symbols([TOKEN])
                print(f"[live] streaming {SYMBOL} — Ctrl+C to stop")
                while True:
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n[live] stopped by user")
            self.ens.report()


if __name__ == "__main__":
    engine = LiveEngine()
    engine.run_with_api()