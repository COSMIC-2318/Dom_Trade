"""
simulate_replay.py — DOM Trade
================================
Replays new_day.xlsx tick-by-tick, driving engine components directly
with CORRECT labels so FTRL and PA actually learn.

THE KEY FIX:
    - Does NOT call _orig_tick (which always learns with label=0)
    - Drives engineer_tick → regime → predict → learn(true_label) manually
    - true_label = 1 if mid_price_now > mid_price_30_ticks_ago else 0
    - This matches exactly what test_new_data.py does via ens.step()

Run:
    python simulate_replay.py --input data/new_day.xlsx --speed 0.05
    python simulate_replay.py --input data/new_day.xlsx --speed 0
"""

import argparse
import json
import os
import sys
import time
import collections
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# Must be imported before pickle.load runs inside LiveEngine.__init__
from regime.regime_detector_v2 import RegimeDetector  # noqa: F401

STATE_FILE  = "sim_state.json"
HISTORY_LEN = 300
WARMUP_ROWS = 60

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "tick": 0, "total_ticks": 0, "running": False, "done": False,
    "ltp": 0.0, "mid_price": 0.0, "obi": 0.0, "spread": 0.0,
    "regime": "Warming up...", "prediction": "—", "score": 0.5,
    "weights": {"ftrl": 0.33, "hoeff": 0.33, "pa": 0.33},
    "drift_count": 0, "accuracy": 0.0,
    "correct": 0, "total_preds": 0, "up_count": 0, "down_count": 0,
    "price_history": [], "obi_history": [], "pred_history": [],
    "regime_history": [], "time_history": [], "acc_history": [],
}


def write_state():
    if os.name == "nt":
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    else:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)


def push_history(key, val):
    state[key].append(val)
    if len(state[key]) > HISTORY_LEN:
        state[key].pop(0)


# ── Accuracy tracker (pred vs actual 30 ticks later) ─────────────────────────
class AccuracyTracker:
    def __init__(self):
        self.pending = {}
        self.correct = 0
        self.total   = 0

    def record(self, tick_idx, prediction, mid):
        self.pending[tick_idx] = (prediction, mid)
        resolve = tick_idx - 30
        if resolve in self.pending:
            pred_past, mid_past = self.pending.pop(resolve)
            if mid > 0 and mid_past > 0:
                actual = 1 if mid > mid_past else 0
                self.total += 1
                if pred_past == actual:
                    self.correct += 1

    @property
    def accuracy(self):
        return self.correct / self.total if self.total > 0 else 0.0


# ── Simulation ────────────────────────────────────────────────────────────────
def run_simulation(input_path: str, speed: float):
    print("=" * 60)
    print("  DOM TRADE — SIMULATION REPLAY")
    print("=" * 60)
    print(f"  Input : {input_path}")
    print(f"  Speed : {speed}s/tick" if speed > 0 else "  Speed : instant")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[load] reading data...")
    if input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)
    df = df[df["symbol"] == "RELIANCE-EQ"].reset_index(drop=True)
    total = len(df)
    print(f"[load] {total:,} RELIANCE-EQ ticks")
    print(f"[load] {df['time'].iloc[0]}  to  {df['time'].iloc[-1]}")

    state["total_ticks"] = total
    write_state()

    # ── Init engine components ────────────────────────────────────────────────
    print("\n[init] loading models + forecaster warmup on Day 1...")
    from regime.live_engine import LiveEngine, engineer_tick, REGIME_NAMES
    engine = LiveEngine()
    # We use engine.buf, engine.detector, engine.fc, engine.ens directly
    # We do NOT call engine.on_tick — it learns with label=0 always

    # ── Pre-fill TickBuffer with Day 1 tail ───────────────────────────────────
    warmup_candidates = [
        "reliance_features.parquet",
        "reliance_features_regime.parquet",
        "data/reliance_features.parquet",
    ]
    warmup_path = next((p for p in warmup_candidates if os.path.exists(p)), None)
    if warmup_path:
        df_w = pd.read_parquet(warmup_path).tail(WARMUP_ROWS)
        for _, row in df_w.iterrows():
            engine.buf.push(
                float(row.get("mid_price", 0) or 0),
                float(row.get("spread",    0) or 0),
                float(row.get("volume",    0) or 0),
                float(row.get("obi",       0) or 0),
            )
        print(f"[init] TickBuffer pre-filled with {WARMUP_ROWS} Day 1 tail rows")
        print(f"[debug] buf.mid_prices={len(engine.buf.mid_prices)}  "
              f"buf.obis={len(engine.buf.obis)}  buf.spreads={len(engine.buf.spreads)}")
    else:
        print("[init] WARNING: no warmup parquet found — rolling windows start cold")

    print("[init] ready\n")

    # ── Rolling mid-price window for true label computation ───────────────────
    # true_label at tick t = 1 if mid[t] > mid[t-30] else 0
    mid_window = collections.deque(maxlen=31)

    tracker  = AccuracyTracker()
    up_count = 0
    dn_count = 0
    tick_num = 0   # our own counter (engine.tick_count not used — we bypass on_tick)

    # ── Replay loop ───────────────────────────────────────────────────────────
    try:
        for i, row in df.iterrows():
            raw = {k: (v.item() if hasattr(v, "item") else v)
                   for k, v in row.to_dict().items()}
            timestamp = str(raw.get("time", ""))
            tick_num += 1

            # ── Step 1: feature engineering (buf.push happens exactly once here)
            features = engineer_tick(raw, engine.buf)

            # ── Basic price values ────────────────────────────────────────────
            bp1    = raw.get("bid_price_1", 0) or 0
            ap1    = raw.get("ask_price_1", 0) or 0
            mid    = (ap1 + bp1) / 2.0 if (ap1 + bp1) > 0 else float(raw.get("ltp", 0))
            bq     = [raw.get(f"bid_qty_{i}", 0) or 0 for i in range(1, 6)]
            aq     = [raw.get(f"ask_qty_{i}", 0) or 0 for i in range(1, 6)]
            bs     = sum(bq); as_ = sum(aq)
            obi    = (bs - as_) / (bs + as_) if (bs + as_) > 0 else 0.0
            spread = ap1 - bp1

            # ── Step 2: compute TRUE label from 30-tick lookahead window ──────
            mid_window.append(mid)
            if len(mid_window) >= 31:
                true_label = 1 if mid_window[-1] > mid_window[0] else 0
            else:
                true_label = 0   # not enough history yet, dummy label

            # ── Step 3: regime detection ──────────────────────────────────────
            regime = engine.detector.predict_one(features)
            regime = regime if regime is not None else 2
            rname  = REGIME_NAMES.get(regime, "Unknown")

            # ── Step 4: predict BEFORE learning (correct online order) ────────
            if tick_num >= 60:
                preds   = engine.fc.predict_all(features)
                pred_f  = int(preds["ftrl"])  if preds["ftrl"]  is not None else 0
                pred_h  = int(preds["hoeff"]) if preds["hoeff"] is not None else 0
                pred_p  = int(preds["pa"])    if preds["pa"]    is not None else 0

                weights = engine.ens._compute_weights(regime)
                score   = (weights["ftrl"] * pred_f +
                           weights["hoeff"] * pred_h +
                           weights["pa"]   * pred_p)
                final   = 1 if score >= 0.5 else 0
                direction = "UP" if final == 1 else "DOWN"
            else:
                pred_f = pred_h = pred_p = 0
                weights   = {"ftrl": 0.33, "hoeff": 0.33, "pa": 0.33}
                score     = 0.5
                final     = 0
                direction = "—"

            # ── Step 5: learn with TRUE label ─────────────────────────────────
            engine.fc.learn_all(features, true_label)

            # ── Step 6: ensemble drift detection + weight update ──────────────
            try:
                engine.ens.step(features, true_label)
            except Exception:
                pass   # ens.step may not exist in all versions

            # ── Terminal print ────────────────────────────────────────────────
            if tick_num >= 60:
                dir_str = "▲ UP  " if final == 1 else "▼ DOWN"
                print(
                    f"[tick {tick_num:>6}] "
                    f"LTP={float(raw.get('ltp', 0)):>8.2f}  "
                    f"OBI={obi:>+.3f}  "
                    f"Regime={rname:8s}  "
                    f"Pred={dir_str}  "
                    f"w=[F:{weights['ftrl']:.2f} "
                    f"H:{weights['hoeff']:.2f} "
                    f"P:{weights['pa']:.2f}]"
                )

            # ── Accuracy tracking ─────────────────────────────────────────────
            if tick_num >= 60:
                if direction == "UP":
                    up_count += 1
                elif direction == "DOWN":
                    dn_count += 1
                tracker.record(tick_num, final, mid)

            # ── Write shared state for dashboard ──────────────────────────────
            state["tick"]        = tick_num
            state["running"]     = True
            state["ltp"]         = float(raw.get("ltp", 0))
            state["mid_price"]   = float(mid)
            state["obi"]         = float(obi)
            state["spread"]      = float(spread)
            state["regime"]      = rname
            state["prediction"]  = direction
            state["score"]       = float(score)
            state["weights"]     = {k: float(v) for k, v in weights.items()}
            state["drift_count"] = int(getattr(engine.ens, "drift_count", 0))
            state["accuracy"]    = float(tracker.accuracy)
            state["correct"]     = int(tracker.correct)
            state["total_preds"] = int(tracker.total)
            state["up_count"]    = int(up_count)
            state["down_count"]  = int(dn_count)

            push_history("price_history",  float(mid))
            push_history("obi_history",    float(obi))
            push_history("pred_history",   1 if direction == "UP" else 0)
            push_history("regime_history", rname)
            push_history("time_history",   timestamp)
            push_history("acc_history",    float(tracker.accuracy))

            write_state()

            if speed > 0:
                time.sleep(speed)

    except KeyboardInterrupt:
        print("\n[sim] stopped by user")

    # ── Final summary ─────────────────────────────────────────────────────────
    state["running"] = False
    state["done"]    = True
    write_state()

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Ticks processed : {tick_num:,}")
    print(f"  Predictions     : {tracker.total:,}")
    if tracker.total > 0:
        print(f"  Accuracy        : {tracker.accuracy:.2%}")
    print(f"  UP signals      : {up_count:,}")
    print(f"  DOWN signals    : {dn_count:,}")
    print(f"  Drift events    : {getattr(engine.ens, 'drift_count', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/new_day.xlsx")
    parser.add_argument("--speed", type=float, default=0.05)
    args = parser.parse_args()
    run_simulation(args.input, args.speed)