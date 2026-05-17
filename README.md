# DOM Trade — Regime-Aware Forecasting System

> **Order Book Data Collection & Stock Price Prediction for NSE Equities**
>
> Built on live Level-2 order book data · Real-time regime detection · Fully online learning · No retraining

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Exchange](https://img.shields.io/badge/exchange-NSE-orange.svg)](https://www.nseindia.com/)
[![Broker](https://img.shields.io/badge/broker-Angel%20One%20SmartAPI-red.svg)](https://smartapi.angelbroking.com/)
[![Models](https://img.shields.io/badge/models-GMM%20·%20ARF%20·%20FTRL%20·%20Hoeffding%20·%20PA-green.svg)]()

---

## Performance at a Glance

|  | Day 1 — May 13 2026 | Day 2 — May 14 2026 (out-of-sample) |
|---|---|---|
| **Naive Baseline** | 41.15% | 41.88% |
| **Ensemble Accuracy** | **85.04%** | **83.41%** |
| **Lift over Baseline** | +43.89 pp | +41.53 pp |
| **Mean Inference Latency** | 0.315 ms | 0.152 ms |
| **Drift Events Detected** | 7 | 9 |
| **Ticks** | 34,743 train / 6,943 test | 10,250 |

> Day 2 accuracy drop: only **−1.63 percentage points** on a completely unseen trading session with online learning enabled.

---

## Documentation

Full technical documentation is in [`DOM.pdf`](./DOM.pdf), covering:

- Complete mathematical derivations for all 46 features
- Regime detection pipeline (GMM → ARF) with formal algorithm specifications
- FTRL, Hoeffding Tree, and PA Classifier update rules
- ADWIN drift detection statistical test
- All design decisions with rationale and rejected alternatives (Kalman, HMM, DeepLOB, LinUCB, AROW)
- Honest limitations and prioritised future work roadmap

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Why Order Book Data](#2-why-order-book-data)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [How to Run](#6-how-to-run)
7. [Feature Engineering](#7-feature-engineering)
8. [Regime Detection](#8-regime-detection)
9. [Online Forecasting Models](#9-online-forecasting-models)
10. [Dynamic Ensemble & Drift Detection](#10-dynamic-ensemble--drift-detection)
11. [Results](#11-results)
12. [Design Decisions](#12-design-decisions)
13. [Limitations & Future Work](#13-limitations--future-work)
14. [Live API Setup](#14-live-api-setup)

---

## 1. What This Project Is

DOM Trade is an **end-to-end adaptive forecasting system** for NSE equities built on live Level-2 order book data. It:

- Streams real-time bid-ask depth from Angel One's WebSocket API (Mode 3)
- Engineers **46 microstructure features** from raw ticks
- Detects the current market regime using a **two-stage ML pipeline** (GMM → ARF)
- Produces **directional predictions (UP/DOWN)** using three parallel online models fused by a dynamic weighted ensemble
- Adapts **continuously per tick** — no retraining, no batch jobs, no manual intervention

**Central thesis:** A single static model trained once cannot adapt to changing market conditions. Markets behave fundamentally differently when trending, falling, or consolidating. This system detects which condition is active and routes predictions through models specifically suited to that regime.

---

## 2. Why Order Book Data

Price alone tells you what *happened*. The order book tells you what is *about to happen*.

When 50,000 shares are sitting at a bid level, that is a wall — price is unlikely to fall through it easily. When the ask side is thin, price can run up fast. No OHLCV or chart-based model sees this. Order book microstructure is the closest thing to reading institutional intent in real time.

**Key insight:** Level-2 data (5 bid levels + 5 ask levels) reveals supply-demand imbalances *before* they manifest in price. OBI (Order Book Imbalance) is statistically predictive of the next 30-tick mid-price direction with a lead time that allows meaningful action within the WebSocket tick interval.

---

## 3. System Architecture

```
Raw WebSocket Ticks  ←  Angel One SmartAPI (Mode 3, 27 columns/tick)
         │
         ▼
┌─────────────────────────────────┐
│  A   feature_engine.py          │  46 microstructure features from raw ticks
│      Price · OBI · S/R · Rolling│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  B   regime_detector_v2.py      │  Stage 1: GMM  (offline label generation)
│                                 │  Stage 2: ARF  (online real-time detection)
└──────────────┬──────────────────┘
               │  regime ∈ { Bullish=0, Bearish=1, Volatile=2 }
               ▼
┌──────────────────────────────────────────────────────┐
│  C   forecasters.py                                  │
│      ├── FTRL           (trending regimes)           │
│      ├── Hoeffding Tree  (regime boundaries)         │
│      └── PA Classifier   (low-volatility regimes)   │
└──────────────┬───────────────────────────────────────┘
               │  3 parallel UP/DOWN predictions
               ▼
┌─────────────────────────────────┐
│  D   ensemble.py                │  Two-stage dynamic weighting
│                                 │  + ADWIN concept drift detection
└──────────────┬──────────────────┘
               │  final UP / DOWN signal
       ┌───────┴──────────┬──────────────┐
       ▼                  ▼              ▼
┌────────────┐   ┌──────────────┐  ┌───────────┐
│ E backtest │   │ F live_      │  │ G dash-   │
│ .py        │   │ engine.py    │  │ board.py  │
│ evaluation │   │ production   │  │ Streamlit │
└────────────┘   └──────────────┘  └───────────┘
```

---

## 4. Repository Structure

```
Dom_Trade/
│
├── DOM.pdf                         # Full technical documentation
│
├── data/
│   ├── orderbook.csv.xlsx          # Raw NSE L2 order book data (Day 1)
│   └── new_day.xlsx                # Day 2 data for out-of-sample testing
│
├── regime/
│   ├── __init__.py
│   ├── regime_detector_v2.py       # GMM + ARF regime detection
│   ├── forecasters.py              # FTRL + Hoeffding Tree + PA Classifier
│   ├── ensemble.py                 # Dynamic weights + ADWIN drift detection
│   └── live_engine.py              # Live WebSocket streaming and inference
│
├── smartapi/                       # Angel One API wrapper
│   ├── __init__.py
│   ├── config.py
│   ├── rate_limiter.py
│   ├── smartapi_client.py
│   └── totp_generator.py
│
├── plots/                          # Generated visualisations
│
├── feature_engine.py               # 46 features from raw ticks
├── backtest.py                     # Full Day 1 evaluation pipeline
├── test_new_data.py                # Out-of-sample Day 2 test
├── simulate_replay.py              # Tick-by-tick simulation replay
├── main.py                         # Single entry point — runs full pipeline
├── dashboard.py                    # Streamlit visual dashboard
├── analysis.nbconvert.ipynb        # Offline benchmarks and algorithm comparison
│
├── .env                            # API credentials 
├── .gitignore
│
# ── Generated at runtime (not committed) ──────────────────────────────
├── reliance_features.parquet       # Day 1 features (required for Day 2 warmup)
├── reliance_regimes.parquet        # Features + regime labels
├── tmp_new_features.parquet        # Temp features during Day 2 test
├── sim_state.json                  # Simulation state
└── regime/
    ├── regime_detector.pkl         # Trained GMM + ARF model
    ├── reliance_features.parquet
    └── reliance_regimes.parquet
```

> **Critical:** `reliance_features.parquet` is generated by the full pipeline and **must exist** before running Day 2 tests. Without it, all models cold-start and collapse to ~58% accuracy. Always run `main.py` first.

---

## 5. Installation

### Prerequisites

- Python 3.9+
- Angel One SmartAPI account (required for live streaming only)
- Live mode market hours: Mon–Fri 09:15–15:30 IST

### Install Dependencies

```bash
pip install pandas numpy river scikit-learn hmmlearn \
            openpyxl pyarrow streamlit plotly
```

### Environment Setup (Live Mode Only)

Create a `.env` file in the project root:

```env
ANGEL_API_KEY=your_api_key_here
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

> `.env` is listed in `.gitignore`. Never commit it.

---

## 6. How to Run

### Option A — Full Pipeline (Single Command)

Runs feature engineering → regime detection → backtest in sequence. Saves `reliance_features.parquet` for Day 2.

```bash
python main.py --input data/orderbook.csv.xlsx
```

### Option B — Step by Step

```bash
# Step 1: Feature engineering
python feature_engine.py \
    --input data/orderbook.csv.xlsx \
    --output reliance_features.parquet

# Step 2: Regime detection (GMM + ARF)
python regime/regime_detector_v2.py \
    --parquet reliance_features.parquet \
    --output regime/regime_model.pkl

# Step 3: Backtest on Day 1 held-out test set
python backtest.py \
    --parquet reliance_features_regime.parquet \
    --regime_model regime/regime_model.pkl
```

### Out-of-Sample Test — New Day's Data

```bash
# Recommended: online learning (models adapt per tick, as in live trading)
python test_new_data.py --input data/new_day.xlsx

# Frozen inference: weights locked at Day 1 state
# Use only to measure frozen generalisation — not live performance
python test_new_data.py --input data/new_day.xlsx --no_learn
```

| Mode | Expected Accuracy | Notes |
|---|---|---|
| Online learning (default) | 80–85% | Models adapt per tick |
| `--no_learn` frozen | ~58–65% | No intraday adaptation; degrades toward naive baseline |

### Streamlit Dashboard

```bash
streamlit run dashboard.py
```

### Live API Streaming

```bash
# Market hours only: Mon–Fri 09:15–15:30 IST
python regime/live_engine.py
```

---

## 7. Feature Engineering

The Angel One WebSocket Mode 3 delivers **27 raw columns** per tick. `feature_engine.py` derives **46 features** across four groups. All prices arrive in paise and are converted to rupees at the `parse_tick()` boundary before any computation.

### Group 1 — Price Features (5)

| Feature | Formula | Why |
|---|---|---|
| `mid_price` | `(ask_p1 + bid_p1) / 2` | True fair value; LTP bounces between bid/ask and is not directionally predictive |
| `spread` | `ask_p1 − bid_p1` | Liquidity signal; directly sets transaction cost |
| `spread_pct` | `spread / ask_p1 × 100` | Normalised spread for cross-tick comparability |
| `vwmp` | `(bid_p1×ask_q1 + ask_p1×bid_q1) / (bid_q1+ask_q1)` | Pressure-adjusted fair value; skews toward the dominant side |
| `ltp_mid_delta` | `ltp − mid_price` | Buyer aggression (+) vs seller aggression (−) on last trade |

### Group 2 — Order Book Features (11)

| Feature | Purpose |
|---|---|
| `obi_l1` | Level-1 Order Book Imbalance ∈ [−1, +1] — top-of-book pressure |
| `obi` | Full 5-level OBI — strongest short-term directional signal in LOB literature |
| `depth_ratio` | Total bid qty / ask qty — structural buyer vs seller dominance |
| `ask_depth` / `bid_depth` | Book slope — measures market impact of a large order |
| `vwap_pressure` | `(ask_vwap − bid_vwap) / mid_price` — asymmetry across all 5 levels |
| `bid_concentration` / `ask_concentration` | Fraction of interest at top of book — aggressiveness signal |
| `fill_price_buy` | Simulated 500-share market order walk — realistic execution cost |

### Group 3 — Support & Resistance Features (6)

Live walls based on where the largest orders are *right now*, not chart history:

| Feature | Definition |
|---|---|
| `support_price` | Bid price at `argmax(bid_qty)` across 5 levels |
| `resistance_price` | Ask price at `argmax(ask_qty)` across 5 levels |
| `dist_to_support` / `dist_to_resistance` | Mid price distance to each wall |
| `support_strength` / `resistance_strength` | Rolling mean of wall size (w=50) — wall stickiness |

### Group 4 — Multi-Scale Rolling Features (25)

Five metrics × five tick windows `w ∈ {1, 5, 15, 30, 60}`:

| Metric | Captures |
|---|---|
| `realized_vol_{w}` | Volatility at each timescale — primary regime signal |
| `rolling_return_{w}` | Directional momentum at each timescale |
| `obi_mean_{w}` | Sustained directional bias (noise-filtered OBI) |
| `spread_vol_{w}` | Liquidity regime — secondary regime indicator |
| `volume_spike_{w}` | Abnormal volume activity preceding price moves |

**Why five windows:** Regimes operate at multiple time horizons simultaneously. A 60-tick window captures the macro trend; a 5-tick window captures current momentum; a 1-tick window captures microstructure noise. A single window is blind to everything at other scales.

### Prediction Target

```
target_t = 1  if mid_price_(t+30) > mid_price_t  else 0
```

**30-tick horizon:** below 10 ticks you are predicting autocorrelation noise; above 100 ticks OBI decays and microstructure features lose power. 30 ticks is the empirically supported sweet spot in LOB literature.

---

## 8. Regime Detection

A model trained across all market conditions learns the *average* of all regimes and underperforms in each. The solution is a two-stage unsupervised → supervised pipeline.

### Stage 1 — Gaussian Mixture Model (GMM)

Run **once offline** on Day 1 data. Fits K=3 Gaussian components to `[realized_vol_60, rolling_return_60]` and labels clusters by their characteristics:

| Characteristic | Label | Regime ID |
|---|---|---|
| Highest realized volatility | Volatile | 2 |
| Most negative rolling return (of remaining) | Bearish | 1 |
| Remaining | Bullish | 0 |

GMM was chosen over K-Means (assumes spherical clusters; financial features are ellipsoidal) and HMM (not online; Baum-Welch requires the full sequence; temporal dependency assumption broken by sudden news-driven regime shifts).

### Stage 2 — Adaptive Random Forest (ARF)

Trained **online** using GMM labels as supervision. Detects regime on new ticks in real time without retraining.

**ARF input features:** `realized_vol_60`, `realized_vol_15`, `rolling_return_60`, `rolling_return_15`, `obi_mean_60`, `spread_vol_60`

Both 15t and 60t windows let ARF distinguish *fast volatility spike within slow downtrend* vs *fast spike within slow uptrend* — different regimes that a single-window detector conflates.

**Why ARF:** Truly online (`learn_one()` per tick) · Non-linear boundaries · Self-replacing (weakest tree replaced on internal ADWIN drift) · No manual monitoring.

---

## 9. Online Forecasting Models

Three models with **fundamentally different inductive biases** run in parallel on every tick, covering each other's blind spots:

| Model | Core Assumption | Wins When |
|---|---|---|
| **FTRL** | Linear · per-feature adaptive learning rates | Trending regimes; OBI and momentum signals consistent |
| **Hoeffding Tree** | Non-linear boundaries · feature interactions matter | Regime transitions; single features insufficient |
| **PA Classifier** | Update only on mistakes; passive on correct predictions | Low-volatility; prevents noise chasing |

### FTRL (Follow The Regularized Leader)

Google's production algorithm for billion-scale real-time prediction. Key properties:
- Per-feature adaptive learning rates — noisy features (OBI) get suppressed automatically; stable features (rolling Sharpe) trusted more
- L1 regularization → sparse weight vectors → automatic feature selection on the fly
- Continuous updates on every tick → builds strong signal in trending regimes (PA misses 70% of updates in the same scenario)

### Hoeffding Tree (VFDT)

Builds incrementally from a stream using the Hoeffding bound to wait for statistically confident splits. Captures non-linear interactions like:

```
Bull regime:     OBI > 0.3                          → UP
Volatile regime: OBI > 0.3 AND spread_vol < 0.05   → UP
                 OBI > 0.3 AND spread_vol > 0.05   → not UP
```

Linear models (FTRL, PA) cannot represent this. The Hoeffding Tree does automatically.

> **Grace period:** Returns `None` for first 50 predictions while accumulating split statistics. Ensemble defaults these to 0.

### Passive-Aggressive Classifier (PA)

Structurally conservative: `τ_t = 0` when prediction is correct and margin is sufficient — no weight update at all. In low-volatility regimes where signal is weak, PA stabilises while FTRL would jitter weights on every tick.

> **⚠️ Warning on PA's 96% accuracy:** PA alone achieves 95–96% by memorising intraday serial correlation within a single day. Frozen inference on a new day drops to **58%**. This is overfitting to autocorrelation, not generalisation. The meaningful number is the **ensemble (83–85%)**, not PA in isolation.

### Forecaster Input Features (13 total)

`obi_mean_1`, `obi_mean_5`, `obi_mean_15`, `spread_vol_5`, `spread_vol_15`, `ltp_mid_delta`, `vwmp`, `rolling_return_5`, `rolling_return_15`, `dist_to_support`, `dist_to_resistance`, `support_strength`, `resistance_strength`

---

## 10. Dynamic Ensemble & Drift Detection

### Two-Stage Weight Computation

**Stage 1 — Regime Prior** (structural belief about model suitability):

| Regime | FTRL | Hoeffding | PA |
|---|---|---|---|
| Bullish (0) | 0.6 | 0.3 | 0.1 |
| Bearish (1) | 0.3 | 0.5 | 0.2 |
| Volatile (2) | 0.2 | 0.3 | 0.5 |

**Stage 2 — Rolling Accuracy Adjustment** (empirical performance over last 100 ticks):

```
ŵ_m^acc  =  â_m / Σ â_m'          (accuracy-normalised)

w_m^final = 0.5 · w_m^regime + 0.5 · w_m^acc
            ─────────────────────────────────   (normalised to sum to 1)
            Σ (0.5 · w_m'^regime + 0.5 · w_m'^acc)
```

The 50/50 blend respects regime structure while continuously re-evaluating which model actually deserves trust right now. The best model in hour one may be the worst in hour three.

**Final prediction:**
```
score    = w_ftrl · ŷ_ftrl + w_hoeff · ŷ_hoeff + w_pa · ŷ_pa
ŷ_final  = 1  if score ≥ 0.5  else 0
```

### ADWIN Concept Drift Detection

ADWIN (Adaptive Windowing) splits its error window at position `i` when:

```
|µ̂₀ − µ̂₁| ≥ ε_cut = sqrt( (1/2m) · ln(4n/δ) )
```

When drift is detected — **boost learning rates, do not reset:**

| Component | Normal | On Drift |
|---|---|---|
| FTRL α | 0.1 | 0.3 |
| PA C | 0.1 | 0.5 |
| ARF | — | Replaces weakest tree |

Resetting throws away all learned weights. Boosting keeps prior knowledge as a warm start while weighting new observations more heavily — fast adaptation without a recovery blind period.

---

## 11. Results

### Day 1 — May 13 2026 (Backtest on Held-Out 20%)

| Model | Accuracy |
|---|---|
| Naive (always UP) | 41.15% |
| Hoeffding Tree alone | 60.90% |
| FTRL alone | 73.70% |
| **Ensemble** | **85.04%** |
| PA alone ⚠️ | 96.01% |

**Per-regime accuracy:**

| Regime | Ticks | Accuracy |
|---|---|---|
| Bullish | 0 | — (not observed; bearish/choppy day) |
| Bearish | 1,119 | 82.75% |
| Volatile | 5,824 | 85.47% |

**Inference latency:**

| Metric | Value |
|---|---|
| Mean | 315,336 ns (0.315 ms) |
| p50 | 289,900 ns |
| p99 | 643,164 ns |
| Max | 971,900 ns |
| WebSocket tick interval | 100–500 ms |

### Day 2 — May 14 2026 (Completely Unseen Session)

| | Day 1 | Day 2 |
|---|---|---|
| Naive baseline | 41.15% | 41.88% |
| **Ensemble** | **85.04%** | **83.41%** |
| Accuracy drop | — | −1.63 pp |
| Mean latency | 0.315 ms | 0.152 ms |
| Drift events | 7 | 9 |

**Classification report:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DOWN | 0.8401 | 0.8827 | 0.8608 | 5,957 |
| UP | 0.8249 | 0.7668 | 0.7948 | 4,293 |
| **Accuracy** | | | **0.8341** | 10,250 |
| Macro avg | 0.8325 | 0.8247 | 0.8278 | 10,250 |

> **On frozen inference:** Running `--no_learn` on Day 2 produced **58.12%** — identical to the always-DOWN naive baseline. Online learning is not optional for cross-day inference.

---

## 12. Design Decisions

### Time-Ordered Train/Test Split

A random split leaks future data into training. This inflates accuracy by 5–15 pp and produces numbers that are fictitious in live trading. **Split used:** first 80% of rows for training, last 20% for testing.

### Binary Target Over 3-Class

The original 3-class formulation (UP / NEUTRAL / DOWN) produced 81% NEUTRAL rows at H=50. Macro F1 ≈ 0.33 across all offline benchmarks — equal to random chance for 3 classes. Binary UP/DOWN gives a 44%/56% tractable split and 85% ensemble accuracy.

### Boost on Drift, Not Reset

Resetting discards all prior knowledge. If FTRL learned `OBI > 0.4` predicts UP with 70% reliability, resetting deletes that. Recovery takes hundreds of ticks. Increasing the learning rate on drift achieves fast adaptation without a blind period.

### Python Over C++

Python inference averages 315µs (Day 1) / 152µs (Day 2). For a 100–500ms WebSocket feed this is entirely acceptable. C++ becomes necessary for co-location deployments requiring sub-10µs tick-to-order latency — tracked as future work.

---

## 13. Limitations & Future Work

### Honest Limitations

1. **Two days of data** — Validated on May 13–14 2026 only. Longer-term generalisation across varied conditions is unproven.
2. **No Bullish regime observed** — Both sessions were bearish/volatile. Ensemble weights for Bullish (FTRL=0.6) are untested structural priors.
3. **Online learning required** — Frozen inference degrades to the naive baseline. Models must adapt per tick.
4. **PA's 96% ≠ generalisation** — Reflects intraday autocorrelation memorisation. The ensemble (83–85%) is the meaningful metric.
5. **5-level book only** — Institutional players with 20+ levels have an information advantage.
6. **No alternative data** — No news, FII/DII flow, options, or sentiment. Purely technical microstructure.
7. **Single symbol** — RELIANCE-EQ only. Generalisation to other symbols unverified.

### Future Work (Priority Order)

1. Collect 20+ trading days — expose all three regimes to GMM, validate generalisation
2. Walk-forward cross-validation — expanding window retraining for robust OOS estimates
3. LinUCB contextual bandit — replace hardcoded regime-to-weight table with a learned selector
4. C++ inference for co-location (extend existing `pa_engine.cpp` via AVX2 SIMD)
5. Signed order flow imbalance (OFI) + trade tick features
6. Multi-symbol test — train on RELIANCE, evaluate on TCS/HDFCBANK
7. 2-state HMM on `(OBI, log-spread)` to replace GMM once sufficient data is available
8. PnL simulation with realistic transaction costs — convert directional accuracy into a tradeable signal assessment

---

## 14. Live API Setup

```env
# .env
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

The `smartapi/` package handles authentication, TOTP 2FA, rate limiting, and WebSocket lifecycle management. The live engine subscribes to Mode 3 on startup, computes features on every incoming tick, detects the current regime, and emits UP/DOWN predictions in real time.

```bash
# Market hours only: Mon–Fri 09:15–15:30 IST
python regime/live_engine.py
```

---

## Dataset

All data collected on NSE via Angel One SmartAPI WebSocket Mode 3:

| Symbol | Company | Token | Rows (May 13 2026) |
|---|---|---|---|
| RELIANCE-EQ | Reliance Industries | 2885 | 34,743 |
| HDFCBANK-EQ | HDFC Bank | 3045 | 35,934 |
| TCS-EQ | Tata Consultancy Services | 11536 | 27,943 |
| INFY-EQ | Infosys | 1594 | 23,149 |
| ICICIBANK-EQ | ICICI Bank | 1660 | 14,155 |
| **Total** | | | **135,924** |

All modelling uses **RELIANCE-EQ only** — the most liquid name in the set, minimising adverse selection and maximising OBI signal quality.

---

*DOM Trade · Internal Research · May 2026*
