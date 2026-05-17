# DOM Trade — Regime-Aware Forecasting System

> **Order Book Data Collection & Stock Price Prediction**  
> NSE · Angel One SmartAPI · WebSocket Mode 3 · River · LightGBM  
> GMM · ARF · FTRL · Hoeffding Tree · PA Classifier · ADWIN

---

## Performance at a Glance

| | Day 1 (May 13 2026) | Day 2 (May 14 2026) |
|---|---|---|
| **Naive Baseline** | 41.15% | 41.88% |
| **Ensemble Accuracy** | **85.04%** | **83.41%** |
| **Improvement** | +43.89 pp | +41.53 pp |
| **Mean Latency** | 0.315 ms | 0.152 ms |
| **Drift Events** | 7 | 9 |
| **Ticks** | 34,743 (train) / 6,943 (test) | 10,250 (out-of-sample) |

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Why Order Book Data](#2-why-order-book-data)
3. [Project Architecture](#3-project-architecture)
4. [File Structure](#4-file-structure)
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
- Adapts continuously — no retraining, no manual intervention

**Central thesis:** A single static model trained once cannot adapt to changing market conditions. Markets behave fundamentally differently when trending up, trending down, or consolidating. This system detects which condition is active and routes predictions through models specifically suited to that regime.

---

## 2. Why Order Book Data

Price alone tells you what *happened*. The order book tells you what is *about to happen*.

When 50,000 shares are sitting at a bid level, that is a wall — price is unlikely to fall through it easily. When the ask side is thin, price can run up fast. No chart-based or price-only model sees this. Order book microstructure is the closest thing to reading institutional intent in real time.

**Key insight:** Level-2 data (5 bid levels + 5 ask levels) reveals supply-demand imbalances before they manifest in price. OBI (Order Book Imbalance) is statistically predictive of the next 30-tick mid-price direction with a lead time that allows meaningful action within the WebSocket tick interval.

---

## 3. Project Architecture

The system is organized into seven layers that run sequentially in the pipeline:

```
Raw WebSocket Ticks (Angel One API)
         │
         ▼
┌─────────────────────────────┐
│  A  feature_engine.py       │  46 microstructure features from raw ticks
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  B  regime_detector_v2.py   │  GMM (offline labels) + ARF (online detection)
└─────────────┬───────────────┘
              │  regime ∈ {Bullish, Bearish, Volatile}
              ▼
┌─────────────────────────────────────────────────────┐
│  C  forecasters.py                                  │
│     ├── FTRL    (trending regimes)                  │
│     ├── Hoeffding Tree  (regime boundaries)         │
│     └── PA Classifier   (low-volatility regimes)   │
└─────────────┬───────────────────────────────────────┘
              │  3 parallel predictions
              ▼
┌─────────────────────────────┐
│  D  ensemble.py             │  Dynamic weighted fusion + ADWIN drift detection
└─────────────┬───────────────┘
              │  final UP / DOWN signal
              ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ E backtest   │   │ F live_engine│   │ G dashboard  │
│ (evaluation) │   │ (production) │   │ (Streamlit)  │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## 4. File Structure

```
PROJECT/
│
├── data/
│   ├── orderbook.csv.xlsx          # Raw NSE order book data (Day 1 training)
│   └── new_day.xlsx                # Day 2 data for out-of-sample testing
│
├── regime/
│   ├── __init__.py
│   ├── regime_detector_v2.py       # GMM + ARF regime detection
│   ├── forecasters.py              # FTRL + Hoeffding Tree + PA Classifier
│   ├── ensemble.py                 # Dynamic weights + ADWIN drift detection
│   └── live_engine.py              # Live API streaming and inference
│
├── smartapi/                       # Angel One API wrapper
│   ├── __pycache__/
│   ├── tests/
│   ├── __init__.py
│   ├── config.py
│   ├── rate_limiter.py
│   ├── smartapi_client.py
│   └── totp_generator.py
│
├── plots/                          # Generated visualisations
│
├── feature_engine.py               # 46 features from raw ticks
├── backtest.py                     # Full Day 1 evaluation
├── test_new_data.py                # Out-of-sample Day 2 test
├── main.py                         # Single entry point (full pipeline)
├── dashboard.py                    # Streamlit visual dashboard
├── simulate_replay.py              # Tick-by-tick simulation replay
├── analysis.nbconvert.ipynb        # Offline analysis and benchmarks
├── .env                            # API credentials — NEVER commit this
├── .gitignore
│
# Generated at runtime:
├── reliance_features.parquet       # Day 1 features (required for Day 2 warmup)
├── reliance_regimes.parquet        # Features + regime labels
├── tmp_new_features.parquet        # Temporary features during Day 2 test
├── sim_state.json                  # Simulation state
├── sim_state.json.tmp
└── regime/
    ├── regime_detector.pkl         # Trained GMM + ARF model
    ├── reliance_features.parquet
    └── reliance_regimes.parquet
```

> **Important:** `reliance_features.parquet` is generated by the full pipeline and is required before running Day 2 tests. Always run `main.py` first.

---

## 5. Installation

### Prerequisites

- Python 3.9+
- Angel One SmartAPI account (for live streaming only)
- Market hours for live mode: Mon–Fri 09:15–15:30 IST

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

> **Never commit `.env` to version control.** It is already listed in `.gitignore`.

---

## 6. How to Run

### Option A — Full Pipeline (One Command)

Runs feature engineering → regime detection → backtest in sequence. Also saves `reliance_features.parquet` needed for Day 2 testing.

```bash
python main.py --input data/orderbook.csv.xlsx
```

### Option B — Step by Step

```bash
# Step 1: Feature engineering
# Produces reliance_features.parquet required for Day 2 warmup
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

### Test on New Day's Data (Day 2)

> **Prerequisite:** Run the full pipeline first. `test_new_data.py` requires `reliance_features.parquet` to exist. Without it, all models collapse to predicting one class (~58% accuracy).

```bash
# Recommended: online learning mode
# Models adapt to the new day's stream as they would in live trading
python test_new_data.py --input data/new_day.xlsx

# Inference only: models frozen at Day 1 weights
# Use only to measure frozen generalisation, not live performance
python test_new_data.py --input data/new_day.xlsx --no_learn
```

| Mode | Expected Accuracy | Notes |
|---|---|---|
| `(default)` online learning | 80–85% | Models adapt per tick |
| `--no_learn` frozen | ~58–65% | No intraday adaptation |

### Streamlit Dashboard

```bash
streamlit run dashboard.py
```

### Live API Streaming

```bash
# Requires .env credentials and market hours (Mon–Fri 09:15–15:30 IST)
python regime/live_engine.py
```

---

## 7. Feature Engineering

The Angel One WebSocket Mode 3 delivers **27 raw columns** per tick. From these, `feature_engine.py` computes **46 features** across four groups.

> All prices arrive in paise and are converted to rupees at the `parse_tick()` boundary before any feature computation.

### Group 1 — Price Features (5 features)

| Feature | Formula | Purpose |
|---|---|---|
| `mid_price` | `(ask_price_1 + bid_price_1) / 2` | True fair value; avoids LTP noise |
| `spread` | `ask_price_1 − bid_price_1` | Liquidity and transaction cost signal |
| `spread_pct` | `spread / ask_price_1 × 100` | Normalised spread |
| `vwmp` | `(bid_p1 × ask_q1 + ask_p1 × bid_q1) / (bid_q1 + ask_q1)` | Pressure-adjusted fair value |
| `ltp_mid_delta` | `ltp − mid_price` | Direction of most recent informed order |

### Group 2 — Order Book Features (11 features)

| Feature | Purpose |
|---|---|
| `obi_l1` | Level-1 Order Book Imbalance ∈ [−1, +1] |
| `obi` | Full 5-level OBI — strongest short-term directional signal |
| `depth_ratio` | Total bid qty / ask qty — structural dominance |
| `ask_depth` / `bid_depth` | Book slope — measures market impact |
| `vwap_pressure` | Asymmetry in liquidity depth across all levels |
| `bid_concentration` / `ask_concentration` | Aggressiveness of top-of-book buyers/sellers |
| `fill_price_buy` | Simulated 500-share market order execution price |

### Group 3 — Support & Resistance Features (6 features)

Live support/resistance based on where the largest orders are *right now*, not historical chart levels:

- `support_price` — bid price at the largest bid quantity
- `resistance_price` — ask price at the largest ask quantity
- `dist_to_support`, `dist_to_resistance` — distance from mid price
- `support_strength`, `resistance_strength` — rolling mean of wall size (w=50)

### Group 4 — Multi-Scale Rolling Features (25 features)

Five metrics computed at five tick windows `w ∈ {1, 5, 15, 30, 60}`:

| Metric | Captures |
|---|---|
| `realized_vol_{w}` | Volatility regime signal at each timescale |
| `rolling_return_{w}` | Directional momentum at each timescale |
| `obi_mean_{w}` | Sustained directional bias (noise-filtered OBI) |
| `spread_vol_{w}` | Liquidity regime indicator |
| `volume_spike_{w}` | Abnormal volume activity preceding price moves |

**Why five windows:** A 60-tick window captures the macro trend. A 5-tick window captures current momentum. A 1-tick window captures microstructure noise. Using only one window means the model is blind to everything happening at other scales.

### Target

```
target_t = 1 if mid_price_(t+30) > mid_price_t else 0
```

Binary UP/DOWN at a **30-tick horizon** — the empirically supported sweet spot where OBI has predictive power but autocorrelation noise is sufficiently dampened.

---

## 8. Regime Detection

A single model trained on all market conditions learns the *average* of all regimes and performs poorly in each. The solution is a two-stage pipeline:

### Stage 1 — Gaussian Mixture Model (GMM)

GMM is run **once offline** on Day 1 data to generate regime labels. It fits K=3 Gaussian components to `[realized_vol_60, rolling_return_60]` and assigns:

| Cluster | Label | Regime ID |
|---|---|---|
| Highest realized volatility | Volatile | 2 |
| Most negative rolling return (of remaining) | Bearish | 1 |
| Remaining | Bullish | 0 |

GMM was chosen over K-Means (which assumes spherical clusters) and HMM (which is not online and struggles with sudden regime changes like news events).

### Stage 2 — Adaptive Random Forest (ARF)

ARF is trained **online** using GMM labels as supervision. It detects regime in real time on new ticks without retraining.

**Input features for ARF:**
`realized_vol_60`, `realized_vol_15`, `rolling_return_60`, `rolling_return_15`, `obi_mean_60`, `spread_vol_60`

ARF advantages over simpler alternatives:
- Truly online — updates per tick via `learn_one()`
- Non-linear — captures complex regime boundaries
- Self-replacing — automatically replaces its weakest tree when ADWIN detects drift
- No manual monitoring required

---

## 9. Online Forecasting Models

Three models with **fundamentally different inductive biases** run in parallel on every tick:

| Model | Core Assumption | Wins When |
|---|---|---|
| **FTRL** | Linear, per-feature adaptive learning rates | Trending regimes; OBI and momentum signals are consistent |
| **Hoeffding Tree** | Non-linear decision boundaries; feature interactions matter | Regime boundaries; single features alone are insufficient |
| **PA Classifier** | Update only on mistakes; do nothing on correct predictions | Low-volatility regimes; conservative updating prevents noise chasing |

### FTRL (Follow The Regularized Leader)

Google's production algorithm for real-time prediction (originally for ad click prediction). Key properties:
- Per-feature adaptive learning rates — noisy features (OBI) get smaller rates automatically
- L1 regularization produces sparse weights — automatic feature selection on the fly
- Continuous updates on every tick — builds strong signal in trending regimes

### Hoeffding Tree (VFDT)

Builds incrementally from a stream using the Hoeffding bound to determine statistically confident splits. Captures non-linear feature interactions that linear models miss (e.g. `OBI > 0.3 AND spread_vol < 0.05` predicts UP; `OBI > 0.3 AND spread_vol > 0.05` does not).

> **Note:** Returns `None` for the first 50 predictions (grace period). The ensemble defaults these to 0.

### Passive-Aggressive Classifier (PA)

Updates only when a mistake is made or the margin is insufficient. In low-volatility regimes where signal is weak, PA stabilises weights instead of chasing noise.

> **Warning on PA's 96% accuracy:** PA alone achieves 95–96% accuracy by memorising intraday serial correlation within a single trading day. Frozen inference on a new day drops to 58%. **The meaningful number is the ensemble (83–85%), not PA in isolation.**

### Input Features for All Three Forecasters (13 total)

`obi_1`, `obi_5`, `obi_15`, `spread_vol_5`, `spread_vol_15`, `ltp_mid_delta`, `vwmp`, `rolling_return_5`, `rolling_return_15`, `dist_to_support`, `dist_to_resistance`, `support_strength`, `resistance_strength`

---

## 10. Dynamic Ensemble & Drift Detection

### Two-Stage Weight Computation

**Stage 1 — Regime Prior** (initial weights based on regime):

| Regime | FTRL | Hoeffding | PA |
|---|---|---|---|
| Bullish (0) | 0.6 | 0.3 | 0.1 |
| Bearish (1) | 0.3 | 0.5 | 0.2 |
| Volatile (2) | 0.2 | 0.3 | 0.5 |

**Stage 2 — Rolling Accuracy Adjustment** (over last 100 ticks):

Final weights are a 50/50 blend of regime priors and rolling accuracy-normalised weights. This means the ensemble continuously re-evaluates which model deserves trust *right now*, not just based on historical assumptions.

**Final prediction:**
```
score = w_ftrl × ŷ_ftrl + w_hoeff × ŷ_hoeff + w_pa × ŷ_pa
ŷ_final = 1 if score ≥ 0.5 else 0
```

### ADWIN Drift Detection

ADWIN (Adaptive Windowing) maintains a variable-size window of recent error observations and uses a statistical test to detect when the recent error rate differs significantly from the older half of the window.

**On drift — boost learning rates, do not reset:**

Resetting throws away everything the model learned. Instead:
- FTRL α: `0.1 → 0.3` (faster adaptation)
- PA C: `0.1 → 0.5` (more aggressive on mistakes)
- ARF: replaces its weakest tree automatically

This keeps prior knowledge as a starting point while weighting new observations more heavily — fast adaptation without a blind period.

---

## 11. Results

### Day 1 (May 13 2026) — Training & Evaluation

| Model | Accuracy |
|---|---|
| Naive (always UP) | 41.15% |
| Hoeffding Tree alone | 60.90% |
| FTRL alone | 73.70% |
| **Ensemble** | **85.04%** |
| PA alone* | 96.01%* |

*See PA warning above

**Per-Regime Accuracy:**

| Regime | Ticks | Accuracy |
|---|---|---|
| Bullish | 0 | — (not observed on this day) |
| Bearish | 1,119 | 82.75% |
| Volatile | 5,824 | 85.47% |

**Latency:**

| Metric | Value |
|---|---|
| Mean | 315,336 ns (0.315 ms) |
| p50 | 289,900 ns |
| p99 | 643,164 ns |
| Max | 971,900 ns |
| WebSocket tick interval | 100–500 ms |

All predictions complete well within the WebSocket tick interval.

### Day 2 (May 14 2026) — Out-of-Sample Generalisation

| | Day 1 | Day 2 |
|---|---|---|
| Naive baseline | 41.15% | 41.88% |
| **Ensemble accuracy** | **85.04%** | **83.41%** |
| Accuracy drop | — | −1.63 pp |
| Mean latency | 0.315 ms | 0.152 ms |
| Drift events | 7 | 9 |

**Day 2 Classification Report:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DOWN | 0.8401 | 0.8827 | 0.8608 | 5,957 |
| UP | 0.8249 | 0.7668 | 0.7948 | 4,293 |
| **Accuracy** | | | **0.8341** | 10,250 |
| Macro avg | 0.8325 | 0.8247 | 0.8278 | 10,250 |

The model generalises. Only 1.63 percentage points were lost moving to a completely unseen trading session.

---

## 12. Design Decisions

### Why Time-Ordered Train/Test Split

A random split leaks future data into training — the morning of a trading day would be in test while the afternoon is in training, meaning the model has "seen the future." This inflates accuracy by 5–15 percentage points and produces numbers that are entirely fictitious in live trading.

**Split used:** First 80% of rows for training, last 20% for testing.

### Why Binary Target Instead of 3-Class

The original pipeline used 3-class labels (UP / NEUTRAL / DOWN). At H=50, 81% of rows were NEUTRAL, making the actionable minority classes nearly impossible to predict (macro F1 ≈ 0.33 ≈ random chance).

Binary UP/DOWN removes the NEUTRAL class. The resulting 44%/56% split is tractable for all three online models.

### Why Python and Not C++

Python inference averages 315µs (Day 1) and 152µs (Day 2). For a WebSocket feed with 100–500ms tick intervals, this is entirely acceptable.

A production C++ engine (`pa_engine.cpp`) using AVX2 SIMD achieves sub-microsecond inference and is relevant for co-location deployments requiring tick-to-order latency under 10µs — that is future work.

### Why Online Models and Not LightGBM/XGBoost

Offline benchmark results (macro F1 on 3-class formulation):

| Algorithm | H=50 | H=100 | H=200 | H=300 |
|---|---|---|---|---|
| LightGBM | 0.3148 | 0.3025 | 0.3276 | 0.3032 |
| XGBoost | 0.3106 | 0.2997 | 0.3099 | 0.2963 |
| CatBoost | 0.2549 | 0.2890 | 0.2839 | 0.2690 |

All scores ≈ 0.33 = random chance for 3 classes. Additionally, these models cannot update per tick without full retraining. The online binary formulation achieves 85% accuracy vs 41% naive baseline.

---

## 13. Limitations & Future Work

### Honest Limitations

1. **Two days of data** — The system has been validated on May 13–14 2026 only. Longer-term generalisation across varied market conditions is unproven.
2. **No Bullish regime observed** — Both test days were bearish/volatile. Ensemble weights for Bullish regime (FTRL=0.6) are untested assumptions.
3. **Online learning required for cross-day inference** — Frozen (`--no_learn`) inference on a new day produces ~58% accuracy. Models need per-tick updates to adapt to a new session's rolling feature distribution.
4. **PA's 96% accuracy is not a generalizable result** — It reflects memorised intraday autocorrelation, not genuine predictive skill. The ensemble (83–85%) is the meaningful metric.
5. **5-level book only** — Institutional players with 20+ level access have an information advantage.
6. **No alternative data** — No news, FII/DII flow, options data, or sentiment. Purely technical.
7. **Single symbol** — Trained and tested on RELIANCE-EQ only.

### Future Work (Priority Order)

1. Collect 20+ days of data — expose all three regimes to GMM, validate generalisation across varied market conditions
2. Implement walk-forward cross-validation — expanding window retraining for robust out-of-sample estimates
3. LinUCB contextual bandit — replace hardcoded regime-to-weight mapping with a learned selector
4. C++ inference for co-location deployments (extend existing `pa_engine.cpp`)
5. Add signed order flow imbalance (OFI) and trade tick features
6. Multi-symbol generalisation — train on RELIANCE, evaluate on TCS/HDFC
7. 2-state HMM on (OBI, log-spread) to replace GMM once sufficient data is available
8. PnL simulation with realistic transaction costs to convert directional accuracy into a tradeable signal assessment

---

## 14. Live API Setup

Live streaming requires a valid Angel One SmartAPI account and a `.env` file:

```env
ANGEL_API_KEY=your_api_key_here
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

The `smartapi/` package handles authentication, TOTP generation, rate limiting, and WebSocket connection management.

```bash
# Start live streaming (market hours only: Mon–Fri 09:15–15:30 IST)
python regime/live_engine.py
```

The live engine streams Mode 3 tick data, computes features on every tick, detects the current regime, and outputs UP/DOWN predictions in real time.

---

## Data Collected (May 13 2026)

| Symbol | Company | Token | Rows |
|---|---|---|---|
| RELIANCE-EQ | Reliance Industries | 2885 | 34,743 |
| HDFCBANK-EQ | HDFC Bank | 3045 | 35,934 |
| TCS-EQ | Tata Consultancy Services | 11536 | 27,943 |
| INFY-EQ | Infosys | 1594 | 23,149 |
| ICICIBANK-EQ | ICICI Bank | 1660 | 14,155 |
| **Total** | | | **135,924** |

All modelling uses RELIANCE-EQ only (most liquid name in the set).

---

*Internal research document — DOM Trade, May 2026*
