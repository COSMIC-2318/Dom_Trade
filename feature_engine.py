"""
feature_engine.py — DOM Trade
==============================
Layer A: Feature Engineering on raw orderbook data.
Filters to RELIANCE-EQ, computes 46 features across 4 groups,
generates binary target (30-tick horizon), exports to parquet.

Run:
    python feature_engine.py --input data/orderbook.csv.xlsx --output reliance_features.parquet
"""

import argparse
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

SYMBOL      = "RELIANCE-EQ"
HORIZON     = 30
FILL_SHARES = 500
WINDOWS_S   = [1, 5, 15, 30, 60]


def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["avg_price"], inplace=True)
    df = df[df["symbol"] == SYMBOL].copy()
    df = df.sort_values("time").reset_index(drop=True)
    print(f"[load] {len(df):,} rows for {SYMBOL}")
    print(f"[load] time range: {df['time'].min()} → {df['time'].max()}")
    return df


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_price"]     = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    df["spread"]        = df["ask_price_1"] - df["bid_price_1"]
    df["spread_pct"]    = (df["spread"] / df["ask_price_1"].replace(0, np.nan)) * 100.0
    df["ltp_mid_delta"] = df["ltp"] - df["mid_price"]
    num        = df["bid_price_1"] * df["ask_qty_1"] + df["ask_price_1"] * df["bid_qty_1"]
    den        = (df["bid_qty_1"] + df["ask_qty_1"]).replace(0, np.nan)
    df["vwmp"] = num / den
    return df


def orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    df["obi_l1"] = (
        (df["bid_qty_1"] - df["ask_qty_1"]) /
        (df["bid_qty_1"] + df["ask_qty_1"]).replace(0, np.nan)
    )
    bid_sum = sum(df[f"bid_qty_{i}"].fillna(0) for i in range(1, 6))
    ask_sum = sum(df[f"ask_qty_{i}"].fillna(0) for i in range(1, 6))
    df["obi"]         = (bid_sum - ask_sum) / (bid_sum + ask_sum).replace(0, np.nan)
    df["depth_ratio"] = bid_sum / ask_sum.replace(0, np.nan)
    df["ask_depth"]   = df["ask_price_5"] - df["ask_price_1"]
    df["bid_depth"]   = df["bid_price_1"] - df["bid_price_5"]
    ask_vwap = sum(df[f"ask_price_{i}"] * df[f"ask_qty_{i}"].fillna(0) for i in range(1, 6)) / ask_sum.replace(0, np.nan)
    bid_vwap = sum(df[f"bid_price_{i}"] * df[f"bid_qty_{i}"].fillna(0) for i in range(1, 6)) / bid_sum.replace(0, np.nan)
    df["vwap_pressure"]     = (ask_vwap - bid_vwap) / df["mid_price"].replace(0, np.nan)
    mean_bid                = pd.concat([df[f"bid_qty_{i}"] for i in range(1, 6)], axis=1).mean(axis=1).replace(0, np.nan)
    mean_ask                = pd.concat([df[f"ask_qty_{i}"] for i in range(1, 6)], axis=1).mean(axis=1).replace(0, np.nan)
    df["bid_concentration"] = df["bid_qty_1"] / mean_bid
    df["ask_concentration"] = df["ask_qty_1"] / mean_ask
    df["fill_price_buy"]    = _fill_vectorized(df, "buy")
    df["fill_price_sell"]   = _fill_vectorized(df, "sell")
    return df


def _fill_vectorized(df: pd.DataFrame, side: str) -> pd.Series:
    prefix     = "ask" if side == "buy" else "bid"
    total_cost = pd.Series(0.0, index=df.index)
    remaining  = pd.Series(float(FILL_SHARES), index=df.index)
    for i in range(1, 6):
        px  = df[f"{prefix}_price_{i}"].fillna(0)
        qty = df[f"{prefix}_qty_{i}"].fillna(0)
        filled      = np.minimum(qty, remaining)
        total_cost += filled * px
        remaining  -= filled
    px5         = df[f"{prefix}_price_5"].fillna(df[f"{prefix}_price_1"].fillna(0))
    total_cost += remaining * px5
    return total_cost / FILL_SHARES


def support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    bid_qtys   = df[[f"bid_qty_{i}"   for i in range(1, 6)]].values
    bid_prices = df[[f"bid_price_{i}" for i in range(1, 6)]].values
    ask_qtys   = df[[f"ask_qty_{i}"   for i in range(1, 6)]].values
    ask_prices = df[[f"ask_price_{i}" for i in range(1, 6)]].values
    si = np.argmax(bid_qtys, axis=1)
    ri = np.argmax(ask_qtys, axis=1)
    df["support_price"]       = bid_prices[np.arange(len(df)), si]
    df["resistance_price"]    = ask_prices[np.arange(len(df)), ri]
    df["dist_to_support"]     = df["ltp"] - df["support_price"]
    df["dist_to_resistance"]  = df["resistance_price"] - df["ltp"]
    df["support_strength"]    = pd.Series(bid_qtys[np.arange(len(df)), si], index=df.index).rolling(50, min_periods=1).mean()
    df["resistance_strength"] = pd.Series(ask_qtys[np.arange(len(df)), ri], index=df.index).rolling(50, min_periods=1).mean()
    return df


def multiscale_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in WINDOWS_S:
        s  = f"_{w}t"
        mp = min(2, w) if w > 1 else 1
        df[f"realized_vol{s}"]   = df["mid_price"].rolling(max(w, 2), min_periods=mp).std().fillna(0)
        df[f"rolling_return{s}"] = ((df["mid_price"] - df["mid_price"].shift(w)) / df["mid_price"].shift(w).replace(0, np.nan)).fillna(0)
        df[f"obi_mean{s}"]       = df["obi"].rolling(w, min_periods=1).mean()
        df[f"spread_vol{s}"]     = df["spread"].rolling(max(w, 2), min_periods=mp).std().fillna(0)
        vol_r                    = df["volume"].rolling(w, min_periods=1)
        df[f"volume_spike{s}"]   = vol_r.sum() - vol_r.mean()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = (df["mid_price"].shift(-HORIZON) > df["mid_price"]).astype("Int8")
    df = df.iloc[:-HORIZON].copy()
    print(f"[target] horizon={HORIZON} ticks | UP={df['target'].sum():,} | DOWN={(df['target']==0).sum():,}")
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    cut   = int(len(df) * (1 - test_frac))
    train = df.iloc[:cut].copy()
    test  = df.iloc[cut:].copy()
    print(f"[split] train={len(train):,} | test={len(test):,}")
    return train, test


def quality_check(df: pd.DataFrame):
    null_max = df.isnull().mean().max() * 100
    print(f"[quality] max_null={null_max:.2f}% | spread<=0: {(df['spread']<=0).sum()} | OBI=[{df['obi'].min():.3f}, {df['obi'].max():.3f}]")


def run(input_path: str, output_path: str) -> pd.DataFrame:
    df = load(input_path)
    print("[features] price...")
    df = price_features(df)
    print("[features] orderbook...")
    df = orderbook_features(df)
    print("[features] support/resistance...")
    df = support_resistance_features(df)
    print("[features] multi-scale rolling...")
    df = multiscale_features(df)
    print("[target] generating...")
    df = add_target(df)
    quality_check(df)
    train, test = time_split(df)
    df.to_parquet(output_path, index=False)
    print(f"\n[DONE] {df.shape[0]:,} rows x {df.shape[1]} columns → {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/orderbook.csv.xlsx",   help="Path to xlsx file")
    parser.add_argument("--output", default="reliance_features.parquet", help="Output parquet path")
    args = parser.parse_args()
    run(args.input, args.output)