import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import ta
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, ".yfinance_cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(CACHE_DIR)

DEFAULT_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "AMD",
    "SPY", "QQQ"
]

def fetch_symbol_ohlcv(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    print(f"[*] Fetching {symbol} ({interval}, period={period})...")
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty or len(data) < 60:
        print(f"    [!] Warning: Insufficient or empty data for {symbol}")
        return pd.DataFrame()

    df = data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]

    df = df.rename(
        columns={
            "Date": "time",
            "Datetime": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)

    df["symbol"] = symbol
    df = df[["time", "symbol", "open", "high", "low", "close", "volume"]].dropna()
    return df.sort_values("time").reset_index(drop=True)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Price Returns & Momentum
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["roc_12"] = ta.momentum.roc(df["close"], window=12)
    
    # MACD
    macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # 2. Moving Averages & Trend Relationships
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
    
    df["dist_ema_20"] = (df["close"] - df["ema_20"]) / (df["ema_20"] + 1e-12)
    df["dist_ema_50"] = (df["close"] - df["ema_50"]) / (df["ema_50"] + 1e-12)
    df["dist_ema_200"] = (df["close"] - df["ema_200"]) / (df["ema_200"] + 1e-12)
    df["ema_trend_ratio"] = df["ema_20"] / (df["ema_50"] + 1e-12)
    
    # 3. Volatility & Compression (Squeeze)
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_ratio"] = df["atr_14"] / (df["atr_14"].rolling(50).mean() + 1e-12)
    
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pband"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()
    
    df["adx_14"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # 4. Volume & Flow Metrics
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / (df["vol_ma_20"] + 1e-12)
    
    # 5. Support / Resistance & Breakout
    df["highest_20"] = df["high"].rolling(20).max()
    df["lowest_20"] = df["low"].rolling(20).min()
    df["dist_to_20d_high"] = (df["highest_20"] - df["close"]) / (df["close"] + 1e-12)

    return df

def generate_triple_barrier_labels(
    df: pd.DataFrame, 
    tp_pct: float = 0.10, 
    sl_pct: float = -0.04, 
    max_holding_bars: int = 15
) -> pd.DataFrame:
    df = df.copy()
    targets = []
    forward_max_returns = []
    forward_min_drawdowns = []

    n = len(df)
    for i in range(n):
        if i + max_holding_bars >= n:
            targets.append(np.nan)
            forward_max_returns.append(np.nan)
            forward_min_drawdowns.append(np.nan)
            continue

        entry_price = df["close"].iloc[i]
        future_highs = df["high"].iloc[i + 1 : i + 1 + max_holding_bars]
        future_lows = df["low"].iloc[i + 1 : i + 1 + max_holding_bars]

        max_up = (future_highs.max() - entry_price) / entry_price
        max_down = (future_lows.min() - entry_price) / entry_price

        forward_max_returns.append(max_up)
        forward_min_drawdowns.append(max_down)

        hit_tp = False
        hit_sl = False
        for bar_idx in range(i + 1, i + 1 + max_holding_bars):
            bar_high_pct = (df["high"].iloc[bar_idx] - entry_price) / entry_price
            bar_low_pct = (df["low"].iloc[bar_idx] - entry_price) / entry_price

            if bar_low_pct <= sl_pct:
                hit_sl = True
                break
            if bar_high_pct >= tp_pct:
                hit_tp = True
                break

        if hit_tp and not hit_sl:
            targets.append(1)
        else:
            targets.append(0)

    df["target"] = targets
    df["forward_max_return"] = forward_max_returns
    df["forward_max_drawdown"] = forward_min_drawdowns
    return df

def build_full_dataset(
    symbols=None, 
    period: str = "5y", 
    interval: str = "1d",
    tp_pct: float = 0.10, 
    sl_pct: float = -0.04, 
    max_holding_bars: int = 15
) -> pd.DataFrame:
    symbols = symbols or DEFAULT_UNIVERSE
    processed_dfs = []

    for symbol in symbols:
        raw_df = fetch_symbol_ohlcv(symbol, period=period, interval=interval)
        if raw_df.empty:
            continue
        
        feature_df = calculate_indicators(raw_df)
        labeled_df = generate_triple_barrier_labels(
            feature_df, tp_pct=tp_pct, sl_pct=sl_pct, max_holding_bars=max_holding_bars
        )
        
        cleaned = labeled_df.dropna().reset_index(drop=True)
        print(f"    -> Processed {len(cleaned)} rows for {symbol} | Win rate (Class 1): {cleaned['target'].mean()*100:.1f}%")
        processed_dfs.append(cleaned)

    if not processed_dfs:
        print("[!] No data was successfully processed.")
        return pd.DataFrame()

    full_df = pd.concat(processed_dfs, ignore_index=True)
    return full_df

def main():
    parser = argparse.ArgumentParser(description="Stock Data Fetcher & Feature Pipeline for ML Trading")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE, help="List of stock tickers")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period")
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval")
    parser.add_argument("--tp", type=float, default=0.10, help="Take profit threshold percentage")
    parser.add_argument("--sl", type=float, default=-0.04, help="Stop loss threshold percentage")
    parser.add_argument("--hold", type=int, default=15, help="Max holding bars")
    args = parser.parse_args()

    print("=" * 60)
    print(" STOCK ML DATASET BUILDER (SWING TRADING)")
    print(f" Symbols: {args.symbols}")
    print(f" Period: {args.period} | Interval: {args.interval}")
    print(f" Target: Take Profit +{args.tp*100:.1f}% | Stop Loss {args.sl*100:.1f}% | Max Hold: {args.hold} bars")
    print("=" * 60)

    dataset = build_full_dataset(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        tp_pct=args.tp,
        sl_pct=args.sl,
        max_holding_bars=args.hold
    )

    if dataset.empty:
        print("[!] Dataset generation failed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"swing_dataset_{timestamp}.csv")
    parquet_path = os.path.join(DATA_DIR, "swing_dataset_latest.parquet")

    dataset.to_csv(csv_path, index=False)
    dataset.to_parquet(parquet_path, index=False)

    print("\n" + "=" * 60)
    print(" DATASET SUMMARY")
    print(f" Total Rows (Data points): {len(dataset):,}")
    print(f" Total Features: {len(dataset.columns)}")
    print(f" Win Class (Target=1): {(dataset['target'] == 1).sum():,} ({dataset['target'].mean()*100:.2f}%)")
    print(f" Loss/Timeout Class (Target=0): {(dataset['target'] == 0).sum():,} ({(1 - dataset['target'].mean())*100:.2f}%)")
    print(f" Saved CSV: {csv_path}")
    print(f" Saved Parquet: {parquet_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
