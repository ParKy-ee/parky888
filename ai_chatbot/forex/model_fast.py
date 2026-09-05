import os
import json
import shutil
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

MODEL_VERSION = "0.30-two-layer-quality-exits"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
HUMAN_LOG_SCORE_ONLY = True
CONSOLE_VERBOSE = False

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data", "forex_intraday")
NEWS_BLACKOUT_PATH = os.path.join(BASE_DIR, "data", "news_blackout.csv")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

YFINANCE_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

DAYTRADE_SYMBOLS = [
    "USDJPY",
    "EURUSD",
    "GBPUSD",
]
DAYTRADE_INTERVALS = ["5m", "15m"]

LOCAL_CSV_FIRST = True
YFINANCE_PERIOD_BY_INTERVAL = {
    "5m": "60d",
    "15m": "60d",
}

ACCOUNT_EQUITY = 10000
RISK_PER_TRADE = 0.002
MAX_DAILY_LOSS = 0.006
MAX_TRADES_PER_DAY = 5
MIN_TRADES_PER_DAY_TARGET = 1
MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_BARS = {
    "5m": 18,
    "15m": 8,
}
MIN_BARS_BETWEEN_TRADES = {
    "5m": 4,
    "15m": 2,
}
MAX_HOLD_BARS = {
    "5m": 18,
    "15m": 10,
}

SESSION_HOURS_BKK = set(range(13, 24)) | {0, 1}
ROLLOVER_HOURS_BKK = {4, 5, 6}

EXPERIMENT_MODES = {
    "EXP_01_RAW_LOOSE": {
        "enabled": True,
        "scope": "research_only",
        "adx_min": 15,
        "atr_ratio_min": 0.70,
        "atr_ratio_max": 2.60,
        "body_ratio_min": 0.30,
        "quality_score_min": 0,
        "allowed_hours_bkk": sorted(SESSION_HOURS_BKK),
        "use_session_filter": False,
        "use_regime_filter": False,
        "use_htf_trend_filter": False,
        "use_quality_filter": False,
        "use_dynamic_targets": False,
        "use_breakeven": False,
        "use_partial_tp": False,
        "max_trades_per_day": 5,
    },
    "EXP_02_SESSION": {
        "enabled": True,
        "scope": "candidate",
        "adx_min": 15,
        "atr_ratio_min": 0.70,
        "atr_ratio_max": 2.60,
        "body_ratio_min": 0.30,
        "quality_score_min": 0,
        "allowed_hours_bkk": [14, 15, 16, 19, 20, 21, 22],
        "use_session_filter": True,
        "use_regime_filter": False,
        "use_htf_trend_filter": False,
        "use_quality_filter": False,
        "use_dynamic_targets": False,
        "use_breakeven": False,
        "use_partial_tp": False,
        "max_trades_per_day": 3,
    },
    "EXP_03_SESSION_HTF": {
        "enabled": True,
        "scope": "candidate",
        "adx_min": 15,
        "atr_ratio_min": 0.70,
        "atr_ratio_max": 2.60,
        "body_ratio_min": 0.30,
        "quality_score_min": 0,
        "allowed_hours_bkk": [14, 15, 16, 19, 20, 21, 22],
        "use_session_filter": True,
        "use_regime_filter": False,
        "use_htf_trend_filter": True,
        "htf_adx_min": 18,
        "use_quality_filter": False,
        "use_dynamic_targets": False,
        "use_breakeven": False,
        "use_partial_tp": False,
        "max_trades_per_day": 3,
    },
    "EXP_04_QUALITY": {
        "enabled": True,
        "scope": "candidate",
        "adx_min": 15,
        "atr_ratio_min": 0.70,
        "atr_ratio_max": 2.60,
        "body_ratio_min": 0.30,
        "quality_score_min": 5,
        "allowed_hours_bkk": [14, 15, 16, 19, 20, 21, 22],
        "use_session_filter": True,
        "use_regime_filter": True,
        "use_htf_trend_filter": True,
        "htf_adx_min": 18,
        "use_quality_filter": True,
        "use_dynamic_targets": False,
        "use_breakeven": False,
        "use_partial_tp": False,
        "max_trades_per_day": 3,
    },
    "EXP_05_DYNAMIC_EXIT": {
        "enabled": True,
        "scope": "candidate",
        "adx_min": 15,
        "atr_ratio_min": 0.70,
        "atr_ratio_max": 2.60,
        "body_ratio_min": 0.30,
        "quality_score_min": 5,
        "allowed_hours_bkk": [14, 15, 16, 19, 20, 21, 22],
        "use_session_filter": True,
        "use_regime_filter": True,
        "use_htf_trend_filter": True,
        "htf_adx_min": 18,
        "use_quality_filter": True,
        "use_dynamic_targets": True,
        "use_breakeven": False,
        "use_partial_tp": False,
        "max_trades_per_day": 3,
    },
    "EXP_06_BE_PARTIAL": {
        "enabled": True,
        "scope": "candidate",
        "adx_min": 15,
        "atr_ratio_min": 0.70,
        "atr_ratio_max": 2.60,
        "body_ratio_min": 0.30,
        "quality_score_min": 5,
        "allowed_hours_bkk": [14, 15, 16, 19, 20, 21, 22],
        "use_session_filter": True,
        "use_regime_filter": True,
        "use_htf_trend_filter": True,
        "htf_adx_min": 18,
        "use_quality_filter": True,
        "use_dynamic_targets": True,
        "use_breakeven": True,
        "use_partial_tp": True,
        "max_trades_per_day": 3,
    },
}

PAIR_CONFIG = {
    "USDJPY": {
        "pip_size": 0.01,
        "estimated_spread_pips": 1.1,
        "slippage_pips": 0.4,
        "min_adx": 18,
        "min_atr_ratio": 0.75,
        "max_atr_ratio": 2.40,
        "tp_atr": 0.95,
        "sl_atr": 0.65,
    },
    "EURUSD": {
        "pip_size": 0.0001,
        "estimated_spread_pips": 0.9,
        "slippage_pips": 0.3,
        "min_adx": 17,
        "min_atr_ratio": 0.70,
        "max_atr_ratio": 2.35,
        "tp_atr": 0.90,
        "sl_atr": 0.65,
    },
    "GBPUSD": {
        "pip_size": 0.0001,
        "estimated_spread_pips": 1.2,
        "slippage_pips": 0.4,
        "min_adx": 18,
        "min_atr_ratio": 0.75,
        "max_atr_ratio": 2.50,
        "tp_atr": 0.95,
        "sl_atr": 0.70,
    },
}

DEFAULT_PAIR_CONFIG = {
    "pip_size": 0.0001,
    "estimated_spread_pips": 1.5,
    "slippage_pips": 0.5,
    "min_adx": 18,
    "min_atr_ratio": 0.75,
    "max_atr_ratio": 2.40,
    "tp_atr": 0.90,
    "sl_atr": 0.70,
}


def pair_config(symbol):
    return PAIR_CONFIG.get(symbol, DEFAULT_PAIR_CONFIG)


def normalize_forex_symbol(symbol):
    symbol = symbol.replace("/", "").replace("-", "").upper()
    return symbol if symbol.endswith("=X") else f"{symbol}=X"


def clean_symbol(symbol):
    return symbol.replace("=X", "").replace("/", "").replace("-", "").upper()


def estimated_cost_pct(symbol, price):
    cfg = pair_config(symbol)
    spread = cfg["estimated_spread_pips"] * cfg["pip_size"]
    slippage = cfg["slippage_pips"] * cfg["pip_size"]
    round_trip_cost = spread + (2 * slippage)
    return round_trip_cost / np.maximum(price, 1e-12)


def load_local_intraday_csv(symbol, interval):
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    rename_map = {
        "datetime": "time",
        "date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    required = ["time", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Local CSV missing columns for {symbol} {interval}: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    df["symbol"] = symbol
    df["interval"] = interval
    df["data_provider"] = "local_csv"
    return df.sort_values("time").reset_index(drop=True)


def fetch_yfinance_intraday(symbol, interval):
    yf_symbol = normalize_forex_symbol(symbol)
    period = YFINANCE_PERIOD_BY_INTERVAL.get(interval, "60d")

    if CONSOLE_VERBOSE and not HUMAN_LOG_SCORE_ONLY:
        print(f"Fetching {symbol} ({yf_symbol}) interval={interval} period={period}")

    data = yf.download(
        tickers=yf_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return None

    df = data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]

    df = df.rename(columns={
        "Datetime": "time",
        "Date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    required = ["time", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from yfinance for {symbol}: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    df["symbol"] = symbol
    df["interval"] = interval
    df["data_provider"] = "yfinance_fallback"
    return df.sort_values("time").reset_index(drop=True)


def fetch_intraday_data(symbol, interval):
    symbol = clean_symbol(symbol)
    if LOCAL_CSV_FIRST:
        local_df = load_local_intraday_csv(symbol, interval)
        if local_df is not None:
            return local_df

    return fetch_yfinance_intraday(symbol, interval)


def load_news_blackout():
    if not os.path.exists(NEWS_BLACKOUT_PATH):
        return pd.DataFrame(columns=["time", "currency", "impact", "minutes_before", "minutes_after"])

    df = pd.read_csv(NEWS_BLACKOUT_PATH)
    if "time" not in df.columns:
        return pd.DataFrame(columns=["time", "currency", "impact", "minutes_before", "minutes_after"])

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)

    for col, default in [("currency", ""), ("impact", "high"), ("minutes_before", 30), ("minutes_after", 30)]:
        if col not in df.columns:
            df[col] = default

    return df


def symbol_currencies(symbol):
    symbol = clean_symbol(symbol)
    if len(symbol) != 6:
        return set()
    return {symbol[:3], symbol[3:]}


def mark_news_blackout(df, symbol, news_df):
    df = df.copy()
    df["news_blackout"] = False
    if news_df.empty:
        return df

    currencies = symbol_currencies(symbol)
    relevant = news_df[
        news_df["currency"].astype(str).str.upper().isin(currencies) |
        (news_df["currency"].astype(str).str.upper() == "ALL")
    ].copy()
    if relevant.empty:
        return df

    blackout = np.zeros(len(df), dtype=bool)
    times = df["time"]
    for _, event in relevant.iterrows():
        before = int(event.get("minutes_before", 30))
        after = int(event.get("minutes_after", 30))
        start = event["time"] - timedelta(minutes=before)
        end = event["time"] + timedelta(minutes=after)
        blackout |= ((times >= start) & (times <= end)).to_numpy()

    df["news_blackout"] = blackout
    return df


def add_higher_timeframe_trend(df):
    source = df[["time", "open", "high", "low", "close", "volume"]].copy()
    htf = (
        source
        .set_index("time")
        .resample("1h", label="right", closed="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
        .reset_index()
    )
    if len(htf) < 220:
        df["htf_long_allowed"] = False
        df["htf_short_allowed"] = False
        df["htf_adx"] = 0.0
        return df

    htf["htf_ema50"] = ta.trend.ema_indicator(htf["close"], window=50)
    htf["htf_ema200"] = ta.trend.ema_indicator(htf["close"], window=200)
    htf["htf_adx"] = ta.trend.adx(htf["high"], htf["low"], htf["close"], window=14)
    htf["htf_long_allowed"] = (
        (htf["close"] > htf["htf_ema200"]) &
        (htf["htf_ema50"] > htf["htf_ema200"])
    )
    htf["htf_short_allowed"] = (
        (htf["close"] < htf["htf_ema200"]) &
        (htf["htf_ema50"] < htf["htf_ema200"])
    )

    merged = pd.merge_asof(
        df.sort_values("time"),
        htf[["time", "htf_long_allowed", "htf_short_allowed", "htf_adx"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    merged["htf_long_allowed"] = merged["htf_long_allowed"].fillna(False).astype(bool)
    merged["htf_short_allowed"] = merged["htf_short_allowed"].fillna(False).astype(bool)
    merged["htf_adx"] = merged["htf_adx"].fillna(0.0)
    return merged


def calculate_quality_score(row):
    score = 0

    if row["adx"] >= 25:
        score += 2
    elif row["adx"] >= 20:
        score += 1

    if row["ATR_ratio"] >= 1.1:
        score += 2
    elif row["ATR_ratio"] >= 0.9:
        score += 1

    if row["body_ratio"] >= 0.55:
        score += 2
    elif row["body_ratio"] >= 0.40:
        score += 1

    if abs(row["ret_12"]) >= 0.001:
        score += 1

    if row.get("htf_adx", 0) >= 20:
        score += 1

    if row["adx"] < 15:
        score -= 2

    if row["spread_cost_pct"] > 0 and row["ATR"] / row["close"] < row["spread_cost_pct"] * 4:
        score -= 2

    return score


def detect_market_regime(row):
    if row["adx"] >= 25 and row["ATR_ratio"] >= 1.0:
        return "TREND"
    if row["adx"] < 18 and row["ATR_ratio"] < 1.0:
        return "RANGE"
    return "NO_TRADE"


def get_dynamic_targets(row):
    if row["adx"] >= 30 and row["ATR_ratio"] >= 1.2 and row["quality_score"] >= 6:
        return 2.2, 1.1, 12
    if row["adx"] >= 22 and row["quality_score"] >= 5:
        return 1.6, 1.0, 10
    return 1.2, 1.0, 6


def build_features(df, symbol, news_df):
    df = df.copy().sort_values("time").reset_index(drop=True)
    cfg = pair_config(symbol)
    pip_size = cfg["pip_size"]
    spread_price = cfg["estimated_spread_pips"] * pip_size

    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["ATR_mean"] = df["ATR"].rolling(80).mean()
    df["ATR_ratio"] = df["ATR"] / (df["ATR_mean"] + 1e-12)
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["bar_range"] = df["high"] - df["low"]
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["bar_range"] + 1e-12)
    df["hour"] = df["time"].dt.hour
    df["date"] = df["time"].dt.date.astype(str)
    df["in_session"] = df["hour"].isin(SESSION_HOURS_BKK)
    df["rollover_block"] = df["hour"].isin(ROLLOVER_HOURS_BKK)
    df["spread_cost_pct"] = estimated_cost_pct(symbol, df["close"])
    df["spread_ok"] = spread_price <= (df["ATR"] * 0.18)
    df["cost_reward_ok"] = ((df["ATR"] * cfg["tp_atr"]) / df["close"]) >= (df["spread_cost_pct"] * 4)
    df["volatility_ok"] = (
        (df["ATR_ratio"] <= cfg["max_atr_ratio"]) &
        (df["bar_range"] >= spread_price * 2.5)
    )
    df = mark_news_blackout(df, symbol, news_df)
    df = add_higher_timeframe_trend(df)
    df = df.dropna().reset_index(drop=True)
    df["quality_score"] = df.apply(calculate_quality_score, axis=1)
    df["market_regime"] = df.apply(detect_market_regime, axis=1)
    return df


def apply_daytrade_strategy(df, symbol, interval, mode_name, mode_config):
    df = df.copy()
    allowed_hours = set(mode_config["allowed_hours_bkk"])

    df["strategy_long"] = False
    df["strategy_short"] = False
    df["strategy_type"] = "None"
    df["filter_reason"] = "NO_RAW_SIGNAL"
    df["experiment_mode"] = mode_name
    df["experiment_scope"] = mode_config["scope"]
    df["mode_adx_min"] = mode_config["adx_min"]
    df["mode_atr_ratio_min"] = mode_config["atr_ratio_min"]
    df["mode_body_ratio_min"] = mode_config["body_ratio_min"]
    df["raw_signal"] = False
    df["raw_strategy_type"] = "None"

    session_ok = df["hour"].isin(allowed_hours) if mode_config["use_session_filter"] else pd.Series(True, index=df.index)
    htf_adx_ok = df["htf_adx"] >= mode_config.get("htf_adx_min", 0)
    htf_long_ok = (df["htf_long_allowed"] & htf_adx_ok) if mode_config["use_htf_trend_filter"] else pd.Series(True, index=df.index)
    htf_short_ok = (df["htf_short_allowed"] & htf_adx_ok) if mode_config["use_htf_trend_filter"] else pd.Series(True, index=df.index)
    quality_ok = (df["quality_score"] >= mode_config["quality_score_min"]) if mode_config["use_quality_filter"] else pd.Series(True, index=df.index)
    trend_regime_ok = (df["market_regime"] == "TREND") if mode_config["use_regime_filter"] else pd.Series(True, index=df.index)
    range_regime_ok = (df["market_regime"] == "RANGE") if mode_config["use_regime_filter"] else pd.Series(True, index=df.index)

    base_filter = (
        session_ok &
        ~df["rollover_block"] &
        ~df["news_blackout"] &
        df["spread_ok"] &
        df["cost_reward_ok"] &
        (df["ATR_ratio"] >= mode_config["atr_ratio_min"]) &
        (df["ATR_ratio"] <= mode_config["atr_ratio_max"]) &
        df["volatility_ok"] &
        (df["adx"] >= mode_config["adx_min"]) &
        quality_ok
    )

    long_trend = (
        (df["close"] > df["ema200"]) &
        (df["ema20"] > df["ema50"]) &
        (df["close"] > df["ema20"]) &
        (df["ret_3"] > 0) &
        (df["body_ratio"] >= mode_config["body_ratio_min"]) &
        (df["rsi"].between(50, 72))
    )
    short_trend = (
        (df["close"] < df["ema200"]) &
        (df["ema20"] < df["ema50"]) &
        (df["close"] < df["ema20"]) &
        (df["ret_3"] < 0) &
        (df["body_ratio"] >= mode_config["body_ratio_min"]) &
        (df["rsi"].between(28, 50))
    )

    long_pullback = (
        (df["close"] > df["ema200"]) &
        (df["ema20"] > df["ema50"]) &
        (df["low"] <= df["ema20"]) &
        (df["close"] > df["ema20"]) &
        (df["ret_1"] > 0) &
        (df["rsi"].between(42, 62))
    )
    short_pullback = (
        (df["close"] < df["ema200"]) &
        (df["ema20"] < df["ema50"]) &
        (df["high"] >= df["ema20"]) &
        (df["close"] < df["ema20"]) &
        (df["ret_1"] < 0) &
        (df["rsi"].between(38, 58))
    )

    df.loc[long_trend, "filter_reason"] = "RAW_LONG_MOMENTUM"
    df.loc[short_trend, "filter_reason"] = "RAW_SHORT_MOMENTUM"
    df.loc[long_pullback, "filter_reason"] = "RAW_LONG_PULLBACK"
    df.loc[short_pullback, "filter_reason"] = "RAW_SHORT_PULLBACK"

    df.loc[long_trend, "raw_signal"] = True
    df.loc[long_trend, "raw_strategy_type"] = "fast_momentum_long"
    df.loc[short_trend, "raw_signal"] = True
    df.loc[short_trend, "raw_strategy_type"] = "fast_momentum_short"
    df.loc[long_pullback & ~df["raw_signal"], "raw_signal"] = True
    df.loc[long_pullback & (df["raw_strategy_type"] == "None"), "raw_strategy_type"] = "fast_pullback_long"
    df.loc[short_pullback & ~df["raw_signal"], "raw_signal"] = True
    df.loc[short_pullback & (df["raw_strategy_type"] == "None"), "raw_strategy_type"] = "fast_pullback_short"

    long_final = base_filter & htf_long_ok
    short_final = base_filter & htf_short_ok

    df.loc[long_final & trend_regime_ok & long_trend, "strategy_long"] = True
    df.loc[long_final & trend_regime_ok & long_trend, "strategy_type"] = "fast_momentum_long"
    df.loc[short_final & trend_regime_ok & short_trend, "strategy_short"] = True
    df.loc[short_final & trend_regime_ok & short_trend, "strategy_type"] = "fast_momentum_short"

    no_momentum = df["strategy_type"] == "None"
    df.loc[long_final & range_regime_ok & no_momentum & long_pullback, "strategy_long"] = True
    df.loc[long_final & range_regime_ok & no_momentum & long_pullback, "strategy_type"] = "fast_pullback_long"
    df.loc[short_final & range_regime_ok & no_momentum & short_pullback, "strategy_short"] = True
    df.loc[short_final & range_regime_ok & no_momentum & short_pullback, "strategy_type"] = "fast_pullback_short"

    df.loc[df["strategy_type"] != "None", "filter_reason"] = "PASS_ALL_FAST_GATES"
    df.loc[~session_ok, "filter_reason"] = "OUT_OF_SESSION"
    df.loc[df["rollover_block"], "filter_reason"] = "ROLLOVER_BLOCK"
    df.loc[df["news_blackout"], "filter_reason"] = "NEWS_BLACKOUT"
    df.loc[~df["spread_ok"], "filter_reason"] = "SPREAD_TOO_HIGH_FOR_ATR"
    df.loc[~df["cost_reward_ok"], "filter_reason"] = "COST_REWARD_TOO_LOW"
    df.loc[
        (~df["volatility_ok"]) |
        (df["ATR_ratio"] < mode_config["atr_ratio_min"]) |
        (df["ATR_ratio"] > mode_config["atr_ratio_max"]),
        "filter_reason"
    ] = "VOLATILITY_REGIME_BLOCK"
    df.loc[df["adx"] < mode_config["adx_min"], "filter_reason"] = "ADX_TOO_LOW"
    df.loc[mode_config["use_quality_filter"] & ~quality_ok, "filter_reason"] = "QUALITY_SCORE_TOO_LOW"
    df.loc[
        mode_config["use_htf_trend_filter"] &
        df["raw_signal"] &
        (df["strategy_type"] == "None") &
        ~(htf_long_ok | htf_short_ok),
        "filter_reason"
    ] = "HTF_TREND_BLOCK"
    df.loc[
        mode_config["use_regime_filter"] &
        df["raw_signal"] &
        (df["strategy_type"] == "None") &
        (df["market_regime"] == "NO_TRADE"),
        "filter_reason"
    ] = "REGIME_NO_TRADE"

    cfg = pair_config(symbol)
    df["target_tp_atr"] = cfg["tp_atr"]
    df["target_sl_atr"] = cfg["sl_atr"]
    df["target_hold_bars"] = MAX_HOLD_BARS.get(interval, 12)
    if mode_config["use_dynamic_targets"]:
        dynamic_targets = df.apply(lambda row: pd.Series(get_dynamic_targets(row)), axis=1)
        df[["target_tp_atr", "target_sl_atr", "target_hold_bars"]] = dynamic_targets
    return df


def max_drawdown_from_returns(returns):
    if not returns:
        return 0
    equity = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd)) if len(dd) else 0


def max_consecutive_losses(returns):
    max_loss = 0
    streak = 0
    for value in returns:
        if value < 0:
            streak += 1
            max_loss = max(max_loss, streak)
        else:
            streak = 0
    return max_loss


def analyze_trades(trades_df, symbol, interval, mode_name, mode_config):
    if trades_df.empty:
        return {
            "mode": mode_name,
            "scope": mode_config["scope"],
            "symbol": symbol,
            "interval": interval,
            "trades": 0,
            "trades_per_day": 0,
            "active_trade_days": 0,
            "win_rate": 0,
            "return": 0,
            "max_dd": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "tp_rate": 0,
            "sl_rate": 0,
            "timeout_rate": 0,
            "max_consecutive_loss": 0,
            "avg_hold_bars": 0,
        }

    returns = trades_df["return"].to_numpy()
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_win = wins.sum()
    gross_loss = losses.sum()
    profit_factor = np.inf if gross_loss == 0 and gross_win > 0 else gross_win / abs(gross_loss) if gross_loss != 0 else 0
    date_count = trades_df["entry_date"].nunique()

    return {
        "mode": mode_name,
        "scope": mode_config["scope"],
        "symbol": symbol,
        "interval": interval,
        "trades": len(trades_df),
        "trades_per_day": len(trades_df) / max(date_count, 1),
        "active_trade_days": date_count,
        "win_rate": len(wins) / len(returns),
        "return": float(np.prod(1 + returns) - 1),
        "max_dd": max_drawdown_from_returns(returns.tolist()),
        "profit_factor": profit_factor,
        "expectancy": float(np.mean(returns)),
        "tp_rate": float((trades_df["exit_reason"].str.contains("TP")).mean()),
        "sl_rate": float((trades_df["exit_reason"].str.contains("SL")).mean()),
        "timeout_rate": float((trades_df["exit_reason"] == "TIMEOUT").mean()),
        "max_consecutive_loss": max_consecutive_losses(returns.tolist()),
        "avg_hold_bars": float(trades_df["hold_bars"].mean()),
    }


def backtest_daytrade(df, symbol, interval, mode_name, mode_config):
    cost_pct_by_row = estimated_cost_pct(symbol, df["close"])
    position = 0
    entry_price = 0
    entry_time = None
    entry_type = "None"
    entry_index = None
    entry_atr = 0
    entry_tp_atr = 0
    entry_sl_atr = 0
    entry_hold_bars = 0
    entry_quality_score = 0
    entry_market_regime = "UNKNOWN"
    moved_to_be = False
    partial_closed = False
    realized_return = 0.0
    remaining_size = 1.0
    daily_return = {}
    daily_trades = {}
    loss_streak = 0
    cooldown_until = -1
    last_entry_index = -10_000
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_date = str(row["date"])

        if position == 0:
            daily_return.setdefault(current_date, 0.0)
            daily_trades.setdefault(current_date, 0)

            if i < cooldown_until:
                continue
            if daily_return[current_date] <= -MAX_DAILY_LOSS:
                continue
            if daily_trades[current_date] >= mode_config["max_trades_per_day"]:
                continue
            if i - last_entry_index < MIN_BARS_BETWEEN_TRADES.get(interval, 3):
                continue

            is_long = bool(row.get("strategy_long", False))
            is_short = bool(row.get("strategy_short", False))
            if not is_long and not is_short:
                continue

            position = 1 if is_long else -1
            entry_price = row["close"]
            entry_time = row["time"]
            entry_index = i
            entry_type = row["strategy_type"]
            entry_atr = row["ATR"]
            entry_tp_atr = row["target_tp_atr"]
            entry_sl_atr = row["target_sl_atr"]
            entry_hold_bars = row["target_hold_bars"]
            entry_quality_score = row["quality_score"]
            entry_market_regime = row["market_regime"]
            moved_to_be = False
            partial_closed = False
            realized_return = 0.0
            remaining_size = 1.0
            last_entry_index = i
            daily_trades[current_date] += 1
            continue

        hold_bars = i - entry_index
        cost_pct = cost_pct_by_row.iloc[i]
        tp_distance = entry_atr * entry_tp_atr
        sl_distance = entry_atr * entry_sl_atr
        be_distance = sl_distance * 0.8
        partial_distance = sl_distance

        exited = False
        exit_reason = None
        exit_price = row["close"]

        if position == 1:
            tp_price = entry_price + tp_distance
            initial_sl_price = entry_price - sl_distance
            sl_price = entry_price if moved_to_be else initial_sl_price
            hit_tp = row["high"] >= tp_price
            hit_sl = row["low"] <= sl_price
            if hit_tp and hit_sl:
                exit_price = sl_price
                exit_reason = "BE_SAME_BAR" if moved_to_be else "SL_SAME_BAR"
                exited = True
            elif hit_sl:
                exit_price = sl_price
                exit_reason = "BE" if moved_to_be else "SL"
                exited = True
            elif hit_tp:
                exit_price = tp_price
                exit_reason = "TP"
                exited = True
            elif hold_bars >= entry_hold_bars:
                exit_reason = "TIMEOUT"
                exited = True

            if not exited and mode_config["use_partial_tp"] and not partial_closed and row["high"] >= entry_price + partial_distance:
                partial_return = (partial_distance / entry_price)
                realized_return += 0.5 * partial_return
                remaining_size = 0.5
                partial_closed = True

            if not exited and mode_config["use_breakeven"] and not moved_to_be and row["high"] >= entry_price + be_distance:
                moved_to_be = True

            raw_return = (exit_price / entry_price) - 1
        else:
            tp_price = entry_price - tp_distance
            initial_sl_price = entry_price + sl_distance
            sl_price = entry_price if moved_to_be else initial_sl_price
            hit_tp = row["low"] <= tp_price
            hit_sl = row["high"] >= sl_price
            if hit_tp and hit_sl:
                exit_price = sl_price
                exit_reason = "BE_SAME_BAR" if moved_to_be else "SL_SAME_BAR"
                exited = True
            elif hit_sl:
                exit_price = sl_price
                exit_reason = "BE" if moved_to_be else "SL"
                exited = True
            elif hit_tp:
                exit_price = tp_price
                exit_reason = "TP"
                exited = True
            elif hold_bars >= entry_hold_bars:
                exit_reason = "TIMEOUT"
                exited = True

            if not exited and mode_config["use_partial_tp"] and not partial_closed and row["low"] <= entry_price - partial_distance:
                partial_return = (partial_distance / entry_price)
                realized_return += 0.5 * partial_return
                remaining_size = 0.5
                partial_closed = True

            if not exited and mode_config["use_breakeven"] and not moved_to_be and row["low"] <= entry_price - be_distance:
                moved_to_be = True

            raw_return = (entry_price / exit_price) - 1

        if not exited:
            continue

        trade_return = realized_return + (remaining_size * raw_return) - cost_pct
        daily_return[current_date] = daily_return.get(current_date, 0.0) + trade_return

        trades.append({
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "mode": mode_name,
            "scope": mode_config["scope"],
            "symbol": symbol,
            "interval": interval,
            "entry_time": entry_time,
            "exit_time": row["time"],
            "entry_date": str(pd.Timestamp(entry_time).date()),
            "side": "LONG" if position == 1 else "SHORT",
            "strategy_type": entry_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": trade_return,
            "raw_return": raw_return,
            "realized_partial_return": realized_return,
            "remaining_size": remaining_size,
            "cost_pct": cost_pct,
            "exit_reason": exit_reason,
            "hold_bars": hold_bars,
            "adx": row["adx"],
            "atr_ratio": row["ATR_ratio"],
            "quality_score": entry_quality_score,
            "market_regime": entry_market_regime,
            "target_tp_atr": entry_tp_atr,
            "target_sl_atr": entry_sl_atr,
            "target_hold_bars": entry_hold_bars,
            "moved_to_be": moved_to_be,
            "partial_closed": partial_closed,
            "risk_per_trade": RISK_PER_TRADE,
        })

        if trade_return < 0:
            loss_streak += 1
        else:
            loss_streak = 0

        if loss_streak >= MAX_CONSECUTIVE_LOSSES:
            cooldown_until = i + COOLDOWN_BARS.get(interval, 10)
            loss_streak = 0

        position = 0

    return pd.DataFrame(trades)


def save_json_append(payload, path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = []
    data.append(payload)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def backup_file(path):
    if not os.path.exists(path):
        return
    backup_dir = os.path.join(LOGS_DIR, "state_backups")
    os.makedirs(backup_dir, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(path))
    backup_path = os.path.join(backup_dir, f"{name}_{RUN_ID}{ext}")
    shutil.copy2(path, backup_path)


def save_run_config():
    path = os.path.join(LOGS_DIR, "fast_run_config_snapshot.json")
    payload = {
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "saved_at": datetime.now().isoformat(),
        "config": {
            "symbols": DAYTRADE_SYMBOLS,
            "intervals": DAYTRADE_INTERVALS,
            "local_csv_first": LOCAL_CSV_FIRST,
            "risk_per_trade": RISK_PER_TRADE,
            "max_daily_loss": MAX_DAILY_LOSS,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "max_consecutive_losses": MAX_CONSECUTIVE_LOSSES,
            "cooldown_bars": COOLDOWN_BARS,
            "session_hours_bkk": sorted(SESSION_HOURS_BKK),
            "rollover_hours_bkk": sorted(ROLLOVER_HOURS_BKK),
            "experiment_modes": EXPERIMENT_MODES,
            "pair_config": PAIR_CONFIG,
        },
    }
    backup_file(path)
    save_json_append(payload, path)


def print_score_summary(rows):
    if not rows:
        print("No fast daytrade score available.")
        return

    df = pd.DataFrame(rows)
    display_cols = [
        "mode", "scope", "symbol", "interval", "raw_signals", "final_signals",
        "trades", "trades_per_day", "win_rate", "return", "max_dd", "profit_factor", "expectancy", "sl_rate",
        "max_consecutive_loss", "avg_hold_bars",
    ]
    df = df.sort_values(["mode", "symbol", "interval"])
    print("\nFAST EXPERIMENT SCORE SUMMARY")
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    viable = df[
        (df["scope"] != "research_only") &
        (df["trades"] >= 150) &
        (df["trades_per_day"] >= MIN_TRADES_PER_DAY_TARGET) &
        (df["trades_per_day"] <= 3) &
        (df["profit_factor"] >= 1.20) &
        (df["expectancy"] > 0) &
        (df["return"] > 0) &
        (df["max_dd"] <= 0.04) &
        (df["sl_rate"] <= 0.58) &
        (df["max_consecutive_loss"] <= 6)
    ].copy()

    print("\nFAST EXPERIMENT READINESS")
    if viable.empty:
        print("WAIT: No experiment passed the minimum paper criteria yet.")
    else:
        print(viable[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


def main():
    save_run_config()
    news_df = load_news_blackout()
    all_trades = []
    summary_rows = []

    enabled_modes = {
        mode: config
        for mode, config in EXPERIMENT_MODES.items()
        if config.get("enabled", False)
    }

    for symbol in DAYTRADE_SYMBOLS:
        for interval in DAYTRADE_INTERVALS:
            df = fetch_intraday_data(symbol, interval)
            if df is None or df.empty:
                if CONSOLE_VERBOSE and not HUMAN_LOG_SCORE_ONLY:
                    print(f"No data for {symbol} {interval}")
                continue

            feature_df = build_features(df, symbol, news_df)
            for mode_name, mode_config in enabled_modes.items():
                strategy_df = apply_daytrade_strategy(feature_df, symbol, interval, mode_name, mode_config)
                trades_df = backtest_daytrade(strategy_df, symbol, interval, mode_name, mode_config)
                all_trades.append(trades_df)
                summary_row = analyze_trades(trades_df, symbol, interval, mode_name, mode_config)
                raw_signals = int(strategy_df["raw_signal"].sum())
                final_signals = int((strategy_df["strategy_long"] | strategy_df["strategy_short"]).sum())
                summary_row.update({
                    "raw_signals": raw_signals,
                    "final_signals": final_signals,
                    "signal_pass_rate": final_signals / raw_signals if raw_signals else 0,
                    "quality_block_count": int((strategy_df["filter_reason"] == "QUALITY_SCORE_TOO_LOW").sum()),
                    "regime_block_count": int((strategy_df["filter_reason"] == "REGIME_NO_TRADE").sum()),
                    "htf_block_count": int((strategy_df["filter_reason"] == "HTF_TREND_BLOCK").sum()),
                })
                summary_rows.append(summary_row)

    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        trades_path = os.path.join(LOGS_DIR, "fast_daytrade_trades.csv")
        trades.to_csv(trades_path, index=False)

    run_summary = pd.DataFrame(summary_rows)
    run_summary["run_id"] = RUN_ID
    run_summary["model_version"] = MODEL_VERSION
    summary_path = os.path.join(LOGS_DIR, "fast_daytrade_score_summary.csv")
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        if set(run_summary.columns).issubset(set(old.columns)):
            summary = pd.concat([old, run_summary], ignore_index=True)
        else:
            backup_file(summary_path)
            summary = run_summary
    else:
        summary = run_summary
    summary.to_csv(summary_path, index=False)

    print_score_summary(summary_rows)


if __name__ == "__main__":
    main()
