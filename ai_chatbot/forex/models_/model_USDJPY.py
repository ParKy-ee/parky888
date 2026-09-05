import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

MODEL_VERSION = "USDJPY-forecast-v0.10"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SYMBOL = "USDJPY"
YF_SYMBOL = "USDJPY=X"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

YFINANCE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

FORECAST_INTERVALS = ["30m", "1h", "4h"]
PERIOD_BY_INTERVAL = {
    "30m": "60d",
    "1h": "2y",
    "4h": "2y",
}
HTF_RULE_BY_INTERVAL = {
    "30m": "1h",
    "1h": "4h",
    "4h": "1d",
}

PAIR_CONFIG = {
    "pip_size": 0.01,
    "estimated_spread_pips": 1.1,
    "slippage_pips": 0.4,
}


def resample_ohlcv(df, interval):
    rule_map = {
        "4h": "4h",
    }
    rule = rule_map.get(interval.lower())
    if rule is None:
        return df

    return (
        df.set_index("time")
        .resample(rule, label="right", closed="right")
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


def fetch_usdjpy(interval):
    period = PERIOD_BY_INTERVAL.get(interval, "60d")
    request_interval = "1h" if interval == "4h" else interval
    data = yf.download(
        tickers=YF_SYMBOL,
        period=period,
        interval=request_interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"No data returned for {YF_SYMBOL} interval={interval}")

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
        raise RuntimeError(f"Missing columns for {YF_SYMBOL}: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    df = resample_ohlcv(df, interval)
    return df.sort_values("time").reset_index(drop=True)


def add_higher_timeframe_context(df, interval):
    htf_rule = HTF_RULE_BY_INTERVAL.get(interval, "1h")
    htf = (
        df[["time", "open", "high", "low", "close", "volume"]]
        .set_index("time")
        .resample(htf_rule, label="right", closed="right")
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

    if len(htf) < 80:
        df["htf_bias"] = "NEUTRAL"
        df["htf_adx"] = 0.0
        return df

    fast_window = 20 if htf_rule == "1d" else 50
    slow_window = 50 if htf_rule == "1d" else 200
    htf["htf_ema_fast"] = ta.trend.ema_indicator(htf["close"], window=fast_window)
    htf["htf_ema_slow"] = ta.trend.ema_indicator(htf["close"], window=slow_window)
    htf["htf_adx"] = ta.trend.adx(htf["high"], htf["low"], htf["close"], window=14)
    htf["htf_bias"] = "NEUTRAL"
    htf.loc[
        (htf["close"] > htf["htf_ema_slow"]) &
        (htf["htf_ema_fast"] > htf["htf_ema_slow"]),
        "htf_bias"
    ] = "BULLISH"
    htf.loc[
        (htf["close"] < htf["htf_ema_slow"]) &
        (htf["htf_ema_fast"] < htf["htf_ema_slow"]),
        "htf_bias"
    ] = "BEARISH"

    merged = pd.merge_asof(
        df.sort_values("time"),
        htf[["time", "htf_bias", "htf_adx"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    merged["htf_bias"] = merged["htf_bias"].fillna("NEUTRAL")
    merged["htf_adx"] = merged["htf_adx"].fillna(0.0)
    return merged


def build_features(df, interval):
    df = df.copy().sort_values("time").reset_index(drop=True)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["ATR_mean"] = df["ATR"].rolling(80).mean()
    df["ATR_ratio"] = df["ATR"] / (df["ATR_mean"] + 1e-12)
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / ((df["high"] - df["low"]) + 1e-12)
    df["support_20"] = df["low"].rolling(20).min()
    df["resistance_20"] = df["high"].rolling(20).max()
    df["support_80"] = df["low"].rolling(80).min()
    df["resistance_80"] = df["high"].rolling(80).max()
    df = add_higher_timeframe_context(df, interval)
    return df.dropna().reset_index(drop=True)


def detect_regime(row):
    if row["adx"] >= 25 and row["ATR_ratio"] >= 1.0:
        return "TREND"
    if row["adx"] < 18 and row["ATR_ratio"] < 1.0:
        return "RANGE"
    return "MIXED"


def score_forecast(row):
    score = 0
    reasons = []

    if row["close"] > row["ema200"]:
        score += 2
        reasons.append("price_above_ema200")
    else:
        score -= 2
        reasons.append("price_below_ema200")

    if row["ema20"] > row["ema50"] > row["ema200"]:
        score += 2
        reasons.append("ema_stack_bullish")
    elif row["ema20"] < row["ema50"] < row["ema200"]:
        score -= 2
        reasons.append("ema_stack_bearish")

    if row["ret_6"] > 0 and row["ret_12"] > 0:
        score += 1
        reasons.append("momentum_positive")
    elif row["ret_6"] < 0 and row["ret_12"] < 0:
        score -= 1
        reasons.append("momentum_negative")

    if row["macd"] > row["macd_signal"]:
        score += 1
        reasons.append("macd_bullish")
    else:
        score -= 1
        reasons.append("macd_bearish")

    if row["rsi"] >= 68:
        score -= 1
        reasons.append("rsi_near_overbought")
    elif row["rsi"] <= 32:
        score += 1
        reasons.append("rsi_near_oversold")

    if row["htf_bias"] == "BULLISH" and row["htf_adx"] >= 18:
        score += 2
        reasons.append("higher_tf_bullish")
    elif row["htf_bias"] == "BEARISH" and row["htf_adx"] >= 18:
        score -= 2
        reasons.append("higher_tf_bearish")

    if row["adx"] < 15:
        reasons.append("weak_trend")
    elif row["adx"] >= 25:
        reasons.append("strong_trend")

    return score, reasons


def build_forecast_row(df, interval):
    row = df.iloc[-1]
    score, reasons = score_forecast(row)
    regime = detect_regime(row)

    if score >= 4:
        bias = "BULLISH"
        action = "LONG_ONLY_ON_BREAKOUT_OR_PULLBACK"
    elif score <= -4:
        bias = "BEARISH"
        action = "SHORT_ONLY_ON_BREAKDOWN_OR_RETEST"
    else:
        bias = "NEUTRAL"
        action = "WAIT"

    confidence = min(abs(score) / 8, 1.0)
    atr = row["ATR"]
    price = row["close"]

    if bias == "BULLISH":
        entry_zone = max(row["ema20"], row["support_20"])
        stop_loss = price - (1.2 * atr)
        take_profit_1 = price + (1.6 * atr)
        take_profit_2 = price + (2.4 * atr)
        invalidation = min(row["support_20"], row["ema50"])
    elif bias == "BEARISH":
        entry_zone = min(row["ema20"], row["resistance_20"])
        stop_loss = price + (1.2 * atr)
        take_profit_1 = price - (1.6 * atr)
        take_profit_2 = price - (2.4 * atr)
        invalidation = max(row["resistance_20"], row["ema50"])
    else:
        entry_zone = np.nan
        stop_loss = np.nan
        take_profit_1 = np.nan
        take_profit_2 = np.nan
        invalidation = np.nan

    return {
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "interval": interval,
        "last_bar_time": row["time"],
        "price": price,
        "bias": bias,
        "action": action,
        "confidence": confidence,
        "score": score,
        "market_regime": regime,
        "htf_bias": row["htf_bias"],
        "htf_adx": row["htf_adx"],
        "adx": row["adx"],
        "rsi": row["rsi"],
        "atr": atr,
        "atr_ratio": row["ATR_ratio"],
        "ema20": row["ema20"],
        "ema50": row["ema50"],
        "ema200": row["ema200"],
        "support_20": row["support_20"],
        "resistance_20": row["resistance_20"],
        "support_80": row["support_80"],
        "resistance_80": row["resistance_80"],
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "invalidation": invalidation,
        "reasons": ",".join(reasons),
    }


def combine_forecasts(forecast_df):
    weights = {
        "30m": 1.0,
        "1h": 1.5,
        "4h": 2.0,
    }
    weighted_score = 0.0
    total_weight = 0.0
    for _, row in forecast_df.iterrows():
        weight = weights.get(row["interval"], 1.0)
        weighted_score += row["score"] * weight
        total_weight += weight

    final_score = weighted_score / max(total_weight, 1e-12)
    if final_score >= 4:
        final_bias = "BULLISH"
    elif final_score <= -4:
        final_bias = "BEARISH"
    else:
        final_bias = "NEUTRAL"

    alignment = forecast_df["bias"].value_counts().to_dict()
    return {
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "final_bias": final_bias,
        "final_score": final_score,
        "confidence": min(abs(final_score) / 8, 1.0),
        "alignment": alignment,
        "decision": "WAIT" if final_bias == "NEUTRAL" else f"{final_bias}_WATCH",
    }


def save_reports(forecast_df, summary):
    forecast_path = os.path.join(LOGS_DIR, "usdjpy_forecast_report.csv")
    json_path = os.path.join(LOGS_DIR, "usdjpy_forecast_latest.json")

    out = forecast_df.copy()
    if os.path.exists(forecast_path):
        old = pd.read_csv(forecast_path)
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(forecast_path, index=False)

    payload = {
        "summary": summary,
        "timeframes": forecast_df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def print_forecast(forecast_df, summary):
    display_cols = [
        "interval", "last_bar_time", "price", "bias", "confidence", "score",
        "market_regime", "htf_bias", "adx", "rsi", "atr_ratio",
        "support_20", "resistance_20", "action",
    ]
    print("\nUSDJPY FORECAST SUMMARY")
    print(
        f"FINAL: {summary['final_bias']} | decision={summary['decision']} | "
        f"score={summary['final_score']:.2f} | confidence={summary['confidence']:.2%}"
    )
    print(f"alignment={summary['alignment']}")
    print("\nTIMEFRAME DETAIL")
    print(forecast_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


def main():
    rows = []
    for interval in FORECAST_INTERVALS:
        df = fetch_usdjpy(interval)
        df = build_features(df, interval)
        rows.append(build_forecast_row(df, interval))

    forecast_df = pd.DataFrame(rows)
    summary = combine_forecasts(forecast_df)
    save_reports(forecast_df, summary)
    print_forecast(forecast_df, summary)


if __name__ == "__main__":
    main()
