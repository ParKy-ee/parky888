import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

YFINANCE_CACHE_DIR = os.path.join(BASE_DIR, ".yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

INTERVALS = ["15m", "1h", "4h", "1d"]
PERIOD = {"15m": "60d", "1h": "730d", "4h": "730d", "1d": "5y"}
WEIGHT = {"15m": 0.45, "1h": 1.2, "4h": 2.0, "1d": 2.35}

DEFAULT_PARAMS = {
    "trade_score": 6,
    "watch_score": 4,
    "adx_min": 20,
    "volume_min": 1.0,
    "atr_high": 1.75,
    "atr_low": 0.48,
    "rsi_long_max": 66,
    "rsi_short_min": 34,
    "sl_atr": 1.75,
    "tp_atr": 2.05,
    "max_hold_bars": 20,
    "trade_enabled": False,
}

COIN_PARAMS = {
    "BTCUSD": {**DEFAULT_PARAMS, "trade_score": 5.5, "adx_min": 18, "sl_atr": 1.35, "tp_atr": 1.85},
    "ETHUSD": {**DEFAULT_PARAMS, "trade_score": 5.8, "adx_min": 19, "sl_atr": 1.45, "tp_atr": 1.95},
    "SOLUSD": {**DEFAULT_PARAMS, "trade_score": 6.4, "adx_min": 21, "atr_high": 1.65, "sl_atr": 1.75, "tp_atr": 2.45},
    "BNBUSD": {
        **DEFAULT_PARAMS,
        "trade_score": 6.2,
        "adx_min": 21,
        "volume_min": 1.03,
        "sl_atr": 1.45,
        "tp_atr": 1.9,
        "trade_enabled": True,
    },
    "XRPUSD": {**DEFAULT_PARAMS, "trade_score": 6.5, "adx_min": 22, "atr_high": 1.6, "sl_atr": 1.85, "tp_atr": 2.6},
}


def params_for(symbol):
    return COIN_PARAMS.get(symbol, DEFAULT_PARAMS)


def fetch_data(yf_symbol, interval):
    request_interval = "1h" if interval == "4h" else interval
    data = yf.download(yf_symbol, period=PERIOD[interval], interval=request_interval, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {yf_symbol} {interval}")

    df = data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]

    df = df.rename(
        columns={
            "Datetime": "time",
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()

    if interval == "4h":
        df = (
            df.set_index("time")
            .resample("4h", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

    return df.sort_values("time").reset_index(drop=True)


def add_features(df):
    df = df.copy()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_ratio"] = df["atr"] / (df["atr"].rolling(80).mean() + 1e-12)
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema100"] = ta.trend.ema_indicator(df["close"], window=100)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["roc"] = ta.momentum.roc(df["close"], window=12)
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(40).mean() + 1e-12)
    macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_hist"] = macd.macd_diff()
    df["support"] = df["low"].rolling(48).min()
    df["resistance"] = df["high"].rolling(48).max()
    df["range_pct"] = (df["resistance"] - df["support"]) / (df["close"] + 1e-12)
    return df.dropna().reset_index(drop=True)


def detect_regime(row, params):
    if row["atr_ratio"] >= params["atr_high"]:
        return "HIGH_VOLATILITY"
    if row["atr_ratio"] <= params["atr_low"] or row["adx"] < 15:
        return "LOW_EDGE_RANGE"
    if row["adx"] >= params["adx_min"] and row["range_pct"] >= 0.035:
        return "TREND"
    if row["volume_ratio"] >= 1.35 and abs(row["roc"]) >= 2.0:
        return "BREAKOUT"
    return "RANGE"


def score_row(row, params=None):
    params = params or DEFAULT_PARAMS
    score, reasons = 0, []

    if row["close"] > row["ema200"]:
        score += 2
        reasons.append("above_ema200")
    else:
        score -= 2
        reasons.append("below_ema200")

    if row["ema20"] > row["ema50"] > row["ema100"]:
        score += 2
        reasons.append("ema_bull_stack")
    elif row["ema20"] < row["ema50"] < row["ema100"]:
        score -= 2
        reasons.append("ema_bear_stack")

    if row["macd_hist"] > 0 and row["roc"] > 0:
        score += 2
        reasons.append("momentum_positive")
    elif row["macd_hist"] < 0 and row["roc"] < 0:
        score -= 2
        reasons.append("momentum_negative")

    if row["adx"] >= params["adx_min"] and score > 0:
        score += 1
        reasons.append("trend_strength_bull")
    elif row["adx"] >= params["adx_min"] and score < 0:
        score -= 1
        reasons.append("trend_strength_bear")

    if row["volume_ratio"] >= params["volume_min"] and abs(score) >= 4:
        score += 1 if score > 0 else -1
        reasons.append("volume_confirmation")

    if row["rsi"] >= params["rsi_long_max"] and score > 0:
        score -= 2
        reasons.append("rsi_long_too_hot")
    elif row["rsi"] <= params["rsi_short_min"] and score < 0:
        score += 2
        reasons.append("rsi_short_too_cold")

    regime = detect_regime(row, params)
    if regime == "HIGH_VOLATILITY":
        score = int(score * 0.65)
        reasons.append("high_volatility_filter")
    elif regime == "LOW_EDGE_RANGE":
        score = int(score * 0.55)
        reasons.append("low_edge_range_filter")
    elif regime == "BREAKOUT":
        score += 1 if score > 0 else -1 if score < 0 else 0
        reasons.append("breakout_regime")

    return score, reasons


def bias_from_score(score, params):
    if score >= params["watch_score"]:
        return "BULLISH"
    if score <= -params["watch_score"]:
        return "BEARISH"
    return "NEUTRAL"


def trade_levels(price, atr, bias, params):
    if bias == "BULLISH":
        return price - params["sl_atr"] * atr, price + params["tp_atr"] * atr
    if bias == "BEARISH":
        return price + params["sl_atr"] * atr, price - params["tp_atr"] * atr
    return np.nan, np.nan


def evaluate_trade_setup(rows, symbol):
    params = params_for(symbol)
    by_interval = {row["interval"]: row for row in rows}
    one_h = by_interval.get("1h")
    four_h = by_interval.get("4h")
    one_d = by_interval.get("1d")

    weighted_score = sum(row["score"] * WEIGHT[row["interval"]] for row in rows) / sum(WEIGHT.values())
    weighted_bias = bias_from_score(weighted_score, params)
    alignment = pd.Series([row["bias"] for row in rows]).value_counts().to_dict()

    if weighted_bias == "NEUTRAL":
        return weighted_score, weighted_bias, "WAIT", alignment, ["weighted_score_neutral"]

    if weighted_bias == "BEARISH":
        return weighted_score, weighted_bias, "BEARISH_WATCH", alignment, ["short_disabled_crypto_long_bias"]

    if not params.get("trade_enabled", False):
        return weighted_score, weighted_bias, "BULLISH_WATCH", alignment, ["trade_disabled_until_backtest_edge"]

    direction = "BULLISH" if weighted_bias == "BULLISH" else "BEARISH"
    opposing = "BEARISH" if direction == "BULLISH" else "BULLISH"
    reasons = []

    core_aligned = four_h and four_h["bias"] == direction and (one_h is None or one_h["bias"] != opposing)
    daily_not_opposed = one_d is None or one_d["bias"] in [direction, "NEUTRAL"]
    strong_enough = abs(weighted_score) >= params["trade_score"]
    regime_ok = four_h is not None and four_h["regime"] in ["TREND", "BREAKOUT"]
    rsi_ok = four_h is not None and four_h["rsi"] <= params["rsi_long_max"]
    volume_ok = four_h is not None and four_h["volume_ratio"] >= params["volume_min"] * 0.85

    if not core_aligned:
        reasons.append("4h_not_confirmed_or_1h_opposed")
    if not daily_not_opposed:
        reasons.append("1d_opposes_signal")
    if not strong_enough:
        reasons.append("score_below_trade_threshold")
    if not regime_ok:
        reasons.append("4h_regime_not_tradeable")
    if not rsi_ok:
        reasons.append("4h_rsi_too_hot")
    if not volume_ok:
        reasons.append("volume_below_trade_threshold")

    if core_aligned and daily_not_opposed and strong_enough and regime_ok and rsi_ok and volume_ok:
        return weighted_score, weighted_bias, f"{direction}_TRADE_READY", alignment, ["trade_ready_confirmed"]

    return weighted_score, weighted_bias, f"{direction}_WATCH", alignment, reasons


def forecast_interval(symbol, yf_symbol, model_version, run_id, interval):
    params = params_for(symbol)
    df = add_features(fetch_data(yf_symbol, interval))
    row = df.iloc[-1]
    score, reasons = score_row(row, params)
    bias = bias_from_score(score, params)
    regime = detect_regime(row, params)
    stop_loss, take_profit = trade_levels(float(row["close"]), float(row["atr"]), bias, params)

    return {
        "run_id": run_id,
        "model_version": model_version,
        "symbol": symbol,
        "interval": interval,
        "last_bar_time": row["time"],
        "price": row["close"],
        "bias": bias,
        "action": "WAIT" if bias == "NEUTRAL" else f"{bias}_WATCH",
        "confidence": min(abs(score) / 10, 1.0),
        "score": score,
        "regime": regime,
        "adx": row["adx"],
        "rsi": row["rsi"],
        "atr": row["atr"],
        "atr_ratio": row["atr_ratio"],
        "volume_ratio": row["volume_ratio"],
        "support": row["support"],
        "resistance": row["resistance"],
        "stop_loss": stop_loss,
        "take_profit_1": take_profit,
        "reasons": ",".join(reasons + [f"regime_{regime.lower()}"]),
    }


def run_crypto_model(symbol, yf_symbol):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"{symbol}-crypto-forecast-v0.20-regime-mtf-trade-ready"
    rows = [forecast_interval(symbol, yf_symbol, model_version, run_id, interval) for interval in INTERVALS]
    final_score, final_bias, decision, alignment, decision_reasons = evaluate_trade_setup(rows, symbol)
    df = pd.DataFrame(rows)
    summary = {
        "run_id": run_id,
        "model_version": model_version,
        "symbol": symbol,
        "final_bias": final_bias,
        "final_score": final_score,
        "confidence": min(abs(final_score) / 10, 1.0),
        "decision": decision,
        "alignment": alignment,
        "decision_reasons": decision_reasons,
        "params": params_for(symbol),
    }

    prefix = symbol.lower().replace("-", "")
    report = os.path.join(LOGS_DIR, f"{prefix}_forecast_report.csv")
    latest = os.path.join(LOGS_DIR, f"{prefix}_forecast_latest.json")
    out = pd.concat([pd.read_csv(report), df], ignore_index=True) if os.path.exists(report) else df
    out.to_csv(report, index=False)
    with open(latest, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "timeframes": df.to_dict(orient="records")}, file, indent=2, ensure_ascii=False, default=str)

    print(f"\n{symbol} CRYPTO FORECAST SUMMARY")
    print(
        f"FINAL: {summary['final_bias']} | decision={summary['decision']} | "
        f"score={summary['final_score']:.2f} | confidence={summary['confidence']:.2%}"
    )
    print(
        df[
            [
                "interval",
                "last_bar_time",
                "price",
                "bias",
                "action",
                "confidence",
                "score",
                "regime",
                "adx",
                "rsi",
                "atr_ratio",
                "volume_ratio",
                "support",
                "resistance",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:0.5f}")
    )


def main(symbol, yf_symbol):
    run_crypto_model(symbol, yf_symbol)
