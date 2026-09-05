import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

FOREX_PAIRS = {
    "AUDCAD": "AUDCAD=X",
    "AUDCHF": "AUDCHF=X",
    "AUDJPY": "AUDJPY=X",
    "AUDNZD": "AUDNZD=X",
    "AUDUSD": "AUDUSD=X",
    "CADCHF": "CADCHF=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "EURAUD": "EURAUD=X",
    "EURCAD": "EURCAD=X",
    "EURCHF": "EURCHF=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "EURNZD": "EURNZD=X",
    "EURUSD": "EURUSD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPCHF": "GBPCHF=X",
    "GBPJPY": "GBPJPY=X",
    "GBPNZD": "GBPNZD=X",
    "GBPUSD": "GBPUSD=X",
    "NZDCAD": "NZDCAD=X",
    "NZDCHF": "NZDCHF=X",
    "NZDJPY": "NZDJPY=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "USDJPY": "USDJPY=X",
}

SYMBOL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FOREX_SYMBOL", "EURUSD")).upper()
if SYMBOL not in FOREX_PAIRS:
    raise SystemExit(f"Unsupported forex symbol: {SYMBOL}")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
YF_SYMBOL = FOREX_PAIRS[SYMBOL]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "pair_profiles.json")
os.makedirs(LOGS_DIR, exist_ok=True)

YFINANCE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

HTF_RULE = {"30m": "1h", "1h": "4h", "4h": "1d"}


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_profile(symbol):
    with open(PROFILE_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return deep_merge(payload["default"], payload.get("pairs", {}).get(symbol, {}))


PROFILE = load_profile(SYMBOL)
MODEL_VERSION = f"{SYMBOL}-forecast-v0.40-{PROFILE['profile_name']}"
INTERVALS = PROFILE["intervals"]
PERIOD = PROFILE["period"]
WEIGHT = PROFILE["timeframe_weights"]


def fetch_data(interval):
    request_interval = "1h" if interval == "4h" else interval
    data = yf.download(YF_SYMBOL, period=PERIOD[interval], interval=request_interval, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {YF_SYMBOL} {interval}")
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
    if "volume" not in df.columns:
        df["volume"] = 0.0
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


def add_htf(df, interval):
    rule = HTF_RULE[interval]
    htf = (
        df.set_index("time")
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    if len(htf) < 80:
        df["htf_bias"] = "NEUTRAL"
        df["htf_adx"] = 0.0
        return df

    fast, slow = (
        (PROFILE["htf_fast_daily"], PROFILE["htf_slow_daily"])
        if rule == "1d"
        else (PROFILE["htf_fast_intraday"], PROFILE["htf_slow_intraday"])
    )
    htf["fast"] = ta.trend.ema_indicator(htf["close"], window=fast)
    htf["slow"] = ta.trend.ema_indicator(htf["close"], window=slow)
    htf["htf_adx"] = ta.trend.adx(htf["high"], htf["low"], htf["close"], window=PROFILE["adx_window"])
    htf["htf_bias"] = "NEUTRAL"
    htf.loc[(htf["close"] > htf["slow"]) & (htf["fast"] > htf["slow"]), "htf_bias"] = "BULLISH"
    htf.loc[(htf["close"] < htf["slow"]) & (htf["fast"] < htf["slow"]), "htf_bias"] = "BEARISH"
    out = pd.merge_asof(
        df.sort_values("time"),
        htf[["time", "htf_bias", "htf_adx"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    out["htf_bias"] = out["htf_bias"].fillna("NEUTRAL")
    out["htf_adx"] = out["htf_adx"].fillna(0.0)
    return out


def build_features(df, interval):
    df = df.copy()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=PROFILE["atr_window"])
    df["atr_ratio"] = df["atr"] / (df["atr"].rolling(PROFILE["atr_ratio_window"]).mean() + 1e-12)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=PROFILE["ema_fast"])
    df["ema55"] = ta.trend.ema_indicator(df["close"], window=PROFILE["ema_mid"])
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=PROFILE["ema_slow"])
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=PROFILE["adx_window"])
    df["rsi"] = ta.momentum.rsi(df["close"], window=PROFILE["rsi_window"])
    macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["ret_6"] = df["close"].pct_change(PROFILE["momentum_lookback"])
    df["support"] = df["low"].rolling(PROFILE["support_resistance_window"]).min()
    df["resistance"] = df["high"].rolling(PROFILE["support_resistance_window"]).max()
    df = add_htf(df, interval)
    return df.dropna().reset_index(drop=True)


def score_row(row):
    weights = PROFILE["score_weights"]
    score, reasons = 0, []
    if row["close"] > row["ema200"]:
        score += weights["ema200"]
        reasons.append("above_ema200")
    else:
        score -= weights["ema200"]
        reasons.append("below_ema200")
    if row["ema21"] > row["ema55"] > row["ema200"]:
        score += weights["ema_stack"]
        reasons.append("ema_bull_stack")
    elif row["ema21"] < row["ema55"] < row["ema200"]:
        score -= weights["ema_stack"]
        reasons.append("ema_bear_stack")
    if row["macd_hist"] > 0 and row["ret_6"] > 0:
        score += weights["momentum"]
        reasons.append("macd_momentum_up")
    elif row["macd_hist"] < 0 and row["ret_6"] < 0:
        score -= weights["momentum"]
        reasons.append("macd_momentum_down")
    if PROFILE["rsi_bull_min"] <= row["rsi"] <= PROFILE["rsi_bull_max"] and score > 0:
        score += weights["rsi_balance"]
        reasons.append("rsi_bullish_balanced")
    elif PROFILE["rsi_bear_min"] <= row["rsi"] <= PROFILE["rsi_bear_max"] and score < 0:
        score -= weights["rsi_balance"]
        reasons.append("rsi_bearish_balanced")
    if row["htf_bias"] == "BULLISH" and row["htf_adx"] >= PROFILE["htf_adx_min"]:
        score += weights["htf_alignment"]
        reasons.append("htf_bullish")
    elif row["htf_bias"] == "BEARISH" and row["htf_adx"] >= PROFILE["htf_adx_min"]:
        score -= weights["htf_alignment"]
        reasons.append("htf_bearish")
    if row["adx"] >= PROFILE["adx_trade_min"] and abs(score) >= PROFILE["watch_threshold"]:
        score += weights["adx_trade_bonus"] if score > 0 else -weights["adx_trade_bonus"]
        reasons.append("adx_trade_bonus")
    if row["atr_ratio"] < PROFILE["atr_low_discount"]:
        score = score * 0.6
        reasons.append("low_volatility_discount")
    elif row["atr_ratio"] > PROFILE["atr_high_penalty"]:
        score = score * 0.8
        reasons.append("high_volatility_penalty")
    return score, reasons


def forecast_interval(interval):
    df = build_features(fetch_data(interval), interval)
    row = df.iloc[-1]
    score, reasons = score_row(row)
    if row["atr_ratio"] > PROFILE["atr_extreme_block"]:
        score = 0
        reasons.append("extreme_volatility_block")
    bias = "BULLISH" if score >= PROFILE["watch_threshold"] else "BEARISH" if score <= -PROFILE["watch_threshold"] else "NEUTRAL"
    action = "WAIT" if bias == "NEUTRAL" else f"{bias}_WATCH"
    return {
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "interval": interval,
        "last_bar_time": row["time"],
        "price": row["close"],
        "bias": bias,
        "action": action,
        "confidence": min(abs(score) / PROFILE["confidence_denominator"], 1.0),
        "score": score,
        "adx": row["adx"],
        "rsi": row["rsi"],
        "atr_ratio": row["atr_ratio"],
        "htf_bias": row["htf_bias"],
        "htf_adx": row["htf_adx"],
        "support": row["support"],
        "resistance": row["resistance"],
        "stop_loss": row["close"] - PROFILE["sl_atr_multiplier"] * row["atr"] if bias == "BULLISH" else row["close"] + PROFILE["sl_atr_multiplier"] * row["atr"] if bias == "BEARISH" else np.nan,
        "take_profit_1": row["close"] + PROFILE["tp1_atr_multiplier"] * row["atr"] if bias == "BULLISH" else row["close"] - PROFILE["tp1_atr_multiplier"] * row["atr"] if bias == "BEARISH" else np.nan,
        "reasons": ",".join(reasons),
        "profile_name": PROFILE["profile_name"],
        "watch_threshold": PROFILE["watch_threshold"],
        "trade_ready_threshold": PROFILE["trade_ready_threshold"],
    }


def final_decision(final_bias, final_score):
    if final_bias == "NEUTRAL":
        return "WAIT"
    if abs(final_score) >= PROFILE["trade_ready_threshold"]:
        return f"{final_bias}_TRADE_READY"
    return f"{final_bias}_WATCH"


def main():
    rows = [forecast_interval(interval) for interval in INTERVALS]
    df = pd.DataFrame(rows)
    final_score = sum(row["score"] * WEIGHT[row["interval"]] for row in rows) / sum(WEIGHT.values())
    final_bias = "BULLISH" if final_score >= PROFILE["watch_threshold"] else "BEARISH" if final_score <= -PROFILE["watch_threshold"] else "NEUTRAL"
    summary = {
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "final_bias": final_bias,
        "final_score": final_score,
        "confidence": min(abs(final_score) / PROFILE["confidence_denominator"], 1.0),
        "decision": final_decision(final_bias, final_score),
        "alignment": df["bias"].value_counts().to_dict(),
        "profile_name": PROFILE["profile_name"],
        "watch_threshold": PROFILE["watch_threshold"],
        "trade_ready_threshold": PROFILE["trade_ready_threshold"],
    }
    report = os.path.join(LOGS_DIR, f"{SYMBOL.lower()}_forecast_report.csv")
    latest = os.path.join(LOGS_DIR, f"{SYMBOL.lower()}_forecast_latest.json")
    out = pd.concat([pd.read_csv(report), df], ignore_index=True) if os.path.exists(report) else df
    out.to_csv(report, index=False)
    with open(latest, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "timeframes": df.to_dict(orient="records")}, file, indent=2, ensure_ascii=False, default=str)
    print(f"\n{SYMBOL} FORECAST SUMMARY")
    print(
        f"FINAL: {summary['final_bias']} | decision={summary['decision']} | "
        f"score={summary['final_score']:.2f} | confidence={summary['confidence']:.2%}"
    )


if __name__ == "__main__":
    main()
