import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

MODEL_VERSION = "EURUSD-forecast-v0.20-liquid-trend-balance"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SYMBOL = "EURUSD"
YF_SYMBOL = "EURUSD=X"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

YFINANCE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

INTERVALS = ["30m", "1h", "4h"]
PERIOD = {"30m": "60d", "1h": "2y", "4h": "2y"}
HTF_RULE = {"30m": "1h", "1h": "4h", "4h": "1d"}
WEIGHT = {"30m": 1.0, "1h": 1.5, "4h": 2.0}


def fetch_data(interval):
    request_interval = "1h" if interval == "4h" else interval
    data = yf.download(YF_SYMBOL, period=PERIOD[interval], interval=request_interval, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {YF_SYMBOL} {interval}")
    df = data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]
    df = df.rename(columns={"Datetime": "time", "Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    if interval == "4h":
        df = df.set_index("time").resample("4h", label="right", closed="right").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
    return df.sort_values("time").reset_index(drop=True)


def add_htf(df, interval):
    rule = HTF_RULE[interval]
    htf = df.set_index("time").resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    if len(htf) < 80:
        df["htf_bias"] = "NEUTRAL"
        df["htf_adx"] = 0.0
        return df
    fast, slow = (20, 50) if rule == "1d" else (55, 200)
    htf["fast"] = ta.trend.ema_indicator(htf["close"], window=fast)
    htf["slow"] = ta.trend.ema_indicator(htf["close"], window=slow)
    htf["htf_adx"] = ta.trend.adx(htf["high"], htf["low"], htf["close"], window=14)
    htf["htf_bias"] = "NEUTRAL"
    htf.loc[(htf["close"] > htf["slow"]) & (htf["fast"] > htf["slow"]), "htf_bias"] = "BULLISH"
    htf.loc[(htf["close"] < htf["slow"]) & (htf["fast"] < htf["slow"]), "htf_bias"] = "BEARISH"
    out = pd.merge_asof(df.sort_values("time"), htf[["time", "htf_bias", "htf_adx"]].sort_values("time"), on="time", direction="backward")
    out["htf_bias"] = out["htf_bias"].fillna("NEUTRAL")
    out["htf_adx"] = out["htf_adx"].fillna(0.0)
    return out


def build_features(df, interval):
    df = df.copy()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_ratio"] = df["atr"] / (df["atr"].rolling(80).mean() + 1e-12)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema55"] = ta.trend.ema_indicator(df["close"], window=55)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["support"] = df["low"].rolling(40).min()
    df["resistance"] = df["high"].rolling(40).max()
    df = add_htf(df, interval)
    return df.dropna().reset_index(drop=True)


def score_row(row):
    score, reasons = 0, []
    if row["close"] > row["ema200"]:
        score += 2; reasons.append("above_ema200")
    else:
        score -= 2; reasons.append("below_ema200")
    if row["ema21"] > row["ema55"] > row["ema200"]:
        score += 2; reasons.append("ema_bull_stack")
    elif row["ema21"] < row["ema55"] < row["ema200"]:
        score -= 2; reasons.append("ema_bear_stack")
    if row["macd_hist"] > 0 and row["ret_6"] > 0:
        score += 1; reasons.append("macd_momentum_up")
    elif row["macd_hist"] < 0 and row["ret_6"] < 0:
        score -= 1; reasons.append("macd_momentum_down")
    if 45 <= row["rsi"] <= 62 and score > 0:
        score += 1; reasons.append("rsi_bullish_balanced")
    elif 38 <= row["rsi"] <= 55 and score < 0:
        score -= 1; reasons.append("rsi_bearish_balanced")
    if row["htf_bias"] == "BULLISH" and row["htf_adx"] >= 16:
        score += 2; reasons.append("htf_bullish")
    elif row["htf_bias"] == "BEARISH" and row["htf_adx"] >= 16:
        score -= 2; reasons.append("htf_bearish")
    if row["atr_ratio"] < 0.65:
        score = int(score * 0.6); reasons.append("low_volatility_discount")
    return score, reasons


def forecast_interval(interval):
    df = build_features(fetch_data(interval), interval)
    row = df.iloc[-1]
    score, reasons = score_row(row)
    bias = "BULLISH" if score >= 4 else "BEARISH" if score <= -4 else "NEUTRAL"
    action = "WAIT" if bias == "NEUTRAL" else f"{bias}_WATCH"
    return {
        "run_id": RUN_ID, "model_version": MODEL_VERSION, "symbol": SYMBOL, "interval": interval,
        "last_bar_time": row["time"], "price": row["close"], "bias": bias, "action": action,
        "confidence": min(abs(score) / 8, 1.0), "score": score,
        "adx": row["adx"], "rsi": row["rsi"], "atr_ratio": row["atr_ratio"],
        "htf_bias": row["htf_bias"], "htf_adx": row["htf_adx"],
        "support": row["support"], "resistance": row["resistance"],
        "stop_loss": row["close"] - 1.2 * row["atr"] if bias == "BULLISH" else row["close"] + 1.2 * row["atr"] if bias == "BEARISH" else np.nan,
        "take_profit_1": row["close"] + 1.6 * row["atr"] if bias == "BULLISH" else row["close"] - 1.6 * row["atr"] if bias == "BEARISH" else np.nan,
        "reasons": ",".join(reasons),
    }


def main():
    rows = [forecast_interval(interval) for interval in INTERVALS]
    df = pd.DataFrame(rows)
    final_score = sum(row["score"] * WEIGHT[row["interval"]] for row in rows) / sum(WEIGHT.values())
    final_bias = "BULLISH" if final_score >= 4 else "BEARISH" if final_score <= -4 else "NEUTRAL"
    summary = {
        "run_id": RUN_ID, "model_version": MODEL_VERSION, "symbol": SYMBOL,
        "final_bias": final_bias, "final_score": final_score,
        "confidence": min(abs(final_score) / 8, 1.0),
        "decision": "WAIT" if final_bias == "NEUTRAL" else f"{final_bias}_WATCH",
        "alignment": df["bias"].value_counts().to_dict(),
    }
    report = os.path.join(LOGS_DIR, "eurusd_forecast_report.csv")
    latest = os.path.join(LOGS_DIR, "eurusd_forecast_latest.json")
    out = pd.concat([pd.read_csv(report), df], ignore_index=True) if os.path.exists(report) else df
    out.to_csv(report, index=False)
    with open(latest, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "timeframes": df.to_dict(orient="records")}, file, indent=2, ensure_ascii=False, default=str)
    print(f"\n{SYMBOL} FORECAST SUMMARY")
    print(f"FINAL: {summary['final_bias']} | decision={summary['decision']} | score={summary['final_score']:.2f} | confidence={summary['confidence']:.2%}")
    print(df[["interval", "last_bar_time", "price", "bias", "confidence", "score", "htf_bias", "adx", "rsi", "atr_ratio", "support", "resistance", "action"]].to_string(index=False, float_format=lambda x: f"{x:0.5f}"))


if __name__ == "__main__":
    main()
