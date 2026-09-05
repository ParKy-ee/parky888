import pandas as pd
import numpy as np
import ta
import os
import json
import shutil
import yfinance as yf
from datetime import datetime
import warnings

warnings.filtervectors = True 
warnings.filterwarnings("ignore")

MODEL_VERSION = "0.80-stable-watchlist-governance"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CONSOLE_VERBOSE = False

YFINANCE_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

FOREX_COSTS = {
    "EURUSD": {"fee": 0.0, "slippage": 0.00003, "spread": 0.00002},
    "GBPUSD": {"fee": 0.0, "slippage": 0.00004, "spread": 0.00003},
    "USDJPY": {"fee": 0.0, "slippage": 0.00004, "spread": 0.00003},
    "USDCHF": {"fee": 0.0, "slippage": 0.00004, "spread": 0.00004},
    "AUDUSD": {"fee": 0.0, "slippage": 0.00004, "spread": 0.00003},
    "USDCAD": {"fee": 0.0, "slippage": 0.00004, "spread": 0.00004},
}
DEFAULT_FOREX_COST = {"fee": 0.0, "slippage": 0.00005, "spread": 0.00005}

PAPER_TRADE_MODE = True
PAPER_TRADE_RISK_PER_TRADE = 0.004
MAX_NOTIONAL_LEVERAGE = 3.0
SIGNAL_BAR_OFFSET = 2
BACKTEST_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
]
TRADE_WATCH_SYMBOLS = [
    "USDJPY",
]
SCAN_ONLY_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
]
PAPER_SIGNAL_INTERVAL = "1h"
TRADE_INTERVALS = ["1h"]
MONITOR_INTERVALS = ["30m"]
DISABLED_INTERVALS = ["2h", "4h"]
BACKTEST_INTERVALS = TRADE_INTERVALS + MONITOR_INTERVALS
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_BARS = 12

APPROVED_SETUPS = {
    ("USDJPY", "1h", "momentum"),
    ("USDJPY", "1h", "pullback_short"),
    ("USDJPY", "30m", "momentum"),
}

SETUP_MODE = {
    ("USDJPY", "1h", "momentum"): "WATCH_ONLY",
    ("USDJPY", "1h", "pullback_short"): "WAIT",
    ("USDJPY", "30m", "momentum"): "MONITOR_ONLY",
}

PAPER_TRADE_GATES = {
    ("USDJPY", "1h", "pullback_short"): {
        "min_adx": 25,
        "min_atr_ratio": 0.8,
        "max_atr_ratio": 1.2,
        "risk_scale": 0.75,
        "note": "Soft watchlist: 1h short pullback in controlled volatility.",
    },
    ("USDJPY", "1h", "momentum"): {
        "min_adx": 20,
        "min_atr_ratio": 1.2,
        "max_atr_ratio": 2.0,
        "risk_scale": 0.60,
        "note": "Soft watchlist: 1h high-volatility trend continuation setup.",
    },
    ("USDJPY", "30m", "momentum"): {
        "min_adx": 20,
        "min_atr_ratio": 1.2,
        "max_atr_ratio": 2.0,
        "risk_scale": 0.40,
        "note": "Monitor only: 30m momentum needs more evidence.",
    },
}

RUN_CONFIG = {
    "model_version": MODEL_VERSION,
    "backtest_symbols": BACKTEST_SYMBOLS,
    "trade_watch_symbols": TRADE_WATCH_SYMBOLS,
    "scan_only_symbols": SCAN_ONLY_SYMBOLS,
    "trade_intervals": TRADE_INTERVALS,
    "monitor_intervals": MONITOR_INTERVALS,
    "disabled_intervals": DISABLED_INTERVALS,
    "approved_setups": [list(item) for item in sorted(APPROVED_SETUPS)],
    "setup_mode": {"|".join(key): value for key, value in SETUP_MODE.items()},
    "paper_trade_risk_per_trade": PAPER_TRADE_RISK_PER_TRADE,
    "max_notional_leverage": MAX_NOTIONAL_LEVERAGE,
    "max_consecutive_losses": MAX_CONSECUTIVE_LOSSES,
    "cooldown_bars": COOLDOWN_BARS,
    "strict_gate": {
        "min_trades": 30,
        "min_profit_factor": 1.30,
        "min_win_rate": 0.40,
        "max_sl_rate": 0.60,
        "max_dd": 0.03,
        "expectancy_gt": 0,
    },
    "soft_gate": {
        "min_trades": 15,
        "min_profit_factor": 1.15,
        "max_dd": 0.04,
        "expectancy_gt": 0,
    },
    "promotion_required_pass_count": 2,
}

# ─────────────────────────────────────────────────────────────
# 0. Data Engine
# ─────────────────────────────────────────────────────────────
def _normalize_forex_symbol(symbol):
    symbol = symbol.replace("/", "").replace("-", "").upper()
    return symbol if symbol.endswith("=X") else f"{symbol}=X"


def _normalize_yfinance_period(lookback):
    if isinstance(lookback, str):
        text = lookback.strip().lower()
        legacy_map = {
            "1 year ago": "1y",
            "2 years ago": "2y",
            "3 years ago": "2y",  # Yahoo limits hourly forex history to roughly two years.
            "30 days ago": "30d",
            "60 days ago": "60d",
            "90 days ago": "90d",
        }
        return legacy_map.get(text, lookback)
    return lookback


def _period_for_interval(period, interval):
    if interval.lower() == "30m" and isinstance(period, str) and period.endswith("y"):
        return "60d"
    return period


def resample_ohlcv(df, interval):
    rule_map = {
        "2h": "2h",
        "4h": "4h",
    }
    rule = rule_map.get(interval.lower())
    if rule is None:
        return df

    resampled = (
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
    return resampled


def fetch_data_api(symbol="EURUSD", interval="1h", lookback="2y"):
    try:
        yf_symbol = _normalize_forex_symbol(symbol)
        display_symbol = yf_symbol.replace("=X", "")
        period = _period_for_interval(_normalize_yfinance_period(lookback), interval)
        request_interval = "1h" if interval.lower() in ["2h", "4h"] else interval
        if CONSOLE_VERBOSE:
            print(f"Fetching forex data: {display_symbol} ({yf_symbol}) | interval={interval} | period={period}")

        data = yf.download(
            tickers=yf_symbol,
            period=period,
            interval=request_interval,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print(f"No data returned for {yf_symbol}")
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

        required_cols = ["time", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns for {yf_symbol}: {missing_cols}")
            return None

        if "volume" not in df.columns:
            df["volume"] = 0.0

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)

        df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
        df = resample_ohlcv(df, interval)
        df["symbol"] = display_symbol
        df["fetched_symbol"] = yf_symbol
        df["fetch_interval"] = interval
        df["data_provider"] = "Yahoo Finance"
        if CONSOLE_VERBOSE:
            print(f"Fetched {len(df)} bars for {display_symbol} ({yf_symbol})")
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def build_features(df):
    df = df.copy()
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ATR_ratio'] = df['ATR'] / (df['ATR'].rolling(50).mean() + 1e-9)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['ret_12'] = df['close'].pct_change(12)
    df['vol_spike'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    df['hour'] = df['time'].dt.hour
    
    # Core Indicators
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['ret_1'] = df['close'].pct_change() # For correlation
    
    return df.dropna()

def create_advanced_labels(df, h=12):
    df = df.copy()
    df['future_max'] = df['high'].rolling(h).max().shift(-h)
    df['future_min'] = df['low'].rolling(h).min().shift(-h)
    df['up_move'] = (df['future_max'] / df['close'] - 1)
    df['down_move'] = (df['future_min'] / df['close'] - 1)
    df['target'] = np.log(df['close'].shift(-h) / df['close'])
    return df


def apply_strategy_signals(df, symbol, interval=PAPER_SIGNAL_INTERVAL, paper_trade_mode=True):
    clean_symbol = symbol.replace("=X", "").replace("/", "").replace("-", "").upper()
    df = df.copy()

    df['strategy_long'] = False
    df['strategy_short'] = False
    df['strategy_type'] = "None"
    df['raw_strategy_type'] = "None"
    df['target_tp'] = 0.0
    df['target_sl'] = 0.0
    df['target_hold'] = 0
    df['paper_gate_pass'] = False
    df['paper_gate_note'] = ""
    df['quality_risk_scale'] = 1.0

    df['trade_score'] = (df['adx'] / 40.0).clip(0.5, 1.5)

    momentum_cond = (
        (df['close'] > df['ema200']) &
        (df['ema50'] > df['ema200']) &
        (df['ret_12'] > 0) &
        (df['ATR_ratio'] > 0.9) &
        (df['body_ratio'] > 0.45) &
        (df['adx'] > 22)
    )

    pullback_cond = (
        (df['close'] > df['ema200']) &
        (df['ret_12'] < 0) &
        (df['ATR_ratio'] > 1.0) &
        (df['adx'] > 20)
    )

    momentum_short_cond = (
        (df['close'] < df['ema200']) &
        (df['ema50'] < df['ema200']) &
        (df['ret_12'] < 0) &
        (df['ATR_ratio'] > 0.9) &
        (df['body_ratio'] > 0.45) &
        (df['adx'] > 22)
    )

    pullback_short_cond = (
        (df['close'] < df['ema200']) &
        (df['ret_12'] > 0) &
        (df['ATR_ratio'] > 1.0) &
        (df['adx'] > 20)
    )

    time_filter = df['hour'].isin([16, 17, 18, 19, 20, 21])

    df.loc[momentum_cond & time_filter, 'strategy_long'] = True
    df.loc[momentum_cond & time_filter, 'strategy_type'] = "momentum"
    df.loc[momentum_cond & time_filter, ['target_tp', 'target_sl', 'target_hold']] = [2.0, 1.2, 12]

    df.loc[pullback_cond & time_filter & ~momentum_cond, 'strategy_long'] = True
    df.loc[pullback_cond & time_filter & ~momentum_cond, 'strategy_type'] = "pullback"
    df.loc[pullback_cond & time_filter & ~momentum_cond, ['target_tp', 'target_sl', 'target_hold']] = [1.6, 1.0, 10]

    df.loc[momentum_short_cond & time_filter, 'strategy_short'] = True
    df.loc[momentum_short_cond & time_filter, 'strategy_type'] = "momentum_short"
    df.loc[momentum_short_cond & time_filter, ['target_tp', 'target_sl', 'target_hold']] = [2.0, 1.2, 12]

    df.loc[pullback_short_cond & time_filter & ~momentum_short_cond, 'strategy_short'] = True
    df.loc[pullback_short_cond & time_filter & ~momentum_short_cond, 'strategy_type'] = "pullback_short"
    df.loc[pullback_short_cond & time_filter & ~momentum_short_cond, ['target_tp', 'target_sl', 'target_hold']] = [1.6, 1.0, 10]

    no_trade = (df['adx'] < 15) & (df['ATR_ratio'] < 0.8)
    df.loc[no_trade, 'strategy_long'] = False
    df.loc[no_trade, 'strategy_short'] = False
    df.loc[no_trade, 'strategy_type'] = "None"

    df['raw_strategy_type'] = df['strategy_type']

    if paper_trade_mode:
        if not any(gate_symbol == clean_symbol and gate_interval == interval for gate_symbol, gate_interval, _ in APPROVED_SETUPS):
            df['strategy_long'] = False
            df['strategy_short'] = False
            df['strategy_type'] = "None"
            df['paper_gate_note'] = "Symbol/interval not approved for paper trading."
            return df

        raw_long = df['strategy_long'].copy()
        raw_short = df['strategy_short'].copy()
        raw_type = df['strategy_type'].copy()

        df['strategy_long'] = False
        df['strategy_short'] = False
        df['strategy_type'] = "None"
        df['target_tp'] = 0.0
        df['target_sl'] = 0.0
        df['target_hold'] = 0

        for (gate_symbol, gate_interval, strategy_type), gate in PAPER_TRADE_GATES.items():
            if (gate_symbol, gate_interval, strategy_type) not in APPROVED_SETUPS:
                continue
            if gate_symbol != clean_symbol or gate_interval != interval:
                continue

            gate_mask = (
                (raw_type == strategy_type) &
                (df['adx'] >= gate["min_adx"]) &
                (df['ATR_ratio'] >= gate["min_atr_ratio"]) &
                (df['ATR_ratio'] <= gate["max_atr_ratio"])
            )

            df.loc[gate_mask, 'strategy_long'] = raw_long[gate_mask]
            df.loc[gate_mask, 'strategy_short'] = raw_short[gate_mask]
            df.loc[gate_mask, 'strategy_type'] = strategy_type
            df.loc[gate_mask, 'paper_gate_pass'] = True
            df.loc[gate_mask, 'paper_gate_note'] = gate["note"]
            df.loc[gate_mask, 'quality_risk_scale'] = gate["risk_scale"]
            df.loc[gate_mask, 'trade_score'] = 1.0

            if strategy_type in ["momentum", "momentum_short"]:
                df.loc[gate_mask, ['target_tp', 'target_sl', 'target_hold']] = [2.0, 1.2, 12]
            else:
                df.loc[gate_mask, ['target_tp', 'target_sl', 'target_hold']] = [1.6, 1.0, 10]

    return df


def get_pair_cost(symbol):
    clean_symbol = symbol.replace("=X", "").replace("/", "").replace("-", "").upper()
    return FOREX_COSTS.get(clean_symbol, DEFAULT_FOREX_COST)


def get_symbol_mode(symbol):
    clean_symbol = symbol.replace("=X", "").replace("/", "").replace("-", "").upper()
    if clean_symbol in TRADE_WATCH_SYMBOLS:
        return "ACTIVE_WATCH"
    if clean_symbol in SCAN_ONLY_SYMBOLS:
        return "SCAN_ONLY"
    return "DISABLED"


def explain_wait_reason(symbol, interval, strategy_type, setup_mode, latest_row):
    if strategy_type in [None, "None"]:
        return "No raw entry signal on the latest closed bar."

    if setup_mode == "WAIT":
        return f"Raw signal exists but setup is WAIT: {symbol} {interval} {strategy_type}."

    if setup_mode == "MONITOR_ONLY":
        return "Setup is MONITOR_ONLY, signal is logged for observation only."

    if setup_mode == "WATCH_ONLY":
        return "Setup is WATCH_ONLY, not promoted to PAPER_TRADE yet."

    return "No approved paper-trade setup on the latest closed bar."


def build_order_ticket(row, symbol, account_equity=10000, allocation=1.0, interval=PAPER_SIGNAL_INTERVAL, setup_modes=None, symbol_mode=None):
    clean_symbol = symbol.replace("=X", "").replace("/", "").replace("-", "").upper()
    signal_long = bool(row.get("strategy_long", False))
    signal_short = bool(row.get("strategy_short", False))
    setup_modes = setup_modes or SETUP_MODE
    symbol_mode = symbol_mode or get_symbol_mode(symbol)
    raw_strategy_type = row.get("raw_strategy_type", "None")
    raw_setup_key = (clean_symbol, interval, raw_strategy_type)
    raw_setup_mode = setup_modes.get(raw_setup_key, "WAIT")
    raw_signal = raw_strategy_type not in [None, "None"]

    if symbol_mode == "SCAN_ONLY":
        return {
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "time": row["time"],
            "symbol": symbol,
            "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
            "data_provider": row.get("data_provider", "Yahoo Finance"),
            "interval": interval,
            "symbol_mode": symbol_mode,
            "signal_time": row["time"],
            "status": "WAIT",
            "strategy_type": raw_strategy_type,
            "raw_signal": raw_signal,
            "setup_mode": "SCAN_ONLY",
            "stage": "SCAN_ONLY",
            "strict_pass_count": None,
            "is_appendable": False,
            "reason": "Symbol is in SCAN_ONLY mode. Backtest/report only, no paper signal allowed.",
        }

    if symbol_mode == "DISABLED":
        return {
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "time": row["time"],
            "symbol": symbol,
            "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
            "data_provider": row.get("data_provider", "Yahoo Finance"),
            "interval": interval,
            "symbol_mode": symbol_mode,
            "signal_time": row["time"],
            "status": "WAIT",
            "strategy_type": raw_strategy_type,
            "raw_signal": raw_signal,
            "setup_mode": "DISABLED",
            "stage": "DISABLED",
            "strict_pass_count": None,
            "is_appendable": False,
            "reason": "Symbol is DISABLED. No backtest paper signal allowed.",
        }

    if not signal_long and not signal_short:
        return {
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "time": row["time"],
            "symbol": symbol,
            "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
            "data_provider": row.get("data_provider", "Yahoo Finance"),
            "interval": interval,
            "symbol_mode": symbol_mode,
            "signal_time": row["time"],
            "status": "WAIT",
            "strategy_type": raw_strategy_type,
            "raw_signal": raw_signal,
            "setup_mode": raw_setup_mode,
            "stage": raw_setup_mode if raw_signal else "NO_RAW_SIGNAL",
            "strict_pass_count": None,
            "is_appendable": False,
            "reason": explain_wait_reason(symbol, interval, raw_strategy_type, raw_setup_mode, row),
        }

    side = "LONG" if signal_long else "SHORT"
    strategy_type = row["strategy_type"]
    setup_key = (clean_symbol, interval, strategy_type)
    if setup_key not in APPROVED_SETUPS:
        return {
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "time": row["time"],
            "symbol": symbol,
            "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
            "data_provider": row.get("data_provider", "Yahoo Finance"),
            "interval": interval,
            "symbol_mode": symbol_mode,
            "signal_time": row["time"],
            "status": "WAIT",
            "strategy_type": strategy_type,
            "raw_signal": True,
            "setup_mode": "WAIT",
            "stage": "UNAPPROVED_SETUP",
            "strict_pass_count": None,
            "is_appendable": False,
            "reason": "Setup not approved for paper trading.",
        }

    status = setup_modes.get(setup_key, "WAIT")
    if status == "WAIT":
        return {
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "time": row["time"],
            "symbol": symbol,
            "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
            "data_provider": row.get("data_provider", "Yahoo Finance"),
            "interval": interval,
            "symbol_mode": symbol_mode,
            "signal_time": row["time"],
            "status": "WAIT",
            "strategy_type": strategy_type,
            "raw_signal": True,
            "setup_mode": status,
            "stage": "WAIT",
            "strict_pass_count": None,
            "is_appendable": False,
            "reason": "Setup mode is WAIT.",
        }

    entry_price = row["close"]
    atr = row["ATR"]
    tp_pct = (row["target_tp"] * atr) / entry_price
    sl_pct = (row["target_sl"] * atr) / entry_price

    if signal_long:
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    risk_budget = account_equity * allocation * PAPER_TRADE_RISK_PER_TRADE
    raw_notional = risk_budget / sl_pct if sl_pct > 0 else 0
    capped_notional = min(raw_notional, account_equity * allocation * MAX_NOTIONAL_LEVERAGE)
    scaled_notional = capped_notional * row.get("quality_risk_scale", 1.0)

    return {
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "time": row["time"],
        "symbol": symbol,
        "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
        "data_provider": row.get("data_provider", "Yahoo Finance"),
        "interval": interval,
        "symbol_mode": symbol_mode,
        "signal_time": row["time"],
        "status": status,
        "side": side,
        "strategy_type": strategy_type,
        "raw_signal": True,
        "setup_mode": status,
        "stage": status,
        "strict_pass_count": None,
        "is_appendable": status != "WAIT",
        "entry": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "target_hold_bars": row["target_hold"],
        "risk_budget": risk_budget,
        "notional_size": scaled_notional,
        "adx": row["adx"],
        "atr_ratio": row["ATR_ratio"],
        "paper_gate_note": row.get("paper_gate_note", ""),
    }


def save_paper_signal(signal, path=None):
    if signal.get("status") == "WAIT":
        return False

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs",
            "paper_signals.csv",
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame([signal])

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(path, index=False)
    return True


def apply_signal_audit_fields(signals_df, promotion_state):
    df = signals_df.copy()

    if "strict_pass_count" not in df.columns:
        df["strict_pass_count"] = None

    if "stage" not in df.columns:
        df["stage"] = None

    if "setup_mode" not in df.columns:
        df["setup_mode"] = df["status"]

    if "raw_signal" not in df.columns:
        df["raw_signal"] = df["strategy_type"].notna() & (df["strategy_type"] != "None")

    if "is_appendable" not in df.columns:
        df["is_appendable"] = df["status"] != "WAIT"

    for idx, row in df.iterrows():
        strategy_type = row.get("strategy_type")
        if pd.isna(strategy_type) or strategy_type in [None, "None"]:
            continue

        key_text = _setup_key_to_string((row["symbol"], row["interval"], strategy_type))
        state_item = promotion_state.get(key_text, {})
        df.at[idx, "strict_pass_count"] = state_item.get("strict_pass_count", 0)
        df.at[idx, "stage"] = state_item.get("last_stage", df.at[idx, "stage"])
        df.at[idx, "setup_mode"] = state_item.get("last_mode", df.at[idx, "setup_mode"])

    return df


def save_daily_signal_summary(signal_report_df, path=None):
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs",
            "daily_signal_summary.csv",
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    summary = (
        signal_report_df
        .groupby(["symbol", "interval", "status"])
        .size()
        .reset_index(name="count")
    )
    summary["date"] = today
    summary["run_id"] = RUN_ID
    summary["model_version"] = MODEL_VERSION

    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, summary], ignore_index=True)
    else:
        out = summary

    out.to_csv(path, index=False)
    return summary


def build_no_trade_reason_summary(signal_report_df):
    return (
        signal_report_df
        .groupby(["symbol", "interval", "reason"])
        .size()
        .reset_index(name="count")
    )


def save_no_trade_reason_summary(signal_report_df, path=None):
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs",
            "no_trade_reason_summary.csv",
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    summary = build_no_trade_reason_summary(signal_report_df)
    summary["date"] = today
    summary["run_id"] = RUN_ID
    summary["model_version"] = MODEL_VERSION

    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, summary], ignore_index=True)
    else:
        out = summary

    out.to_csv(path, index=False)
    return summary


def _json_safe(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_run_config_snapshot(config, run_id, path=None):
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs",
            "run_config_snapshot.json",
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "saved_at": datetime.now().isoformat(),
        "config": _json_safe(config),
    }

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = []
    else:
        old = []

    old.append(payload)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2, ensure_ascii=False)

    return path


def is_strong_paper_ready(row):
    return (
        row["trades"] >= 30 and
        row["profit_factor"] >= 1.30 and
        row["expectancy"] > 0 and
        row["trade_return_max_dd"] <= 0.03 and
        row["strategy_return_sum"] > 0 and
        row["win_rate"] >= 0.40 and
        row["sl_rate"] <= 0.60
    )


def is_soft_paper_ready(row):
    return (
        row["trades"] >= 15 and
        row["profit_factor"] >= 1.15 and
        row["expectancy"] > 0 and
        row["trade_return_max_dd"] <= 0.04 and
        row["strategy_return_sum"] > 0 and
        row["sl_rate"] <= 0.60
    )


def setup_stage(row):
    setup_key = (row["symbol"], row["interval"], row["strategy_type"])
    base_mode = SETUP_MODE.get(setup_key, "WAIT")

    if base_mode == "WAIT":
        return "WAIT"

    if row["trades"] < 10:
        return "IGNORE_SAMPLE_TOO_LOW"

    if row["trades"] < 30:
        if is_soft_paper_ready(row):
            return "MONITOR_ONLY" if base_mode == "MONITOR_ONLY" else "WATCH_ONLY"
        return "WAIT"

    if is_strong_paper_ready(row):
        return "PAPER_TRADE"

    return "WAIT"


def _setup_key_to_string(setup_key):
    return "|".join(setup_key)


def _setup_key_from_string(value):
    symbol, interval, strategy_type = value.split("|")
    return (symbol, interval, strategy_type)


def load_promotion_state(path):
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_promotion_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backup_state_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def backup_state_file(state_path):
    if not os.path.exists(state_path):
        return None

    backup_dir = os.path.join(os.path.dirname(state_path), "state_backups")
    os.makedirs(backup_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"paper_promotion_state_{ts}.json")
    shutil.copy2(state_path, backup_path)
    return backup_path


def _json_float(value):
    value = float(value)
    if np.isinf(value) or np.isnan(value):
        return None
    return value


def update_promotion_state(state, key_text, strict_pass, current_stage, current_mode, row, now):
    item = state.get(key_text, {
        "strict_pass_count": 0,
        "last_stage": None,
        "last_mode": None,
        "last_checked": None,
        "last_strict_pass_time": None,
        "promoted_at": None,
    })

    if "pass_count" in item and "strict_pass_count" not in item:
        item["strict_pass_count"] = item.pop("pass_count")

    item.setdefault("strict_pass_count", 0)
    item.setdefault("last_stage", None)
    item.setdefault("last_mode", None)
    item.setdefault("last_checked", None)
    item.setdefault("last_strict_pass_time", None)
    item.setdefault("promoted_at", None)

    if strict_pass:
        item["strict_pass_count"] = item.get("strict_pass_count", 0) + 1
        item["last_strict_pass_time"] = str(now)
    else:
        item["strict_pass_count"] = 0

    item["last_stage"] = current_stage
    item["last_mode"] = current_mode
    item["last_checked"] = str(now)
    item["last_trades"] = int(row["trades"])
    item["last_profit_factor"] = _json_float(row["profit_factor"])
    item["last_expectancy"] = _json_float(row["expectancy"])

    if item["strict_pass_count"] >= 2 and item.get("promoted_at") is None:
        item["promoted_at"] = str(now)

    state[key_text] = item
    return state


def build_setup_modes(setup_report, promotion_state_path, now=None):
    setup_modes = {key: "WAIT" for key in APPROVED_SETUPS}
    promotion_state = load_promotion_state(promotion_state_path)
    now = now or datetime.now()

    oos_report = setup_report[setup_report["walk_forward_set"] == "OOS 2025+"].copy()
    for _, row in oos_report.iterrows():
        setup_key = (row["symbol"], row["interval"], row["strategy_type"])
        if setup_key not in APPROVED_SETUPS:
            continue

        key_text = _setup_key_to_string(setup_key)
        base_mode = SETUP_MODE.get(setup_key, "WAIT")
        stage = setup_stage(row)
        strict_pass = is_strong_paper_ready(row)
        state_item = promotion_state.get(key_text, {})
        pass_count = state_item.get("strict_pass_count", state_item.get("pass_count", 0))
        next_pass_count = pass_count + 1 if strict_pass else 0

        if base_mode == "MONITOR_ONLY" and stage != "WAIT":
            final_mode = "MONITOR_ONLY"
        elif strict_pass and next_pass_count >= 2:
            final_mode = "PAPER_TRADE"
        elif strict_pass:
            final_mode = "WATCH_ONLY"
        elif stage in ["WATCH_ONLY", "MONITOR_ONLY"]:
            final_mode = stage
        else:
            final_mode = "WAIT"

        promotion_state = update_promotion_state(
            promotion_state,
            key_text,
            strict_pass,
            stage,
            final_mode,
            row,
            now,
        )
        setup_modes[setup_key] = final_mode

    save_promotion_state(promotion_state_path, promotion_state)
    return setup_modes, promotion_state


def analyze_pnl(pnl_list, trades_df=None):
    pnl = np.array(pnl_list, dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    if len(pnl) == 0:
        base_stats = {
            "trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "max_consecutive_loss": 0,
            "tp_rate": 0,
            "sl_rate": 0,
            "timeout_rate": 0,
            "trades_per_month": 0,
        }
        return base_stats

    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    max_consecutive_loss = 0
    current_loss_streak = 0
    for ret in pnl:
        if ret <= 0:
            current_loss_streak += 1
            max_consecutive_loss = max(max_consecutive_loss, current_loss_streak)
        else:
            current_loss_streak = 0

    stats = {
        "trades": len(pnl),
        "win_rate": len(wins) / len(pnl),
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "profit_factor": profit_factor,
        "expectancy": pnl.mean(),
        "max_consecutive_loss": max_consecutive_loss,
        "tp_rate": 0,
        "sl_rate": 0,
        "timeout_rate": 0,
        "trades_per_month": 0,
    }

    if trades_df is not None and not trades_df.empty:
        exit_reason = trades_df["exit_reason"].astype(str)
        stats["tp_rate"] = exit_reason.str.startswith("TP").mean()
        stats["sl_rate"] = exit_reason.str.startswith("SL").mean()
        stats["timeout_rate"] = (exit_reason == "TIMEOUT").mean()

        months = trades_df["exit_time"].dt.to_period("M").nunique()
        stats["trades_per_month"] = len(trades_df) / months if months else 0

    return stats


def backtest_strategy(
    df,
    capital=10000,
    allocation=1.0,
    reference_returns=None,
    symbol="EURUSD",
    risk_per_trade=PAPER_TRADE_RISK_PER_TRADE,
    max_notional_leverage=MAX_NOTIONAL_LEVERAGE,
):
    position = 0
    entry_price = 0
    entry_atr = 0
    entry_time = None
    entry_type = "None"
    entry_adx = 0
    entry_atr_ratio = 0
    hold_time = 0
    current_tp_mult = 0
    current_sl_mult = 0
    current_max_hold = 0
    pyramided = False
    base_size = 0
    size = 0
    
    pnl = []
    trades_log = []
    equity = capital * allocation
    start_equity = equity
    peak_equity = equity
    max_dd = 0
    
    costs = get_pair_cost(symbol)
    fee = costs["fee"]
    slippage = costs["slippage"]
    spread = costs["spread"]
    total_cost = fee + slippage + spread
    last_date = None
    daily_pnl = 0
    loss_streak = 0
    cooldown_until = -1

    # Correlation calculation
    if reference_returns is not None:
        asset_ret = df['ret_1'].reset_index(drop=True)
        aligned_reference = pd.Series(reference_returns).reset_index(drop=True)
        rolling_corr = asset_ret.rolling(50).corr(aligned_reference)
    else:
        rolling_corr = None

    for i in range(len(df)):
        row = df.iloc[i]
        curr_date = row['time'].date()
        
        # Daily Loss Limit (2%)
        if curr_date != last_date:
            daily_pnl = 0
            last_date = curr_date
        
        if daily_pnl < -0.02 * equity: 
            continue

        if position == 0 and i < cooldown_until:
            continue

        # Global Kill Switch (10% from Start)
        if equity < start_equity * 0.9:
            break

        # Asset-level Kill Switch (12% DD)
        if max_dd > 0.12:
            break

        if position == 0:
            is_long_signal = bool(row.get('strategy_long', False))
            is_short_signal = bool(row.get('strategy_short', False))
            if not is_long_signal and not is_short_signal:
                continue

            position = 1 if is_long_signal else -1
            entry_price = row['close']
            entry_atr = row['ATR']
            entry_time = row['time']
            entry_type = row['strategy_type']
            entry_adx = row['adx']
            entry_atr_ratio = row['ATR_ratio']
            hold_time = 0
            pyramided = False
            
            current_tp_mult = row['target_tp']
            current_sl_mult = row['target_sl']
            current_max_hold = row['target_hold']
            
            score = row.get('trade_score', 1.0) * row.get('quality_risk_scale', 1.0)
            stop_loss_pct = (current_sl_mult * entry_atr) / entry_price if entry_price > 0 else 0.02
            
            # Base Size
            size = ((equity * risk_per_trade) / stop_loss_pct) * score if stop_loss_pct > 0 else 0
            size = min(size, equity * max_notional_leverage)
            
            # 1. Correlation Filter: Fix (Rolling 50p > 0.7)
            if rolling_corr is not None:
                c = rolling_corr.iloc[i]
                if pd.notna(c) and c > 0.7:
                    size *= 0.6 

            # 2. Behavior Filter: Last 10 trades winrate < 40%
            if len(pnl) >= 10:
                wr_10 = len([x for x in pnl[-10:] if x > 0]) / 10
                if wr_10 < 0.4:
                    size *= 0.7

            # 3. Tighten DD Guard: if max_dd > 6%
            if max_dd > 0.06:
                size *= 0.7
            
            # 4. Volatility Scaling
            if row['ATR_ratio'] > 1.5:
                size *= 0.7

            base_size = size

        else:
            hold_time += 1
            if position == 1:
                raw_return = (row['close'] / entry_price) - 1
            else:
                raw_return = (entry_price / row['close']) - 1
            
            tp_pct = (current_tp_mult * entry_atr) / entry_price
            sl_pct = (current_sl_mult * entry_atr) / entry_price

            # 3. Pyramid (Fix): Add if win > 50% of TP
            if not pyramided and raw_return > (tp_pct * 0.5) and row['adx'] > 35:
                add_size = base_size * 0.25
                if (size + add_size) <= (base_size * 2.0):
                    size += add_size
                    pyramided = True

            ret = 0
            exited = False
            exit_reason = None
            exit_price = row['close']

            if position == 1:
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)

                if row['low'] <= sl_price and row['high'] >= tp_price:
                    ret = -sl_pct - total_cost
                    exit_price = sl_price
                    exit_reason = "SL_SAME_BAR"
                    exited = True
                elif row['low'] <= sl_price:
                    ret = -sl_pct - total_cost
                    exit_price = sl_price
                    exit_reason = "SL"
                    exited = True
                elif row['high'] >= tp_price:
                    ret = tp_pct - total_cost
                    exit_price = tp_price
                    exit_reason = "TP"
                    exited = True
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

                if row['high'] >= sl_price and row['low'] <= tp_price:
                    ret = -sl_pct - total_cost
                    exit_price = sl_price
                    exit_reason = "SL_SAME_BAR"
                    exited = True
                elif row['high'] >= sl_price:
                    ret = -sl_pct - total_cost
                    exit_price = sl_price
                    exit_reason = "SL"
                    exited = True
                elif row['low'] <= tp_price:
                    ret = tp_pct - total_cost
                    exit_price = tp_price
                    exit_reason = "TP"
                    exited = True

            if not exited and hold_time >= current_max_hold:
                ret = raw_return - total_cost
                exit_price = row['close']
                exit_reason = "TIMEOUT"
                exited = True

            if exited:
                pnl.append(ret)
                trade_profit = size * ret
                equity += trade_profit
                daily_pnl += trade_profit
                trades_log.append({
                    "symbol": symbol,
                    "fetched_symbol": row.get("fetched_symbol", _normalize_forex_symbol(symbol)),
                    "data_provider": row.get("data_provider", "Yahoo Finance"),
                    "entry_time": entry_time,
                    "exit_time": row['time'],
                    "side": "LONG" if position == 1 else "SHORT",
                    "strategy_type": entry_type,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": ret,
                    "profit": trade_profit,
                    "exit_reason": exit_reason,
                    "hold_time": hold_time,
                    "size": size,
                    "fee": fee,
                    "slippage": slippage,
                    "spread": spread,
                    "adx": row['adx'],
                    "atr_ratio": row['ATR_ratio'],
                    "entry_adx": entry_adx,
                    "entry_atr_ratio": entry_atr_ratio,
                })
                
                if equity > peak_equity: peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_dd: max_dd = dd

                if ret < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0

                if loss_streak >= MAX_CONSECUTIVE_LOSSES:
                    cooldown_until = i + COOLDOWN_BARS
                    loss_streak = 0

                position = 0

    return pnl, max_dd, equity, pd.DataFrame(trades_log)

import random


def format_metric_pct(value):
    return f"{value:6.2%}"


def print_backtest_summary(label, pnl_list, max_dd, final_eq, starting_equity, trades_df):
    stats = analyze_pnl(pnl_list, trades_df)
    port_ret = (final_eq - starting_equity) / starting_equity if starting_equity else 0
    profit_factor = stats["profit_factor"]
    pf_text = "inf" if np.isinf(profit_factor) else f"{profit_factor:5.2f}"
    summary = {
        "walk_forward_set": label,
        "trades": stats["trades"],
        "win_rate": stats["win_rate"],
        "return": port_ret,
        "max_dd": max_dd,
        "profit_factor": profit_factor,
        "expectancy": stats["expectancy"],
        "sl_rate": stats["sl_rate"],
        "max_consecutive_loss": stats["max_consecutive_loss"],
        "trades_per_month": stats["trades_per_month"],
    }

    if CONSOLE_VERBOSE:
        print(
            f"{label:<18} | Trades: {stats['trades']:3} | "
            f"WR: {stats['win_rate']:5.2%} | Return: {port_ret:7.2%} | "
            f"MaxDD: {max_dd:6.2%} | PF: {pf_text} | "
            f"Exp: {stats['expectancy']:8.4%} | "
            f"TP/SL/TO: {stats['tp_rate']:4.1%}/{stats['sl_rate']:4.1%}/{stats['timeout_rate']:4.1%} | "
            f"MaxL: {stats['max_consecutive_loss']:2} | T/M: {stats['trades_per_month']:4.1f}"
        )

    if CONSOLE_VERBOSE and trades_df is not None and not trades_df.empty:
        by_strategy = trades_df.groupby("strategy_type")["return"].agg(["count", "mean", "sum"])
        print(by_strategy.to_string(float_format=lambda x: f"{x:0.4%}"))

    return summary


def build_setup_report(trades_result):
    rows = []
    for (symbol, interval, strategy_type, walk_forward_set), group in trades_result.groupby(
        ["symbol", "interval", "strategy_type", "walk_forward_set"]
    ):
        stats = analyze_pnl(group["return"].tolist(), group)
        returns = group["return"].to_numpy()
        eq_curve = np.cumprod(1 + returns) if len(returns) else np.array([])
        peak = np.maximum.accumulate(eq_curve) if len(eq_curve) else np.array([])
        dd = np.max((peak - eq_curve) / peak) if len(eq_curve) else 0
        rows.append({
            "symbol": symbol,
            "interval": interval,
            "strategy_type": strategy_type,
            "walk_forward_set": walk_forward_set,
            "trades": stats["trades"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "expectancy": stats["expectancy"],
            "max_consecutive_loss": stats["max_consecutive_loss"],
            "strategy_return_sum": group["return"].sum(),
            "trade_return_max_dd": dd,
            "tp_rate": stats["tp_rate"],
            "sl_rate": stats["sl_rate"],
            "timeout_rate": stats["timeout_rate"],
            "trades_per_month": stats["trades_per_month"],
        })

    return pd.DataFrame(rows)


def print_key_backtest_summary(summary_rows):
    if not summary_rows:
        print("No backtest summary available.")
        return

    summary_df = pd.DataFrame(summary_rows)
    oos = summary_df[summary_df["walk_forward_set"] == "OOS 2025+"].copy()
    if oos.empty:
        print("No OOS summary available.")
        return

    oos = oos.sort_values(["symbol_mode", "symbol", "interval"])
    display_cols = [
        "symbol", "interval", "symbol_mode", "trades", "win_rate",
        "profit_factor", "expectancy", "max_dd", "sl_rate", "return",
    ]
    print("\nKEY OOS BACKTEST SUMMARY")
    print(oos[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

if __name__ == "__main__":
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    config_snapshot_path = save_run_config_snapshot(RUN_CONFIG, RUN_ID)
    print(f"MODEL VERSION: {MODEL_VERSION} | RUN_ID: {RUN_ID}")
    print(f"Saved run config snapshot: {config_snapshot_path}")

    symbols = BACKTEST_SYMBOLS
    portfolio_config = {symbol: 1.0 for symbol in BACKTEST_SYMBOLS}
    
    all_trades = []
    paper_signals = []
    latest_signal_rows = {}
    promotion_state = {}
    backtest_summary_rows = []
    
    for sym in symbols:
        for interval in BACKTEST_INTERVALS:
            symbol_mode = get_symbol_mode(sym)
            if CONSOLE_VERBOSE:
                print(f"\n" + "="*70)
                print(f"SYMBOL: {sym} | INTERVAL: {interval} | MODE: {symbol_mode}")
                print(f"STRATEGY BACKTEST: {sym} | {interval}")
                print("="*70)
            df = fetch_data_api(symbol=sym, interval=interval, lookback="2y")
            if df is None:
                continue
            df = build_features(df)
            df = apply_strategy_signals(
                df,
                sym,
                interval=interval,
                paper_trade_mode=PAPER_TRADE_MODE and symbol_mode == "ACTIVE_WATCH",
            )

            alloc = portfolio_config.get(sym, 1.0)

            if interval in TRADE_INTERVALS + MONITOR_INTERVALS:
                signal_index = -SIGNAL_BAR_OFFSET if len(df) >= SIGNAL_BAR_OFFSET else -1
                latest_signal_rows[(sym, interval)] = df.iloc[signal_index]
            
            if CONSOLE_VERBOSE:
                print(f"\n--- Walk-forward Analysis (Allocation: {alloc*100}%) ---")
            walk_forward_sets = [
                ("Tune 2023", "2023-01-01", "2024-01-01"),
                ("Valid 2024", "2024-01-01", "2025-01-01"),
                ("OOS 2025+", "2025-01-01", None),
            ]

            for label, test_start, test_end in walk_forward_sets:
                if test_end is None:
                    test = df[df['time'] >= test_start]
                else:
                    test = df[(df['time'] >= test_start) & (df['time'] < test_end)]
                if len(test) == 0:
                    continue

                pnl_list, max_dd, final_eq, trades_df = backtest_strategy(
                    test,
                    capital=10000,
                    allocation=alloc,
                    reference_returns=None,
                    symbol=sym,
                )
                summary_row = print_backtest_summary(label, pnl_list, max_dd, final_eq, 10000 * alloc, trades_df)
                summary_row.update({
                    "symbol": sym,
                    "interval": interval,
                    "symbol_mode": symbol_mode,
                })
                backtest_summary_rows.append(summary_row)

                if not trades_df.empty:
                    trades_df["walk_forward_set"] = label
                    trades_df["interval"] = interval
                    all_trades.append(trades_df)

                if pnl_list:
                    mc_drawdowns = []
                    for _ in range(30):
                        mc_pnl = pnl_list.copy()
                        random.shuffle(mc_pnl)
                        mc_eq = np.cumprod(1 + np.array(mc_pnl))
                        mc_peak = np.maximum.accumulate(mc_eq)
                        mc_dd = np.max((mc_peak - mc_eq) / mc_peak) if len(mc_eq) > 0 else 0
                        mc_drawdowns.append(mc_dd)
                    if CONSOLE_VERBOSE:
                        print(f"{'Monte Carlo':<18} | Avg MC DD: {np.mean(mc_drawdowns):6.4f}")

    print_key_backtest_summary(backtest_summary_rows)

    if all_trades:
        trades_result = pd.concat(all_trades, ignore_index=True)
        trades_result["run_id"] = RUN_ID
        trades_result["model_version"] = MODEL_VERSION
        trades_path = os.path.join(logs_dir, "forex_backtest_trades.csv")
        trades_result.to_csv(trades_path, index=False)
        if CONSOLE_VERBOSE:
            print(f"\nSaved trade log: {trades_path}")

        setup_report = build_setup_report(trades_result)
        setup_report["stage"] = setup_report.apply(setup_stage, axis=1)
        setup_report["run_id"] = RUN_ID
        setup_report["model_version"] = MODEL_VERSION
        setup_report_path = os.path.join(logs_dir, "forex_paper_setup_report.csv")
        setup_report.to_csv(setup_report_path, index=False)

        oos_report = setup_report[setup_report["walk_forward_set"] == "OOS 2025+"].copy()
        oos_report["is_approved_setup"] = oos_report.apply(
            lambda row: (row["symbol"], row["interval"], row["strategy_type"]) in APPROVED_SETUPS,
            axis=1,
        )
        approved_oos_report = oos_report[oos_report["is_approved_setup"]].copy()
        promotion_state_path = os.path.join(logs_dir, "paper_promotion_state.json")
        dynamic_setup_modes, promotion_state = build_setup_modes(setup_report, promotion_state_path)
        strong_ready_report = approved_oos_report[approved_oos_report.apply(is_strong_paper_ready, axis=1)]
        soft_ready_report = approved_oos_report[
            approved_oos_report.apply(is_soft_paper_ready, axis=1) &
            ~approved_oos_report.apply(is_strong_paper_ready, axis=1)
        ]

        if strong_ready_report.empty:
            print("Strong paper readiness: WAIT. No setup passed the strict OOS gate.")
        else:
            print("Strong paper readiness: READY for selective paper trading setups:")
            print(strong_ready_report[[
                "symbol", "interval", "strategy_type", "trades", "win_rate",
                "profit_factor", "expectancy", "trade_return_max_dd"
            ]].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

        if soft_ready_report.empty:
            print("Soft paper watchlist: empty.")
        else:
            print("Soft paper watchlist (monitor only until trades >= 30 and PF >= 1.30):")
            print(soft_ready_report[[
                "symbol", "interval", "strategy_type", "trades", "win_rate",
                "profit_factor", "expectancy", "trade_return_max_dd"
            ]].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

        print("\nCURRENT SETUP MODES")
        for setup_key, mode in sorted(dynamic_setup_modes.items()):
            key_text = _setup_key_to_string(setup_key)
            state_item = promotion_state.get(key_text, {})
            pass_count = state_item.get("strict_pass_count", state_item.get("pass_count", 0))
            print(f"{setup_key}: {mode} | strict_pass_count={pass_count}")

        for (sym, interval), row in latest_signal_rows.items():
                paper_signals.append(build_order_ticket(
                    row,
                    sym,
                    account_equity=10000,
                    allocation=portfolio_config.get(sym, 1.0),
                    interval=interval,
                    setup_modes=dynamic_setup_modes,
                    symbol_mode=get_symbol_mode(sym),
                ))

    if paper_signals:
        signals_path = os.path.join(logs_dir, "forex_paper_signal_report.csv")
        signals_df = pd.DataFrame(paper_signals)
        signals_df = apply_signal_audit_fields(signals_df, promotion_state if all_trades else {})
        signals_df.to_csv(signals_path, index=False)
        daily_summary = save_daily_signal_summary(signals_df)
        reason_summary = save_no_trade_reason_summary(signals_df)

        appended_signals = 0
        for signal in paper_signals:
            if save_paper_signal(signal):
                appended_signals += 1

        print("\n" + "="*70)
        print("LATEST SIGNALS")
        print("="*70)
        console_signals = signals_df[
            (signals_df["symbol_mode"] == "ACTIVE_WATCH") |
            (signals_df["status"] != "WAIT") |
            (signals_df["raw_signal"] == True)
        ].copy()
        if console_signals.empty:
            console_signals = signals_df[signals_df["symbol_mode"] == "ACTIVE_WATCH"].copy()

        display_cols = [col for col in [
            "symbol", "interval", "symbol_mode", "status", "strategy_type",
            "raw_signal", "setup_mode", "stage", "reason"
        ] if col in console_signals.columns]
        print(console_signals[display_cols].to_string(index=False))
        if CONSOLE_VERBOSE:
            print(f"\nSaved latest signal report: {signals_path}")
        print("\nSIGNAL COUNT SUMMARY")
        signal_count_summary = (
            signals_df
            .groupby(["symbol_mode", "status"])
            .size()
            .reset_index(name="count")
        )
        print(signal_count_summary.to_string(index=False))
        print("\nNO-TRADE REASON SUMMARY")
        compact_reason_summary = (
            reason_summary
            .groupby("reason")["count"]
            .sum()
            .reset_index()
            .sort_values("count", ascending=False)
        )
        print(compact_reason_summary.to_string(index=False))
        if appended_signals > 0:
            print(f"Appended actionable/monitor signals: {appended_signals}")
        else:
            print("No non-WAIT signal. Nothing appended to paper_signals.csv.")
