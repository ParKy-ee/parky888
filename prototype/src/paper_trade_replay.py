import os
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from .cost_model import trade_cost_pct
from .csv_store import read_csv, write_csv_atomic
from .paths import AUDIT_DIR, PAPER_DIR
from .paper_trade_tracker import CLOSED_FIELDS, OPEN_FIELDS
from .settings import load_settings

OPEN_TRADES_FILE = PAPER_DIR / "open_trades.csv"
CLOSED_TRADES_FILE = PAPER_DIR / "closed_trades.csv"
REPLAY_LOG_FILE = AUDIT_DIR / "paper_trade_replay_log.csv"

REPLAY_TIMEFRAME = mt5.TIMEFRAME_M1

SYMBOL_MAP = {
    # Populate when a broker-specific suffix/prefix is required.
}


def clean_value(value):
    if pd.isna(value) or value is None:
        return ""
    return value


def normalize_trade_row(row, fields):
    normalized = {field: clean_value(row.get(field, "")) for field in fields}

    if "initial_stop_loss" in fields and not normalized["initial_stop_loss"]:
        normalized["initial_stop_loss"] = clean_value(row.get("stop_loss", ""))
    if "initial_risk_distance" in fields and not normalized["initial_risk_distance"]:
        normalized["initial_risk_distance"] = clean_value(row.get("risk_distance", ""))
    if "schema_version" in fields and not normalized["schema_version"]:
        normalized["schema_version"] = "2.0.0"

    return normalized


def merge_rows(existing_rows, new_rows, fields, key_field):
    merged = []
    seen = set()

    for row in existing_rows + new_rows:
        normalized = normalize_trade_row(row, fields)
        key = normalized.get(key_field, "")
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(normalized)

    return merged


def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    return True


def shutdown_mt5():
    mt5.shutdown()


def parse_time(value):
    if pd.isna(value) or value == "":
        return None
    try:
        dt = pd.to_datetime(value, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None


def fetch_candles(symbol, start_time, end_time, timeframe=REPLAY_TIMEFRAME):
    real_symbol = SYMBOL_MAP.get(symbol, symbol)
    if not mt5.symbol_select(real_symbol, True):
        print(f"Failed to select symbol {real_symbol}")
        return pd.DataFrame()

    rates = mt5.copy_rates_range(real_symbol, timeframe, start_time, end_time)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    return df[["time", "open", "high", "low", "close"]]


def replay_one_trade(trade, candles):
    direction = str(trade["direction"]).upper()
    sl = float(trade["stop_loss"])
    tp = float(trade["take_profit_1"])

    for _, candle in candles.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        candle_time = candle["time"]

        if direction == "LONG":
            hit_sl = low <= sl
            hit_tp = high >= tp

            if hit_sl and hit_tp:
                return {
                    "closed": True,
                    "exit_reason": "SL_INTRABAR_AMBIGUOUS",
                    "exit_price": sl,
                    "exit_time": candle_time,
                    "hit_sl": True,
                    "hit_tp": True,
                }
            if hit_sl:
                return {
                    "closed": True,
                    "exit_reason": "SL",
                    "exit_price": sl,
                    "exit_time": candle_time,
                    "hit_sl": True,
                    "hit_tp": False,
                }
            if hit_tp:
                return {
                    "closed": True,
                    "exit_reason": "TP1",
                    "exit_price": tp,
                    "exit_time": candle_time,
                    "hit_sl": False,
                    "hit_tp": True,
                }

        elif direction == "SHORT":
            hit_sl = high >= sl
            hit_tp = low <= tp

            if hit_sl and hit_tp:
                return {
                    "closed": True,
                    "exit_reason": "SL_INTRABAR_AMBIGUOUS",
                    "exit_price": sl,
                    "exit_time": candle_time,
                    "hit_sl": True,
                    "hit_tp": True,
                }
            if hit_sl:
                return {
                    "closed": True,
                    "exit_reason": "SL",
                    "exit_price": sl,
                    "exit_time": candle_time,
                    "hit_sl": True,
                    "hit_tp": False,
                }
            if hit_tp:
                return {
                    "closed": True,
                    "exit_reason": "TP1",
                    "exit_price": tp,
                    "exit_time": candle_time,
                    "hit_sl": False,
                    "hit_tp": True,
                }

    return {
        "closed": False,
        "exit_reason": None,
        "exit_price": None,
        "exit_time": None,
        "hit_sl": False,
        "hit_tp": False,
    }


def calc_pnl_pct(trade, exit_price):
    direction = str(trade["direction"]).upper()
    entry = float(trade["entry_price"])
    if direction == "LONG":
        return ((exit_price - entry) / entry) * 100
    if direction == "SHORT":
        return ((entry - exit_price) / entry) * 100
    return 0.0


def log_replay_event(trade_id, symbol, direction, start_time, end_time, result, exit_reason, exit_price, exit_time):
    log_row = {
        "run_time": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_id,
        "symbol": symbol,
        "direction": direction,
        "start_time": start_time.isoformat() if start_time else "",
        "end_time": end_time.isoformat() if end_time else "",
        "result": result,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "exit_time": exit_time.isoformat() if exit_time else "",
    }
    df_new = pd.DataFrame([log_row])
    file_exists = os.path.exists(REPLAY_LOG_FILE)
    df_new.to_csv(REPLAY_LOG_FILE, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


def run_replay_sync():
    if not os.path.exists(OPEN_TRADES_FILE):
        print("No open_trades.csv found.")
        return

    open_rows = [normalize_trade_row(row, OPEN_FIELDS) for row in read_csv(OPEN_TRADES_FILE)]
    if not open_rows:
        print("No open trades.")
        return

    if not init_mt5():
        return

    settings = load_settings()

    try:
        now = datetime.now(timezone.utc)
        still_open_rows = []
        closed_rows = []

        for trade in open_rows:
            status = str(trade.get("status", "PAPER_OPEN")).upper()
            if status not in {"PAPER_OPEN", "OPEN"}:
                still_open_rows.append(trade)
                continue

            trade_id = trade.get("trade_id", "unknown")
            symbol = str(trade["symbol"])
            direction = str(trade["direction"]).upper()
            market = str(trade.get("market", "forex"))

            entry_time = parse_time(trade["entry_time"])
            last_checked = parse_time(trade.get("last_checked_at", ""))
            start_time = last_checked if last_checked else entry_time

            if start_time is None:
                print(f"Skipping trade {trade_id} due to missing entry_time")
                still_open_rows.append(trade)
                continue

            print(f"\nReplay trade: {trade_id} {symbol}")
            print(f"From {start_time} to {now}")

            candles = fetch_candles(symbol, start_time, now)

            if candles.empty:
                print(f"No candles found for {symbol}. Keep trade open and retry later.")
                still_open_rows.append(trade)
                continue

            result = replay_one_trade(trade, candles)

            if result["closed"]:
                exit_price = float(result["exit_price"])
                exit_time = result["exit_time"]
                gross_pnl = calc_pnl_pct(trade, exit_price)
                cost = trade_cost_pct(market, symbol, settings)
                net_pnl = gross_pnl - cost

                closed_row = dict(trade)
                closed_row["status"] = "PAPER_CLOSED"
                closed_row["exit_reason"] = result["exit_reason"]
                closed_row["exit_price"] = round(exit_price, 8)
                closed_row["exit_time"] = exit_time.isoformat()
                closed_row["exit_source"] = "historical_replay"
                closed_row["gross_return_pct"] = round(gross_pnl, 4)
                closed_row["cost_pct"] = round(cost, 4)
                closed_row["net_return_pct"] = round(net_pnl, 4)
                closed_row["win"] = "True" if net_pnl > 0 else "False"

                if entry_time:
                    closed_row["hold_minutes"] = int((exit_time - entry_time).total_seconds() / 60)

                closed_row["label_quality"] = "HIGH"
                closed_row["is_trainable"] = "true"
                closed_row["replay_verified"] = "True"
                closed_row["replay_timeframe"] = "M1"
                closed_row["ambiguous_intrabar"] = "True" if result["exit_reason"] == "SL_INTRABAR_AMBIGUOUS" else "False"
                closed_row["evaluation_included"] = "True"

                closed_rows.append(closed_row)

                log_replay_event(
                    trade_id,
                    symbol,
                    direction,
                    start_time,
                    now,
                    "CLOSED",
                    result["exit_reason"],
                    exit_price,
                    exit_time,
                )

                print(f"CLOSED {symbol} | reason={result['exit_reason']} | exit={exit_price} | pnl={net_pnl:.4f}%")
            else:
                trade["last_checked_at"] = now.isoformat()
                still_open_rows.append(trade)
                print(f"Still OPEN {symbol}. Updated last_checked_at.")

        if closed_rows:
            existing_closed = [normalize_trade_row(row, CLOSED_FIELDS) for row in read_csv(CLOSED_TRADES_FILE)]
            merged_closed = merge_rows(existing_closed, closed_rows, CLOSED_FIELDS, "trade_id")
            write_csv_atomic(CLOSED_TRADES_FILE, merged_closed, CLOSED_FIELDS)

        normalized_open = [normalize_trade_row(row, OPEN_FIELDS) for row in still_open_rows]
        write_csv_atomic(OPEN_TRADES_FILE, normalized_open, OPEN_FIELDS)

        print("\nReplay sync completed.")

    except Exception as e:
        print(f"Error during replay sync: {e}")
        import traceback

        traceback.print_exc()
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    run_replay_sync()
