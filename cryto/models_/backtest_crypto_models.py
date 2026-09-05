import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from crypto_forecast_common import (
    LOGS_DIR,
    add_features,
    detect_regime,
    evaluate_trade_setup,
    fetch_data,
    params_for,
    score_row,
    trade_levels,
)


SYMBOLS = [
    ("BTCUSD", "BTC-USD"),
    ("ETHUSD", "ETH-USD"),
    ("SOLUSD", "SOL-USD"),
    ("BNBUSD", "BNB-USD"),
    ("XRPUSD", "XRP-USD"),
]

FEE_RATE = 0.001


def bias_from_score(score, params):
    if score >= params["watch_score"]:
        return "BULLISH"
    if score <= -params["watch_score"]:
        return "BEARISH"
    return "NEUTRAL"


def pct_return(direction, entry, exit_price):
    if direction == "LONG":
        return (exit_price - entry) / entry
    return (entry - exit_price) / entry


def row_snapshot(row, interval, symbol):
    params = params_for(symbol)
    score, reasons = score_row(row, params)
    bias = bias_from_score(score, params)
    regime = detect_regime(row, params)
    stop_loss, take_profit = trade_levels(float(row["close"]), float(row["atr"]), bias, params)
    return {
        "interval": interval,
        "price": float(row["close"]),
        "bias": bias,
        "action": "WAIT" if bias == "NEUTRAL" else f"{bias}_WATCH",
        "score": score,
        "regime": regime,
        "adx": float(row["adx"]),
        "rsi": float(row["rsi"]),
        "atr": float(row["atr"]),
        "atr_ratio": float(row["atr_ratio"]),
        "volume_ratio": float(row["volume_ratio"]),
        "support": float(row["support"]),
        "resistance": float(row["resistance"]),
        "stop_loss": stop_loss,
        "take_profit_1": take_profit,
        "reasons": ",".join(reasons + [f"regime_{regime.lower()}"]),
    }


def prepare_data(yf_symbol):
    data = {}
    for interval in ["1h", "4h", "1d"]:
        data[interval] = add_features(fetch_data(yf_symbol, interval)).sort_values("time").reset_index(drop=True)
    return data


def latest_before(df, timestamp):
    eligible = df[df["time"] <= timestamp]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def simulate_symbol(symbol, yf_symbol):
    params = params_for(symbol)
    data = prepare_data(yf_symbol)
    trigger_df = data["4h"]
    trades = []
    index = 220

    while index < len(trigger_df) - 2:
        signal = trigger_df.iloc[index]
        signal_time = signal["time"]
        rows = []

        for interval in ["1h", "4h", "1d"]:
            source_row = signal if interval == "4h" else latest_before(data[interval], signal_time)
            if source_row is None:
                rows = []
                break
            rows.append(row_snapshot(source_row, interval, symbol))

        if not rows:
            index += 1
            continue

        final_score, final_bias, decision, _alignment, decision_reasons = evaluate_trade_setup(rows, symbol)
        if "TRADE_READY" not in decision:
            index += 1
            continue

        entry_bar = trigger_df.iloc[index + 1]
        entry = float(entry_bar["open"])
        direction = "LONG" if "BULLISH" in decision else "SHORT"
        stop_loss, take_profit = trade_levels(entry, float(signal["atr"]), final_bias, params)
        exit_price = float(trigger_df.iloc[min(index + params["max_hold_bars"], len(trigger_df) - 1)]["close"])
        exit_reason = "TIME_EXIT"
        exit_index = min(index + params["max_hold_bars"], len(trigger_df) - 1)

        for forward_index in range(index + 1, min(index + params["max_hold_bars"] + 1, len(trigger_df))):
            bar = trigger_df.iloc[forward_index]
            high = float(bar["high"])
            low = float(bar["low"])

            if direction == "LONG":
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                    exit_index = forward_index
                    break
                if high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP1"
                    exit_index = forward_index
                    break
            else:
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                    exit_index = forward_index
                    break
                if low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP1"
                    exit_index = forward_index
                    break

        gross_return = pct_return(direction, entry, exit_price)
        net_return = gross_return - (FEE_RATE * 2)
        trades.append(
            {
                "symbol": symbol,
                "timeframe": "4h",
                "entry_time": entry_bar["time"],
                "direction": direction,
                "decision": decision,
                "final_score": final_score,
                "entry": entry,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "net_return_pct": net_return * 100,
                "win": net_return > 0,
                "decision_reasons": ",".join(decision_reasons),
            }
        )
        index = exit_index + 1

    return trades


def summarize(trades):
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate_pct": 0.0,
            "total_return_pct": 0.0,
            "avg_return_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "profit_factor": 0.0,
        }

    returns = [float(trade["net_return_pct"]) for trade in trades]
    wins = sum(1 for trade in trades if trade["win"])
    gross_win = sum(value for value in returns if value > 0)
    gross_loss = abs(sum(value for value in returns if value < 0))
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "winrate_pct": wins / len(trades) * 100,
        "total_return_pct": sum(returns),
        "avg_return_pct": sum(returns) / len(returns),
        "best_trade_pct": max(returns),
        "worst_trade_pct": min(returns),
        "profit_factor": gross_win / gross_loss if gross_loss else 0.0,
    }


def main():
    all_trades = []
    summary_rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for symbol, yf_symbol in SYMBOLS:
        trades = simulate_symbol(symbol, yf_symbol)
        all_trades.extend(trades)
        row = {"run_id": run_id, "symbol": symbol, "timeframe": "4h_mtf_confirmed", **summarize(trades)}
        summary_rows.append(row)

    portfolio = {"run_id": run_id, "symbol": "PORTFOLIO", "timeframe": "4h_mtf_confirmed", **summarize(all_trades)}
    summary_rows.append(portfolio)

    trades_path = Path(LOGS_DIR) / "crypto_model_trade_backtest_trades.csv"
    summary_path = Path(LOGS_DIR) / "crypto_model_trade_backtest_summary.csv"

    if all_trades:
        pd.DataFrame(all_trades).to_csv(trades_path, index=False)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Crypto model trade backtest")
    print("Timeframe: 4h with 1h/1d confirmation")
    print(f"Saved summary: {summary_path}")
    print(f"Saved trades: {trades_path}")
    print("")
    for row in summary_rows:
        print(
            f"{row['symbol']}: trades={row['trades']} | winrate={row['winrate_pct']:.2f}% | "
            f"return={row['total_return_pct']:.2f}% | avg={row['avg_return_pct']:.2f}% | "
            f"pf={row['profit_factor']:.2f}"
        )


if __name__ == "__main__":
    main()
