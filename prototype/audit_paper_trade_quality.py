import pandas as pd
from pathlib import Path

# Adjust paths to match project structure
BASE_DIR = Path(__file__).parent
CLOSED_FILE = BASE_DIR / "data" / "paper_trades" / "closed_trades.csv"
OPEN_FILE = BASE_DIR / "data" / "paper_trades" / "open_trades.csv"

VALID_EXIT_SOURCES = {
    "candle_path",
    "latest_candle_close",
    "time_exit",
    "historical_replay",
}

DISABLED_TIMEFRAMES = {"4h"}
MONITOR_ONLY_TIMEFRAMES = {"30m"}


def load_csv(path):
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_text(series):
    return series.astype(str).str.strip().str.lower()


def audit_closed_trades(df):
    print("\n==============================")
    print("CLOSED TRADES AUDIT")
    print("==============================")

    if df.empty:
        print("No closed trades.")
        return

    total = len(df)
    print(f"Total closed trades: {total}")

    # 1. latest_log_fallback check
    if "exit_source" in df.columns:
        exit_source = normalize_text(df["exit_source"])
        fallback_count = (exit_source == "latest_log_fallback").sum()
        print(f"latest_log_fallback count: {fallback_count}")

        verified_df = df[exit_source.isin(VALID_EXIT_SOURCES)]
        print(f"Verified evaluation trades: {len(verified_df)}")

        if not verified_df.empty and "net_return_pct" in verified_df.columns:
            net_sum = verified_df["net_return_pct"].sum()
            avg_return = verified_df["net_return_pct"].mean()

            if "win" in verified_df.columns:
                # Handle string Booleans 'True'/'False'
                win_bool = normalize_text(verified_df["win"]) == "true"
                win_rate = win_bool.mean() * 100
            else:
                win_rate = None

            print(f"Verified net return: {net_sum:.4f}%")
            print(f"Verified avg/trade: {avg_return:.4f}%")

            if win_rate is not None:
                print(f"Verified win rate: {win_rate:.2f}%")

    else:
        print("[WARN] Missing column: exit_source")

    # 2. timeframe nan check
    if "timeframe" in df.columns:
        tf = normalize_text(df["timeframe"])
        nan_tf = df[tf.isin(["nan", "", "none"])]
        print(f"timeframe nan/empty count: {len(nan_tf)}")

        if len(nan_tf) > 0:
            print("[WARN] Found rows with missing timeframe:")
            print(nan_tf[["trade_id", "symbol", "reason"]].head(10).to_string(index=False))
    else:
        print("[WARN] Missing column: timeframe")

    # 3. replay metadata check
    required_cols = ["replay_verified", "replay_timeframe", "ambiguous_intrabar", "evaluation_included"]

    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] Missing replay metadata column: {col}")
        else:
            missing_count = df[col].isna().sum()
            print(f"{col} missing count: {missing_count}")

    # 4. disabled timeframe check
    if "timeframe" in df.columns:
        disabled_rows = df[normalize_text(df["timeframe"]).isin(DISABLED_TIMEFRAMES)]
        print(f"Closed trades from disabled timeframes: {len(disabled_rows)}")

        if len(disabled_rows) > 0:
            print("[WARN] Found closed trades from disabled timeframe.")
            print(disabled_rows[["trade_id", "symbol", "timeframe", "entry_time", "exit_time"]].head(10).to_string(index=False))


def audit_open_trades(df):
    print("\n==============================")
    print("OPEN TRADES AUDIT")
    print("==============================")

    if df.empty:
        print("No open trades.")
        return

    print(f"Total open trades: {len(df)}")

    if "timeframe" in df.columns:
        disabled_rows = df[normalize_text(df["timeframe"]).isin(DISABLED_TIMEFRAMES)]
        print(f"Open trades from disabled timeframes: {len(disabled_rows)}")

        if len(disabled_rows) > 0:
            print("[ERROR] New order from disabled timeframe still exists.")
            print(disabled_rows[["trade_id", "symbol", "timeframe", "entry_time"]].head(10).to_string(index=False))
    else:
        print("[WARN] Missing column: timeframe")

    if "status" in df.columns:
        print(df["status"].value_counts(dropna=False).to_string())


def main():
    closed_df = load_csv(CLOSED_FILE)
    open_df = load_csv(OPEN_FILE)

    audit_closed_trades(closed_df)
    audit_open_trades(open_df)

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
