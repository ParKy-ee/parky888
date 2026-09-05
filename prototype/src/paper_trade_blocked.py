from .csv_store import read_csv, write_csv_atomic
from .paths import PAPER_DIR
from .risk_manager import direction_from_signal


BLOCKED_PATH = PAPER_DIR / "paper_trade_blocked.csv"

BLOCKED_FIELDS = [
    "blocked_at",
    "outcome",
    "mode",
    "signal_id",
    "signal_cluster_id",
    "cluster_status",
    "market",
    "symbol",
    "direction",
    "decision",
    "signal_type",
    "source_run_at",
    "model_score",
    "confidence_pct",
    "block_rule",
    "block_reason",
    "price_source",
    "current_price",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "timeframe",
    "trade_thesis",
    "reason",
    "price_drift_pct",
    "schema_version",
]


def read_blocked():
    return read_csv(BLOCKED_PATH)


def build_blocked_row(
    signal,
    *,
    outcome,
    mode,
    block_rule,
    block_reason,
    signal_cluster_id="",
    cluster_status="",
    price_source="",
    current_price=None,
    drift_pct=None,
    trade_thesis="",
    blocked_at,
):
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    return {
        "blocked_at": blocked_at,
        "outcome": outcome,
        "mode": mode,
        "signal_id": signal.get("signal_id", ""),
        "signal_cluster_id": signal_cluster_id,
        "cluster_status": cluster_status,
        "market": signal.get("market", "forex"),
        "symbol": signal.get("symbol", ""),
        "direction": direction,
        "decision": signal.get("decision", ""),
        "signal_type": signal.get("signal_type", ""),
        "source_run_at": signal.get("source_run_at", ""),
        "model_score": signal.get("final_score", ""),
        "confidence_pct": signal.get("confidence_pct", ""),
        "block_rule": block_rule,
        "block_reason": block_reason,
        "price_source": price_source,
        "current_price": current_price if current_price is not None else "",
        "entry_price": signal.get("entry_price") or signal.get("price", ""),
        "stop_loss": signal.get("stop_loss", ""),
        "take_profit_1": signal.get("take_profit_1", ""),
        "timeframe": signal.get("timeframe", ""),
        "trade_thesis": trade_thesis,
        "reason": signal.get("reason", ""),
        "price_drift_pct": round(drift_pct, 4) if drift_pct is not None else "",
    }


def append_blocked_rows(existing_rows, new_rows):
    if not new_rows:
        return
    write_csv_atomic(BLOCKED_PATH, existing_rows + new_rows, BLOCKED_FIELDS)
