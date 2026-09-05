import hashlib
from datetime import datetime, timezone

from .audit import audit_decision
from .csv_store import append_unique_csv, read_csv
from .market_data import generated_at_from_human_log
from .paths import CRYPTO_LOG_DIR, FOREX_LOG_DIR, SIGNALS_DIR, ensure_dirs


SIGNAL_FIELDS = [
    "signal_id",
    "market",
    "symbol",
    "source_run_at",
    "ingested_at",
    "status",
    "final_bias",
    "decision",
    "signal_type",
    "final_score",
    "confidence_pct",
    "price",
    "best_timeframe",
    "entry_price",
    "entry_zone",
    "stop_loss",
    "take_profit_1",
    "model_profile",
    "watch_threshold",
    "trade_ready_threshold",
    "reason",
    "source_file",
]

SOURCES = {
    "forex": {
        "summary": FOREX_LOG_DIR / "forex_models_summary.csv",
        "human": FOREX_LOG_DIR / "forex_models_human_log.md",
        "out": SIGNALS_DIR / "forex_signals.csv",
    },
    "crypto": {
        "summary": CRYPTO_LOG_DIR / "crypto_models_summary.csv",
        "human": CRYPTO_LOG_DIR / "crypto_models_human_log.md",
        "out": SIGNALS_DIR / "crypto_signals.csv",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def confidence_to_number(value):
    if value is None:
        return ""
    return str(value).replace("%", "").strip()


def signal_type(decision):
    if "TRADE_READY" in decision:
        return "TRADE_READY"
    if "WATCH" in decision:
        return "WATCH"
    return "WAIT"


def make_signal_id(market, row, source_run_at):
    raw = "|".join(
        [
            market,
            row.get("symbol", ""),
            source_run_at,
            row.get("decision", ""),
            row.get("final_score", ""),
            row.get("price", ""),
            row.get("reason", ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def normalize_signal(market, row, source_run_at, source_file):
    return {
        "signal_id": make_signal_id(market, row, source_run_at),
        "market": market,
        "symbol": row.get("symbol", ""),
        "source_run_at": source_run_at,
        "ingested_at": utc_now(),
        "status": "SIGNAL",
        "final_bias": row.get("final_bias", ""),
        "decision": row.get("decision", ""),
        "signal_type": signal_type(row.get("decision", "")),
        "final_score": row.get("final_score", ""),
        "confidence_pct": confidence_to_number(row.get("confidence", "")),
        "price": row.get("price", ""),
        "best_timeframe": row.get("best_timeframe", ""),
        "entry_price": row.get("entry_price", ""),
        "entry_zone": row.get("entry_zone", ""),
        "stop_loss": row.get("stop_loss", ""),
        "take_profit_1": row.get("take_profit_1", ""),
        "model_profile": row.get("model_profile", ""),
        "watch_threshold": row.get("watch_threshold", ""),
        "trade_ready_threshold": row.get("trade_ready_threshold", ""),
        "reason": row.get("reason", ""),
        "source_file": str(source_file),
    }


def ingest_market(market):
    ensure_dirs()
    source = SOURCES[market]
    if not source["summary"].exists():
        return 0
    source_run_at = generated_at_from_human_log(source["human"]) or utc_now()
    rows = [
        normalize_signal(market, row, source_run_at, source["summary"])
        for row in read_csv(source["summary"])
        if row.get("status") == "OK"
    ]
    added = append_unique_csv(source["out"], rows, SIGNAL_FIELDS, "signal_id")
    audit_decision(
        "signals_ingested",
        {
            "market": market,
            "source_run_at": source_run_at,
            "rows_seen": len(rows),
            "rows_added": added,
        },
    )
    return added


def ingest_all():
    results = {market: ingest_market(market) for market in SOURCES}
    return sum(results.values())
