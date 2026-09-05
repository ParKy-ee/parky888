from datetime import datetime, timedelta, timezone

from .cost_model import trade_cost_pct
from .csv_store import read_csv, write_csv_atomic
from .live_market_feed import fetch_candles
from .dataset_labels import apply_dataset_labels
from .paper_trade_blocked import read_blocked
from .paths import AI_DIR, PAPER_DIR, SIGNALS_DIR, ensure_dirs
from .risk_manager import direction_from_signal, to_float
from .settings import load_settings


DATASET_FIELDS = [
    "dataset_label",
    "execution_type",
    "trade_id",
    "signal_id",
    "market",
    "symbol",
    "direction",
    "decision",
    "signal_type",
    "source_run_at",
    "model_score",
    "confidence_pct",
    "timeframe",
    "model_profile",
    "watch_threshold",
    "trade_ready_threshold",
    "entry_time",
    "exit_time",
    "last_checked_at",
    "market_regime",
    "spread_cost_pct",
    "slippage_cost_pct",
    "total_cost_pct",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "risk_reward_ratio",
    "distance_to_sl_pct",
    "distance_to_tp_pct",
    "mfe_pct",
    "mae_pct",
    "forward_return_1h_pct",
    "forward_return_4h_pct",
    "forward_return_24h_pct",
    "exit_price",
    "exit_reason",
    "exit_source",
    "net_return_pct",
    "win",
    "hold_minutes",
    "entry_source",
    "reason",
    "mode",
    "signal_cluster_id",
    "cluster_status",
    "outcome",
    "train_tier",
    "should_enter_label",
    "max_favorable_r",
    "max_adverse_r",
    "schema_version",
]

BLOCKED_DATASET_FIELDS = [
    "dataset_label",
    "execution_type",
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
    "block_rule",
    "block_reason",
    "price_source",
    "train_tier",
    "should_enter_label",
    "model_score",
    "confidence_pct",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "risk_reward_ratio",
    "total_cost_pct",
    "trade_thesis",
    "reason",
    "schema_version",
]

OBSERVATION_FIELDS = [
    "dataset_label",
    "execution_type",
    "signal_id",
    "market",
    "symbol",
    "source_run_at",
    "ingested_at",
    "decision",
    "signal_type",
    "observation_role",
    "direction",
    "final_bias",
    "model_score",
    "confidence_pct",
    "timeframe",
    "model_profile",
    "watch_threshold",
    "trade_ready_threshold",
    "market_regime",
    "spread_cost_pct",
    "slippage_cost_pct",
    "total_cost_pct",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "risk_reward_ratio",
    "distance_to_sl_pct",
    "distance_to_tp_pct",
    "mfe_pct",
    "mae_pct",
    "forward_return_1h_pct",
    "forward_return_4h_pct",
    "forward_return_24h_pct",
    "reason",
    "source_file",
    "mode",
    "signal_cluster_id",
    "outcome",
    "train_tier",
    "should_enter_label",
    "schema_version",
]


from zoneinfo import ZoneInfo

def parse_time(value):
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Bangkok"))
    return parsed.astimezone(timezone.utc)


def round_or_blank(value, digits=4):
    if value is None:
        return ""
    return round(value, digits)


def pct_change(direction, entry, price):
    if entry is None or price is None or entry <= 0:
        return None
    if direction == "SHORT":
        return (entry - price) / entry * 100
    return (price - entry) / entry * 100


def extract_market_regime(reason):
    text = str(reason or "")
    for part in text.replace("|", ",").split(","):
        token = part.strip()
        if token.startswith("regime_"):
            return token
    return ""


PROFILE_REGIME_MAP = {
    "oil_linked_major": "regime_trend",
    "liquid_major": "regime_trend",
    "safe_haven_major": "regime_range",
    "commodity_major": "regime_trend",
    "volatile_major": "regime_chop",
    "high_vol_jpy_major": "regime_chop",
}

SYMBOL_REGIME_MAP = {
    "USDCAD": "regime_trend",
    "EURUSD": "regime_trend",
    "GBPUSD": "regime_chop",
    "USDJPY": "regime_chop",
    "USDCHF": "regime_range",
    "AUDUSD": "regime_trend",
}


def resolve_market_regime(row):
    explicit = str(row.get("market_regime", "") or "").strip()
    if explicit and explicit.upper() != "UNKNOWN":
        return explicit

    profile = str(
        row.get("model_profile")
        or row.get("profile_name")
        or ""
    ).strip().lower()
    if profile in PROFILE_REGIME_MAP:
        return PROFILE_REGIME_MAP[profile]

    symbol = str(row.get("symbol", "") or "").strip().upper()
    if symbol in SYMBOL_REGIME_MAP:
        return SYMBOL_REGIME_MAP[symbol]

    parsed = extract_market_regime(row.get("reason", ""))
    if parsed:
        return parsed
    return "UNKNOWN"


def deduplicate_cluster_rows(enriched_closed, enriched_blocked, observation_rows):
    precedence = {
        "opened_trade": 3,
        "blocked_trade": 2,
        "observation_only": 1
    }
    pool = []
    for r in enriched_closed + enriched_blocked + observation_rows:
        if r.get("signal_cluster_id") and r.get("signal_id"):
            pool.append(r)
    by_signal = {}
    for r in pool:
        sig_id = r["signal_id"]
        outcome = r.get("outcome", "observation_only")
        if sig_id not in by_signal:
            by_signal[sig_id] = r
        else:
            existing_outcome = by_signal[sig_id].get("outcome", "observation_only")
            p_new = precedence.get(outcome, 0)
            p_old = precedence.get(existing_outcome, 0)
            if p_new > p_old:
                by_signal[sig_id] = r
    return list(by_signal.values())


def risk_features(direction, entry, stop_loss, take_profit):
    entry_value = to_float(entry)
    sl = to_float(stop_loss)
    tp = to_float(take_profit)
    if entry_value is None or entry_value <= 0:
        return {
            "risk_reward_ratio": "",
            "distance_to_sl_pct": "",
            "distance_to_tp_pct": "",
        }

    sl_distance = abs(entry_value - sl) if sl is not None else None
    tp_distance = abs(tp - entry_value) if tp is not None else None
    rr = tp_distance / sl_distance if sl_distance and tp_distance is not None else None

    return {
        "risk_reward_ratio": round_or_blank(rr),
        "distance_to_sl_pct": round_or_blank(sl_distance / entry_value * 100 if sl_distance is not None else None),
        "distance_to_tp_pct": round_or_blank(tp_distance / entry_value * 100 if tp_distance is not None else None),
    }


CANDLE_CACHE = {}


def candle_features(market, symbol, start_time, timeframe, direction, entry_price):
    entry = to_float(entry_price)
    parsed_start = parse_time(start_time)
    if entry is None or parsed_start is None:
        return {
            "mfe_pct": "",
            "mae_pct": "",
            "forward_return_1h_pct": "",
            "forward_return_4h_pct": "",
            "forward_return_24h_pct": "",
        }

    start_iso = parsed_start.isoformat()
    cache_key = f"{market}|{symbol}|{timeframe or '1h'}|{start_iso}"
    if cache_key in CANDLE_CACHE:
        candles = CANDLE_CACHE[cache_key]
    else:
        try:
            candles = fetch_candles(market, symbol, start_iso, timeframe or "1h")
            CANDLE_CACHE[cache_key] = candles
        except Exception:
            candles = []

    if not candles:
        return {
            "mfe_pct": "",
            "mae_pct": "",
            "forward_return_1h_pct": "",
            "forward_return_4h_pct": "",
            "forward_return_24h_pct": "",
        }

    favorable = []
    adverse = []
    for candle in candles:
        high = to_float(candle.get("high"))
        low = to_float(candle.get("low"))
        if direction == "SHORT":
            favorable.append(pct_change(direction, entry, low))
            adverse.append(pct_change(direction, entry, high))
        else:
            favorable.append(pct_change(direction, entry, high))
            adverse.append(pct_change(direction, entry, low))

    def close_after(hours):
        target = parsed_start + timedelta(hours=hours)
        selected = None
        for candle in candles:
            candle_time = parse_time(candle.get("time"))
            if candle_time and candle_time <= target:
                selected = candle
            elif candle_time and candle_time > target:
                break
        return to_float((selected or candles[-1]).get("close"))

    return {
        "mfe_pct": round_or_blank(max(value for value in favorable if value is not None), 4) if favorable else "",
        "mae_pct": round_or_blank(min(value for value in adverse if value is not None), 4) if adverse else "",
        "forward_return_1h_pct": round_or_blank(pct_change(direction, entry, close_after(1)), 4),
        "forward_return_4h_pct": round_or_blank(pct_change(direction, entry, close_after(4)), 4),
        "forward_return_24h_pct": round_or_blank(pct_change(direction, entry, close_after(24)), 4),
    }


def cost_features(market, symbol, settings):
    spread = float(settings.get("market_cost_pct", {}).get(market, 0))
    slippage = float(settings.get("slippage_pct", {}).get(market, 0))
    total = trade_cost_pct(market, symbol, settings)
    return {
        "spread_cost_pct": spread,
        "slippage_cost_pct": slippage,
        "total_cost_pct": total,
    }


def feature_row_base(row, settings, start_time):
    market = row.get("market", "")
    symbol = row.get("symbol", "")
    decision = row.get("decision", "")
    final_bias = row.get("final_bias", "")
    direction = row.get("direction") or direction_from_signal(decision, final_bias)
    entry_price = row.get("entry_price") or row.get("price")
    timeframe = row.get("timeframe") or row.get("best_timeframe")
    reason = row.get("reason", "")

    features = {
        "direction": direction,
        "timeframe": timeframe,
        "market_regime": resolve_market_regime(row),
        "model_profile": row.get("model_profile") if str(row.get("model_profile")).lower() != "nan" else row.get("symbol", "UNKNOWN"),
        "watch_threshold": row.get("watch_threshold", ""),
        "trade_ready_threshold": row.get("trade_ready_threshold", ""),
        **cost_features(market, symbol, settings),
        **risk_features(direction, entry_price, row.get("stop_loss"), row.get("take_profit_1")),
        **candle_features(market, symbol, start_time, timeframe, direction, entry_price),
    }
    return features


def enrich_closed_trade(row, settings, signal_lookup=None):
    # If source_run_at is missing, try to look it up from signals
    if not row.get("source_run_at") and signal_lookup:
        sig = signal_lookup.get(row.get("signal_id"))
        if sig:
            row["source_run_at"] = sig.get("source_run_at")

    features = feature_row_base(row, settings, row.get("entry_time") or row.get("source_run_at"))
    
    # Separate results of candle_path / historical_replay / live paper
    exit_src = row.get("exit_source", "unknown")
    if exit_src == "candle_path":
        exec_type = "paper_candle_path"
    elif exit_src == "historical_replay":
        exec_type = "paper_historical_replay"
    elif exit_src == "latest_log_fallback" or exit_src == "latest_candle_close":
        exec_type = "paper_live_track"
    else:
        exec_type = f"paper_{exit_src}"

    labeled = {
        **row,
        **features,
        "dataset_label": "paper_simulated",
        "execution_type": exec_type,
        "outcome": "opened_trade",
    }
    return apply_dataset_labels(labeled)


def observation_role(signal):
    signal_type = signal.get("signal_type", "")
    if signal_type == "TRADE_READY":
        return "trade_candidate"
    if signal_type == "WATCH":
        return "watch_observation"
    return "wait_observation"


def enrich_blocked_row(row, settings):
    pseudo = {
        **row,
        "market": row.get("market", ""),
        "symbol": row.get("symbol", ""),
        "decision": row.get("decision", ""),
        "final_bias": "",
        "direction": row.get("direction", ""),
        "entry_price": row.get("entry_price", ""),
        "stop_loss": row.get("stop_loss", ""),
        "take_profit_1": row.get("take_profit_1", ""),
        "reason": row.get("reason", ""),
        "timeframe": row.get("timeframe", ""),
    }
    features = feature_row_base(pseudo, settings, row.get("source_run_at") or row.get("blocked_at"))
    labeled = {
        **pseudo,
        **features,
        "dataset_label": "blocked_or_observation",
        "execution_type": f"blocked_{row.get('outcome', 'unknown')}",
        "outcome": row.get("outcome", ""),
        "mode": row.get("mode", ""),
        "signal_cluster_id": row.get("signal_cluster_id", ""),
        "cluster_status": row.get("cluster_status", ""),
        "block_rule": row.get("block_rule", ""),
        "block_reason": row.get("block_reason", ""),
        "price_source": row.get("price_source", ""),
        "trade_thesis": row.get("trade_thesis", ""),
        "source_run_at": row.get("source_run_at", ""),
    }
    return apply_dataset_labels(labeled)


def enrich_observation(signal, settings, blocked_lookup=None):
    features = feature_row_base(signal, settings, signal.get("source_run_at") or signal.get("ingested_at"))
    blocked = (blocked_lookup or {}).get(signal.get("signal_id"))
    outcome = blocked.get("outcome", "observation_only") if blocked else "observation_only"
    labeled = {
        "dataset_label": "signal_observation",
        "execution_type": "not_executed_observation",
        "signal_id": signal.get("signal_id", ""),
        "market": signal.get("market", ""),
        "symbol": signal.get("symbol", ""),
        "source_run_at": signal.get("source_run_at", ""),
        "ingested_at": signal.get("ingested_at", ""),
        "decision": signal.get("decision", ""),
        "signal_type": signal.get("signal_type", ""),
        "observation_role": observation_role(signal),
        "final_bias": signal.get("final_bias", ""),
        "model_score": signal.get("final_score", ""),
        "confidence_pct": signal.get("confidence_pct", ""),
        "entry_price": signal.get("entry_price") or signal.get("price", ""),
        "stop_loss": signal.get("stop_loss", ""),
        "take_profit_1": signal.get("take_profit_1", ""),
        "reason": signal.get("reason", ""),
        "source_file": signal.get("source_file", ""),
        "mode": blocked.get("mode", "research_mode") if blocked else "research_mode",
        "signal_cluster_id": blocked.get("signal_cluster_id", "") if blocked else "",
        "outcome": outcome,
        **features,
    }
    return apply_dataset_labels(labeled)


def export_blocked_analytics(blocked_rows):
    summary = {}
    for row in blocked_rows:
        key = row.get("block_rule", "unknown")
        bucket = summary.setdefault(
            key,
            {"block_rule": key, "count": 0, "blocked": 0, "observation_only": 0},
        )
        bucket["count"] += 1
        if row.get("outcome") == "blocked_trade":
            bucket["blocked"] += 1
        else:
            bucket["observation_only"] += 1
    return list(summary.values())


def export_ai_dataset():
    ensure_dirs()
    settings = load_settings()
    CANDLE_CACHE.clear()
    
    # Load signals for lookup
    signal_rows = read_csv(SIGNALS_DIR / "forex_signals.csv") + read_csv(SIGNALS_DIR / "crypto_signals.csv")
    signal_lookup = {s.get("signal_id"): s for s in signal_rows if s.get("signal_id")}

    # Process closed trades
    closed_rows = read_csv(PAPER_DIR / "closed_trades.csv")
    enriched_closed = []
    for row in closed_rows:
        if to_float(row.get("hold_minutes")) == 0:
            continue
        
        # Filter if model_profile is nan
        profile = row.get("model_profile") or ""
        if str(profile).lower() == "nan" or not profile:
            continue
            
        enriched_closed.append(enrich_closed_trade(row, settings, signal_lookup))
    
    out_path = AI_DIR / "ai_training_dataset.csv"
    write_csv_atomic(out_path, enriched_closed, DATASET_FIELDS)

    blocked_rows = read_blocked()
    blocked_lookup = {r.get("signal_id"): r for r in blocked_rows if r.get("signal_id")}
    enriched_blocked = [enrich_blocked_row(row, settings) for row in blocked_rows]
    blocked_path = AI_DIR / "blocked_trade_dataset.csv"
    write_csv_atomic(blocked_path, enriched_blocked, BLOCKED_DATASET_FIELDS)

    analytics = export_blocked_analytics(blocked_rows)
    analytics_path = AI_DIR / "blocked_trade_analytics.csv"
    write_csv_atomic(
        analytics_path,
        analytics,
        ["block_rule", "count", "blocked", "observation_only"],
    )

    observation_rows = [
        enrich_observation(row, settings, blocked_lookup) for row in signal_rows
    ]
    obs_path = AI_DIR / "signal_observation_dataset.csv"
    write_csv_atomic(obs_path, observation_rows, OBSERVATION_FIELDS)

    cluster_path = AI_DIR / "cluster_opportunity_dataset.csv"
    cluster_rows = deduplicate_cluster_rows(enriched_closed, enriched_blocked, observation_rows)
    write_csv_atomic(
        cluster_path,
        cluster_rows,
        ["signal_cluster_id", "cluster_status", "outcome", "symbol", "decision", "train_tier"],
    )

    return {
        "training_rows": len(enriched_closed),
        "observation_rows": len(observation_rows),
        "blocked_rows": len(enriched_blocked),
        "cluster_rows": len(cluster_rows),
    }
