import hashlib
import time
from datetime import datetime, timezone

from .audit import audit_execution, log_system_error
from .cost_model import trade_cost_pct
from .csv_store import read_csv, write_csv_atomic
from .live_market_feed import fallback_price, fetch_candles, fetch_executable_price
from .paper_trade_blocked import append_blocked_rows, build_blocked_row, read_blocked
from .paths import PAPER_DIR, SIGNALS_DIR, ensure_dirs
from .risk_manager import calculate_position, direction_from_signal, to_float
from .settings import load_settings
from .entry_quality import run_entry_quality_guards
from .signal_clustering import (
    build_cluster_index,
    compute_signal_cluster_id,
    register_cluster,
    resolve_cluster_status,
)
from .trade_lifecycle import apply_lifecycle
from .trade_context import extract_timeframe_from_reason, get_trade_thesis, parse_source_run_at

MODE_RESEARCH = "research_mode"
MODE_REALISTIC = "realistic_paper_mode"


OPEN_FIELDS = [
    "trade_id",
    "signal_id",
    "mode",
    "signal_cluster_id",
    "cluster_status",
    "market",
    "symbol",
    "status",
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
    "entry_price",
    "stop_loss",
    "initial_stop_loss",
    "initial_risk_distance",
    "take_profit_1",
    "quantity",
    "lot_size",
    "notional_usd",
    "risk_pct",
    "risk_amount_usd",
    "risk_distance",
    "entry_source",
    "last_checked_at",
    "last_market_source",
    "reason",
    "trade_thesis",
    "correlation_group",
    "active_same_thesis_count",
    "recent_sl_same_thesis_4h",
    "entry_delay_minutes",
    "entry_price_drift_pct",
    "trading_session",
    "market_regime",
    "volatility_bucket",
    "replay_verified",
    "replay_timeframe",
    "ambiguous_intrabar",
    "evaluation_included",
    "highest_favorable_price",
    "lowest_adverse_price",
    "max_favorable_r",
    "max_adverse_r",
    "breakeven_activated",
    "partial_close_done",
    "remaining_quantity_pct",
    "realized_partial_pct",
    "partial_close_count",
    "trailing_stop_active",
    "schema_version",
]

CLOSED_FIELDS = OPEN_FIELDS + [
    "exit_time",
    "exit_price",
    "exit_reason",
    "exit_source",
    "gross_return_pct",
    "cost_pct",
    "net_return_pct",
    "win",
    "hold_minutes",
    "label_quality",
    "is_trainable",
]

EVENT_FIELDS = [
    "event_id",
    "trade_id",
    "signal_id",
    "event_time",
    "event_type",
    "market",
    "symbol",
    "message",
]

SUMMARY_FIELDS = [
    "generated_at",
    "open_trades",
    "closed_trades",
    "wins",
    "losses",
    "winrate_pct",
    "total_net_return_pct",
    "avg_net_return_pct",
]
DECISION_FIELDS = [
    "decision_time",
    "outcome",
    "mode",
    "signal_cluster_id",
    "cluster_status",
    "signal_id",
    "symbol",
    "direction",
    "decision",
    "model_score",
    "confidence_pct",
    "final_action",
    "skip_reason",
    "block_rule",
    "price_source",
    "entry_allowed",
    "risk_allowed",
    "correlation_allowed",
    "stale_signal",
    "price_drift_pct",
    "current_price",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "schema_version",
]



OPEN_PATH = PAPER_DIR / "open_trades.csv"
CLOSED_PATH = PAPER_DIR / "closed_trades.csv"
EVENT_PATH = PAPER_DIR / "trade_events.csv"
SUMMARY_PATH = PAPER_DIR / "paper_trade_summary.csv"
DECISION_PATH = PAPER_DIR / "paper_trade_decisions.csv"


def utc_now():
    return datetime.now(timezone.utc)


def utc_now_text():
    return utc_now().isoformat()


def make_id(*parts):
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


def get_trading_session():
    hour = utc_now().hour
    if 0 <= hour < 7: return "ASIA"
    if 7 <= hour < 12: return "LONDON"
    if 12 <= hour < 16: return "LONDON_NY_OVERLAP"
    if 16 <= hour < 21: return "NEW_YORK"
    return "TRANSITION"


def log_decision(
    signal,
    action,
    skip_reason,
    current_price=None,
    drift_pct=None,
    *,
    outcome="",
    mode="",
    signal_cluster_id="",
    cluster_status="",
    block_rule="",
    price_source="",
):
    existing = read_csv(DECISION_PATH)
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    row = {
        "decision_time": utc_now_text(),
        "outcome": outcome,
        "mode": mode,
        "signal_cluster_id": signal_cluster_id,
        "cluster_status": cluster_status,
        "signal_id": signal.get("signal_id"),
        "symbol": signal.get("symbol"),
        "direction": direction,
        "decision": signal.get("decision"),
        "model_score": signal.get("final_score"),
        "confidence_pct": signal.get("confidence_pct"),
        "final_action": action,
        "skip_reason": skip_reason,
        "block_rule": block_rule,
        "price_source": price_source,
        "entry_allowed": "true" if action == "OPEN_TRADE" else "false",
        "risk_allowed": "true" if skip_reason != "max_positions_total" else "false",
        "correlation_allowed": "true" if skip_reason != "max_positions_per_symbol" else "false",
        "stale_signal": "true" if skip_reason == "stale_signal" else "false",
        "price_drift_pct": round(drift_pct, 4) if drift_pct is not None else "",
        "current_price": current_price,
        "entry_price": signal.get("entry_price") or signal.get("price"),
        "stop_loss": signal.get("stop_loss"),
        "take_profit_1": signal.get("take_profit_1"),
    }
    try:
        write_csv_atomic(DECISION_PATH, existing + [row], DECISION_FIELDS)
    except Exception as error:
        log_system_error(
            "log_decision",
            error,
            signal_id=signal.get("signal_id"),
            action=action,
            skip_reason=skip_reason,
        )
        raise


def get_recent_sl_count(thesis, closed_trades, hours=4):
    now = utc_now()
    count = 0
    for trade in closed_trades:
        if trade.get("trade_thesis") == thesis and trade.get("exit_reason") == "SL":
            try:
                exit_time = datetime.fromisoformat(trade["exit_time"])
                if (now - exit_time).total_seconds() / 3600 <= hours:
                    count += 1
            except:
                continue
    return count


def passes_data_leakage_guard(signal, planned_entry_time):
    source_time = parse_source_run_at(signal.get("source_run_at"))
    if source_time is None:
        return False, "missing_or_invalid_source_run_at"
    if planned_entry_time <= source_time:
        return False, "entry_time_not_after_signal_time"
    return True, "ok"


def read_signals():
    return read_csv(SIGNALS_DIR / "forex_signals.csv") + read_csv(SIGNALS_DIR / "crypto_signals.csv")


def _parse_iso_dt(value):
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _latest_decision_time(decision_rows):
    latest = None
    for row in decision_rows:
        ts = _parse_iso_dt(row.get("decision_time"))
        if ts and (latest is None or ts > latest):
            latest = ts
    return latest


def write_event(trade_id, signal_id, event_type, market, symbol, message):
    existing = read_csv(EVENT_PATH)
    event = {
        "event_id": make_id(trade_id, event_type, utc_now_text()),
        "trade_id": trade_id,
        "signal_id": signal_id,
        "event_time": utc_now_text(),
        "event_type": event_type,
        "market": market,
        "symbol": symbol,
        "message": message,
    }
    write_csv_atomic(EVENT_PATH, existing + [event], EVENT_FIELDS)


def can_open_signal(signal, open_trades, settings):
    if signal.get("decision") not in settings["paper_entry_decisions"]:
        return False, "decision_not_allowed"
    market = signal.get("market", "forex")
    symbol = signal.get("symbol", "unknown")
    reason = signal.get("reason", "")
    timeframe = signal.get("timeframe") or extract_timeframe_from_reason(reason)

    if timeframe in settings.get("disabled_timeframes", []):
        return False, f"timeframe_{timeframe}_disabled"
    
    if timeframe in settings.get("monitor_only_timeframes", []):
        return False, f"timeframe_{timeframe}_monitor_only"

    if any(row.get("signal_id") == signal.get("signal_id") for row in open_trades):
        return False, "signal_already_open"
    if len(open_trades) >= int(settings["max_positions_total"]):
        return False, "max_positions_total"
    same_symbol = [row for row in open_trades if row.get("market") == signal.get("market") and row.get("symbol") == signal.get("symbol")]
    if len(same_symbol) >= int(settings["max_positions_per_symbol"]):
        return False, "max_positions_per_symbol"
    
    closed_trades = read_csv(CLOSED_PATH)

    # Cooldown Guard: Prevent re-entry if the symbol was recently traded
    last_symbol_trade = None
    for trade in reversed(closed_trades):
        if trade.get("symbol") == symbol:
            last_symbol_trade = trade
            break
    
    if last_symbol_trade:
        try:
            exit_time = datetime.fromisoformat(last_symbol_trade["exit_time"].replace("Z", "+00:00"))
            cooldown_hrs = float(settings.get("cooldown_hours_standard", 6))
            if "GBP" in symbol:
                cooldown_hrs = float(settings.get("cooldown_hours_gbp", 12))
            
            hours_since_exit = (utc_now() - exit_time).total_seconds() / 3600
            if hours_since_exit < cooldown_hrs:
                return False, f"cooldown_active_{round(cooldown_hrs - hours_since_exit, 1)}h_remaining"
        except Exception:
            pass

    return True, "ok"


def _signal_timeframe(signal):
    reason = signal.get("reason", "")
    timeframe = signal.get("timeframe", "")
    if not timeframe or str(timeframe).lower() == "nan":
        timeframe = extract_timeframe_from_reason(reason) or "1h"
    return timeframe


def _record_blocked(ctx, signal, *, outcome, mode, block_rule, block_reason, cluster_id, cluster_status, price_source="", current_price=None, drift_pct=None):
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    thesis = get_trade_thesis(signal.get("symbol", ""), direction, signal.get("market", "forex"))
    row = build_blocked_row(
        signal,
        outcome=outcome,
        mode=mode,
        block_rule=block_rule,
        block_reason=block_reason,
        signal_cluster_id=cluster_id,
        cluster_status=cluster_status,
        price_source=price_source,
        current_price=current_price,
        drift_pct=drift_pct,
        trade_thesis=thesis,
        blocked_at=ctx["now_text"],
    )
    ctx["blocked_new"].append(row)
    if outcome == "blocked_trade":
        register_cluster(ctx["cluster_index"], cluster_id, signal.get("signal_id"))

    action = "OPEN_TRADE" if outcome == "opened_trade" else f"SKIP_{block_rule.upper()}"
    log_decision(
        signal,
        action,
        block_reason,
        current_price,
        drift_pct,
        outcome=outcome,
        mode=mode,
        signal_cluster_id=cluster_id,
        cluster_status=cluster_status,
        block_rule=block_rule,
        price_source=price_source,
    )


def simulate_candle_exit(trade, candles):
    sl = to_float(trade.get("stop_loss"))
    tp = to_float(trade.get("take_profit_1"))
    if sl is None or tp is None:
        return None

    for candle in candles:
        high = to_float(candle.get("high"))
        low = to_float(candle.get("low"))
        if high is None or low is None:
            continue

        if trade["direction"] == "LONG":
            sl_hit = low <= sl
            tp_hit = high >= tp
            if sl_hit and tp_hit:
                return sl, "SL_AMBIGUOUS_FIRST", "candle_path_pessimistic", candle.get("time")
            if sl_hit:
                return sl, "SL", "candle_path", candle.get("time")
            if tp_hit:
                return tp, "TP1", "candle_path", candle.get("time")
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp
            if sl_hit and tp_hit:
                return sl, "SL_AMBIGUOUS_FIRST", "candle_path_pessimistic", candle.get("time")
            if sl_hit:
                return sl, "SL", "candle_path", candle.get("time")
            if tp_hit:
                return tp, "TP1", "candle_path", candle.get("time")

    return None


def open_from_signals():
    ensure_dirs()
    settings = load_settings()
    signals = read_signals()
    if not signals:
        return 0

    open_trades = read_csv(OPEN_PATH)
    closed_trades = read_csv(CLOSED_PATH)
    blocked_existing = read_blocked()
    decision_existing = read_csv(DECISION_PATH)
    closed_signal_ids = {row.get("signal_id") for row in closed_trades}
    decided_signal_ids = {row.get("signal_id") for row in decision_existing if row.get("signal_id")}
    latest_decision_ts = _latest_decision_time(decision_existing)
    research_on = bool(settings.get("research_mode", False))
    realistic_on = bool(settings.get("realistic_paper_mode", True))
    process_only_new_signals = bool(settings.get("process_only_new_signals", True))

    if process_only_new_signals:
        filtered = []
        for signal in signals:
            signal_id = signal.get("signal_id")
            if signal_id in decided_signal_ids:
                continue
            if latest_decision_ts is not None:
                ingested_at = _parse_iso_dt(signal.get("ingested_at"))
                if ingested_at is None or ingested_at <= latest_decision_ts:
                    continue
            filtered.append(signal)
        signals = filtered

    if not signals:
        return 0

    latest_ts_map = {}
    for s in signals:
        key = (s.get("market", "forex"), s.get("symbol", "unknown"))
        ts = parse_source_run_at(s.get("source_run_at"))
        if ts and (key not in latest_ts_map or ts > latest_ts_map[key]):
            latest_ts_map[key] = ts

    ctx = {
        "settings": settings,
        "research_on": research_on,
        "realistic_on": realistic_on,
        "now": utc_now(),
        "now_text": utc_now_text(),
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "blocked_new": [],
        "cluster_index": build_cluster_index(open_trades, blocked_existing, closed_trades, settings),
        "latest_ts_map": latest_ts_map,
        "closed_signal_ids": closed_signal_ids,
        "ignore_superseded_rule": bool(settings.get("ignore_superseded_rule", False)),
    }
    added = 0

    for signal in signals:
        if _try_open_signal(signal, ctx):
            added += 1

    append_blocked_rows(blocked_existing, ctx["blocked_new"])
    return added


def _try_open_signal(signal, ctx):
    settings = ctx["settings"]
    signal_id = signal.get("signal_id")
    if signal_id in ctx["closed_signal_ids"]:
        cluster_id = compute_signal_cluster_id(signal, ctx["settings"])
        cluster_status = resolve_cluster_status(cluster_id, ctx["cluster_index"], signal_id)
        _record_blocked(
            ctx,
            signal,
            outcome="observation_only",
            mode=MODE_RESEARCH,
            block_rule="already_closed",
            block_reason="signal_already_has_closed_trade",
            cluster_id=cluster_id,
            cluster_status=cluster_status,
        )
        return False

    market = signal.get("market", "forex")
    symbol = signal.get("symbol", "unknown")
    planned_entry_time = ctx["now"]
    source_time = parse_source_run_at(signal.get("source_run_at"))
    cluster_id = compute_signal_cluster_id(signal, settings)
    cluster_status = resolve_cluster_status(cluster_id, ctx["cluster_index"], signal_id)
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    signal_type = signal.get("signal_type", "")

    latest_ts = ctx["latest_ts_map"].get((market, symbol))
    if (not ctx["ignore_superseded_rule"]) and source_time and latest_ts and source_time < latest_ts:
        _record_blocked(
            ctx,
            signal,
            outcome="observation_only",
            mode=MODE_RESEARCH,
            block_rule="superseded_signal",
            block_reason="not_latest_source_run_for_symbol",
            cluster_id=cluster_id,
            cluster_status=cluster_status,
        )
        return False

    if ctx["research_on"] and signal_type == "WATCH":
        _record_blocked(
            ctx,
            signal,
            outcome="observation_only",
            mode=MODE_RESEARCH,
            block_rule="watch_observation",
            block_reason="watch_signals_not_opened_in_realistic_mode",
            cluster_id=cluster_id,
            cluster_status=cluster_status,
            price_source="none",
        )
        return False

    if not ctx["realistic_on"]:
        if ctx["research_on"]:
            _record_blocked(
                ctx,
                signal,
                outcome="observation_only",
                mode=MODE_RESEARCH,
                block_rule="realistic_mode_disabled",
                block_reason="realistic_paper_mode_off",
                cluster_id=cluster_id,
                cluster_status=cluster_status,
            )
        return False

    if source_time:
        age_minutes = (planned_entry_time - source_time).total_seconds() / 60
        if age_minutes > 60:
            _record_blocked(
                ctx,
                signal,
                outcome="blocked_trade",
                mode=MODE_REALISTIC,
                block_rule="stale_signal",
                block_reason="signal_older_than_60_minutes",
                cluster_id=cluster_id,
                cluster_status=cluster_status,
            )
            return False

    leakage_ok, leakage_reason = passes_data_leakage_guard(signal, planned_entry_time)
    if not leakage_ok:
        _record_blocked(
            ctx,
            signal,
            outcome="blocked_trade",
            mode=MODE_REALISTIC,
            block_rule="data_leakage",
            block_reason=leakage_reason,
            cluster_id=cluster_id,
            cluster_status=cluster_status,
        )
        return False

    if cluster_status == "same_cluster":
        _record_blocked(
            ctx,
            signal,
            outcome="blocked_trade",
            mode=MODE_REALISTIC,
            block_rule="duplicate_cluster",
            block_reason="signal_cluster_already_has_trade_or_block",
            cluster_id=cluster_id,
            cluster_status=cluster_status,
        )
        return False

    timeframe = _signal_timeframe(signal)
    current_price, price_source = fetch_executable_price(market, symbol, timeframe)
    if current_price is None or price_source == "latest_log_fallback":
        _record_blocked(
            ctx,
            signal,
            outcome="blocked_trade",
            mode=MODE_REALISTIC,
            block_rule="no_executable_price",
            block_reason="missing_live_or_candle_close_price",
            cluster_id=cluster_id,
            cluster_status=cluster_status,
            price_source=price_source or "none",
        )
        return False

    sl = to_float(signal.get("stop_loss"))
    tp = to_float(signal.get("take_profit_1"))
    entry_val = to_float(signal.get("entry_price") or signal.get("price"))

    if sl and tp:
        if direction == "LONG":
            if current_price <= sl:
                _record_blocked(
                    ctx,
                    signal,
                    outcome="blocked_trade",
                    mode=MODE_REALISTIC,
                    block_rule="price_hit_sl",
                    block_reason="current_price_at_or_below_stop",
                    cluster_id=cluster_id,
                    cluster_status=cluster_status,
                    price_source=price_source,
                    current_price=current_price,
                )
                return False
            if current_price >= tp:
                _record_blocked(
                    ctx,
                    signal,
                    outcome="blocked_trade",
                    mode=MODE_REALISTIC,
                    block_rule="price_hit_tp",
                    block_reason="current_price_at_or_above_tp1",
                    cluster_id=cluster_id,
                    cluster_status=cluster_status,
                    price_source=price_source,
                    current_price=current_price,
                )
                return False
        elif direction == "SHORT":
            if current_price >= sl:
                _record_blocked(
                    ctx,
                    signal,
                    outcome="blocked_trade",
                    mode=MODE_REALISTIC,
                    block_rule="price_hit_sl",
                    block_reason="current_price_at_or_above_stop",
                    cluster_id=cluster_id,
                    cluster_status=cluster_status,
                    price_source=price_source,
                    current_price=current_price,
                )
                return False
            if current_price <= tp:
                _record_blocked(
                    ctx,
                    signal,
                    outcome="blocked_trade",
                    mode=MODE_REALISTIC,
                    block_rule="price_hit_tp",
                    block_reason="current_price_at_or_below_tp1",
                    cluster_id=cluster_id,
                    cluster_status=cluster_status,
                    price_source=price_source,
                    current_price=current_price,
                )
                return False

    drift_pct = None
    if entry_val:
        drift_pct = abs(current_price - entry_val) / entry_val * 100
        max_drift = float(settings.get("max_entry_price_drift_pct", 0.05))
        if drift_pct > max_drift:
            _record_blocked(
                ctx,
                signal,
                outcome="blocked_trade",
                mode=MODE_REALISTIC,
                block_rule="excessive_drift",
                block_reason="entry_price_drift_above_limit",
                cluster_id=cluster_id,
                cluster_status=cluster_status,
                price_source=price_source,
                current_price=current_price,
                drift_pct=drift_pct,
            )
            return False

    allowed, reason = can_open_signal(signal, ctx["open_trades"], settings)
    if not allowed:
        _record_blocked(
            ctx,
            signal,
            outcome="blocked_trade",
            mode=MODE_REALISTIC,
            block_rule=reason,
            block_reason=reason,
            cluster_id=cluster_id,
            cluster_status=cluster_status,
            price_source=price_source,
            current_price=current_price,
            drift_pct=drift_pct,
        )
        return False

    entry_price = current_price
    position = calculate_position(
        market=market,
        symbol=signal["symbol"],
        direction=direction,
        entry_price=entry_price,
        stop_loss=signal.get("stop_loss"),
        balance_usd=float(settings["account_balance_usd"]),
        risk_pct=float(settings["risk_pct_per_trade"]),
        settings=settings,
    )
    if position is None:
        _record_blocked(
            ctx,
            signal,
            outcome="blocked_trade",
            mode=MODE_REALISTIC,
            block_rule="position_sizing_failed",
            block_reason="could_not_compute_position_size",
            cluster_id=cluster_id,
            cluster_status=cluster_status,
            price_source=price_source,
            current_price=current_price,
            drift_pct=drift_pct,
        )
        return False

    quality_ok, quality_rule, quality_reason = run_entry_quality_guards(
        signal,
        ctx["open_trades"],
        ctx["closed_trades"],
        settings,
        current_price,
        position=position,
    )
    if not quality_ok:
        _record_blocked(
            ctx,
            signal,
            outcome="blocked_trade",
            mode=MODE_REALISTIC,
            block_rule=quality_rule,
            block_reason=quality_reason,
            cluster_id=cluster_id,
            cluster_status=cluster_status,
            price_source=price_source,
            current_price=current_price,
            drift_pct=drift_pct,
        )
        return False

    reason = signal.get("reason", "")
    profile = signal.get("model_profile") or signal.get("profile_name") or signal.get("symbol", "UNKNOWN")
    if str(profile).lower() == "nan":
        profile = signal.get("symbol", "UNKNOWN")

    trade_id = make_id(signal["signal_id"], market, signal["symbol"], entry_price)
    thesis = get_trade_thesis(signal["symbol"], direction, market)
    trade = {
        "trade_id": trade_id,
        "signal_id": signal["signal_id"],
        "mode": MODE_REALISTIC,
        "signal_cluster_id": cluster_id,
        "cluster_status": cluster_status,
        "market": market,
        "symbol": signal["symbol"],
        "status": "PAPER_OPEN",
        "direction": direction,
        "decision": signal["decision"],
        "signal_type": signal_type,
        "source_run_at": signal.get("source_run_at", ""),
        "model_score": signal.get("final_score", ""),
        "confidence_pct": signal.get("confidence_pct", ""),
        "timeframe": timeframe,
        "model_profile": profile,
        "watch_threshold": signal.get("watch_threshold", ""),
        "trade_ready_threshold": signal.get("trade_ready_threshold", ""),
        "entry_time": planned_entry_time.isoformat(),
        "entry_price": entry_price,
        "stop_loss": signal.get("stop_loss", ""),
        "initial_stop_loss": signal.get("stop_loss", ""),
        "initial_risk_distance": position["risk_distance"],
        "take_profit_1": signal.get("take_profit_1", ""),
        "quantity": position["quantity"],
        "lot_size": position["lot_size"],
        "notional_usd": position["notional_usd"],
        "risk_pct": settings["risk_pct_per_trade"],
        "risk_amount_usd": position["risk_amount_usd"],
        "risk_distance": position["risk_distance"],
        "entry_source": price_source,
        "last_checked_at": planned_entry_time.isoformat(),
        "last_market_source": price_source,
        "reason": reason,
        "trade_thesis": thesis,
        "correlation_group": thesis,
        "active_same_thesis_count": len(
            [
                t
                for t in ctx["open_trades"]
                if get_trade_thesis(t["symbol"], t["direction"], t.get("market", "forex")) == thesis
            ]
        ),
        "recent_sl_same_thesis_4h": get_recent_sl_count(thesis, ctx["closed_trades"], 4),
        "entry_delay_minutes": round((planned_entry_time - source_time).total_seconds() / 60, 2) if source_time else 0,
        "entry_price_drift_pct": round(drift_pct, 4) if drift_pct is not None else 0,
        "trading_session": get_trading_session(),
        "market_regime": signal.get("market_regime", "UNKNOWN"),
        "volatility_bucket": signal.get("volatility_bucket", "NORMAL"),
        "replay_verified": "False",
        "replay_timeframe": "",
        "ambiguous_intrabar": "False",
        "highest_favorable_price": entry_price,
        "lowest_adverse_price": entry_price,
        "max_favorable_r": 0,
        "max_adverse_r": 0,
        "breakeven_activated": "False",
        "partial_close_done": "False",
        "remaining_quantity_pct": 1.0,
        "realized_partial_pct": 0.0,
        "partial_close_count": 0,
        "trailing_stop_active": "False",
    }
    ctx["open_trades"].append(trade)
    write_csv_atomic(OPEN_PATH, ctx["open_trades"], OPEN_FIELDS)
    register_cluster(ctx["cluster_index"], cluster_id, signal_id)
    write_event(trade_id, signal["signal_id"], "PAPER_OPEN", market, signal["symbol"], f"Opened at {entry_price}")
    log_decision(
        signal,
        "OPEN_TRADE",
        "",
        current_price,
        drift_pct,
        outcome="opened_trade",
        mode=MODE_REALISTIC,
        signal_cluster_id=cluster_id,
        cluster_status=cluster_status,
        block_rule="",
        price_source=price_source,
    )
    audit_execution("paper_trade_opened", trade)
    return True


def gross_return(direction, entry, exit_price):
    if direction == "LONG":
        return (exit_price - entry) / entry * 100
    return (entry - exit_price) / entry * 100


def close_trade(trade, exit_price, exit_reason, exit_source, settings, exit_time=None):
    entry = to_float(trade["entry_price"])
    exit_value = to_float(exit_price)
    if entry is None or exit_value is None:
        return None
    if exit_time is None:
        exit_time = utc_now()
    elif isinstance(exit_time, str):
        exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

    gross = gross_return(trade["direction"], entry, exit_value)
    cost = trade_cost_pct(trade["market"], trade["symbol"], settings)
    net = gross - cost
    opened = datetime.fromisoformat(trade["entry_time"])
    hold_minutes = int((exit_time - opened).total_seconds() / 60)
    
    label_quality = "HIGH" if exit_source == "candle_path" and hold_minutes > 0 else "LOW"
    is_trainable = "true" if label_quality == "HIGH" and hold_minutes > 1 and exit_reason in ["SL", "TP1"] else "false"

    return {
        **trade,
        "status": "PAPER_CLOSED",
        "last_checked_at": exit_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "exit_price": round(exit_value, 8),
        "exit_reason": exit_reason,
        "exit_source": exit_source,
        "gross_return_pct": round(gross, 4),
        "cost_pct": round(cost, 4),
        "net_return_pct": round(net, 4),
        "win": "True" if net > 0 else "False",
        "hold_minutes": hold_minutes,
        "label_quality": label_quality,
        "is_trainable": is_trainable,
        "replay_verified": trade.get("replay_verified", "False"),
        "replay_timeframe": trade.get("replay_timeframe", ""),
        "ambiguous_intrabar": trade.get("ambiguous_intrabar", "False"),
        "evaluation_included": "True" if exit_source in ["candle_path", "latest_candle_close", "time_exit", "historical_replay"] else "False",
        "highest_favorable_price": trade.get("highest_favorable_price"),
        "breakeven_activated": trade.get("breakeven_activated"),
    }


def update_open_trades():
    ensure_dirs()
    settings = load_settings()
    open_trades = read_csv(OPEN_PATH)
    closed_trades = read_csv(CLOSED_PATH)
    still_open = []
    closed_now = 0

    for trade in open_trades:
        checked_at = utc_now_text()
        entry_time = datetime.fromisoformat(trade["entry_time"])
        hold_minutes = int((utc_now() - entry_time).total_seconds() / 60)
        exit_reason = ""
        exit_source = ""
        price = None

        try:
            candles = fetch_candles(trade["market"], trade["symbol"], trade["entry_time"], trade.get("timeframe", "1h"))
        except Exception as error:
            candles = []
            audit_execution(
                "market_feed_failed",
                {
                    "trade_id": trade.get("trade_id"),
                    "market": trade.get("market"),
                    "symbol": trade.get("symbol"),
                    "error": str(error),
                },
            )

        trade = apply_lifecycle(trade, candles, settings, write_event)

        candle_exit = simulate_candle_exit(trade, candles)
        exit_time_val = None
        if candle_exit:
            price, exit_reason, exit_source, exit_time_val = candle_exit
        else:
            if candles:
                price = candles[-1]["close"]
                exit_source = "latest_candle_close"
            else:
                price, exit_source = fallback_price(trade["market"], trade["symbol"])
                exit_source = exit_source or "latest_log_fallback"
            sl = to_float(trade.get("stop_loss"))
            tp = to_float(trade.get("take_profit_1"))
            if trade["direction"] == "LONG":
                if price is not None and sl is not None and price <= sl:
                    exit_reason = "SL"
                elif price is not None and tp is not None and price >= tp:
                    exit_reason = "TP1"
            else:
                if price is not None and sl is not None and price >= sl:
                    exit_reason = "SL"
                elif price is not None and tp is not None and price <= tp:
                    exit_reason = "TP1"

        if not exit_reason and hold_minutes >= int(settings["max_hold_minutes"]):
            exit_reason = "TIME_EXIT"
            exit_source = exit_source or "time_exit"

        if exit_reason and price is not None:
            closed = close_trade(trade, price, exit_reason, exit_source, settings, exit_time=exit_time_val)
            if closed:
                closed_trades.append(closed)
                write_event(trade["trade_id"], trade["signal_id"], "PAPER_CLOSED", trade["market"], trade["symbol"], exit_reason)
                audit_execution("paper_trade_closed", closed)
                closed_now += 1
        else:
            trade["last_checked_at"] = checked_at
            trade["last_market_source"] = exit_source
            still_open.append(trade)
        
        # Throttle to respect API limits
        time.sleep(1)

    write_csv_atomic(OPEN_PATH, still_open, OPEN_FIELDS)
    write_csv_atomic(CLOSED_PATH, closed_trades, CLOSED_FIELDS)
    write_summary()
    return closed_now


def write_summary():
    open_trades = read_csv(OPEN_PATH)
    closed_trades_all = read_csv(CLOSED_PATH)
    
    # Eval filter: Only count credible exit sources
    credible_sources = ["candle_path", "latest_candle_close", "time_exit", "historical_replay"]
    closed_trades = [r for row in closed_trades_all if (r := row).get("exit_source") in credible_sources]
    
    returns = [to_float(row.get("net_return_pct")) or 0 for row in closed_trades]
    wins = sum(1 for row in closed_trades if row.get("win") == "True")
    losses = len(closed_trades) - wins
    summary = {
        "generated_at": utc_now_text(),
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
        "wins": wins,
        "losses": losses,
        "winrate_pct": round(wins / len(closed_trades) * 100, 4) if closed_trades else 0,
        "total_net_return_pct": round(sum(returns), 4),
        "avg_net_return_pct": round(sum(returns) / len(returns), 4) if returns else 0,
    }
    write_csv_atomic(SUMMARY_PATH, [summary], SUMMARY_FIELDS)
