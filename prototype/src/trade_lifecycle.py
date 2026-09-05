"""Phase 3: trade lifecycle management (BE, partial, trail, event log)."""

from .cost_model import trade_cost_pct
from .risk_manager import to_float


def _risk_distance(trade):
    entry = to_float(trade.get("entry_price"))
    sl = to_float(trade.get("stop_loss"))
    if entry is None or sl is None:
        return None
    return abs(entry - sl)


def update_r_metrics(trade, candles):
    entry = to_float(trade.get("entry_price"))
    risk = to_float(trade.get("initial_risk_distance"))
    if risk is None or risk <= 0:
        # Fallback to initial stop loss or current stop loss distance
        initial_sl = to_float(trade.get("initial_stop_loss") or trade.get("stop_loss"))
        if entry is not None and initial_sl is not None:
            risk = abs(entry - initial_sl)
        else:
            risk = _risk_distance(trade)

    if entry is None or risk is None or risk <= 0:
        return trade

    highest = to_float(trade.get("highest_favorable_price") or entry)
    lowest = to_float(trade.get("lowest_adverse_price") or entry)
    direction = trade.get("direction")

    for candle in candles:
        high = to_float(candle.get("high"))
        low = to_float(candle.get("low"))
        if high is None or low is None:
            continue
        if direction == "LONG":
            highest = max(highest, high)
            lowest = min(lowest, low)
        else:
            # highest_favorable_price stores best (lowest) price for shorts
            highest = min(highest, low)
            lowest = max(lowest, high)

    trade["highest_favorable_price"] = highest
    trade["lowest_adverse_price"] = lowest
    if direction == "LONG":
        trade["max_favorable_r"] = round((highest - entry) / risk, 4)
        trade["max_adverse_r"] = round((entry - lowest) / risk, 4)
    else:
        trade["max_favorable_r"] = round((entry - highest) / risk, 4)
        trade["max_adverse_r"] = round((lowest - entry) / risk, 4)
    return trade


def apply_breakeven_cost_aware(trade, candles, settings, write_event_fn):
    entry = to_float(trade.get("entry_price"))
    tp = to_float(trade.get("take_profit_1"))
    if not candles or entry is None or tp is None:
        return trade
    if trade.get("breakeven_activated") == "True":
        return trade

    dist_to_tp = abs(tp - entry)
    trigger = float(settings.get("breakeven_trigger_pct", 0.75))
    highest = to_float(trade.get("highest_favorable_price") or entry)
    direction = trade.get("direction")

    if direction == "LONG":
        mfe = highest - entry
    else:
        mfe = entry - highest

    if dist_to_tp <= 0 or mfe / dist_to_tp < trigger:
        return trade

    cost_pct = trade_cost_pct(trade.get("market", "forex"), trade.get("symbol", ""), settings)
    cost_price = entry * (cost_pct / 100)
    if direction == "LONG":
        new_sl = entry + cost_price
    else:
        new_sl = entry - cost_price

    trade["stop_loss"] = new_sl
    trade["breakeven_activated"] = "True"
    write_event_fn(
        trade["trade_id"],
        trade["signal_id"],
        "BREAKEVEN_COST_AWARE",
        trade["market"],
        trade["symbol"],
        f"SL moved to BE+cost ({new_sl})",
    )
    return trade


def apply_partial_close(trade, candles, settings, write_event_fn):
    if trade.get("partial_close_done") == "True":
        return trade
    if not settings.get("enable_partial_close", True):
        return trade

    entry = to_float(trade.get("entry_price"))
    tp = to_float(trade.get("take_profit_1"))
    trigger_pct = float(settings.get("partial_close_trigger_pct", 0.5))
    size_pct = float(settings.get("partial_close_size_pct", 0.5))
    highest = to_float(trade.get("highest_favorable_price") or entry)
    direction = trade.get("direction")

    if entry is None or tp is None:
        return trade
    dist = abs(tp - entry)
    mfe = (highest - entry) if direction == "LONG" else (entry - highest)
    if dist <= 0 or mfe / dist < trigger_pct:
        return trade

    qty = to_float(trade.get("quantity"))
    if qty is None:
        return trade
    remaining_pct = 1.0 - size_pct
    
    trade["quantity"] = round(qty * remaining_pct, 8)
    
    # Scale down other position size metrics
    lot_size = to_float(trade.get("lot_size"))
    if lot_size is not None:
        trade["lot_size"] = round(lot_size * remaining_pct, 4)
        
    notional = to_float(trade.get("notional_usd"))
    if notional is not None:
        trade["notional_usd"] = round(notional * remaining_pct, 4)
        
    risk_amount = to_float(trade.get("risk_amount_usd"))
    if risk_amount is not None:
        trade["risk_amount_usd"] = round(risk_amount * remaining_pct, 4)

    trade["partial_close_done"] = "True"
    trade["remaining_quantity_pct"] = round(remaining_pct, 4)
    trade["realized_partial_pct"] = round(size_pct, 4)
    trade["partial_close_count"] = int(trade.get("partial_close_count") or 0) + 1
    
    write_event_fn(
        trade["trade_id"],
        trade["signal_id"],
        "PARTIAL_CLOSE",
        trade["market"],
        trade["symbol"],
        f"Reduced size by {size_pct*100:.0f}% at {trigger_pct*100:.0f}% of TP path",
    )
    return trade


def apply_trailing_stop(trade, candles, settings, write_event_fn):
    if not settings.get("enable_trailing_stop", True):
        return trade

    entry = to_float(trade.get("entry_price"))
    tp = to_float(trade.get("take_profit_1"))
    trigger_pct = float(settings.get("trailing_stop_trigger_pct", 0.6))
    lock_pct = float(settings.get("trailing_stop_lock_pct", 0.35))
    highest = to_float(trade.get("highest_favorable_price") or entry)
    direction = trade.get("direction")

    if entry is None or tp is None:
        return trade
    dist = abs(tp - entry)
    mfe = (highest - entry) if direction == "LONG" else (entry - highest)
    if dist <= 0 or mfe / dist < trigger_pct:
        return trade

    lock_move = mfe * lock_pct
    if direction == "LONG":
        new_sl = entry + lock_move
        current_sl = to_float(trade.get("stop_loss")) or 0
        if new_sl > current_sl:
            trade["stop_loss"] = new_sl
            trade["trailing_stop_active"] = "True"
            write_event_fn(
                trade["trade_id"],
                trade["signal_id"],
                "TRAILING_STOP",
                trade["market"],
                trade["symbol"],
                f"Trail SL -> {round(new_sl, 6)}",
            )
    else:
        new_sl = entry - lock_move
        current_sl = to_float(trade.get("stop_loss")) or float("inf")
        if new_sl < current_sl:
            trade["stop_loss"] = new_sl
            trade["trailing_stop_active"] = "True"
            write_event_fn(
                trade["trade_id"],
                trade["signal_id"],
                "TRAILING_STOP",
                trade["market"],
                trade["symbol"],
                f"Trail SL -> {round(new_sl, 6)}",
            )
    return trade


def apply_lifecycle(trade, candles, settings, write_event_fn):
    trade = update_r_metrics(trade, candles)
    trade = apply_breakeven_cost_aware(trade, candles, settings, write_event_fn)
    trade = apply_partial_close(trade, candles, settings, write_event_fn)
    trade = apply_trailing_stop(trade, candles, settings, write_event_fn)
    return trade
