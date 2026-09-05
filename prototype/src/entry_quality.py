"""Phase 2: entry quality gates for realistic_paper_mode."""

from .ai_dataset import resolve_market_regime, risk_features
from .cost_model import trade_cost_pct
from .risk_manager import direction_from_signal, to_float
from .trade_context import get_trade_thesis


def _regime_from_signal(signal):
    return resolve_market_regime(signal)


def check_rr_and_cost(signal, settings, current_price):
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    entry = to_float(current_price) or to_float(signal.get("entry_price") or signal.get("price"))
    sl = to_float(signal.get("stop_loss"))
    tp = to_float(signal.get("take_profit_1"))
    if entry is None or sl is None or tp is None:
        return False, "missing_sl_or_tp"

    features = risk_features(direction, entry, sl, tp)
    rr = to_float(features.get("risk_reward_ratio"))
    if rr is None:
        return False, "invalid_risk_reward"

    market = signal.get("market", "forex")
    symbol = signal.get("symbol", "")
    cost_pct = trade_cost_pct(market, symbol, settings)
    sl_dist_pct = to_float(features.get("distance_to_sl_pct")) or 0
    if sl_dist_pct <= 0:
        return False, "invalid_sl_distance"

    cost_in_r = cost_pct / sl_dist_pct
    net_rr = rr - cost_in_r
    min_net_rr = float(settings.get("min_net_risk_reward", 1.2))
    if net_rr < min_net_rr:
        return False, f"net_rr_below_{min_net_rr}"
    return True, "ok"


def check_regime_gate(signal, settings):
    regime = _regime_from_signal(signal)
    blocked = [r.lower() for r in settings.get("blocked_market_regimes", [])]
    if regime.lower() in blocked:
        return False, f"regime_blocked_{regime}"

    allowed = [r.lower() for r in settings.get("allowed_market_regimes", []) if r]
    if allowed:
        if regime.lower() not in allowed:
            return False, "SKIP_REGIME_UNKNOWN_OR_NOT_ALLOWED"
    return True, "ok"


def check_active_thesis_limit(signal, open_trades, settings):
    market = signal.get("market", "forex")
    symbol = signal.get("symbol", "")
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    thesis = get_trade_thesis(symbol, direction, market)
    same = [
        t
        for t in open_trades
        if get_trade_thesis(t.get("symbol", ""), t.get("direction", ""), t.get("market", "forex")) == thesis
    ]
    max_same = int(settings.get("max_active_same_thesis", 2))
    if len(same) >= max_same:
        return False, f"max_active_thesis_{len(same)}"
    return True, "ok"


def check_portfolio_risk(signal, open_trades, settings, position):
    balance = float(settings.get("account_balance_usd", 10000))
    max_risk_pct = float(settings.get("max_portfolio_risk_pct", 0.03))
    max_notional_pct = float(settings.get("max_portfolio_notional_pct", 0.5))

    open_risk = sum(to_float(t.get("risk_amount_usd")) or 0 for t in open_trades)
    open_notional = sum(to_float(t.get("notional_usd")) or 0 for t in open_trades)
    new_risk = to_float(position.get("risk_amount_usd")) or 0
    new_notional = to_float(position.get("notional_usd")) or 0

    if (open_risk + new_risk) / balance > max_risk_pct:
        return False, "portfolio_risk_cap"
    if (open_notional + new_notional) / balance > max_notional_pct:
        return False, "portfolio_notional_cap"
    return True, "ok"


def check_recent_sl_guard(signal, closed_trades, settings):
    """Weighted SL count: same-symbol SL weighs more than thesis-only."""
    market = signal.get("market", "forex")
    symbol = signal.get("symbol", "")
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    thesis = get_trade_thesis(symbol, direction, market)
    hours = float(settings.get("recent_sl_guard_hours", 4))
    symbol_weight = float(settings.get("recent_sl_weight_same_symbol", 2))
    max_weighted = float(settings.get("max_weighted_sl_per_thesis", 2))

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    weighted = 0.0
    for trade in closed_trades:
        if trade.get("exit_reason") not in ("SL", "SL_AMBIGUOUS_FIRST"):
            continue
        try:
            exit_time = datetime.fromisoformat(str(trade["exit_time"]).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if (now - exit_time).total_seconds() / 3600 > hours:
            continue
        trade_thesis = trade.get("trade_thesis") or get_trade_thesis(
            trade.get("symbol", ""), trade.get("direction", ""), trade.get("market", "forex")
        )
        if trade_thesis != thesis:
            continue
        weight = symbol_weight if trade.get("symbol") == symbol else 1.0
        weighted += weight

    if weighted >= max_weighted:
        return False, f"weighted_sl_guard_{weighted}"
    return True, "ok"


def run_entry_quality_guards(signal, open_trades, closed_trades, settings, current_price, position=None):
    checks = [
        ("rr_and_cost", lambda: check_rr_and_cost(signal, settings, current_price)),
        ("regime_gate", lambda: check_regime_gate(signal, settings)),
        ("active_thesis_limit", lambda: check_active_thesis_limit(signal, open_trades, settings)),
    ]
    if position is not None:
        checks.append(
            ("portfolio_risk", lambda: check_portfolio_risk(signal, open_trades, settings, position))
        )
    checks.append(("recent_sl_guard", lambda: check_recent_sl_guard(signal, closed_trades, settings)))

    for rule, fn in checks:
        ok, reason = fn()
        if not ok:
            return False, rule, reason
    return True, "ok", "ok"
