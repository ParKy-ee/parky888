import json

from .paths import CONFIG_DIR


DEFAULT_SETTINGS = {
    "research_mode": True,
    "realistic_paper_mode": True,
    "signal_cluster_window_minutes": 240,
    "account_balance_usd": 10000.0,
    "risk_pct_per_trade": 0.005,
    "max_positions_total": 10,
    "max_positions_per_symbol": 1,
    "max_hold_minutes": 1440,
    "paper_entry_decisions": ["BULLISH_TRADE_READY", "BEARISH_TRADE_READY"],
    "max_recent_sl_per_thesis_4h": 1,
    "max_entry_price_drift_pct": 0.05,
    "market_cost_pct": {"forex": 0.03, "crypto": 0.20},
    "slippage_pct": {"forex": 0.01, "crypto": 0.05},
    "symbol_cost_pct": {},
    "forex_symbol_specs": {},
    "disabled_timeframes": [],
    "monitor_only_timeframes": [],
    "ignore_superseded_rule": False,
    "process_only_new_signals": True,
    "cooldown_hours_standard": 6.0,
    "cooldown_hours_gbp": 12.0,
    "breakeven_trigger_pct": 0.75,
    "min_net_risk_reward": 1.2,
    "allowed_market_regimes": [],
    "blocked_market_regimes": [],
    "max_active_same_thesis": 2,
    "max_portfolio_risk_pct": 0.03,
    "max_portfolio_notional_pct": 0.5,
    "recent_sl_guard_hours": 4,
    "recent_sl_weight_same_symbol": 2,
    "max_weighted_sl_per_thesis": 2,
    "enable_partial_close": True,
    "partial_close_trigger_pct": 0.5,
    "partial_close_size_pct": 0.5,
    "enable_trailing_stop": True,
    "trailing_stop_trigger_pct": 0.6,
    "trailing_stop_lock_pct": 0.35,
}


def load_settings():
    path = CONFIG_DIR / "settings.json"
    if not path.exists():
        return DEFAULT_SETTINGS
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    merged = {**DEFAULT_SETTINGS, **loaded}
    merged["market_cost_pct"] = {**DEFAULT_SETTINGS["market_cost_pct"], **loaded.get("market_cost_pct", {})}
    merged["slippage_pct"] = {**DEFAULT_SETTINGS["slippage_pct"], **loaded.get("slippage_pct", {})}
    merged["symbol_cost_pct"] = {**DEFAULT_SETTINGS["symbol_cost_pct"], **loaded.get("symbol_cost_pct", {})}
    merged["forex_symbol_specs"] = {**DEFAULT_SETTINGS["forex_symbol_specs"], **loaded.get("forex_symbol_specs", {})}
    merged["disabled_timeframes"] = loaded.get("disabled_timeframes", DEFAULT_SETTINGS["disabled_timeframes"])
    merged["monitor_only_timeframes"] = loaded.get("monitor_only_timeframes", DEFAULT_SETTINGS["monitor_only_timeframes"])
    return merged
