def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def direction_from_signal(decision, final_bias):
    text = f"{decision} {final_bias}".upper()
    if "BULLISH" in text:
        return "LONG"
    if "BEARISH" in text:
        return "SHORT"
    return ""


DEFAULT_FOREX_SPECS = {
    "EURUSD": {"pip_size": 0.0001, "pip_value_per_lot_usd": 10.0, "contract_size": 100000},
    "GBPUSD": {"pip_size": 0.0001, "pip_value_per_lot_usd": 10.0, "contract_size": 100000},
    "USDJPY": {"pip_size": 0.01, "pip_value_per_lot_usd": 9.2, "contract_size": 100000},
    "USDCHF": {"pip_size": 0.0001, "pip_value_per_lot_usd": 11.1, "contract_size": 100000},
    "AUDUSD": {"pip_size": 0.0001, "pip_value_per_lot_usd": 10.0, "contract_size": 100000},
    "USDCAD": {"pip_size": 0.0001, "pip_value_per_lot_usd": 7.3, "contract_size": 100000},
}

FOREX_SYMBOLS = [
    "AUDCAD",
    "AUDCHF",
    "AUDJPY",
    "AUDNZD",
    "AUDUSD",
    "CADCHF",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURNZD",
    "EURUSD",
    "GBPAUD",
    "GBPCAD",
    "GBPCHF",
    "GBPJPY",
    "GBPNZD",
    "GBPUSD",
    "NZDCAD",
    "NZDCHF",
    "NZDJPY",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
]

for _symbol in FOREX_SYMBOLS:
    DEFAULT_FOREX_SPECS.setdefault(
        _symbol,
        {
            "pip_size": 0.01 if _symbol.endswith("JPY") else 0.0001,
            "pip_value_per_lot_usd": 10.0,
            "contract_size": 100000,
        },
    )


def calculate_position(market, symbol, direction, entry_price, stop_loss, balance_usd, risk_pct, settings=None):
    entry = to_float(entry_price)
    sl = to_float(stop_loss)
    if not direction or entry is None or sl is None or entry <= 0:
        return None
    risk_distance = abs(entry - sl)
    if risk_distance <= 0:
        return None

    risk_amount = balance_usd * risk_pct

    if market == "forex":
        configured_specs = (settings or {}).get("forex_symbol_specs", {})
        specs = {**DEFAULT_FOREX_SPECS.get(symbol, {}), **configured_specs.get(symbol, {})}
        pip_size = float(specs.get("pip_size", 0.0001))
        pip_value = float(specs.get("pip_value_per_lot_usd", 10.0))
        contract_size = float(specs.get("contract_size", 100000))
        stop_pips = risk_distance / pip_size
        if stop_pips <= 0 or pip_value <= 0:
            return None
        lot_size = risk_amount / (stop_pips * pip_value)
        quantity = lot_size * contract_size
        notional_usd = quantity * entry
    else:
        raw_quantity = risk_amount / risk_distance
        quantity = raw_quantity
        lot_size = ""
        notional_usd = quantity * entry

    return {
        "risk_amount_usd": round(risk_amount, 4),
        "risk_distance": round(risk_distance, 8),
        "quantity": round(quantity, 8),
        "lot_size": round(lot_size, 4) if lot_size != "" else "",
        "notional_usd": round(notional_usd, 4),
    }
