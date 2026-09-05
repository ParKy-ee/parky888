def trade_cost_pct(market, symbol, settings):
    symbol_cost = settings.get("symbol_cost_pct", {}).get(symbol)
    if symbol_cost is not None:
        return float(symbol_cost)
    return float(settings.get("market_cost_pct", {}).get(market, 0)) + float(
        settings.get("slippage_pct", {}).get(market, 0)
    )
