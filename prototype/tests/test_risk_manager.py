from prototype.src.risk_manager import calculate_position, direction_from_signal


def test_direction_from_signal():
    assert direction_from_signal("BULLISH_WATCH", "BULLISH") == "LONG"
    assert direction_from_signal("BEARISH_TRADE_READY", "BEARISH") == "SHORT"
    assert direction_from_signal("WAIT", "NEUTRAL") == ""


def test_calculate_crypto_position():
    result = calculate_position("crypto", "BTCUSD", "LONG", 100.0, 95.0, 10000.0, 0.01)
    assert result["risk_amount_usd"] == 100.0
    assert result["quantity"] == 20.0


def test_calculate_forex_position_uses_pip_value():
    result = calculate_position("forex", "EURUSD", "LONG", 1.1, 1.098, 10000.0, 0.01)
    assert result["risk_amount_usd"] == 100.0
    assert round(result["lot_size"], 2) == 0.5
