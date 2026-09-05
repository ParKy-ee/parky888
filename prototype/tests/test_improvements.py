import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from prototype.src.ai_dataset import candle_features, CANDLE_CACHE
from prototype.src.entry_quality import check_recent_sl_guard
from prototype.src.paper_trade_tracker import get_trade_thesis, can_open_signal

def test_candle_features_caching_diff_times():
    # Clear the cache first
    CANDLE_CACHE.clear()
    
    # Mock fetch_candles to return different candles for different start times
    # so we can verify they get cached separately and don't reuse each other's results.
    with patch("prototype.src.ai_dataset.fetch_candles") as mock_fetch:
        # Mock fetch_candles returns different candles based on start_time
        def side_effect(market, symbol, start_time_iso, timeframe):
            if "10:00:00" in start_time_iso:
                return [{"high": 105.0, "low": 95.0, "close": 101.0, "time": start_time_iso}]
            else:
                return [{"high": 205.0, "low": 195.0, "close": 201.0, "time": start_time_iso}]
        mock_fetch.side_effect = side_effect
        
        # Call for time 1
        res1 = candle_features("crypto", "BTCUSD", "2026-05-19T10:00:00+00:00", "1h", "LONG", 100.0)
        # Call for time 2
        res2 = candle_features("crypto", "BTCUSD", "2026-05-19T11:00:00+00:00", "1h", "LONG", 100.0)
        
        # Verify fetch_candles was called twice
        assert mock_fetch.call_count == 2
        # Verify results are different and not cached/reused
        assert res1["mfe_pct"] != res2["mfe_pct"]
        assert res1["mfe_pct"] == 5.0  # (105 - 100) / 100 * 100
        assert res2["mfe_pct"] == 105.0 # (205 - 100) / 100 * 100

def test_get_trade_thesis_separation():
    # Forex uses USD_WEAK/USD_STRONG/SYMBOL_SPECIFIC
    assert get_trade_thesis("EURUSD", "LONG", "forex") == "USD_WEAK"
    assert get_trade_thesis("EURUSD", "SHORT", "forex") == "USD_STRONG"
    assert get_trade_thesis("GBPUSD", "LONG", "forex") == "USD_WEAK"
    assert get_trade_thesis("AUDJPY", "LONG", "forex") == "SYMBOL_SPECIFIC"
    
    # Crypto uses CRYPTO_BTC, CRYPTO_ETH, or CRYPTO_SYMBOL_SPECIFIC
    assert get_trade_thesis("BTCUSD", "LONG", "crypto") == "CRYPTO_BTC"
    assert get_trade_thesis("BTCUSD", "SHORT", "crypto") == "CRYPTO_BTC"
    assert get_trade_thesis("ETHUSD", "LONG", "crypto") == "CRYPTO_ETH"
    assert get_trade_thesis("SOLUSD", "LONG", "crypto") == "CRYPTO_SOLUSD"

def test_recent_sl_guard_for_crypto_not_affected_by_forex():
    # We want to test that a trade on a forex symbol (e.g. EURUSD) does not affect
    # SL guard for a crypto symbol (e.g. BTCUSD).
    # SL Guard uses get_trade_thesis().
    # If a forex trade on EURUSD exits, it should not trigger SL guard on BTCUSD
    # since EURUSD has thesis USD_WEAK/STRONG, while BTCUSD has thesis CRYPTO_BTC.
    
    # Let's test that get_trade_thesis is indeed different:
    thesis_forex = get_trade_thesis("EURUSD", "LONG", "forex")
    thesis_crypto = get_trade_thesis("BTCUSD", "LONG", "crypto")
    assert thesis_forex != thesis_crypto
    
    # Let's mock closed trades to have a recent SL on EURUSD
    closed_trades = [
        {
            "symbol": "EURUSD",
            "market": "forex",
            "trade_thesis": "USD_WEAK",
            "exit_reason": "SL",
            "exit_time": datetime.now(timezone.utc).isoformat(),
        }
    ]
    
    settings = {
        "recent_sl_guard_hours": 4,
        "recent_sl_weight_same_symbol": 2,
        "max_weighted_sl_per_thesis": 2,
    }

    signal_crypto = {
        "decision": "BULLISH_TRADE_READY",
        "market": "crypto",
        "symbol": "BTCUSD",
        "final_bias": "BULLISH",
    }

    signal_forex = {
        "decision": "BULLISH_TRADE_READY",
        "market": "forex",
        "symbol": "EURUSD",
        "final_bias": "BULLISH",
    }

    ok_forex, reason_forex = check_recent_sl_guard(signal_forex, closed_trades, settings)
    assert ok_forex is False
    assert "weighted_sl_guard" in reason_forex

    ok_crypto, reason_crypto = check_recent_sl_guard(signal_crypto, closed_trades, settings)
    assert ok_crypto is True
    assert reason_crypto == "ok"

def test_cooldown_guard_behavior():
    # Mock settings with 6 hours cooldown
    settings = {
        "paper_entry_decisions": ["BULLISH_TRADE_READY", "BEARISH_TRADE_READY"],
        "max_positions_total": 5,
        "max_positions_per_symbol": 2,
        "max_recent_sl_per_thesis_4h": 1,
        "cooldown_hours_standard": 6,
    }
    
    # Mock closed trades:
    # 1. BTCUSD (crypto) closed 1 hour ago
    # 2. EURUSD (forex) closed 2 hours ago
    closed_trades = [
        {
            "symbol": "BTCUSD",
            "market": "crypto",
            "trade_thesis": "CRYPTO_BTC",
            "exit_reason": "TP1",
            "exit_time": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        },
        {
            "symbol": "EURUSD",
            "market": "forex",
            "trade_thesis": "USD_WEAK",
            "exit_reason": "TP1",
            "exit_time": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        }
    ]
    
    # Mock read_csv to return these closed trades
    with patch("prototype.src.paper_trade_tracker.read_csv", return_value=closed_trades):
        # Case A: Same symbol (BTCUSD) within cooldown period should be BLOCKED
        signal_btc = {
            "decision": "BULLISH_TRADE_READY",
            "market": "crypto",
            "symbol": "BTCUSD",
            "final_bias": "BULLISH",
        }
        allowed_btc, reason_btc = can_open_signal(signal_btc, [], settings)
        assert allowed_btc is False
        assert "cooldown_active" in reason_btc

        # Case B: Different symbol (ETHUSD) not traded recently should be ALLOWED
        signal_eth = {
            "decision": "BULLISH_TRADE_READY",
            "market": "crypto",
            "symbol": "ETHUSD",
            "final_bias": "BULLISH",
        }
        allowed_eth, reason_eth = can_open_signal(signal_eth, [], settings)
        assert allowed_eth is True
        assert reason_eth == "ok"

        # Case C: Forex (EURUSD) within cooldown should be BLOCKED
        signal_eurusd = {
            "decision": "BULLISH_TRADE_READY",
            "market": "forex",
            "symbol": "EURUSD",
            "final_bias": "BULLISH",
        }
        allowed_eurusd, reason_eurusd = can_open_signal(signal_eurusd, [], settings)
        assert allowed_eurusd is False
        assert "cooldown_active" in reason_eurusd
