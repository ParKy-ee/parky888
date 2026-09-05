from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from prototype.src.paper_trade_tracker import MODE_REALISTIC, MODE_RESEARCH, _try_open_signal
from prototype.src.signal_clustering import compute_signal_cluster_id, resolve_cluster_status
from prototype.src.live_market_feed import fetch_executable_price, fallback_price


def test_compute_signal_cluster_id_stable_within_window():
    settings = {"signal_cluster_window_minutes": 240}
    signal = {
        "market": "forex",
        "symbol": "EURUSD",
        "decision": "BULLISH_TRADE_READY",
        "final_bias": "BULLISH",
        "source_run_at": "2026-05-20 10:00:00",
    }
    other = {**signal, "source_run_at": "2026-05-20 10:30:00"}
    assert compute_signal_cluster_id(signal, settings) == compute_signal_cluster_id(other, settings)


def test_fallback_price_returns_log_source_tuple():
    with patch("prototype.src.live_market_feed.latest_price", return_value=1.2345):
        price, source = fallback_price("forex", "EURUSD")
        assert price == 1.2345
        assert source == "latest_log_fallback"


def test_try_open_blocks_without_executable_price():
    settings = {
        "research_mode": True,
        "realistic_paper_mode": True,
        "paper_entry_decisions": ["BULLISH_TRADE_READY", "BEARISH_TRADE_READY"],
        "max_positions_total": 5,
        "max_positions_per_symbol": 2,
        "max_recent_sl_per_thesis_4h": 5,
        "signal_cluster_window_minutes": 240,
        "max_entry_price_drift_pct": 5.0,
        "account_balance_usd": 10000,
        "risk_pct_per_trade": 0.005,
    }
    run_at = datetime.now(timezone.utc)
    signal = {
        "signal_id": "abc123",
        "market": "forex",
        "symbol": "EURUSD",
        "decision": "BULLISH_TRADE_READY",
        "signal_type": "TRADE_READY",
        "final_bias": "BULLISH",
        "source_run_at": run_at.isoformat(),
        "entry_price": "1.10",
        "stop_loss": "1.09",
        "take_profit_1": "1.12",
        "final_score": "7",
        "confidence_pct": "80",
    }
    planned = run_at + timedelta(minutes=1)
    ctx = {
        "settings": settings,
        "research_on": True,
        "realistic_on": True,
        "now": planned,
        "now_text": planned.isoformat(),
        "open_trades": [],
        "closed_trades": [],
        "blocked_new": [],
        "cluster_index": {},
        "latest_ts_map": {("forex", "EURUSD"): run_at},
        "closed_signal_ids": set(),
    }
    with patch("prototype.src.paper_trade_tracker.fetch_executable_price", return_value=(None, None)):
        opened = _try_open_signal(signal, ctx)
    assert opened is False
    assert len(ctx["blocked_new"]) == 1
    row = ctx["blocked_new"][0]
    assert row["outcome"] == "blocked_trade"
    assert row["mode"] == MODE_REALISTIC
    assert row["block_rule"] == "no_executable_price"


def test_watch_signal_is_observation_only():
    settings = {
        "research_mode": True,
        "realistic_paper_mode": True,
        "signal_cluster_window_minutes": 240,
    }
    signal = {
        "signal_id": "watch1",
        "market": "forex",
        "symbol": "EURUSD",
        "decision": "BULLISH_WATCH",
        "signal_type": "WATCH",
        "source_run_at": "2026-05-20 10:00:00",
    }
    ctx = {
        "settings": settings,
        "research_on": True,
        "realistic_on": True,
        "now": datetime.now(timezone.utc),
        "now_text": datetime.now(timezone.utc).isoformat(),
        "open_trades": [],
        "closed_trades": [],
        "blocked_new": [],
        "cluster_index": {},
        "latest_ts_map": {("forex", "EURUSD"): datetime.now(timezone.utc)},
        "closed_signal_ids": set(),
    }
    opened = _try_open_signal(signal, ctx)
    assert opened is False
    assert ctx["blocked_new"][0]["outcome"] == "observation_only"
    assert ctx["blocked_new"][0]["mode"] == MODE_RESEARCH


def test_resolve_cluster_status_same_cluster():
    index = {"cluster_a": ["sig1"]}
    assert resolve_cluster_status("cluster_a", index, "sig2") == "same_cluster"
    assert resolve_cluster_status("cluster_a", index, "sig1") == "new_opportunity"
