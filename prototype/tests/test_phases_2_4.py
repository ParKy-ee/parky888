from prototype.src.dataset_labels import apply_dataset_labels, compute_train_tier, should_enter_label
from prototype.src.entry_quality import check_rr_and_cost, check_regime_gate
from prototype.src.ai_dataset import resolve_market_regime, deduplicate_cluster_rows
from prototype.src.trade_lifecycle import update_r_metrics, apply_partial_close
from prototype.src.paper_trade_replay import merge_rows, normalize_trade_row
from prototype.src.paper_trade_tracker import CLOSED_FIELDS, OPEN_FIELDS


def test_check_rr_and_cost_blocks_low_rr():
    settings = {
        "min_net_risk_reward": 2.0,
        "market_cost_pct": {"forex": 0.03},
        "slippage_pct": {"forex": 0.01},
        "symbol_cost_pct": {},
    }
    signal = {
        "market": "forex",
        "symbol": "EURUSD",
        "decision": "BULLISH_TRADE_READY",
        "final_bias": "BULLISH",
        "entry_price": "1.10",
        "stop_loss": "1.09",
        "take_profit_1": "1.105",
    }
    ok, reason = check_rr_and_cost(signal, settings, 1.10)
    assert ok is False
    assert "net_rr" in reason


def test_regime_gate_blocks_listed_regime():
    settings = {"blocked_market_regimes": ["regime_chop"], "allowed_market_regimes": []}
    signal = {"market_regime": "regime_chop", "reason": ""}
    ok, reason = check_regime_gate(signal, settings)
    assert ok is False
    assert "regime_blocked" in reason


def test_train_tier_and_should_enter_label():
    opened = apply_dataset_labels(
        {
            "mode": "realistic_paper_mode",
            "outcome": "opened_trade",
            "label_quality": "HIGH",
            "exit_source": "candle_path",
            "is_trainable": "true",
            "trade_id": "t1",
        }
    )
    assert opened["train_tier"] == "HIGH"
    assert opened["should_enter_label"] == "true"

    blocked = apply_dataset_labels(
        {
            "outcome": "blocked_trade",
            "decision": "BULLISH_TRADE_READY",
            "signal_type": "TRADE_READY",
        }
    )
    assert blocked["train_tier"] == "BLOCKED_SAMPLE"
    assert blocked["should_enter_label"] == "false"
    assert compute_train_tier(blocked) == "BLOCKED_SAMPLE"
    assert should_enter_label(blocked) == "false"
    assert compute_train_tier(blocked) == "BLOCKED_SAMPLE"
    assert should_enter_label(blocked) == "false"


def test_regime_gate_allowlist_unknown_and_trend():
    # Allowlist set to ["regime_trend"]
    settings = {
        "blocked_market_regimes": [],
        "allowed_market_regimes": ["regime_trend"]
    }
    
    # 1. UNKNOWN regime should be BLOCKED with reason "SKIP_REGIME_UNKNOWN_OR_NOT_ALLOWED"
    signal_unknown = {"market_regime": "UNKNOWN", "reason": ""}
    ok, reason = check_regime_gate(signal_unknown, settings)
    assert ok is False
    assert reason == "SKIP_REGIME_UNKNOWN_OR_NOT_ALLOWED"

    # 2. Non-matching regime should be BLOCKED with reason "SKIP_REGIME_UNKNOWN_OR_NOT_ALLOWED"
    signal_chop = {"market_regime": "regime_chop", "reason": ""}
    ok, reason = check_regime_gate(signal_chop, settings)
    assert ok is False
    assert reason == "SKIP_REGIME_UNKNOWN_OR_NOT_ALLOWED"

    # 3. Matching regime should pass
    signal_trend = {"market_regime": "regime_trend", "reason": ""}
    ok, reason = check_regime_gate(signal_trend, settings)
    assert ok is True

    # 4. If no allowlist is configured, UNKNOWN should pass
    settings_no_allow = {
        "blocked_market_regimes": [],
        "allowed_market_regimes": []
    }
    ok, reason = check_regime_gate(signal_unknown, settings_no_allow)
    assert ok is True


def test_resolve_market_regime():
    # Test resolving from market_regime field directly
    row1 = {"market_regime": "regime_trend", "reason": ""}
    assert resolve_market_regime(row1) == "regime_trend"

    # Test resolving from reason fallback
    row2 = {"market_regime": "UNKNOWN", "reason": "some text, regime_chop, other text"}
    assert resolve_market_regime(row2) == "regime_chop"

    # Test resolving with no matching reason and UNKNOWN regime
    row3 = {"market_regime": "UNKNOWN", "reason": "no regime keyword here"}
    assert resolve_market_regime(row3) == "UNKNOWN"

    # Test resolving with empty dictionary
    assert resolve_market_regime({}) == "UNKNOWN"


def test_r_metrics_baseline_stability():
    trade = {
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "initial_stop_loss": 90.0,
        "initial_risk_distance": 10.0,
        "direction": "LONG",
        "highest_favorable_price": 120.0,
        "lowest_adverse_price": 95.0,
    }
    # Before update_r_metrics, max_favorable_r = (highest - entry) / risk = (120 - 100) / 10 = 2.0
    trade = update_r_metrics(trade, [{"high": 130.0, "low": 98.0}])
    assert trade["max_favorable_r"] == 3.0  # (130 - 100) / 10 = 3.0
    assert trade["max_adverse_r"] == 0.5   # (100 - 95) / 10 = 0.5
    
    # Simulating trailing stop moving the stop loss to 110.0
    trade["stop_loss"] = 110.0
    
    # Calling update_r_metrics again should still yield the same R metrics based on initial risk distance of 10.0
    trade = update_r_metrics(trade, [{"high": 140.0, "low": 105.0}])
    assert trade["max_favorable_r"] == 4.0  # (140 - 100) / 10 = 4.0
    assert trade["max_adverse_r"] == 0.5   # lowest remains 95.0, risk is still 10.0


def test_partial_close_accounting():
    trade = {
        "trade_id": "t1",
        "signal_id": "s1",
        "market": "forex",
        "symbol": "EURUSD",
        "entry_price": 1.10,
        "stop_loss": 1.09,
        "take_profit_1": 1.12,
        "quantity": 100000.0,
        "lot_size": 1.0,
        "notional_usd": 110000.0,
        "risk_amount_usd": 1000.0,
        "highest_favorable_price": 1.115, # trigger path is TP path = 1.12 - 1.10 = 0.02, mfe is 0.015 (75% of path)
        "direction": "LONG",
        "partial_close_done": "False",
    }
    
    settings = {
        "enable_partial_close": True,
        "partial_close_trigger_pct": 0.5,
        "partial_close_size_pct": 0.5,
    }
    
    events = []
    def mock_write_event(trade_id, signal_id, event_type, market, symbol, message):
        events.append((trade_id, event_type, message))
        
    updated = apply_partial_close(trade, [], settings, mock_write_event)
    
    assert updated["partial_close_done"] == "True"
    assert updated["quantity"] == 50000.0
    assert updated["lot_size"] == 0.5
    assert updated["notional_usd"] == 55000.0
    assert updated["risk_amount_usd"] == 500.0
    assert updated["realized_partial_pct"] == 0.5
    assert updated["partial_close_count"] == 1
    assert len(events) == 1
    assert events[0][1] == "PARTIAL_CLOSE"


def test_cluster_dataset_deduplication():
    # 1. Closed/opened trade row for signal 'sig1'
    closed = [
        {
            "signal_id": "sig1",
            "signal_cluster_id": "clusterA",
            "outcome": "opened_trade",
            "symbol": "EURUSD",
            "decision": "BULLISH_TRADE_READY",
            "train_tier": "HIGH"
        }
    ]
    
    # 2. Blocked trade row for signal 'sig1' and 'sig2'
    blocked = [
        {
            "signal_id": "sig1",
            "signal_cluster_id": "clusterA",
            "outcome": "blocked_trade",
            "symbol": "EURUSD",
            "decision": "BULLISH_TRADE_READY",
            "train_tier": "BLOCKED_SAMPLE"
        },
        {
            "signal_id": "sig2",
            "signal_cluster_id": "clusterA",
            "outcome": "blocked_trade",
            "symbol": "EURUSD",
            "decision": "BULLISH_TRADE_READY",
            "train_tier": "BLOCKED_SAMPLE"
        }
    ]
    
    # 3. Observation rows for signal 'sig1', 'sig2', and 'sig3'
    observations = [
        {
            "signal_id": "sig1",
            "signal_cluster_id": "clusterA",
            "outcome": "observation_only",
            "symbol": "EURUSD",
            "decision": "BULLISH_TRADE_READY",
            "train_tier": "OBSERVATION"
        },
        {
            "signal_id": "sig2",
            "signal_cluster_id": "clusterA",
            "outcome": "observation_only",
            "symbol": "EURUSD",
            "decision": "BULLISH_TRADE_READY",
            "train_tier": "OBSERVATION"
        },
        {
            "signal_id": "sig3",
            "signal_cluster_id": "clusterA",
            "outcome": "observation_only",
            "symbol": "EURUSD",
            "decision": "BULLISH_TRADE_READY",
            "train_tier": "OBSERVATION"
        }
    ]
    
    # Deduplicate them
    result = deduplicate_cluster_rows(closed, blocked, observations)
    
    # We should have exactly 3 rows (sig1, sig2, sig3)
    assert len(result) == 3
    
    # Group results by signal_id
    by_sig = {r["signal_id"]: r for r in result}
    
    # sig1 should have outcome 'opened_trade' (precedence 3 > 2 > 1)
    assert by_sig["sig1"]["outcome"] == "opened_trade"
    # sig2 should have outcome 'blocked_trade' (precedence 2 > 1)
    assert by_sig["sig2"]["outcome"] == "blocked_trade"
    # sig3 should have outcome 'observation_only' (precedence 1)
    assert by_sig["sig3"]["outcome"] == "observation_only"


def test_schema_migration_and_archiving(tmp_path, monkeypatch):
    import prototype.run_prototype as rp
    open_file = tmp_path / "open_trades.csv"
    closed_file = tmp_path / "closed_trades.csv"
    decision_file = tmp_path / "paper_trade_decisions.csv"
    blocked_file = tmp_path / "paper_trade_blocked.csv"
    
    monkeypatch.setattr(rp, "OPEN_PATH", open_file)
    monkeypatch.setattr(rp, "CLOSED_PATH", closed_file)
    monkeypatch.setattr(rp, "DECISION_PATH", decision_file)
    monkeypatch.setattr(rp, "BLOCKED_PATH", blocked_file)
    
    # Create legacy CSV without schema_version field
    with open_file.open("w", encoding="utf-8") as f:
        f.write("trade_id,signal_id,status\nt1,s1,OPEN\n")
        
    legacy_detected, state_initialized = rp.check_and_migrate_all_states()
    
    assert legacy_detected is True
    assert state_initialized is True
    
    archive_dir = tmp_path / "legacy_snapshots"
    assert archive_dir.exists()
    archived_files = list(archive_dir.glob("open_trades_*.csv"))
    assert len(archived_files) == 1
    
    assert open_file.exists()
    header = rp.get_csv_header(open_file)
    assert "schema_version" in header


def test_get_config_hash(tmp_path, monkeypatch):
    import prototype.run_prototype as rp
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_file = config_dir / "settings.json"
    settings_file.write_text('{"foo": "bar"}', encoding="utf-8")
    
    monkeypatch.setattr(rp, "CONFIG_DIR", config_dir)
    
    hash1 = rp.get_config_hash()
    assert len(hash1) == 8
    
    settings_file.write_text('{"foo": "baz"}', encoding="utf-8")
    hash2 = rp.get_config_hash()
    assert hash1 != hash2


def test_replay_normalizes_and_preserves_canonical_schema():
    legacy_open = {
        "trade_id": "t1",
        "signal_id": "s1",
        "symbol": "EURUSD",
        "status": "PAPER_OPEN",
        "direction": "LONG",
        "entry_time": "2026-05-20T00:00:00+00:00",
        "entry_price": 1.10,
        "stop_loss": 1.09,
        "take_profit_1": 1.12,
        "risk_distance": 0.01,
    }
    normalized_open = normalize_trade_row(legacy_open, OPEN_FIELDS)
    assert set(normalized_open.keys()) == set(OPEN_FIELDS)
    assert normalized_open["initial_stop_loss"] == 1.09
    assert normalized_open["initial_risk_distance"] == 0.01
    assert normalized_open["schema_version"] == "2.0.0"

    existing_closed = [
        {
            "trade_id": "t1",
            "signal_id": "s1",
            "status": "PAPER_CLOSED",
            "symbol": "EURUSD",
            "stop_loss": 1.09,
            "risk_distance": 0.01,
            "exit_time": "2026-05-20T01:00:00+00:00",
            "exit_price": 1.12,
        }
    ]
    new_closed = [
        {
            "trade_id": "t2",
            "signal_id": "s2",
            "status": "PAPER_CLOSED",
            "symbol": "GBPUSD",
            "stop_loss": 1.25,
            "risk_distance": 0.02,
            "exit_time": "2026-05-20T02:00:00+00:00",
            "exit_price": 1.20,
        }
    ]
    merged_closed = merge_rows(existing_closed, new_closed, CLOSED_FIELDS, "trade_id")
    assert len(merged_closed) == 2
    assert all(set(row.keys()) == set(CLOSED_FIELDS) for row in merged_closed)
    assert all(row["schema_version"] == "2.0.0" for row in merged_closed)
    by_trade = {row["trade_id"]: row for row in merged_closed}
    assert by_trade["t1"]["initial_stop_loss"] == 1.09
    assert by_trade["t2"]["initial_risk_distance"] == 0.02

