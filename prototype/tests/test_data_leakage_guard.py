from datetime import datetime, timedelta, timezone

from prototype.src.paper_trade_tracker import passes_data_leakage_guard


def test_signal_must_precede_entry_time():
    planned_entry = datetime(2026, 5, 9, 16, 0, 0, tzinfo=timezone.utc)
    signal = {"source_run_at": "2026-05-09 22:59:00"}
    ok, reason = passes_data_leakage_guard(signal, planned_entry)
    assert ok is True
    assert reason == "ok"


def test_future_signal_is_blocked():
    planned_entry = datetime(2026, 5, 9, 16, 0, 0, tzinfo=timezone.utc)
    signal = {"source_run_at": "2026-05-10 00:01:00"}
    ok, reason = passes_data_leakage_guard(signal, planned_entry)
    assert ok is False
    assert reason == "entry_time_not_after_signal_time"


def test_missing_signal_time_is_blocked():
    ok, reason = passes_data_leakage_guard({}, datetime.now(timezone.utc) + timedelta(minutes=1))
    assert ok is False
    assert reason == "missing_or_invalid_source_run_at"
