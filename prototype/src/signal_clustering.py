import hashlib
from datetime import datetime, timezone

from .risk_manager import direction_from_signal
from .trade_context import get_trade_thesis, parse_source_run_at


def compute_signal_cluster_id(signal, settings):
    market = signal.get("market", "forex")
    symbol = signal.get("symbol", "unknown")
    direction = direction_from_signal(signal.get("decision", ""), signal.get("final_bias", ""))
    thesis = get_trade_thesis(symbol, direction, market)
    source_time = parse_source_run_at(signal.get("source_run_at"))
    window_minutes = int(settings.get("signal_cluster_window_minutes", 240))
    window_seconds = max(window_minutes, 1) * 60
    bucket = int(source_time.timestamp() // window_seconds) if source_time else 0
    raw = "|".join([market, symbol, thesis, direction, str(bucket)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_cluster_index(open_trades, blocked_rows, closed_trades, settings):
    index = {}
    for row in open_trades:
        cluster_id = row.get("signal_cluster_id")
        if cluster_id:
            index.setdefault(cluster_id, []).append(row.get("signal_id"))
    for row in blocked_rows:
        if row.get("outcome") != "blocked_trade":
            continue
        cluster_id = row.get("signal_cluster_id")
        if cluster_id:
            index.setdefault(cluster_id, []).append(row.get("signal_id"))
    window_minutes = int(settings.get("signal_cluster_window_minutes", 240))
    cutoff = datetime.now(timezone.utc).timestamp() - window_minutes * 60
    for row in closed_trades:
        cluster_id = row.get("signal_cluster_id")
        if not cluster_id:
            continue
        try:
            exit_time = datetime.fromisoformat(str(row.get("exit_time", "")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if exit_time.timestamp() >= cutoff:
            index.setdefault(cluster_id, []).append(row.get("signal_id"))
    return index


def resolve_cluster_status(cluster_id, cluster_index, signal_id):
    members = cluster_index.get(cluster_id, [])
    other_members = [member for member in members if member and member != signal_id]
    if other_members:
        return "same_cluster"
    return "new_opportunity"


def register_cluster(cluster_index, cluster_id, signal_id):
    if not cluster_id:
        return
    members = cluster_index.setdefault(cluster_id, [])
    if signal_id not in members:
        members.append(signal_id)
