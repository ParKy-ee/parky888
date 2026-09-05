"""Phase 4: AI dataset labels and tiering."""

from .risk_manager import to_float


def compute_train_tier(row):
    mode = row.get("mode", "")
    label_quality = row.get("label_quality", "")
    exit_source = row.get("exit_source", "")
    outcome = row.get("outcome", "")
    is_trainable = str(row.get("is_trainable", "")).lower() == "true"

    if outcome == "blocked_trade":
        return "BLOCKED_SAMPLE"
    if outcome == "observation_only":
        return "OBSERVATION"
    if mode != "realistic_paper_mode":
        return "LOW"
    if is_trainable and label_quality == "HIGH" and exit_source == "candle_path":
        return "HIGH"
    if exit_source in ("candle_path", "historical_replay", "latest_candle_close"):
        return "MEDIUM"
    return "LOW"


def should_enter_label(row):
    outcome = row.get("outcome", "")
    if outcome == "opened_trade":
        return "true"
    if outcome in ("blocked_trade", "observation_only"):
        decision = str(row.get("decision", ""))
        if "TRADE_READY" in decision:
            return "false"
    signal_type = row.get("signal_type", "")
    if signal_type == "TRADE_READY" and outcome != "opened_trade":
        return "false"
    return ""


def r_from_row(row):
    fav = to_float(row.get("max_favorable_r"))
    adv = to_float(row.get("max_adverse_r"))
    return fav, adv


def apply_dataset_labels(row):
    fav, adv = r_from_row(row)
    return {
        **row,
        "train_tier": compute_train_tier(row),
        "should_enter_label": should_enter_label(row),
        "max_favorable_r": fav if fav is not None else row.get("max_favorable_r", ""),
        "max_adverse_r": adv if adv is not None else row.get("max_adverse_r", ""),
        "signal_cluster_id": row.get("signal_cluster_id", ""),
        "cluster_status": row.get("cluster_status", ""),
        "outcome": row.get("outcome", "opened_trade" if row.get("trade_id") else ""),
    }
