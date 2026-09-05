import json

from .csv_store import read_csv
from .paths import CRYPTO_LOG_DIR, FOREX_LOG_DIR


SUMMARY_PATH = {
    "forex": FOREX_LOG_DIR / "forex_models_summary.csv",
    "crypto": CRYPTO_LOG_DIR / "crypto_models_summary.csv",
}

LATEST_DIR = {
    "forex": FOREX_LOG_DIR,
    "crypto": CRYPTO_LOG_DIR,
}


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_price(market, symbol):
    summary_path = SUMMARY_PATH[market]
    for row in read_csv(summary_path):
        if row.get("symbol") == symbol:
            price = _to_float(row.get("price"))
            if price is not None:
                return price

    latest_path = LATEST_DIR[market] / f"{symbol.lower()}_forecast_latest.json"
    if latest_path.exists():
        with latest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        timeframes = payload.get("timeframes", [])
        if timeframes:
            return _to_float(timeframes[0].get("price"))
    return None


def generated_at_from_human_log(path):
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Generated at:" in line:
            return line.split("Generated at:", 1)[1].strip()
    return ""
