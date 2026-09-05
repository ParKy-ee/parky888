import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def extract_timeframe_from_reason(reason: str):
    if not reason:
        return None
    match = re.search(r"\b(5m|15m|30m|1h|2h|4h|1d)\b", str(reason))
    return match.group(1) if match else None


def parse_source_run_at(value):
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Bangkok"))
    return parsed.astimezone(timezone.utc)


def get_trade_thesis(symbol, direction, market="forex"):
    market_lower = str(market).lower() if market else "forex"
    if market_lower == "forex":
        if "USD" in symbol:
            if symbol.endswith("USD"):
                return "USD_WEAK" if direction == "LONG" else "USD_STRONG"
            if symbol.startswith("USD"):
                return "USD_WEAK" if direction == "SHORT" else "USD_STRONG"
        return "SYMBOL_SPECIFIC"
    symbol_upper = str(symbol).upper()
    if "BTC" in symbol_upper:
        return "CRYPTO_BTC"
    if "ETH" in symbol_upper:
        return "CRYPTO_ETH"
    return f"CRYPTO_{symbol_upper}"
