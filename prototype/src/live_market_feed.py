from datetime import datetime, timedelta, timezone
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import yfinance as yf

from .market_data import latest_price
from .paths import PROTOTYPE_ROOT


YFINANCE_CACHE_DIR = PROTOTYPE_ROOT / ".yfinance_cache"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


YF_SYMBOLS = {
    "forex": {
        "AUDCAD": "AUDCAD=X",
        "AUDCHF": "AUDCHF=X",
        "AUDJPY": "AUDJPY=X",
        "AUDNZD": "AUDNZD=X",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "CADCHF": "CADCHF=X",
        "CADJPY": "CADJPY=X",
        "CHFJPY": "CHFJPY=X",
        "EURAUD": "EURAUD=X",
        "EURCAD": "EURCAD=X",
        "EURCHF": "EURCHF=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "EURNZD": "EURNZD=X",
        "GBPAUD": "GBPAUD=X",
        "GBPCAD": "GBPCAD=X",
        "GBPCHF": "GBPCHF=X",
        "GBPJPY": "GBPJPY=X",
        "GBPNZD": "GBPNZD=X",
        "NZDCAD": "NZDCAD=X",
        "NZDCHF": "NZDCHF=X",
        "NZDJPY": "NZDJPY=X",
        "NZDUSD": "NZDUSD=X",
    },
    "crypto": {
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "SOLUSD": "SOL-USD",
        "BNBUSD": "BNB-USD",
        "XRPUSD": "XRP-USD",
    },
}

INTERVAL_MAP = {
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",
    "1d": "1d",
}


def parse_time(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_candles(market, symbol, since_time, timeframe):
    yf_symbol = YF_SYMBOLS.get(market, {}).get(symbol)
    if not yf_symbol:
        return []

    interval = INTERVAL_MAP.get(timeframe or "1h", "1h")
    since = parse_time(since_time)
    
    # Optimize: only download from 'since' to now
    # yfinance works best with start/end for specific ranges, but date boundaries are exclusive and timezone-tricky
    start_dt = since - timedelta(days=1) # Wider buffer for timezone differences
    end_dt = datetime.now(timezone.utc) + timedelta(days=2)

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        data = yf.download(
            yf_symbol, 
            start=start_dt.strftime("%Y-%m-%d"), 
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval, 
            auto_adjust=False, 
            progress=False
        )
        
    # If the range is very small (today), yfinance start/end might be tricky with intervals < 1d
    # Fallback to period if data is empty but we expect some
    if data.empty and (datetime.now(timezone.utc) - since).days < 7:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = yf.download(yf_symbol, period="1d", interval=interval, auto_adjust=False, progress=False)

    if data.empty:
        return []

    frame = data.reset_index()
    if hasattr(frame.columns, "levels"):
        frame.columns = [col[0] if col[0] else col[1] for col in frame.columns]
    time_column = "Datetime" if "Datetime" in frame.columns else "Date"
    candles = []
    for _, row in frame.iterrows():
        candle_time = row[time_column]
        if getattr(candle_time, "tzinfo", None) is None:
            candle_time = candle_time.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            candle_time = candle_time.to_pydatetime().astimezone(timezone.utc)
        if candle_time < since:
            continue
        candles.append(
            {
                "time": candle_time.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
        )
    return candles


def fetch_executable_price(market, symbol, timeframe="1h"):
    """Live/current price suitable for paper entry (never log-summary fallback)."""
    since = datetime.now(timezone.utc) - timedelta(hours=3)
    candles = fetch_candles(market, symbol, since.isoformat(), timeframe)
    if candles:
        return float(candles[-1]["close"]), "live_candle_close"
    return None, None


def fallback_price(market, symbol):
    """Log/forecast fallback for observation, monitoring, and exit only."""
    price = latest_price(market, symbol)
    if price is not None:
        return price, "latest_log_fallback"
    return None, None
