import yfinance as yf
from datetime import datetime, timezone

symbol = "AUDCAD=X"
data = yf.download(symbol, period="1d", interval="1h")
print(data)
