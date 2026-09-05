import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.join(BASE_DIR, "service")
APP_PATH = os.path.join(SERVICE_DIR, "app.py")

app_code = """import os
import sys
import time
import asyncio
import sqlite3
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import ta
import yfinance as yf
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_DIR = os.path.join(BASE_DIR, "database")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, ".yfinance_cache")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(CACHE_DIR)

LOG_FILE = os.path.join(LOGS_DIR, "service.log")
DB_PATH = os.path.join(DB_DIR, "market_data.db")
MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_trailing_model.joblib")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("AITradingService")

UNIVERSE = ["NVDA", "AMD", "TSLA", "MSFT", "AVGO", "NFLX", "AMZN", "META", "GOOGL", "SPY"]
SCAN_INTERVAL_MINUTES = int(os.environ.get("SCAN_INTERVAL_MINUTES", "5"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.38"))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            symbol TEXT,
            price REAL,
            ai_confidence REAL,
            sl_price REAL,
            tp_price REAL,
            action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_positions (
            symbol TEXT PRIMARY KEY,
            entry_date TEXT,
            entry_price REAL,
            highest_price REAL,
            sl_price REAL,
            tp_price REAL,
            status_note TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("ฐานข้อมูล SQLite พร้อมใช้งาน!")

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"ไม่สามารถส่ง Telegram alert ได้: {e}")

class AITradingDaemon:
    def __init__(self):
        logger.info("กำลังโหลดโมเดล AI เข้าสู่หน่วยความจำ (RAM)...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
        model_dict = joblib.load(MODEL_PATH)
        self.lgb_model = model_dict["lgb"]
        self.rf_model = model_dict["rf"]
        self.feature_cols = model_dict["features"]
        logger.info("โมเดล AI พร้อมทำงาน! (ใช้ RAM รวมไม่เกิน 120MB)")

    def fetch_latest_data(self):
        results = {}
        for symbol in UNIVERSE:
            try:
                data = yf.download(symbol, period="60d", interval="1d", auto_adjust=False, progress=False)
                if data.empty or len(data) < 30:
                    continue
                df = data.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if col[0] else col[1] for col in df.columns]
                df = df.rename(columns={"Date": "time", "Datetime": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df["time"] = pd.to_datetime(df["time"])
                df["symbol"] = symbol
                results[symbol] = df.sort_values("time").reset_index(drop=True)
            except Exception as e:
                logger.warning(f"ดึงข้อมูล {symbol} ไม่สำเร็จ: {e}")
        return results

    def calculate_features(self, dfs):
        if "SPY" not in dfs:
            return {}
        spy = dfs["SPY"].copy()
        spy["ema_50"] = ta.trend.ema_indicator(spy["close"], window=50)
        spy["ema_200"] = ta.trend.ema_indicator(spy["close"], window=200)
        spy["return_20d"] = spy["close"].pct_change(20)
        spy_last = spy.iloc[-1]
        market_bullish = bool((spy_last["close"] > spy_last["ema_200"]) and (spy_last["ema_50"] > spy_last["ema_200"]))
        spy_ret_20d = float(spy_last["return_20d"])

        processed = {}
        for symbol, df in dfs.items():
            if symbol == "SPY" or len(df) < 30:
                continue
            d = df.copy()
            d["return_1d"] = d["close"].pct_change(1)
            d["return_5d"] = d["close"].pct_change(5)
            d["return_20d"] = d["close"].pct_change(20)
            d["rsi_14"] = ta.momentum.rsi(d["close"], window=14)
            d["roc_12"] = ta.momentum.roc(d["close"], window=12)
            
            macd = ta.trend.MACD(d["close"], window_fast=12, window_slow=26, window_sign=9)
            d["macd_line"] = macd.macd()
            d["macd_signal"] = macd.macd_signal()
            d["macd_hist"] = macd.macd_diff()

            d["ema_20"] = ta.trend.ema_indicator(d["close"], window=20)
            d["ema_50"] = ta.trend.ema_indicator(d["close"], window=50)
            d["ema_200"] = ta.trend.ema_indicator(d["close"], window=200)
            d["dist_ema_20"] = (d["close"] - d["ema_20"]) / (d["ema_20"] + 1e-12)
            d["dist_ema_50"] = (d["close"] - d["ema_50"]) / (d["ema_50"] + 1e-12)
            d["dist_ema_200"] = (d["close"] - d["ema_200"]) / (d["ema_200"] + 1e-12)
            d["ema_trend_ratio"] = d["ema_20"] / (d["ema_50"] + 1e-12)

            d["atr_14"] = ta.volatility.average_true_range(d["high"], d["low"], d["close"], window=14)
            d["atr_ratio"] = d["atr_14"] / (d["atr_14"].rolling(50).mean() + 1e-12)
            
            bb = ta.volatility.BollingerBands(d["close"], window=20, window_dev=2)
            d["bb_pband"] = bb.bollinger_pband()
            d["bb_width"] = bb.bollinger_wband()
            d["adx_14"] = ta.trend.adx(d["high"], d["low"], d["close"], window=14)

            d["vol_ma_20"] = d["volume"].rolling(20).mean()
            d["rvol"] = d["volume"] / (d["vol_ma_20"] + 1e-12)
            d["highest_20"] = d["high"].rolling(20).max()
            d["dist_to_20d_high"] = (d["highest_20"] - d["close"]) / (d["close"] + 1e-12)
            d["rs_20d"] = d["return_20d"] - spy_ret_20d

            last_row = d.iloc[-1].copy()
            last_row["market_bullish"] = market_bullish
            processed[symbol] = (last_row, d)
        return processed

    def evaluate_signals(self, processed_data):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for symbol, (row, full_df) in processed_data.items():
            features = pd.DataFrame([row[self.feature_cols].fillna(0)])
            p1 = self.lgb_model.predict_proba(features)[:, 1][0]
            p2 = self.rf_model.predict_proba(features)[:, 1][0]
            confidence = (0.5 * p1 + 0.5 * p2)

            curr_price = float(row["close"])
            atr = float(row["atr_14"])
            tp_price = curr_price + (4.0 * atr)
            sl_price = curr_price - (1.5 * atr)

            logger.info(f"[{symbol}] ราคา: ${curr_price:.2f} | ความมั่นใจ AI: {confidence*100:.1f}% | RS 20d: {row['rs_20d']*100:+.1f}%")

            if confidence >= CONFIDENCE_THRESHOLD and row["market_bullish"] and row["rs_20d"] > 0:
                cursor.execute("SELECT id FROM signals WHERE symbol = ? AND date(time) = date('now')", (symbol,))
                if not cursor.fetchone():
                    logger.info(f"🔥 [BUY SIGNAL] ตรวจพบสัญญาณซื้อ {symbol} @ ${curr_price:.2f} (ความมั่นใจ {confidence*100:.1f}%)")
                    cursor.execute('''
                        INSERT INTO signals (time, symbol, price, ai_confidence, sl_price, tp_price, action)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, curr_price, confidence, sl_price, tp_price, "BUY"))
                    conn.commit()

                    alert_msg = (
                        f"🚨 *AI Swing Trading Signal: ซื้อ {symbol}*\\n"
                        f"• ราคาปัจจุบัน: `${curr_price:.2f}`\\n"
                        f"• ความมั่นใจ AI: `{confidence*100:.1f}%`\\n"
                        f"• จุด Take Profit (เป้าหมาย): `${tp_price:.2f}` (+{((tp_price-curr_price)/curr_price)*100:.1f}%)\\n"
                        f"• จุด Stop Loss (คัตลอส): `${sl_price:.2f}` ({((sl_price-curr_price)/curr_price)*100:.1f}%)\\n"
                        f"• วันที่: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_alert(alert_msg)

        conn.close()

    async def start(self):
        logger.info("=" * 60)
        logger.info(" AI SWING TRADING BACKGROUND DAEMON STARTED")
        logger.info(f" สแกนตลาดทุกๆ: {SCAN_INTERVAL_MINUTES} นาที")
        logger.info(f" เกณฑ์ความมั่นใจ AI ขั้นต่ำ: {CONFIDENCE_THRESHOLD*100:.1f}%")
        logger.info("=" * 60)

        while True:
            try:
                logger.info(f"[*] เริ่มรอบการสแกนตลาด ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})...")
                raw_dfs = self.fetch_latest_data()
                processed = self.calculate_features(raw_dfs)
                if processed:
                    self.evaluate_signals(processed)
                logger.info(f"[+] รอบการสแกนเสร็จสิ้น หลับพัก {SCAN_INTERVAL_MINUTES} นาที...")
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในรอบการทำงาน: {e}", exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    init_database()
    daemon = AITradingDaemon()
    asyncio.run(daemon.start())
"""

with open(APP_PATH, "w", encoding="utf-8") as f:
    f.write(app_code)

print("[+] อัปเดต backend/service/app.py เรียบร้อย!")
