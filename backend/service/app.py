import os
import sys
import time
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import joblib
import ta
import yfinance as yf
import requests
import pymysql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, ".yfinance_cache")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(CACHE_DIR)

LOG_FILE = os.path.join(LOGS_DIR, "service.log")
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

# Environment Configurations
DB_TYPE = os.environ.get("DB_TYPE", "mysql").lower()
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "ai_trading_db")
MYSQL_USER = os.environ.get("MYSQL_USER", "trading_user")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "trading_pass_123")

UNIVERSE = ["NVDA", "AMD", "TSLA", "MSFT", "AVGO", "NFLX", "AMZN", "META", "GOOGL", "SPY"]
# Yahoo Finance is an unofficial data source and aggressively rate-limits bursts.
# A 15-minute minimum keeps this daemon from repeatedly hitting the same API.
SCAN_INTERVAL_MINUTES = max(15, int(os.environ.get("SCAN_INTERVAL_MINUTES", "15")))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.38"))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def get_db_connection():
    try:
        return pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=10
        )
    except Exception as e:
        logger.error(f"ไม่สามารถเชื่อมต่อ MySQL ({MYSQL_HOST}:{MYSQL_PORT}): {e}")
        return None

def init_database():
    logger.info(f"[*] ตรวจสอบการเชื่อมต่อฐานข้อมูล MySQL ที่ {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}...")
    conn = get_db_connection()
    if not conn:
        logger.warning("⚠️ ไม่สามารถเชื่อมต่อ MySQL ได้ในขณะนี้ จะลองใหม่ในลูปถัดไป")
        return

    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                time DATETIME NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL(12, 4) NOT NULL,
                ai_confidence DECIMAL(6, 4) NOT NULL,
                sl_price DECIMAL(12, 4) NOT NULL,
                tp_price DECIMAL(12, 4) NOT NULL,
                action VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_symbol_time (symbol, time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_positions (
                symbol VARCHAR(20) PRIMARY KEY,
                entry_date DATETIME NOT NULL,
                entry_price DECIMAL(12, 4) NOT NULL,
                highest_price DECIMAL(12, 4) NOT NULL,
                sl_price DECIMAL(12, 4) NOT NULL,
                tp_price DECIMAL(12, 4) NOT NULL,
                status_note VARCHAR(50) DEFAULT 'INITIAL_SL',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_bars (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                time DATETIME NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                open DECIMAL(12, 4),
                high DECIMAL(12, 4),
                low DECIMAL(12, 4),
                close DECIMAL(12, 4),
                volume BIGINT,
                rsi DECIMAL(6, 2),
                atr DECIMAL(12, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_symbol_time (symbol, time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # Existing installations already have market_bars, so CREATE TABLE above
        # cannot add the new freshness column for them.  Migrate it safely here.
        cursor.execute("""
            SELECT COUNT(*) AS column_exists
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'market_bars'
              AND COLUMN_NAME = 'last_scanned_at'
        """, (MYSQL_DATABASE,))
        if cursor.fetchone()["column_exists"] == 0:
            cursor.execute("""
                ALTER TABLE market_bars
                ADD COLUMN last_scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP AFTER created_at
            """)
            logger.info("เพิ่มคอลัมน์ last_scanned_at ใน market_bars แล้ว")
    conn.close()
    logger.info("✅ ฐานข้อมูล MySQL พร้อมใช้งานสมบูรณ์!")

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
        logger.info("โมเดล AI พร้อมทำงาน! (ใช้ RAM ไม่เกิน 120MB)")

    def fetch_latest_data(self):
        results = {}
        try:
            # Request the shared Yahoo session once for the complete universe.
            # `threads=False` prevents a burst of concurrent requests that can trigger HTTP 429.
            data = yf.download(
                tickers=UNIVERSE,
                period="60d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20,
            )
        except Exception as e:
            logger.warning(f"ดึงข้อมูลตลาดไม่สำเร็จ: {e}")
            return results

        if data.empty:
            logger.warning("Yahoo Finance ไม่ส่งข้อมูลกลับมา; จะรอลองใหม่ในรอบถัดไป")
            return results

        for symbol in UNIVERSE:
            try:
                # With group_by='ticker', each symbol has its own OHLCV frame.
                df = data[symbol].dropna(how="all").reset_index()
                if len(df) < 30:
                    logger.warning(f"ข้อมูล {symbol} ไม่เพียงพอ")
                    continue
                df = df.rename(columns={"Date": "time", "Datetime": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df["time"] = pd.to_datetime(df["time"])
                df["symbol"] = symbol
                results[symbol] = df.sort_values("time").reset_index(drop=True)
            except (KeyError, TypeError) as e:
                logger.warning(f"ไม่พบข้อมูล {symbol} ในผลลัพธ์ Yahoo: {e}")
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
        conn = get_db_connection()
        if not conn:
            logger.warning("ไม่สามารถบันทึกลง MySQL ได้เนื่องจากการเชื่อมต่อขาดหาย")
            return

        with conn.cursor() as cursor:
            for symbol, (row, full_df) in processed_data.items():
                features = pd.DataFrame([row[self.feature_cols].fillna(0)])
                p1 = self.lgb_model.predict_proba(features)[:, 1][0]
                p2 = self.rf_model.predict_proba(features)[:, 1][0]
                confidence = (0.5 * p1 + 0.5 * p2)

                curr_price = float(row["close"])
                atr = float(row["atr_14"])
                tp_price = curr_price + (4.0 * atr)
                sl_price = curr_price - (1.5 * atr)

                logger.info(f"[{symbol}] ราคา: ${curr_price:.2f} | ความมั่นใจ AI: {confidence*100:.1f}% | RS: {row['rs_20d']*100:+.1f}%")

                # บันทึก Bar ล่าสุดลง MySQL
                try:
                    cursor.execute("""
                        INSERT INTO market_bars (time, symbol, open, high, low, close, volume, rsi, atr)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            open=VALUES(open), high=VALUES(high), low=VALUES(low),
                            close=VALUES(close), volume=VALUES(volume),
                            rsi=VALUES(rsi), atr=VALUES(atr),
                            last_scanned_at=CURRENT_TIMESTAMP
                    """, (row["time"].strftime("%Y-%m-%d %H:%M:%S"), symbol, float(row["open"]), float(row["high"]), float(row["low"]), curr_price, int(row["volume"]), float(row["rsi_14"]), atr))
                except Exception as e:
                    logger.warning(f"Error saving bar {symbol}: {e}")

                # ตรวจสอบสัญญาณซื้อ
                if confidence >= CONFIDENCE_THRESHOLD and row["market_bullish"] and row["rs_20d"] > 0:
                    cursor.execute("SELECT id FROM signals WHERE symbol = %s AND DATE(time) = CURDATE()", (symbol,))
                    if not cursor.fetchone():
                        logger.info(f"🔥 [BUY SIGNAL] ตรวจพบสัญญาณซื้อ {symbol} @ ${curr_price:.2f} (ความมั่นใจ {confidence*100:.1f}%)")
                        cursor.execute("""
                            INSERT INTO signals (time, symbol, price, ai_confidence, sl_price, tp_price, action)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, curr_price, confidence, sl_price, tp_price, "BUY"))

                        alert_msg = (
                            f"🚨 *AI Swing Trading Signal: ซื้อ {symbol}*\n"
                            f"• ราคาปัจจุบัน: `${curr_price:.2f}`\n"
                            f"• ความมั่นใจ AI: `{confidence*100:.1f}%`\n"
                            f"• จุด Take Profit (เป้าหมาย): `${tp_price:.2f}` (+{((tp_price-curr_price)/curr_price)*100:.1f}%)\n"
                            f"• จุด Stop Loss (คัตลอส): `${sl_price:.2f}` ({((sl_price-curr_price)/curr_price)*100:.1f}%)\n"
                            f"• บันทึกลง MySQL Database เรียบร้อย"
                        )
                        send_telegram_alert(alert_msg)

        conn.close()

    async def start(self):
        logger.info("=" * 60)
        logger.info(" AI SWING TRADING DAEMON (MYSQL ENABLED)")
        logger.info(f" สแกนตลาดทุกๆ: {SCAN_INTERVAL_MINUTES} นาที")
        logger.info(f" เกณฑ์ความมั่นใจ AI ขั้นต่ำ: {CONFIDENCE_THRESHOLD*100:.1f}%")
        logger.info(f" ฐานข้อมูล: MySQL @ {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        logger.info("=" * 60)

        while True:
            try:
                new_york_now = datetime.now(ZoneInfo("America/New_York"))
                if new_york_now.weekday() >= 5:
                    logger.info("ตลาดสหรัฐปิดช่วงสุดสัปดาห์; ข้ามการเรียก Yahoo Finance ในรอบนี้")
                else:
                    logger.info(f"[*] เริ่มรอบการสแกนตลาด ({new_york_now.strftime('%Y-%m-%d %H:%M:%S %Z')})...")
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
