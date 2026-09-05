# AI Chatbot Forex Forecast System

โปรเจกต์นี้เป็นระบบ forecast คู่เงิน Forex สำหรับใช้เป็น backend/ข้อมูลประกอบการทำเว็บ dashboard หรือระบบแจ้งเตือนสัญญาณเทรด โดยเน้นการวิเคราะห์หลาย timeframe และสร้าง log ที่อ่านง่ายสำหรับมนุษย์ รวมถึงไฟล์ CSV/JSON ที่เว็บสามารถนำไปใช้งานต่อได้

> สถานะปัจจุบัน: ใช้สำหรับ research, watchlist, forecast dashboard ยังไม่ใช่ระบบส่ง order จริง

## ความสามารถหลัก

- ดึงข้อมูลราคา Forex จาก Yahoo Finance ผ่าน `yfinance`
- วิเคราะห์คู่เงินแยกเป็น model เฉพาะคู่
- ใช้ timeframe หลัก `30m`, `1h`, `4h`
- ประเมิน bias เป็น `BULLISH`, `BEARISH`, `NEUTRAL`
- สร้าง decision เป็น `BULLISH_WATCH`, `BEARISH_WATCH`, `WAIT`
- คำนวณ score และ confidence ของแต่ละ model
- สร้าง entry price, entry zone, stop loss, take profit 1
- สรุปเหตุผลของสัญญาณจาก indicator ที่ model ใช้
- รันทุก model ได้ในคำสั่งเดียว
- สร้าง log สำหรับคนอ่านและไฟล์ structured data สำหรับเว็บ

## Model ที่มี

ไฟล์ทั้งหมดอยู่ใน `forex/models_`

| Model | ไฟล์ | แนวคิดหลัก |
|---|---|---|
| EURUSD | `model_EURUSD.py` | Liquid trend balance ใช้ EMA21/55/200, MACD, RSI, Bollinger width |
| GBPUSD | `model_GBPUSD.py` | Volatility breakout ใช้ Donchian, EMA20/50/100, body ratio |
| USDJPY | `model_USDJPY.py` | Trend + higher timeframe bias ใช้ EMA, MACD, ADX, ATR |
| USDCHF | `model_USDCHF.py` | Range / safe-haven ใช้ Bollinger position, RSI, Stochastic, SMA |
| AUDUSD | `model_AUDUSD.py` | Risk / commodity momentum ใช้ EMA10/50/200, CCI, ROC, MACD |
| USDCAD | `model_USDCAD.py` | Range-trend hybrid ใช้ EMA20/100/200, Bollinger position, ROC, RSI |

แต่ละ model เป็นไฟล์ standalone ไม่ใช้ไฟล์กลางร่วมกัน เพื่อให้สามารถปรับ parameter และ indicator ให้เหมาะกับนิสัยตลาดของแต่ละคู่เงินได้

## วิธีรันทุก Model

ใช้คำสั่งเดียว:

```powershell
python .\forex\models_\run_all_models.py
```

ผลลัพธ์จะรันครบ:

```text
EURUSD
GBPUSD
USDJPY
USDCHF
AUDUSD
USDCAD
```

## วิธีรันทีละ Model

```powershell
python .\forex\models_\model_EURUSD.py
python .\forex\models_\model_GBPUSD.py
python .\forex\models_\model_USDJPY.py
python .\forex\models_\model_USDCHF.py
python .\forex\models_\model_AUDUSD.py
python .\forex\models_\model_USDCAD.py
```

## Output สำหรับมนุษย์

เมื่อรัน `run_all_models.py` ระบบจะสร้างไฟล์:

```text
logs/forex_models_human_log.md
```

ไฟล์นี้เหมาะสำหรับเปิดอ่านตรง ๆ หรือแสดงบนหน้าเว็บแบบ markdown

ตัวอย่างข้อมูลที่อยู่ใน log:

```text
Final Summary:
- Symbol
- Status
- Bias
- Decision
- Score
- Confidence
- Price
- Strongest TF

Entry And Trade Levels:
- Current Price
- Entry Price
- Entry Zone
- Stop Loss
- Take Profit 1

Reasons:
- Alignment ของ timeframe
- เหตุผลของสัญญาณ เช่น ema_bull_stack, htf_bearish, momentum_negative
```

ตัวอย่าง latest log:

| Symbol | Bias | Decision | Score | Confidence | Entry Zone | SL | TP1 |
|---|---|---|---:|---:|---|---:|---:|
| EURUSD | BULLISH | BULLISH_WATCH | 6.11 | 76.39% | 1.17897 to 1.17911 | 1.17827 | 1.17990 |
| GBPUSD | NEUTRAL | WAIT | 3.11 | 38.89% | WAIT | 1.36218 | 1.36473 |
| USDJPY | BEARISH | BEARISH_WATCH | -6.22 | 77.78% | 156.42500 to 156.62100 | 156.81222 | 156.36604 |
| USDCHF | BEARISH | BEARISH_WATCH | -4.00 | 57.14% | 0.77590 to 0.77602 | 0.77677 | 0.77507 |
| AUDUSD | BULLISH | BULLISH_WATCH | 6.00 | 66.67% | 0.72495 to 0.72516 | 0.72439 | 0.72576 |
| USDCAD | NEUTRAL | WAIT | 0.89 | 11.11% | WAIT | 1.36517 | 1.36973 |

## Output สำหรับทำเว็บ

### 1. Summary CSV

ไฟล์:

```text
logs/forex_models_summary.csv
```

เหมาะสำหรับเว็บที่ต้องการโหลดข้อมูลเป็นตาราง เช่น dashboard, watchlist, signal table

Schema:

| Field | ความหมาย |
|---|---|
| `symbol` | คู่เงิน เช่น EURUSD |
| `status` | สถานะการรัน model เช่น OK, ERROR |
| `final_bias` | ทิศทางหลัก `BULLISH`, `BEARISH`, `NEUTRAL` |
| `decision` | คำตัดสิน เช่น `BULLISH_WATCH`, `BEARISH_WATCH`, `WAIT` |
| `final_score` | คะแนนรวมถ่วงน้ำหนักจากหลาย timeframe |
| `confidence` | ความมั่นใจของ model |
| `alignment` | จำนวน timeframe ที่เห็นตรงกัน |
| `price` | ราคาปัจจุบันจาก timeframe ที่ strongest |
| `best_timeframe` | timeframe ที่ให้ score แรงสุด |
| `entry_price` | ราคาเข้าอ้างอิง ถ้า decision ไม่ใช่ WAIT |
| `entry_zone` | โซนราคาที่พิจารณาเข้า |
| `stop_loss` | ระดับ SL อ้างอิง |
| `take_profit_1` | ระดับ TP1 อ้างอิง |
| `reason` | เหตุผลหลักของ signal |
| `returncode` | exit code ของ script model |

ตัวอย่างการอ่านด้วย Python:

```python
import pandas as pd

df = pd.read_csv("logs/forex_models_summary.csv")
watch = df[df["decision"].str.contains("WATCH")]
print(watch[["symbol", "decision", "entry_price", "stop_loss", "take_profit_1"]])
```

ตัวอย่างการใช้ในเว็บ API:

```python
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/api/forex/signals")
def forex_signals():
    df = pd.read_csv("logs/forex_models_summary.csv")
    return df.to_dict(orient="records")
```

### 2. Latest JSON รายคู่

แต่ละ model จะสร้าง JSON ล่าสุดแยกตามคู่เงิน:

```text
logs/eurusd_forecast_latest.json
logs/gbpusd_forecast_latest.json
logs/usdjpy_forecast_latest.json
logs/usdchf_forecast_latest.json
logs/audusd_forecast_latest.json
logs/usdcad_forecast_latest.json
```

เหมาะสำหรับเว็บที่ต้องการหน้า detail ของคู่เงิน เช่น `/forex/EURUSD`

โครงสร้างหลัก:

```json
{
  "summary": {
    "symbol": "EURUSD",
    "final_bias": "BULLISH",
    "final_score": 6.11,
    "confidence": 0.7639,
    "decision": "BULLISH_WATCH",
    "alignment": {
      "BULLISH": 3
    }
  },
  "timeframes": [
    {
      "interval": "30m",
      "price": 1.17897,
      "bias": "BULLISH",
      "score": 7,
      "confidence": 0.875,
      "adx": 41.38,
      "rsi": 69.77,
      "atr_ratio": 0.86,
      "support": 1.17288,
      "resistance": 1.17911,
      "stop_loss": 1.17827,
      "take_profit_1": 1.17990,
      "reasons": "above_ema200,ema_bull_stack,macd_momentum_up,htf_bullish"
    }
  ]
}
```

### 3. Historical Forecast CSV รายคู่

แต่ละ model append ประวัติ forecast ลง CSV:

```text
logs/eurusd_forecast_report.csv
logs/gbpusd_forecast_report.csv
logs/usdjpy_forecast_report.csv
logs/usdchf_forecast_report.csv
logs/audusd_forecast_report.csv
logs/usdcad_forecast_report.csv
```

เหมาะสำหรับ:

- ทำกราฟ score ย้อนหลัง
- เช็กว่า model เปลี่ยน bias บ่อยไหม
- วิเคราะห์ว่า timeframe ไหนแม่นกว่า
- ทำหน้า history ในเว็บ

## การตีความ Decision

| Decision | ความหมาย |
|---|---|
| `BULLISH_WATCH` | มี bias ขึ้น แต่ยังควรใช้เป็น watch signal ไม่ใช่ยิง order อัตโนมัติ |
| `BEARISH_WATCH` | มี bias ลง แต่ยังควรใช้เป็น watch signal |
| `WAIT` | ยังไม่ควรเข้า เพราะ score ต่ำหรือ timeframe ไม่ confirm |

## ข้อควรระวัง

- ระบบนี้เป็น forecast/watchlist ไม่ใช่ financial advice
- ยังไม่มี broker execution
- ยังไม่มี live spread จริงจาก broker
- TP/SL เป็นระดับอ้างอิงจาก ATR/indicator ไม่ใช่คำสั่งเข้าเทรดทันที
- ก่อนใช้ paper/live ควรมี backtest, forward test, risk limit และ manual review

## Roadmap ที่เหมาะต่อไป

- เพิ่ม API endpoint สำหรับอ่าน `forex_models_summary.csv`
- เพิ่มหน้าเว็บ dashboard แสดง bias, score, confidence, entry, SL, TP
- เพิ่ม chart ต่อคู่เงินจาก `*_forecast_latest.json`
- เพิ่ม scheduler ให้รัน `run_all_models.py` ทุก 30 นาทีหรือ 1 ชั่วโมง
- เพิ่ม alert เมื่อ decision เปลี่ยนจาก `WAIT` เป็น `WATCH`
- เพิ่ม paper-trade tracker สำหรับบันทึกว่า signal แต่ละครั้งชน TP หรือ SL
