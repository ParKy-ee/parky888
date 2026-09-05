# Auto Trade Prototype

### 🎯 ภาพรวมและจุดประสงค์ของโปรเจกต์
โปรเจกต์นี้เป็นระบบ **Auto Trade Prototype** แบบ CSV-first สำหรับตลาด **Forex และ Crypto** ซึ่งปัจจุบันเน้นใช้สำหรับการวิจัยและการจำลองเทรด (Paper Trade) เป็นหลัก 
* **หน้าที่หลัก:** อ่านและรับ Signal จากโมเดล AI นำมาจำลองการเทรดแบบ Paper Trade โดยมีการจัดการความเสี่ยง (Risk Management) และติดตามจุดทำกำไร/ตัดขาดทุน (TP/SL) อย่างเป็นระบบ
* **เป้าหมายปัจจุบัน:** เก็บข้อมูลและประเมินผลการเทรดจากโมเดล (ประมาณ 30-50 ไม้) ก่อนที่จะนำไปพิจารณาเปิดระบบ Auto Trade ด้วยเงินจริงในอนาคต

### 🚀 สถานะปัจจุบัน
- **สถานะ:** อยู่ในโหมด `paper / research prototype` **(ยังไม่มีการเทรดด้วยเงินจริงหรือ Live trading)**
- **การเชื่อมต่อ Broker:** ยังไม่มีการเชื่อมต่อไปยัง Broker หรือ Exchange ใดๆ
- **ระบบฐานข้อมูล:** ใช้ไฟล์ CSV และ JSONL ในการเก็บข้อมูลทั้งหมด
- **หน้า Dashboard:** มี Frontend Monitor พัฒนาด้วย Next.js รันอยู่ที่ `http://127.0.0.1:3000/prototype` สำหรับตรวจสอบสถานะและดูประวัติ
- **สถานะโมเดล:** อยู่ในช่วงเก็บข้อมูล (30-50 ไม้) เพื่อประเมิน Performance ก่อนพิจารณา Auto Trade จริง

## สิ่งที่ทำแล้ว

### 1. Prototype Structure

สร้างโครงหลักใน `prototype/`

```text
prototype/
├─ config/
│  └─ settings.json
├─ data/
│  ├─ signals/
│  ├─ paper_trades/
│  ├─ ai/
│  └─ audit/
├─ scripts/
│  └─ run_once.ps1
├─ src/
│  ├─ ai_dataset.py
│  ├─ audit.py
│  ├─ cost_model.py
│  ├─ csv_store.py
│  ├─ live_market_feed.py
│  ├─ market_data.py
│  ├─ paper_trade_tracker.py
│  ├─ paths.py
│  ├─ risk_manager.py
│  ├─ settings.py
│  └─ signal_ingest.py
├─ tests/
│  ├─ test_data_leakage_guard.py
│  └─ test_risk_manager.py
└─ run_prototype.py
```

### 2. Signal Ingestion

อ่าน signal จาก log เดิม:

- Forex: `ai_chatbot/logs/forex_models_summary.csv`
- Crypto: `cryto/logs/crypto_models_summary.csv`

แล้ว normalize เป็น:

- `prototype/data/signals/forex_signals.csv`
- `prototype/data/signals/crypto_signals.csv`

ระบบสร้าง `signal_id` จาก market, symbol, run time, decision, score, price และ reason เพื่อกัน duplicate

### 3. Paper Trade Tracker

สร้างระบบ paper trade จาก signal ที่ผ่านเงื่อนไขใน `settings.json`

Output:

- `prototype/data/paper_trades/open_trades.csv`
- `prototype/data/paper_trades/closed_trades.csv`
- `prototype/data/paper_trades/trade_events.csv`
- `prototype/data/paper_trades/paper_trade_summary.csv`

ความสามารถ:

- เปิด paper trade จาก `TRADE_READY`
- กันเปิดซ้ำด้วย `signal_id` และ `trade_id`
- จำกัดจำนวน position รวม
- จำกัด position ต่อ symbol
- คำนวณ position size จาก balance, risk %, entry และ stop loss
- **Symbol Cooldown**: ป้องกันการเข้าไม้ซ้ำเร็วเกินไป (Standard 6h / GBP Cross 12h)
- **Breakeven Protection**: เลื่อน SL เป็นจุดคุ้มทุนอัตโนมัติเมื่อราคาถึง 75% ของเป้าหมาย TP1
- ปิด trade เมื่อแตะ TP/SL หรือหมดเวลา

### 4. Market Feed และ TP/SL Path

เพิ่ม `live_market_feed.py`

ความสามารถ:

- ดึง candle จาก `yfinance` แบบ best-effort
- ใช้ high/low ของ candle เพื่อเช็กว่า TP หรือ SL ถูกแตะหรือไม่
- ถ้า candle เดียวแตะทั้ง TP และ SL จะถือว่า SL โดนก่อนแบบ conservative
- ถ้าดึง market feed ไม่ได้ จะ fallback ไปใช้ latest price จาก model log
- บันทึก feed/source ผ่าน audit และ field `last_market_source`

### 5. Risk Manager

เพิ่ม `risk_manager.py`

ความสามารถ:

- แปลง decision เป็น direction: `LONG` / `SHORT`
- คำนวณ position สำหรับ crypto จาก distance ถึง SL
- คำนวณ position สำหรับ Forex ด้วย pip size, pip value และ contract size
- รองรับ config ราย symbol ใน `settings.json`

ตัวอย่าง config Forex:

```json
"EURUSD": {
  "pip_size": 0.0001,
  "pip_value_per_lot_usd": 10.0,
  "contract_size": 100000
}
```

### 6. Cost Model

เพิ่ม `cost_model.py`

ความสามารถ:

- ใช้ cost assumption ราย market
- ใช้ cost assumption ราย symbol ได้
- รวม fee/spread/slippage สำหรับ paper trade return

Config อยู่ใน:

```text
prototype/config/settings.json
```

### 7. Data Leakage Guard

เพิ่ม guard ใน `paper_trade_tracker.py`

กฎ:

- signal ต้องมี `source_run_at`
- planned entry time ต้องมากกว่า source signal time
- ถ้า source time หายหรือเป็นอนาคต จะ block paper trade
- block event จะถูกเขียนลง execution audit

เพิ่ม test:

- `prototype/tests/test_data_leakage_guard.py`

### 8. Audit Logs

เพิ่ม audit log แบบ JSONL:

- `prototype/data/audit/decision_audit_log.jsonl`
- `prototype/data/audit/execution_audit_log.jsonl`

ใช้เก็บ:

- การ ingest signal
- การเปิด paper trade
- การปิด paper trade
- market feed failure
- data leakage guard block

### 9. AI Dataset Export

เพิ่ม `src/ai_dataset.py` สำหรับเตรียมข้อมูลให้ AI

สร้าง dataset 2 ประเภท:

- `prototype/data/ai/ai_training_dataset.csv`: เก็บข้อมูลจาก **Closed Trades** (รายการที่ปิดแล้ว) พร้อมผล Win/Loss
- `prototype/data/ai/signal_observation_dataset.csv`: เก็บข้อมูลจาก **All Signals**

ความสามารถเพิ่มเติม:
- **Execution Type Differentiation**: แยกแหล่งที่มาของข้อมูลชัดเจน (`paper_candle_path`, `paper_historical_replay`, `paper_live_track`)
- **Data Integrity**: กรองข้อมูลที่ `model_profile` เป็น `nan` ออกโดยอัตโนมัติ และใช้ Symbol เป็น Fallback Profile
- **Auto-Lookup**: ค้นหาข้อมูลที่ขาดหายไปจากไฟล์ Signal ต้นฉบับ
- **Feature Enrichment**: เพิ่มฟีเจอร์คำนวณระยะ TP/SL, Model Score, และ MFE/MAE จาก Candle จริง

### 10. Frontend Monitor

เพิ่มหน้า Next.js:

```text
http://127.0.0.1:3000/prototype
```

ความสามารถของหน้า:

- กด `Run Prototype`
- refresh status
- ดูจำนวน Forex/Crypto signals
- ดู open paper trades
- ดู closed trades
- ดู paper trade summary
- ดู AI dataset rows
- ดู decision/execution audit logs
- ดู capability list ของ prototype

API ที่เพิ่ม:

- `GET /api/prototype/status`
- `POST /api/prototype/run`

## วิธีรัน

### รัน prototype จาก terminal

```powershell
python .\prototype\run_prototype.py
```

### รันผ่าน PowerShell script

```powershell
.\prototype\scripts\run_once.ps1
```

### รันผ่านหน้าเว็บ

เปิด:

```text
http://127.0.0.1:3000/prototype
```

แล้วกด:

```text
Run Prototype
```

## Flow การทำงาน

```text
Forex/Crypto model logs
-> signal_ingest.py
-> normalized signal CSV
-> paper_trade_tracker.py
-> risk_manager.py
-> open paper trade CSV
-> market feed / latest log fallback
-> TP/SL/time exit
-> closed trade CSV
-> AI dataset export
-> audit logs
-> frontend monitor
```

## Output Files

### Signals

- `prototype/data/signals/forex_signals.csv`
- `prototype/data/signals/crypto_signals.csv`

### Paper Trades

- `prototype/data/paper_trades/open_trades.csv`
- `prototype/data/paper_trades/closed_trades.csv`
- `prototype/data/paper_trades/trade_events.csv`
- `prototype/data/paper_trades/paper_trade_summary.csv`
- `prototype/data/paper_trades/paper_trade_decisions.csv`
- `prototype/data/paper_trades/paper_trade_blocked.csv`

### AI

- `prototype/data/ai/ai_training_dataset.csv` (Training data from closed trades)
- `prototype/data/ai/signal_observation_dataset.csv` (Observation data from all signals)

### Audit

- `prototype/data/audit/decision_audit_log.jsonl`
- `prototype/data/audit/execution_audit_log.jsonl`

## Current Config

ไฟล์:

```text
prototype/config/settings.json
```

ค่าหลัก:

- `research_mode`: true (wide observation / blocked catalog)
- `realistic_paper_mode`: true (open paper trades only with executable live price + guards)
- `signal_cluster_window_minutes`: 240
- `account_balance_usd`: 10000
- `risk_pct_per_trade`: 0.005
- `max_positions_total`: 10
- `max_positions_per_symbol`: 1
- `paper_entry_decisions`: เฉพาะ `BULLISH_TRADE_READY`, `BEARISH_TRADE_READY`
- `cooldown_hours_standard`: 6.0
- `cooldown_hours_gbp`: 12.0
- `breakeven_trigger_pct`: 0.75

หมายเหตุ: open trades เดิมบางรายการอาจมาจากช่วงที่เปิด `research_mode` และให้ WATCH เปิด paper trade ได้ แต่หลัง config ใหม่จะไม่เปิด WATCH เพิ่มแล้ว

## Phase 2–4 (entry quality, lifecycle, dataset)

- **Phase 2** (`entry_quality.py`): net RR+cost filter, regime gate, active thesis cap, portfolio risk cap, weighted SL guard
- **Phase 3** (`trade_lifecycle.py`): cost-aware breakeven, partial close, trailing stop, management events in `trade_events.csv`
- **Phase 4** (`dataset_labels.py`, `ai_dataset.py`): `train_tier`, `should_enter_label`, `max_favorable_r` / `max_adverse_r`, `blocked_trade_dataset.csv`, `blocked_trade_analytics.csv`, `cluster_opportunity_dataset.csv`

## Phase 1: Data integrity

- ทุก signal จบเป็น `opened_trade`, `blocked_trade`, หรือ `observation_only` (ไม่มี silent skip)
- `research_mode`: WATCH / superseded → `observation_only` ใน `paper_trade_blocked.csv`
- `realistic_paper_mode`: เปิด order เฉพาะเมื่อได้ `live_candle_close` (ห้าม `latest_log_fallback` เป็น entry)
- `signal_cluster_id` + `duplicate_cluster` guard กันเปิดซ้ำใน thesis/time window เดียวกัน
- คอลัมน์ `mode`, `signal_cluster_id`, `cluster_status` ใน open trades และ decision log

## Safety Controls ที่เพิ่มแล้ว

- ไม่ส่ง order จริง
- กัน duplicate signal/trade และ duplicate cluster
- ปิด WATCH entry ใน realistic mode
- data leakage guard
- candle path TP/SL simulation
- conservative SL-first เมื่อ candle แตะทั้ง TP และ SL
- market feed fallback
- audit log
- Forex pip-value based sizing
- symbol-level cost assumption

## ข้อจำกัดที่ยังเหลือ

- ยังไม่มี broker/exchange connector
- ยังไม่มี reconciliation กับ broker/exchange จริง
- ยังไม่มี hard SL/TP บน broker/exchange
- ยังไม่มี partial fill จริง
- market feed ใช้ `yfinance` แบบ best-effort ไม่ใช่ feed ระดับ execution
- CSV เหมาะกับ prototype แต่ถ้าข้อมูลมากควรย้ายไป SQLite/PostgreSQL
- AI dataset จะมีประโยชน์หลังมี closed trades จำนวนมากพอ

## Tests

มี test เบื้องต้น:

- risk manager
- forex pip sizing
- data leakage guard

ถ้าไม่มี `pytest` สามารถรันแบบ import ตรงได้:

```powershell
python -c "from prototype.tests.test_risk_manager import test_direction_from_signal,test_calculate_crypto_position,test_calculate_forex_position_uses_pip_value; from prototype.tests.test_data_leakage_guard import test_signal_must_precede_entry_time,test_future_signal_is_blocked,test_missing_signal_time_is_blocked; [fn() for fn in [test_direction_from_signal,test_calculate_crypto_position,test_calculate_forex_position_uses_pip_value,test_signal_must_precede_entry_time,test_future_signal_is_blocked,test_missing_signal_time_is_blocked]]; print('prototype tests passed')"
```

## ขั้นต่อไปที่แนะนำ

1. เพิ่มปุ่มปรับ config จากหน้า `/prototype`
2. เพิ่ม manual close paper trade
3. เพิ่มหน้า export/download AI dataset
4. เพิ่ม reconciliation placeholder table/file
5. เพิ่ม broker/exchange demo connector แบบ `demo_dry_run`
6. เพิ่ม Binance Testnet สำหรับ crypto
7. เพิ่ม MT5 Demo สำหรับ Forex
8. เพิ่ม closed trade analytics เช่น profit factor, drawdown, return by symbol
