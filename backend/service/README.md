# 🐳 AI Swing Trading Service (Dockerized Background Daemon)

ระบบสแกนและประมวลผลสัญญาณเทรดระยะสั้นด้วย Machine Learning แบบ Background Service กินทรัพยากรต่ำมาก (< 120MB RAM, CPU < 1%)

---

## 🚀 วิธีสั่งรันด้วย Docker (ง่ายที่สุด)

### 1. สั่ง Build และ Start Service ในพื้นหลัง (Background)
```bash
cd backend/service
docker compose up -d --build
```

### 2. ดูสถานะการทำงานสด (Live Logs)
```bash
docker compose logs -f
```

### 3. ตรวจสอบการใช้ RAM และ CPU ของ Container
```bash
docker stats ai_swing_trading_service
```
*(จะเห็นว่ากิน RAM เพียง ~80-120MB และ CPU < 0.1%)*

### 4. สั่งหยุด Service
```bash
docker compose down
```

---

## ⚙️ การตั้งค่าการแจ้งเตือน Telegram (ทางเลือก)
แก้ไขไฟล์ `docker-compose.yml` หรือ `.env`:
```yaml
environment:
  - SCAN_INTERVAL_MINUTES=5       # ความถี่ในการสแกนตลาด (นาที)
  - CONFIDENCE_THRESHOLD=0.38    # ค่าความมั่นใจ AI ขั้นต่ำ
  - TELEGRAM_BOT_TOKEN=123456:ABC-DEF # ใส่ Bot Token
  - TELEGRAM_CHAT_ID=987654321        # ใส่ Chat ID
```

---

## 📁 ฐานข้อมูลที่บันทึก
* ข้อมูลสัญญาณซื้อขายทั้งหมดจะถูกบันทึกลงในไฟล์ SQLite: `backend/service/database/market_data.db` แบบอัตโนมัติ
