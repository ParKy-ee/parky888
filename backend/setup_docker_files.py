import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.join(BASE_DIR, "service")
os.makedirs(SERVICE_DIR, exist_ok=True)
os.makedirs(os.path.join(SERVICE_DIR, "database"), exist_ok=True)
os.makedirs(os.path.join(SERVICE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SERVICE_DIR, "models"), exist_ok=True)

# 1. requirements.txt
req_content = """pandas>=2.0.0
numpy>=1.23.0
lightgbm>=4.0.0
scikit-learn>=1.2.0
joblib>=1.2.0
yfinance>=0.2.35
ta>=0.10.2
requests>=2.31.0
"""
with open(os.path.join(SERVICE_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(req_content)

# 2. Dockerfile
dockerfile_content = """# Base image ขนาดเล็กพิเศษเพื่อประหยัด RAM
FROM python:3.10-slim

# กำหนด Timezone เป็น Asia/Bangkok
ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# ติดตั้งเฉพาะ dependencies พื้นฐาน
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดและโมเดลทั้งหมด
COPY . .

# สั่งรันแบบ unbuffered เพื่อให้ log แสดงผลสดทันที
CMD ["python", "-u", "app.py"]
"""
with open(os.path.join(SERVICE_DIR, "Dockerfile"), "w", encoding="utf-8") as f:
    f.write(dockerfile_content)

# 3. docker-compose.yml
compose_content = """version: '3.8'

services:
  ai_trading_bot:
    build: .
    container_name: ai_swing_trading_service
    restart: always
    deploy:
      resources:
        limits:
          cpus: '0.50'      # จำกัด CPU ไม่เกิน 0.5 core (ไม่แย่งเครื่อง)
          memory: 256M      # จำกัด RAM ไม่เกิน 256 MB (เบามาก)
    environment:
      - SCAN_INTERVAL_MINUTES=5
      - CONFIDENCE_THRESHOLD=0.38
      - TELEGRAM_BOT_TOKEN=
      - TELEGRAM_CHAT_ID=
    volumes:
      - ./database:/app/database
      - ./logs:/app/logs
      - ./models:/app/models
"""
with open(os.path.join(SERVICE_DIR, "docker-compose.yml"), "w", encoding="utf-8") as f:
    f.write(compose_content)

# 4. .env.example
env_content = """# ตั้งค่าความถี่ในการสแกนตลาด (นาที)
SCAN_INTERVAL_MINUTES=5

# ค่าความมั่นใจขั้นต่ำของ AI (0.35 - 0.50)
CONFIDENCE_THRESHOLD=0.38

# (ไม่บังคับ) หากต้องการให้ส่งแจ้งเตือนเข้า Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
with open(os.path.join(SERVICE_DIR, ".env.example"), "w", encoding="utf-8") as f:
    f.write(env_content)

print("[+] สร้างโครงสร้างพื้นฐาน Docker ใน backend/service สำเร็จ!")
