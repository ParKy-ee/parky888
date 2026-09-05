import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.join(BASE_DIR, "service")

# 1. Update requirements.txt
req_content = """pandas>=2.0.0
numpy>=1.23.0
lightgbm>=4.0.0
scikit-learn>=1.2.0
joblib>=1.2.0
yfinance>=0.2.35
ta>=0.10.2
requests>=2.31.0
pymysql>=1.1.0
cryptography>=41.0.0
sqlalchemy>=2.0.0
"""
with open(os.path.join(SERVICE_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(req_content)

# 2. Update .env and .env.example
env_content = """# ==========================================
# AI SWING TRADING SERVICE CONFIGURATION
# ==========================================

# ฐานข้อมูล MySQL Database Configuration
DB_TYPE=mysql
MYSQL_HOST=mysql_db
MYSQL_PORT=3306
MYSQL_DATABASE=ai_trading_db
MYSQL_USER=trading_user
MYSQL_PASSWORD=trading_pass_123
MYSQL_ROOT_PASSWORD=root_secure_password_123

# การสแกนตลาดและวิเคราะห์ AI
SCAN_INTERVAL_MINUTES=5
CONFIDENCE_THRESHOLD=0.38

# แจ้งเตือนผ่าน Telegram (ใส่ Token และ Chat ID ถ้ามี)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
with open(os.path.join(SERVICE_DIR, ".env"), "w", encoding="utf-8") as f:
    f.write(env_content)
with open(os.path.join(SERVICE_DIR, ".env.example"), "w", encoding="utf-8") as f:
    f.write(env_content)

# 3. Update docker-compose.yml with MySQL Container
compose_content = """version: '3.8'

services:
  mysql_db:
    image: mysql:8.0
    container_name: ai_trading_mysql
    restart: always
    environment:
      MYSQL_DATABASE: ${MYSQL_DATABASE:-ai_trading_db}
      MYSQL_USER: ${MYSQL_USER:-trading_user}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD:-trading_pass_123}
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:-root_secure_password_123}
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    deploy:
      resources:
        limits:
          memory: 350M
    healthcheck:
      test: ["CMD", "mysqladmin" ,"ping", "-h", "localhost", "-u", "root", "-p${MYSQL_ROOT_PASSWORD:-root_secure_password_123}"]
      interval: 10s
      timeout: 5s
      retries: 5

  ai_trading_bot:
    build: .
    container_name: ai_swing_trading_service
    restart: always
    depends_on:
      mysql_db:
        condition: service_healthy
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 256M

volumes:
  mysql_data:
"""
with open(os.path.join(SERVICE_DIR, "docker-compose.yml"), "w", encoding="utf-8") as f:
    f.write(compose_content)

print("[+] อัปเดต requirements, .env และ docker-compose.yml สำหรับ MySQL สำเร็จ!")
