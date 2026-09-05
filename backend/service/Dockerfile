# Base image ขนาดเล็กพิเศษเพื่อประหยัด RAM
FROM python:3.10-slim

# กำหนด Timezone เป็น Asia/Bangkok
ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# ติดตั้งเฉพาะ dependencies พื้นฐาน
RUN apt-get update && apt-get install -y --no-install-recommends     libgomp1     curl     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดและโมเดลทั้งหมด
COPY . .

# สั่งรันแบบ unbuffered เพื่อให้ log แสดงผลสดทันที
CMD ["python", "-u", "app.py"]
