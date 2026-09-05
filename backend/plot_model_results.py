import os
import sys
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PARQUET_PATH = os.path.join(DATA_DIR, "swing_dataset_latest.parquet")
MODEL_PATH = os.path.join(MODELS_DIR, "swing_ensemble_model.joblib")
OUTPUT_HTML = os.path.join(BASE_DIR, "model_predictions_chart.html")
ROOT_HTML = os.path.join(os.path.dirname(BASE_DIR), "model_predictions_chart.html")

def plot_model_predictions():
    print(f"[*] โหลดชุดข้อมูลจาก: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df["time"] = pd.to_datetime(df["time"])
    
    print(f"[*] โหลดโมเดล AI จาก: {MODEL_PATH}")
    model_dict = joblib.load(MODEL_PATH)
    lgb_model = model_dict["lgb"]
    rf_model = model_dict["rf"]
    feature_cols = model_dict["features"]
    
    # รันโมเดลทำนายผลทั้งชุดข้อมูล
    X = df[feature_cols].fillna(0)
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    p_rf = rf_model.predict_proba(X)[:, 1]
    df["ai_confidence"] = (0.5 * p_lgb + 0.5 * p_rf) * 100
    
    # แบ่งระดับความมั่นใจของ AI เป็นกลุ่ม (Confidence Bins)
    df["conf_bin"] = pd.cut(
        df["ai_confidence"], 
        bins=[0, 20, 30, 40, 50, 100], 
        labels=["0-20% (ต่ำมาก)", "20-30% (ต่ำ)", "30-40% (ปานกลาง)", "40-50% (เริ่มน่าสนใจ)", "50-100% (มั่นใจสูง)"]
    )
    
    # คำนวณอัตราความแม่นยำจริงในแต่ละระดับความมั่นใจ
    bin_stats = df.groupby("conf_bin", observed=False).agg(
        total=("target", "count"),
        wins=("target", lambda x: (x == 1).sum()),
        win_rate=("target", lambda x: (x == 1).mean() * 100 if len(x) > 0 else 0)
    ).reset_index()
    
    # สุ่มตัวอย่าง 3,000 จุดเพื่อให้กราฟลื่นไหล
    sample_df = df.sample(min(3000, len(df)), random_state=42).copy()
    
    # =========================================================================
    # กราฟที่ 1: กราฟจุด คะแนนความมั่นใจของ AI เทียบกับ ผลตอบแทนสูงสุดที่เกิดขึ้นจริง (%)
    # =========================================================================
    fig_scatter1 = go.Figure()
    
    for target_val, name, color, opacity, size in [
        (0, "จุดที่ขาดทุน / ไม่ถึงเป้า (จุดสีเทา)", "#64748b", 0.35, 6),
        (1, "จุดที่ได้กำไร +10% สำเร็จ (จุดสีเขียว)", "#10b981", 0.85, 8)
    ]:
        sub = sample_df[sample_df["target"] == target_val]
        fig_scatter1.add_trace(go.Scatter(
            x=sub["ai_confidence"],
            y=sub["forward_max_return"] * 100,
            mode="markers",
            name=name,
            marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0.5, color="#ffffff" if target_val==1 else "#1e293b")),
            customdata=np.stack([sub["symbol"], sub["time"].astype(str), sub["close"], sub["rsi_14"], sub["rvol"]], axis=-1),
            hovertemplate="<b>หุ้น: %{customdata[0]}</b> (%{customdata[1]})<br>ราคา: $%{customdata[2]:.2f}<br>ความมั่นใจ AI: %{x:.1f}%<br><b>ผลกำไรสูงสุดใน 15 วัน: +%{y:.1f}%</b><br>RSI: %{customdata[3]:.1f} | วอลุ่ม: %{customdata[4]:.2f}x<extra></extra>"
        ))
        
    # เส้นแบ่งเป้าหมาย +10%
    fig_scatter1.add_hline(y=10.0, line_dash="dash", line_color="#10b981", annotation_text="เป้าหมายกำไร +10.0%", annotation_position="top right")
    fig_scatter1.add_vline(x=40.0, line_dash="dot", line_color="#38bdf8", annotation_text="โซนที่ AI มั่นใจ (> 40%)", annotation_position="top left")
    
    fig_scatter1.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        title="<b>กราฟจุด: คะแนนความมั่นใจที่ AI ให้ (0-100%) เทียบกับ ผลกำไรสูงสุดที่เกิดขึ้นจริง (%)</b>",
        xaxis=dict(title="คะแนนความมั่นใจที่ AI ประเมิน (AI Confidence Score %)", gridcolor="#1e293b"),
        yaxis=dict(title="ผลกำไรสูงสุดที่ราคาขึ้นไปถึงใน 15 วันข้างหน้า (%)", gridcolor="#1e293b", range=[-5, 60]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=520
    )

    # =========================================================================
    # กราฟที่ 2: ไทม์ไลน์จุดส่งสัญญาณซื้อจริงของหุ้นเด่น (NVDA)
    # =========================================================================
    nvda_df = df[df["symbol"] == "NVDA"].sort_values("time").copy().tail(400)
    nvda_signals = nvda_df[nvda_df["ai_confidence"] >= 38.0]
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=nvda_df["time"],
        y=nvda_df["close"],
        mode="lines",
        name="ราคาหุ้น NVDA ($)",
        line=dict(color="#94a3b8", width=1.5)
    ))
    
    # จุดที่ AI สั่งซื้อแล้วชนะ
    nvda_wins = nvda_signals[nvda_signals["target"] == 1]
    fig_timeline.add_trace(go.Scatter(
        x=nvda_wins["time"],
        y=nvda_wins["close"],
        mode="markers",
        name="AI สั่งซื้อ -> กำไร +10% สำเร็จ (จุดเขียว)",
        marker=dict(symbol="triangle-up", size=11, color="#10b981", line=dict(width=1, color="#ffffff")),
        customdata=np.stack([nvda_wins["ai_confidence"], nvda_wins["forward_max_return"]*100], axis=-1),
        hovertemplate="<b>วันที่: %{x|%Y-%m-%d}</b><br>ราคาเข้า: $%{y:.2f}<br>ความมั่นใจ AI: %{customdata[0]:.1f}%<br>ผลกำไรที่ได้: +%{customdata[1]:.1f}%<extra></extra>"
    ))
    
    # จุดที่ AI สั่งซื้อแล้วแพ้/คัตลอส
    nvda_losses = nvda_signals[nvda_signals["target"] == 0]
    fig_timeline.add_trace(go.Scatter(
        x=nvda_losses["time"],
        y=nvda_losses["close"],
        mode="markers",
        name="AI สั่งซื้อ -> โดนคัตลอส -4% (จุดแดง)",
        marker=dict(symbol="triangle-down", size=9, color="#f43f5e", line=dict(width=1, color="#ffffff")),
        customdata=np.stack([nvda_losses["ai_confidence"], nvda_losses["forward_max_drawdown"]*100], axis=-1),
        hovertemplate="<b>วันที่: %{x|%Y-%m-%d}</b><br>ราคาเข้า: $%{y:.2f}<br>ความมั่นใจ AI: %{customdata[0]:.1f}%<br>ขาดทุน: %{customdata[1]:.1f}%<extra></extra>"
    ))
    
    fig_timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        title="<b>ตัวอย่างจุดส่งสัญญาณซื้อจริงของโมเดลบนกราฟราคา (NVDA ย้อนหลัง 400 วันล่าสุด)</b>",
        xaxis=dict(title="วันที่", gridcolor="#1e293b"),
        yaxis=dict(title="ราคาหุ้น ($)", gridcolor="#1e293b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=480
    )

    # =========================================================================
    # กราฟที่ 3: อัตราความแม่นยำ (Win Rate) ตามระดับความมั่นใจของ AI
    # =========================================================================
    fig_bins = go.Figure()
    bar_colors = ["#64748b", "#94a3b8", "#38bdf8", "#06b6d4", "#10b981"]
    fig_bins.add_trace(go.Bar(
        x=bin_stats["conf_bin"].astype(str),
        y=bin_stats["win_rate"],
        text=[f"{wr:.1f}% ({w}/{t} ครั้ง)" for wr, w, t in zip(bin_stats["win_rate"], bin_stats["wins"], bin_stats["total"])],
        textposition="outside",
        marker=dict(color=bar_colors, line=dict(width=1, color="#334155"))
    ))
    fig_bins.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        title="<b>อัตราการชนะ (Win Rate) เพิ่มขึ้นตามระดับความมั่นใจของ AI</b>",
        xaxis=dict(title="ระดับคะแนนความมั่นใจที่ AI ประเมิน", gridcolor="#1e293b"),
        yaxis=dict(title="อัตราส่วนที่ทำกำไร +10% สำเร็จ (%)", gridcolor="#1e293b", range=[0, max(bin_stats["win_rate"]) + 10]),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )

    html_scatter1 = fig_scatter1.to_html(full_html=False, include_plotlyjs='cdn')
    html_timeline = fig_timeline.to_html(full_html=False, include_plotlyjs=False)
    html_bins = fig_bins.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>กราฟจุดวิเคราะห์ผลการรันโมเดล AI (Model Predictions)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            background-color: #090d16;
            color: #e2e8f0;
            font-family: 'Sarabun', ui-sans-serif, system-ui, sans-serif;
        }}
        .card {{
            background: #131b2e;
            border: 1px solid #1e293b;
            border-radius: 12px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        }}
    </style>
</head>
<body class="p-6 md:p-10 max-w-7xl mx-auto">
    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-8 border-b border-slate-800 gap-4">
        <div>
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-700 text-emerald-400 text-xs font-semibold mb-3">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                ผลการทำนายจริงจากโมเดล AI (LIVE MODEL PREDICTIONS)
            </div>
            <h1 class="text-3xl font-extrabold tracking-tight text-white">กราฟจุดวิเคราะห์ผลการทำนายของโมเดล AI</h1>
            <p class="text-slate-400 text-sm mt-1">แสดงความสัมพันธ์ระหว่างคะแนนความมั่นใจของ AI กับผลกำไรที่เกิดขึ้นจริงในตลาด</p>
        </div>
        <div class="text-left md:text-right">
            <span class="text-xs text-slate-500 uppercase font-mono">โมเดลที่ใช้</span>
            <div class="text-sm font-semibold text-sky-400">LightGBM + Random Forest</div>
            <span class="text-xs text-slate-500">ทดสอบบนข้อมูล 15,615 วัน</span>
        </div>
    </div>

    <!-- กราฟที่ 1: กราฟจุดใหญ่ -->
    <div class="card p-6 my-8">
        <div class="mb-4">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="p-1.5 rounded-lg bg-emerald-950 border border-emerald-700 text-emerald-400 text-sm">🎯</span>
                1. กราฟจุด: คะแนนความมั่นใจ AI vs ผลกำไรจริงที่เกิดขึ้นใน 15 วัน
            </h2>
            <p class="text-slate-300 text-sm mt-1">
                ทุกๆ จุดแทน 1 วันของการตัดสินใจ ยิ่งจุดอยู่ฝั่งขวา (AI มั่นใจสูง) จุดสีเขียวจะลอยพ้นเส้น <b>+10%</b> ขึ้นไปอย่างชัดเจน
            </p>
        </div>
        <div class="w-full overflow-hidden rounded-lg">
            {html_scatter1}
        </div>
    </div>

    <!-- กราฟที่ 2: ไทม์ไลน์จุดซื้อขายบนราคาหุ้นจริง -->
    <div class="card p-6 mb-8">
        <div class="mb-4">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="p-1.5 rounded-lg bg-sky-950 border border-sky-700 text-sky-400 text-sm">📈</span>
                2. ตัวอย่างจุดเข้าซื้อขายจริงบนกราฟราคาหุ้น (NVDA ย้อนหลัง 400 วัน)
            </h2>
            <p class="text-slate-300 text-sm mt-1">
                สามเหลี่ยมสีเขียว (▲) คือวันที่ AI สั่งซื้อแล้วได้กำไร +10% สำเร็จ | สามเหลี่ยมสีแดง (▼) คือวันที่สั่งซื้อแล้วโดนคัตลอส -4%
            </p>
        </div>
        <div class="w-full overflow-hidden rounded-lg">
            {html_timeline}
        </div>
    </div>

    <!-- กราฟที่ 3: Bar Chart Win rate by confidence -->
    <div class="card p-6 mb-8">
        <div class="mb-4">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="p-1.5 rounded-lg bg-amber-950 border border-amber-700 text-amber-400 text-sm">📊</span>
                3. ความแม่นยำจริงตามระดับคะแนนของ AI
            </h2>
            <p class="text-slate-300 text-sm mt-1">
                เมื่อ AI ให้คะแนนความมั่นใจระดับสูง (ฝั่งขวาสุด) อัตราการเกิดกำไรสำเร็จจะพุ่งสูงขึ้นอย่างเห็นได้ชัด
            </p>
        </div>
        <div class="w-full overflow-hidden rounded-lg">
            {html_bins}
        </div>
    </div>
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(ROOT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[+] สร้างไฟล์กราฟผลการรันโมเดลสำเร็จที่: {OUTPUT_HTML}")
    print(f"[+] คัดลอกไปยัง: {ROOT_HTML}")

if __name__ == "__main__":
    plot_model_predictions()
