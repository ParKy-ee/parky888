import os
import sys
import json
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
MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_trailing_model.joblib")
REPORT_HTML = os.path.join(BASE_DIR, "portfolio_500usd_report.html")
ROOT_HTML = os.path.join(os.path.dirname(BASE_DIR), "portfolio_500usd_report.html")

def run_500usd_trader():
    print("=" * 70)
    print(" 🚀 เริ่มต้นการจำลองเทรดจริงด้วยเงินทุน $500 USD (AI SWING TRADER)")
    print("=" * 70)
    
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError("ไม่พบไฟล์ชุดข้อมูล")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("ไม่พบไฟล์โมเดล AI")
        
    df = pd.read_parquet(PARQUET_PATH)
    df["time"] = pd.to_datetime(df["time"])
    
    # 1. คำนวณ Market Filter และ Relative Strength vs SPY
    spy_df = df[df["symbol"] == "SPY"][["time", "return_20d", "close", "ema_50", "ema_200"]].copy()
    spy_df = spy_df.rename(columns={"return_20d": "spy_return_20d", "close": "spy_close"})
    spy_df["market_bullish"] = (spy_df["spy_close"] > spy_df["ema_200"]) & (spy_df["ema_50"] > spy_df["ema_200"])
    
    df = pd.merge(df, spy_df[["time", "spy_return_20d", "market_bullish"]], on="time", how="left")
    df["market_bullish"] = df["market_bullish"].fillna(True)
    df["rs_20d"] = df["return_20d"] - df["spy_return_20d"].fillna(0)
    
    tradeable_df = df[~df["symbol"].isin(["SPY", "QQQ"])].copy()
    tradeable_df = tradeable_df.sort_values("time").reset_index(drop=True)
    
    # 2. โหลดโมเดล AI และคำนวณคะแนนทำนาย
    model_dict = joblib.load(MODEL_PATH)
    lgb_model = model_dict["lgb"]
    rf_model = model_dict["rf"]
    feature_cols = model_dict["features"]
    
    X = tradeable_df[feature_cols].fillna(0)
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    p_rf = rf_model.predict_proba(X)[:, 1]
    tradeable_df["ai_score"] = 0.5 * p_lgb + 0.5 * p_rf
    
    # 3. เลือกช่วงเวลาทดสอบเสมือนจริง (Out-of-Sample Period: 1 ปีล่าสุด)
    unique_dates = np.sort(tradeable_df["time"].unique())
    test_start_date = unique_dates[int(len(unique_dates) * 0.80)] # 20% ล่าสุด
    test_df = tradeable_df[tradeable_df["time"] >= test_start_date].copy()
    
    print(f"[*] ช่วงเวลาทดสอบจำลองพอร์ต: {pd.to_datetime(test_df['time'].min()).strftime('%Y-%m-%d')} ถึง {pd.to_datetime(test_df['time'].max()).strftime('%Y-%m-%d')}")
    print(f"[*] เงินทุนเริ่มต้น (Initial Capital): $500.00 USD\n")
    
    # =========================================================================
    # การจำลองการบริหารพอร์ตเงิน $500 USD (Fractional Shares 1-2 Positions)
    # =========================================================================
    INITIAL_CAPITAL = 500.00
    MAX_SIMULTANEOUS_POSITIONS = 2 # ถือครองพร้อมกันไม่เกิน 2 หุ้น (ไม้ละ 50% หรือ ~$250)
    CONFIDENCE_THRESHOLD = np.percentile(tradeable_df["ai_score"], 86) # Top 14% คุณภาพสูง
    
    cash = INITIAL_CAPITAL
    active_positions = {}
    trade_history = []
    equity_curve = []
    
    test_dates = np.sort(test_df["time"].unique())
    
    for current_date in test_dates:
        curr_dt = pd.to_datetime(current_date)
        day_df = test_df[test_df["time"] == current_date]
        
        # ก. ตรวจสอบออเดอร์ที่ถืออยู่ (Trailing Stop & Breakeven)
        closed_symbols = []
        for symbol, pos in active_positions.items():
            sym_row = day_df[day_df["symbol"] == symbol]
            if sym_row.empty:
                continue
                
            high_price = sym_row["high"].iloc[0]
            low_price = sym_row["low"].iloc[0]
            close_price = sym_row["close"].iloc[0]
            atr = sym_row["atr_14"].iloc[0]
            pos["days_held"] += 1
            
            if high_price > pos["highest_price"]:
                pos["highest_price"] = high_price
                
            gain = pos["highest_price"] - pos["entry_price"]
            
            # ล็อกเท่าทุนเมื่อกำไรเกิน 1.5x ATR
            if gain >= 1.5 * atr and pos["sl_price"] < pos["entry_price"]:
                pos["sl_price"] = pos["entry_price"] * 1.002
                pos["status_note"] = "BREAKEVEN_LOCKED"
                
            # Trailing Stop ตามหลัง 1.4x ATR
            if gain >= 2.5 * atr:
                new_trail = pos["highest_price"] - (1.4 * atr)
                if new_trail > pos["sl_price"]:
                    pos["sl_price"] = new_trail
                    pos["status_note"] = "TRAILING_STOP"
                    
            hit_tp = high_price >= pos["tp_price"]
            hit_sl = low_price <= pos["sl_price"]
            timeout = pos["days_held"] >= 20
            
            if hit_tp or hit_sl or timeout:
                if hit_tp:
                    exit_price = pos["tp_price"]
                    reason = "TAKE_PROFIT (+12% ถึง +20%)"
                elif hit_sl:
                    exit_price = pos["sl_price"]
                    reason = f"STOP ({pos['status_note']})"
                else:
                    exit_price = close_price
                    reason = "TIMEOUT (ครบ 20 วัน)"
                    
                pnl_usd = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
                cash += (pos["shares"] * exit_price)
                
                trade_history.append({
                    "symbol": symbol,
                    "entry_date": pd.to_datetime(pos["entry_date"]).strftime("%Y-%m-%d"),
                    "exit_date": curr_dt.strftime("%Y-%m-%d"),
                    "days_held": pos["days_held"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "shares": pos["shares"],
                    "invested": pos["shares"] * pos["entry_price"],
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "win": 1 if pnl_usd > 0 else 0
                })
                closed_symbols.append(symbol)
                
        for s in closed_symbols:
            del active_positions[s]
            
        # ข. คัดเลือกเปิดออเดอร์ใหม่
        buy_candidates = day_df[
            (day_df["ai_score"] >= CONFIDENCE_THRESHOLD) &
            (day_df["market_bullish"] == True) &
            (day_df["rs_20d"] > 0) &
            (~day_df["symbol"].isin(active_positions.keys()))
        ].sort_values("ai_score", ascending=False)
        
        for _, cand in buy_candidates.iterrows():
            if len(active_positions) >= MAX_SIMULTANEOUS_POSITIONS:
                break
                
            curr_equity = cash + sum(p["shares"] * p["entry_price"] for p in active_positions.values())
            alloc_usd = min(cash, curr_equity / MAX_SIMULTANEOUS_POSITIONS)
            
            if alloc_usd < 50.0:
                continue
                
            entry_price = cand["close"]
            atr = cand["atr_14"]
            shares = alloc_usd / entry_price
            cash -= alloc_usd
            
            active_positions[cand["symbol"]] = {
                "entry_date": current_date,
                "entry_price": entry_price,
                "highest_price": entry_price,
                "shares": shares,
                "tp_price": entry_price + (4.0 * atr),
                "sl_price": entry_price - (1.5 * atr),
                "status_note": "INITIAL_SL",
                "days_held": 0
            }
            
        curr_equity = cash + sum(p["shares"] * p["entry_price"] for p in active_positions.values())
        equity_curve.append({"date": curr_dt.strftime("%Y-%m-%d"), "equity": curr_equity})
        
    trades_df = pd.DataFrame(trade_history)
    equity_df = pd.DataFrame(equity_curve)
    
    final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else INITIAL_CAPITAL
    profit_loss_usd = final_equity - INITIAL_CAPITAL
    total_return_pct = (profit_loss_usd / INITIAL_CAPITAL) * 100
    
    total_trades = len(trades_df)
    wins = (trades_df["win"] == 1).sum() if total_trades > 0 else 0
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum() if total_trades > 0 else 0
    gross_loss = abs(trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum()) if total_trades > 0 else 1e-12
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    print("=" * 70)
    print(" 📊 สรุปผลการทดสอบพอร์ตจริงเงิน $500 USD")
    print("=" * 70)
    print(f" 💵 เงินเริ่มต้น:             $500.00 USD")
    print(f" 💰 เงินสุดท้ายในพอร์ต:        ${final_equity:,.2f} USD")
    print(f" {'🟢 กำไรสุทธิ:' if profit_loss_usd >= 0 else '🔴 ขาดทุนสุทธิ:'}        {profit_loss_usd:+,.2f} USD ({total_return_pct:+.2f}%)")
    print(f" 🎯 อัตราความแม่นยำ (Win Rate): {win_rate:.1f}% (ชนะ {wins} ไม้ / แพ้ {losses} ไม้ จากทั้งหมด {total_trades} ไม้)")
    print(f" ⚖️ Profit Factor:            {profit_factor:.2f} เท่า")
    print("=" * 70)
    
    if total_trades > 0:
        print("\n📋 รายละเอียดการเข้าเทรดทุกไม้:")
        print(trades_df[["entry_date", "exit_date", "symbol", "invested", "pnl_usd", "pnl_pct", "reason"]].to_string(index=False))
        
    # สร้างกราฟ HTML Report
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity"],
        mode="lines+markers",
        name="มูลค่าพอร์ต ($ USD)",
        line=dict(color="#10b981" if profit_loss_usd >= 0 else "#f43f5e", width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.08)" if profit_loss_usd >= 0 else "rgba(244, 63, 94, 0.08)"
    ))
    fig_equity.add_hline(y=500.00, line_dash="dash", line_color="#94a3b8", annotation_text="เงินทุนเริ่มต้น $500")
    fig_equity.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        title="<b>กราฟการเติบโตของพอร์ต $500 USD (Equity Curve)</b>",
        xaxis=dict(title="วันที่", gridcolor="#1e293b"),
        yaxis=dict(title="มูลค่าพอร์ต ($ USD)", gridcolor="#1e293b"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    html_chart = fig_equity.to_html(full_html=False, include_plotlyjs='cdn')
    
    trade_rows_html = ""
    for _, t in trades_df.iterrows():
        is_win = t["win"] == 1
        badge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-700">กำไร (WIN)</span>' if is_win else '<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-400 border border-rose-700">ขาดทุน (LOSS)</span>'
        color = "text-emerald-400" if is_win else "text-rose-400"
        trade_rows_html += f"""
        <tr class="border-b border-slate-800 text-xs">
            <td class="py-2.5 font-bold text-white">{t['symbol']}</td>
            <td class="py-2.5 text-slate-400">{t['entry_date']}</td>
            <td class="py-2.5 text-slate-400">{t['exit_date']} ({t['days_held']} วัน)</td>
            <td class="py-2.5 text-slate-300 font-mono">${t['entry_price']:.2f}</td>
            <td class="py-2.5 text-slate-300 font-mono">${t['exit_price']:.2f}</td>
            <td class="py-2.5 text-slate-400 font-mono">${t['invested']:.2f}</td>
            <td class="py-2.5 {color} font-bold font-mono">{t['pnl_pct']:+.2f}% (${t['pnl_usd']:+,.2f})</td>
            <td class="py-2.5 text-slate-400">{t['reason']}</td>
            <td class="py-2.5">{badge}</td>
        </tr>
        """
        
    status_text = "พอร์ตมีกำไรสุทธิ" if profit_loss_usd >= 0 else "พอร์ตขาดทุนสุทธิ"
    status_color = "text-emerald-400" if profit_loss_usd >= 0 else "text-rose-400"

    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>รายงานผลการเทรดพอร์ตจริง $500 USD (AI Swing Trader)</title>
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
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-8 border-b border-slate-800 gap-4">
        <div>
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-700 text-emerald-400 text-xs font-semibold mb-3">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                จำลองพอร์ต $500 USD: ทดสอบบนกราฟจริงย้อนหลัง
            </div>
            <h1 class="text-3xl font-extrabold tracking-tight text-white">รายงานผลการเทรดพอร์ต $500 USD (AI Swing Trader)</h1>
            <p class="text-slate-400 text-sm mt-1">ทดสอบการเข้าซื้อขายจริงตามสัญญาณ AI ด้วยเงินตั้งต้น $500 พร้อมระบบ Trailing Stop</p>
        </div>
        <div class="text-left md:text-right">
            <span class="text-xs text-slate-500 uppercase font-mono">สถานะพอร์ต</span>
            <div class="text-sm font-semibold {status_color}">{status_text}</div>
            <span class="text-xs text-slate-500">เงินทุนเริ่มต้น $500 USD</span>
        </div>
    </div>

    <!-- KPI Summary Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 my-8">
        <div class="card p-5 border-emerald-900/50 bg-gradient-to-br from-[#131b2e] to-[#062d22]">
            <div class="text-xs text-emerald-400 font-medium uppercase">มูลค่าพอร์ตสุทธิสุดท้าย</div>
            <div class="text-3xl font-black {status_color} mt-2 font-mono">${final_equity:,.2f}</div>
            <div class="text-xs text-slate-400 mt-1">กำไร/ขาดทุนสุทธิ: <b class="{status_color}">${profit_loss_usd:+,.2f} ({total_return_pct:+.2f}%)</b></div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">อัตราความแม่นยำ (Win Rate)</div>
            <div class="text-3xl font-black text-white mt-2 font-mono">{win_rate:.1f}%</div>
            <div class="text-xs text-slate-500 mt-1">ชนะ {wins} ไม้ / แพ้ {losses} ไม้</div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">Profit Factor</div>
            <div class="text-3xl font-black text-sky-400 mt-2 font-mono">{profit_factor:.2f} เท่า</div>
            <div class="text-xs text-slate-500 mt-1">อัตราส่วนยอดกำไร / ขาดทุน</div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">จำนวนไม้ที่เทรดจริง</div>
            <div class="text-3xl font-black text-amber-400 mt-2 font-mono">{total_trades} ไม้</div>
            <div class="text-xs text-slate-500 mt-1">ถือครองเฉลี่ย {trades_df['days_held'].mean():.1f} วันต่อไม้</div>
        </div>
    </div>

    <!-- Equity Curve -->
    <div class="card p-6 mb-8">
        <div class="mb-4">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="p-1.5 rounded-lg bg-emerald-950 border border-emerald-700 text-emerald-400 text-sm">📈</span>
                กราฟเส้นมูลค่าเงินทุน $500 USD เติบโตจริงตามเวลา
            </h2>
            <p class="text-slate-300 text-sm mt-1">
                แสดงการเพิ่มขึ้น/ลดลงของเงินทุน $500 ในแต่ละรอบการเข้าเทรดจริง
            </p>
        </div>
        <div class="w-full overflow-hidden rounded-lg">
            {html_chart}
        </div>
    </div>

    <!-- Trade History Table -->
    <div class="card p-6 mb-8">
        <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span>📋</span> ประวัติการเข้าซื้อขายทุกออเดอร์ของเงิน $500
        </h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="border-b border-slate-700 text-xs text-slate-400 uppercase">
                        <th class="py-2">ชื่อหุ้น</th>
                        <th class="py-2">วันที่เข้าซื้อ</th>
                        <th class="py-2">วันที่ปิดออเดอร์</th>
                        <th class="py-2">ราคาซื้อ ($)</th>
                        <th class="py-2">ราคาขาย ($)</th>
                        <th class="py-2">เงินที่ลง ($)</th>
                        <th class="py-2">ผลตอบแทน</th>
                        <th class="py-2">เหตุผลการปิดออเดอร์</th>
                        <th class="py-2">ผลลัพธ์</th>
                    </tr>
                </thead>
                <tbody>
                    {trade_rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(ROOT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n[+] สร้างรายงานพอร์ต $500 USD สำเร็จที่: {REPORT_HTML}")
    print(f"[+] คัดลอกไปยัง: {ROOT_HTML}")

if __name__ == "__main__":
    run_500usd_trader()
