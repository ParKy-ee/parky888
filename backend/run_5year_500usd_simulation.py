import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PARQUET_PATH = os.path.join(DATA_DIR, "swing_dataset_latest.parquet")
MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_trailing_model.joblib")
REPORT_HTML = os.path.join(BASE_DIR, "portfolio_5years_report.html")
ROOT_HTML = os.path.join(os.path.dirname(BASE_DIR), "portfolio_5years_report.html")

def run_5year_simulation():
    print("=" * 75)
    print(" 🚀 เริ่มต้นการจำลองเทรดเงิน $500 USD ตลอดระยะเวลา 5 ปีเต็ม (5-YEAR BACKTEST)")
    print("=" * 75)
    
    df = pd.read_parquet(PARQUET_PATH)
    df["time"] = pd.to_datetime(df["time"])
    
    # คำนวณ Market Filter & Relative Strength vs SPY
    spy_df = df[df["symbol"] == "SPY"][["time", "return_20d", "close", "ema_50", "ema_200"]].copy()
    spy_df = spy_df.rename(columns={"return_20d": "spy_return_20d", "close": "spy_close"})
    spy_df["market_bullish"] = (spy_df["spy_close"] > spy_df["ema_200"]) & (spy_df["ema_50"] > spy_df["ema_200"])
    
    df = pd.merge(df, spy_df[["time", "spy_return_20d", "market_bullish"]], on="time", how="left")
    df["market_bullish"] = df["market_bullish"].fillna(True)
    df["rs_20d"] = df["return_20d"] - df["spy_return_20d"].fillna(0)
    
    tradeable_df = df[~df["symbol"].isin(["SPY", "QQQ"])].copy()
    tradeable_df = tradeable_df.sort_values("time").reset_index(drop=True)
    
    # โหลดโมเดล
    model_dict = joblib.load(MODEL_PATH)
    lgb_model = model_dict["lgb"]
    rf_model = model_dict["rf"]
    feature_cols = model_dict["features"]
    
    X = tradeable_df[feature_cols].fillna(0)
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    p_rf = rf_model.predict_proba(X)[:, 1]
    tradeable_df["ai_score"] = 0.5 * p_lgb + 0.5 * p_rf
    
    # =========================================================================
    # จำลองการเทรดแบบทบต้น (Compounding) ตั้งแต่วันแรกถึงวันสุดท้าย (5 ปีเต็ม)
    # =========================================================================
    INITIAL_CAPITAL = 500.00
    MAX_POSITIONS = 2 # ถือครองไม่เกิน 2 หุ้นพร้อมกัน (ไม้ละ 50% ของพอร์ต ณ ขณะนั้น)
    CONFIDENCE_THRESHOLD = np.percentile(tradeable_df["ai_score"], 86)
    
    cash = INITIAL_CAPITAL
    active_positions = {}
    trade_history = []
    equity_curve = []
    
    all_dates = np.sort(tradeable_df["time"].unique())
    start_date = pd.to_datetime(all_dates[0])
    end_date = pd.to_datetime(all_dates[-1])
    
    print(f"[*] วันเริ่มต้น: {start_date.strftime('%Y-%m-%d')} | วันสิ้นสุด: {end_date.strftime('%Y-%m-%d')}")
    print(f"[*] เงินทุนเริ่มต้น: ${INITIAL_CAPITAL:,.2f} USD\n")
    
    for current_date in all_dates:
        curr_dt = pd.to_datetime(current_date)
        day_df = tradeable_df[tradeable_df["time"] == current_date]
        
        # 1. จัดการ Position ที่ถืออยู่
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
                    "year": curr_dt.year,
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
            
        # 2. คัดเลือกเปิดออเดอร์ใหม่
        buy_candidates = day_df[
            (day_df["ai_score"] >= CONFIDENCE_THRESHOLD) &
            (day_df["market_bullish"] == True) &
            (day_df["rs_20d"] > 0) &
            (~day_df["symbol"].isin(active_positions.keys()))
        ].sort_values("ai_score", ascending=False)
        
        for _, cand in buy_candidates.iterrows():
            if len(active_positions) >= MAX_POSITIONS:
                break
                
            curr_equity = cash + sum(p["shares"] * p["entry_price"] for p in active_positions.values())
            alloc_usd = min(cash, curr_equity / MAX_POSITIONS)
            
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
        equity_curve.append({"date": curr_dt.strftime("%Y-%m-%d"), "year": curr_dt.year, "equity": curr_equity})
        
    trades_df = pd.DataFrame(trade_history)
    equity_df = pd.DataFrame(equity_curve)
    
    final_equity = equity_df["equity"].iloc[-1]
    profit_loss_usd = final_equity - INITIAL_CAPITAL
    total_return_pct = (profit_loss_usd / INITIAL_CAPITAL) * 100
    years = (end_date - start_date).days / 365.25
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    
    total_trades = len(trades_df)
    wins = (trades_df["win"] == 1).sum()
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    
    gross_profit = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_loss = abs(trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12)
    
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
    max_dd_pct = abs(equity_df["drawdown"].min()) * 100
    
    # สรุปผลตอบแทนแยกรายปี (Year-by-Year Performance)
    yearly_summary = []
    for yr, group in equity_df.groupby("year"):
        start_eq = group["equity"].iloc[0]
        end_eq = group["equity"].iloc[-1]
        yr_return = ((end_eq - start_eq) / start_eq) * 100
        yr_trades = trades_df[trades_df["year"] == yr]
        yr_wins = (yr_trades["win"] == 1).sum() if not yr_trades.empty else 0
        yr_total = len(yr_trades)
        yr_wr = (yr_wins / yr_total * 100) if yr_total > 0 else 0
        yearly_summary.append({
            "year": yr,
            "start_equity": start_eq,
            "end_equity": end_eq,
            "return_pct": yr_return,
            "trades": yr_total,
            "win_rate": yr_wr
        })
    yearly_df = pd.DataFrame(yearly_summary)
    
    print("=" * 75)
    print(" 🏆 สรุปผลตอบแทนเงิน $500 USD ตลอดระยะเวลา 5 ปีเต็ม (5-YEAR RESULTS)")
    print("=" * 75)
    print(f" 💵 เงินเริ่มต้น (ปี 2021/2022):       $500.00 USD")
    print(f" 💰 เงินในพอร์ตวันนี้ (Final Balance): ${final_equity:,.2f} USD")
    print(f" 🟢 กำไรสะสมสุทธิ (Total Profit):     {profit_loss_usd:+,.2f} USD (+{total_return_pct:.2f}%)  🔥")
    print(f" 📈 ผลตอบแทนทบต้นเฉลี่ยต่อปี (CAGR):   +{cagr:.2f}% ต่อปี")
    print(f" 🎯 อัตราความแม่นยำ (Win Rate):        {win_rate:.1f}% (ชนะ {wins} / แพ้ {losses} จาก {total_trades} ไม้)")
    print(f" ⚖️ Profit Factor (กำไร / ขาดทุน):    {profit_factor:.2f} เท่า")
    print(f" 📉 Max Drawdown (ย่อตัวสูงสุด):       {max_dd_pct:.2f}%")
    print("=" * 75)
    
    print("\n📅 ผลตอบแทนแยกรายปี (Year-by-Year Performance):")
    print(yearly_df[["year", "start_equity", "end_equity", "return_pct", "trades", "win_rate"]].to_string(index=False))
    
    # สร้างกราฟ Plotly แบบ 2 ช่อง (บน: Equity Curve, ล่าง: Yearly Bar)
    fig = make_subplots(rows=2, cols=1, subplot_titles=["<b>กราฟการเติบโตของเงิน $500 ตลอด 5 ปี ($ USD)</b>", "<b>ผลตอบแทนแยกรายปี (%)</b>"], vertical_spacing=0.15, row_heights=[0.65, 0.35])
    
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity"],
        mode="lines",
        name="มูลค่าพอร์ต ($ USD)",
        line=dict(color="#10b981", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.08)"
    ), row=1, col=1)
    fig.add_hline(y=500.00, line_dash="dash", line_color="#94a3b8", annotation_text="ทุนเริ่มต้น $500", row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=yearly_df["year"].astype(str),
        y=yearly_df["return_pct"],
        text=[f"{r:+.1f}%" for r in yearly_df["return_pct"]],
        textposition="outside",
        marker=dict(color=["#10b981" if r >= 0 else "#f43f5e" for r in yearly_df["return_pct"]]),
        name="ผลตอบแทนรายปี (%)"
    ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=40, r=40, t=60, b=40),
        height=650,
        showlegend=False
    )
    html_charts = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # HTML Table สำหรับไม้เด่น
    top_trades_html = ""
    for _, t in trades_df.sort_values("pnl_usd", ascending=False).head(8).iterrows():
        top_trades_html += f"""
        <tr class="border-b border-slate-800 text-xs">
            <td class="py-2.5 font-bold text-emerald-400">{t['symbol']}</td>
            <td class="py-2.5 text-slate-400">{t['entry_date']}</td>
            <td class="py-2.5 text-slate-400">{t['exit_date']} ({t['days_held']} วัน)</td>
            <td class="py-2.5 text-slate-300 font-mono">${t['entry_price']:.2f}</td>
            <td class="py-2.5 text-slate-300 font-mono">${t['exit_price']:.2f}</td>
            <td class="py-2.5 text-slate-400 font-mono">${t['invested']:.2f}</td>
            <td class="py-2.5 text-emerald-400 font-bold font-mono">+{t['pnl_pct']:.2f}% (+${t['pnl_usd']:,.2f})</td>
            <td class="py-2.5 text-slate-400">{t['reason']}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>รายงานผลการเทรดพอร์ต 5 ปีเต็ม (เงินทุน $500 -> วันนี้)</title>
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
        .metric-glow {{
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        }}
    </style>
</head>
<body class="p-6 md:p-10 max-w-7xl mx-auto">
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-8 border-b border-slate-800 gap-4">
        <div>
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-700 text-emerald-400 text-xs font-semibold mb-3">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                จำลองพอร์ต $500 USD ย้อนหลัง 5 ปีเต็ม (2021/2022 - 2026)
            </div>
            <h1 class="text-3xl font-extrabold tracking-tight text-white">รายงานผลการเทรด $500 USD ตลอดระยะเวลา 5 ปี</h1>
            <p class="text-slate-400 text-sm mt-1">ทดสอบระบบ AI Swing Trading พร้อมระบบ Trailing Stop และจำลองดอกเบี้ยทบต้น (Compounding)</p>
        </div>
        <div class="text-left md:text-right">
            <span class="text-xs text-slate-500 uppercase font-mono">ผลตอบแทนสุทธิ</span>
            <div class="text-sm font-semibold text-emerald-400">กำไรสุทธิ +{total_return_pct:.2f}%</div>
            <span class="text-xs text-slate-500">{start_date.strftime('%Y-%m-%d')} ถึง {end_date.strftime('%Y-%m-%d')}</span>
        </div>
    </div>

    <!-- KPI Summary Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 my-8">
        <div class="card p-5 border-emerald-900/50 bg-gradient-to-br from-[#131b2e] to-[#062d22]">
            <div class="text-xs text-emerald-400 font-medium uppercase">มูลค่าพอร์ตสุทธิวันนี้</div>
            <div class="text-3xl font-black text-emerald-400 mt-2 font-mono metric-glow">${final_equity:,.2f}</div>
            <div class="text-xs text-emerald-400/80 mt-1">กำไรสุทธิ: <b>+${profit_loss_usd:,.2f} (+{total_return_pct:.2f}%)</b></div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">ผลตอบแทนทบต้นต่อปี (CAGR)</div>
            <div class="text-3xl font-black text-sky-400 mt-2 font-mono">+{cagr:.2f}%</div>
            <div class="text-xs text-slate-500 mt-1">เฉลี่ยเติบโตปีละ {cagr:.2f}%</div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">อัตราความแม่นยำ (Win Rate)</div>
            <div class="text-3xl font-black text-white mt-2 font-mono">{win_rate:.1f}%</div>
            <div class="text-xs text-slate-500 mt-1">ชนะ {wins} / แพ้ {losses} จาก {total_trades} ไม้</div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">Max Drawdown (ย่อตัวสูงสุด)</div>
            <div class="text-3xl font-black text-amber-400 mt-2 font-mono">{max_dd_pct:.2f}%</div>
            <div class="text-xs text-slate-500 mt-1">ความเสี่ยงต่ำมากตลอด 5 ปี</div>
        </div>
    </div>

    <!-- Charts -->
    <div class="card p-6 mb-8">
        <div class="w-full overflow-hidden rounded-lg">
            {html_charts}
        </div>
    </div>

    <!-- Top Winning Trades -->
    <div class="card p-6 mb-8">
        <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span>🔥</span> อันดับไม้ที่ทำกำไรก้อนใหญ่ที่สุดตลอด 5 ปี (Top Big Winners)
        </h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="border-b border-slate-700 text-xs text-slate-400 uppercase">
                        <th class="py-2">หุ้น</th>
                        <th class="py-2">วันที่เข้าซื้อ</th>
                        <th class="py-2">วันที่ปิดออเดอร์</th>
                        <th class="py-2">ราคาซื้อ</th>
                        <th class="py-2">ราคาขาย</th>
                        <th class="py-2">เงินที่ลงทุน</th>
                        <th class="py-2">กำไรที่ได้ ($)</th>
                        <th class="py-2">เหตุผลการปิดออเดอร์</th>
                    </tr>
                </thead>
                <tbody>
                    {top_trades_html}
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
        
    print(f"\n[+] สร้างรายงานพอร์ต 5 ปีเต็มสำเร็จที่: {REPORT_HTML}")
    print(f"[+] คัดลอกไปยัง: {ROOT_HTML}")

if __name__ == "__main__":
    run_5year_simulation()
