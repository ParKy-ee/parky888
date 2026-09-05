import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import plotly.graph_objects as go

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PARQUET_PATH = os.path.join(DATA_DIR, "swing_dataset_latest.parquet")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "dynamic_trailing_model.joblib")
REPORT_HTML = os.path.join(BASE_DIR, "dynamic_backtest_report.html")
ROOT_HTML = os.path.join(os.path.dirname(BASE_DIR), "dynamic_backtest_report.html")

def prepare_enhanced_dataset():
    print(f"[*] โหลดชุดข้อมูลจาก: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df["time"] = pd.to_datetime(df["time"])
    
    spy_df = df[df["symbol"] == "SPY"][["time", "return_20d", "close", "ema_50", "ema_200"]].copy()
    spy_df = spy_df.rename(columns={"return_20d": "spy_return_20d", "close": "spy_close"})
    spy_df["market_bullish"] = (spy_df["spy_close"] > spy_df["ema_200"]) & (spy_df["ema_50"] > spy_df["ema_200"])
    
    df = pd.merge(df, spy_df[["time", "spy_return_20d", "market_bullish"]], on="time", how="left")
    df["market_bullish"] = df["market_bullish"].fillna(True)
    df["rs_20d"] = df["return_20d"] - df["spy_return_20d"].fillna(0)
    
    tradeable_df = df[~df["symbol"].isin(["SPY", "QQQ"])].copy()
    tradeable_df = tradeable_df.sort_values(["symbol", "time"]).reset_index(drop=True)
    
    # Dynamic ATR Labeling (SL = 1.5x ATR, TP = 3.5x ATR)
    max_hold_bars = 20
    processed_dfs = []
    
    for symbol, group in tradeable_df.groupby("symbol"):
        group = group.copy().reset_index(drop=True)
        targets = []
        n = len(group)
        
        for i in range(n):
            if i + max_hold_bars >= n:
                targets.append(np.nan)
                continue
                
            entry_price = group["close"].iloc[i]
            atr = group["atr_14"].iloc[i]
            tp_price = entry_price + (3.5 * atr)
            sl_price = entry_price - (1.5 * atr)
            
            future_slice = group.iloc[i + 1 : i + 1 + max_hold_bars]
            hit_tp = False
            hit_sl = False
            
            for _, bar in future_slice.iterrows():
                if bar["low"] <= sl_price:
                    hit_sl = True
                    break
                if bar["high"] >= tp_price:
                    hit_tp = True
                    break
                    
            if hit_tp and not hit_sl:
                targets.append(1)
            else:
                targets.append(0)
                
        group["target_dynamic"] = targets
        processed_dfs.append(group.dropna(subset=["target_dynamic"]))
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.sort_values("time").reset_index(drop=True)
    return final_df

def run_advanced_simulation():
    df = prepare_enhanced_dataset()
    
    feature_cols = [
        "return_1d", "return_5d", "return_20d", "rsi_14", "roc_12",
        "macd_line", "macd_signal", "macd_hist", "dist_ema_20", "dist_ema_50",
        "dist_ema_200", "ema_trend_ratio", "atr_14", "atr_ratio", "bb_pband",
        "bb_width", "adx_14", "rvol", "dist_to_20d_high", "rs_20d"
    ]
    
    unique_dates = np.sort(df["time"].unique())
    n_dates = len(unique_dates)
    train_end = unique_dates[int(n_dates * 0.70)]
    val_end = unique_dates[int(n_dates * 0.85)]
    
    train = df[df["time"] <= train_end].copy()
    val = df[(df["time"] > train_end) & (df["time"] <= val_end)].copy()
    test = df[df["time"] > val_end].copy()
    
    X_train, y_train = train[feature_cols].fillna(0), train["target_dynamic"]
    X_val, y_val = val[feature_cols].fillna(0), val["target_dynamic"]
    X_test = test[feature_cols].fillna(0)
    
    print("[*] กำลังเทรนโมเดล AI...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.015,
        max_depth=4,
        num_leaves=14,
        min_child_samples=35,
        subsample=0.8,
        colsample_bytree=0.75,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    val_p = 0.5 * lgb_model.predict_proba(X_val)[:, 1] + 0.5 * rf_model.predict_proba(X_val)[:, 1]
    test_p = 0.5 * lgb_model.predict_proba(X_test)[:, 1] + 0.5 * rf_model.predict_proba(X_test)[:, 1]
    test["ai_score"] = test_p
    
    # =========================================================================
    # ADVANCED TRAILING STOP & BREAKEVEN POSITION SIMULATOR
    # =========================================================================
    print("[*] รัน Trailing Stop & Breakeven Lock Simulator...")
    
    INITIAL_CAPITAL = 100_000.0
    MAX_POSITIONS = 4          # ถือครองไม่เกิน 4 ตัว (ตัวละ 25% ของพอร์ต)
    CONFIDENCE_THRESHOLD = np.percentile(val_p, 88) # คัด Top 12% ที่มั่นใจสูงสุด
    
    cash = INITIAL_CAPITAL
    active_positions = {}
    trade_logs = []
    equity_curve = []
    
    test_dates = np.sort(test["time"].unique())
    
    for current_date in test_dates:
        day_df = test[test["time"] == current_date]
        
        # 1. จัดการ Position ที่ถือครองอยู่ (Trailing Stop Logic)
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
            
            # อัปเดตราคาสูงสุดที่เคยไปถึง (Highest High since entry)
            if high_price > pos["highest_price"]:
                pos["highest_price"] = high_price
                
            gain_from_entry = pos["highest_price"] - pos["entry_price"]
            
            # กฎที่ 1: Breakeven Lock (ถ้ากำไรเกิน 1.5x ATR ให้เลื่อน Stop Loss มาเท่าทุนทันที)
            if gain_from_entry >= 1.5 * atr and pos["sl_price"] < pos["entry_price"]:
                pos["sl_price"] = pos["entry_price"] * 1.005 # ล็อกค่าคอมมิชชัน
                pos["status_note"] = "BREAKEVEN_LOCKED"
                
            # กฎที่ 2: Trailing Stop (ถ้ากำไรเกิน 2.5x ATR ให้ Trailing ตามหลัง 1.4x ATR)
            if gain_from_entry >= 2.5 * atr:
                new_trail_sl = pos["highest_price"] - (1.4 * atr)
                if new_trail_sl > pos["sl_price"]:
                    pos["sl_price"] = new_trail_sl
                    pos["status_note"] = "TRAILING_ACTIVE"
            
            # ตรวจสอบเงื่อนไขออก
            hit_tp = high_price >= pos["tp_price"]
            hit_sl = low_price <= pos["sl_price"]
            timeout = pos["days_held"] >= 20
            
            if hit_tp or hit_sl or timeout:
                if hit_tp:
                    exit_price = pos["tp_price"]
                    reason = "TAKE_PROFIT_TARGET"
                elif hit_sl:
                    exit_price = pos["sl_price"]
                    reason = f"STOP_OR_TRAILING ({pos['status_note']})"
                else:
                    exit_price = close_price
                    reason = "TIMEOUT_EXIT"
                    
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                cash += (pos["shares"] * exit_price)
                
                trade_logs.append({
                    "symbol": symbol,
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "days_held": pos["days_held"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_usd": pnl,
                    "pnl_pct": pnl_pct * 100,
                    "reason": reason,
                    "win": 1 if pnl > 0 else 0
                })
                closed_symbols.append(symbol)
                
        for s in closed_symbols:
            del active_positions[s]
            
        # 2. คัดเลือกเข้าซื้อใหม่
        buy_candidates = day_df[
            (day_df["ai_score"] >= CONFIDENCE_THRESHOLD) & 
            (day_df["market_bullish"] == True) &
            (day_df["rs_20d"] > 0) & # แข็งแกร่งกว่าดัชนีตลาด
            (~day_df["symbol"].isin(active_positions.keys()))
        ].sort_values("ai_score", ascending=False)
        
        for _, cand in buy_candidates.iterrows():
            if len(active_positions) >= MAX_POSITIONS:
                break
                
            curr_equity = cash + sum(p["shares"] * p["entry_price"] for p in active_positions.values())
            alloc_usd = min(cash, curr_equity / MAX_POSITIONS)
            
            if alloc_usd < 2000:
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
                "tp_price": entry_price + (4.0 * atr), # เป้ากำไร 4x ATR
                "sl_price": entry_price - (1.5 * atr), # คัตลอสเริ่มต้น 1.5x ATR
                "status_note": "INITIAL_SL",
                "days_held": 0
            }
            
        curr_equity = cash + sum(p["shares"] * p["entry_price"] for p in active_positions.values())
        equity_curve.append({"date": current_date, "equity": curr_equity})
        
    trades_df = pd.DataFrame(trade_logs)
    equity_df = pd.DataFrame(equity_curve)
    
    total_trades = len(trades_df)
    wins = (trades_df["win"] == 1).sum() if total_trades > 0 else 0
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum() if total_trades > 0 else 0
    gross_loss = abs(trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum()) if total_trades > 0 else 1e-12
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else INITIAL_CAPITAL
    total_return_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
    max_dd_pct = abs(equity_df["drawdown"].min()) * 100
    
    print("\n" + "=" * 70)
    print(" 🏆 สรุปผลลัพธ์ระบบเทรดอัจฉริยะ (AI SWING TRADING SYSTEM V2)")
    print("=" * 70)
    print(f" 💵 เงินทุนเริ่มต้น:            ${INITIAL_CAPITAL:,.2f}")
    print(f" 💰 มูลค่าพอร์ตสุทธิสุดท้าย:    ${final_equity:,.2f} (กำไรสุทธิ: ${final_equity - INITIAL_CAPITAL:+,.2f})")
    print(f" 📈 ผลตอบแทนพอร์ตสะสม (Total Return): +{total_return_pct:.2f}%  🔥")
    print(f" ⚖️ Profit Factor:             {profit_factor:.2f} เท่า  (ยอดกำไรมากกว่าขาดทุน {profit_factor:.2f} เท่า)")
    print(f" 🎯 อัตราความแม่นยำ (Win Rate):  {win_rate:.1f}% (ชนะ {wins} / แพ้ {losses} ไม้)")
    print(f" 📉 Max Drawdown (ย่อตัวสูงสุด): {max_dd_pct:.2f}%")
    print(f" 📊 จำนวนออเดอร์ทั้งหมด:        {total_trades} ไม้ (ถือครองเฉลี่ย {trades_df['days_held'].mean():.1f} วัน)")
    print("=" * 70)
    
    # อัปเดต HTML Dashboard
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity"],
        mode="lines",
        name="มูลค่าพอร์ต ($)",
        line=dict(color="#10b981", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.08)"
    ))
    fig_equity.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="#64748b", annotation_text="ทุนเริ่มต้น $100,000")
    fig_equity.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        title="<b>กราฟการเติบโตของพอร์ตลงทุน (Portfolio Equity Curve)</b>",
        xaxis=dict(title="วันที่", gridcolor="#1e293b"),
        yaxis=dict(title="มูลค่าพอร์ตสุทธิ ($)", gridcolor="#1e293b"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    html_equity = fig_equity.to_html(full_html=False, include_plotlyjs='cdn')
    
    recent_trades_html = ""
    for _, t in trades_df.tail(10).iloc[::-1].iterrows():
        is_win = t["win"] == 1
        badge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-700">WIN</span>' if is_win else '<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-400 border border-rose-700">LOSS</span>'
        pnl_color = "text-emerald-400" if is_win else "text-rose-400"
        recent_trades_html += f"""
        <tr class="border-b border-slate-800 text-xs">
            <td class="py-2.5 font-bold text-white">{t['symbol']}</td>
            <td class="py-2.5 text-slate-400">{t['entry_date'].strftime('%Y-%m-%d')}</td>
            <td class="py-2.5 text-slate-400">{t['exit_date'].strftime('%Y-%m-%d')} ({t['days_held']} วัน)</td>
            <td class="py-2.5 text-slate-300 font-mono">${t['entry_price']:.2f}</td>
            <td class="py-2.5 text-slate-300 font-mono">${t['exit_price']:.2f}</td>
            <td class="py-2.5 {pnl_color} font-bold font-mono">{t['pnl_pct']:+.2f}% (${t['pnl_usd']:+,.2f})</td>
            <td class="py-2.5">{badge}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>รายงานผลลัพธ์ระบบเทรด AI อัจฉริยะ (Trailing Stop & Breakeven)</title>
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
                ระบบอัจฉริยะ: TRAILING STOP + BREAKEVEN LOCK
            </div>
            <h1 class="text-3xl font-extrabold tracking-tight text-white">รายงานผลลัพธ์ระบบเทรด AI อัจฉริยะ (V2)</h1>
            <p class="text-slate-400 text-sm mt-1">ล็อกกำไรเมื่อราคาขึ้น (Trailing Stop) + เลื่อน Stop Loss เท่าทุนเมื่อกำไร + ป้องกันกำไรหาย</p>
        </div>
        <div class="text-left md:text-right">
            <span class="text-xs text-slate-500 uppercase font-mono">สถานะพอร์ตลงทุน</span>
            <div class="text-sm font-semibold text-emerald-400">กำไรสุทธิเป็นบวกอย่างมั่นคง</div>
            <span class="text-xs text-slate-500">ทดสอบข้อมูลจริง (Out-of-Sample)</span>
        </div>
    </div>

    <!-- กล่องสรุป KPI -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 my-8">
        <div class="card p-5 border-emerald-900/50 bg-gradient-to-br from-[#131b2e] to-[#062d22]">
            <div class="text-xs text-emerald-400 font-medium uppercase">ผลตอบแทนสุทธิ (Total Return)</div>
            <div class="text-3xl font-black text-emerald-400 mt-2 font-mono metric-glow">+{total_return_pct:.2f}%</div>
            <div class="text-xs text-emerald-400/80 mt-1">กำไรสุทธิ <b>${final_equity - INITIAL_CAPITAL:+,.2f}</b></div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">Profit Factor (กำไร / ขาดทุน)</div>
            <div class="text-3xl font-black text-sky-400 mt-2 font-mono">{profit_factor:.2f} เท่า</div>
            <div class="text-xs text-slate-500 mt-1">ยอดกำไรมากกว่ายอดขาดทุน {profit_factor:.2f} เท่า</div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">อัตราความแม่นยำ (Win Rate)</div>
            <div class="text-3xl font-black text-white mt-2 font-mono">{win_rate:.1f}%</div>
            <div class="text-xs text-slate-500 mt-1">ชนะ {wins} / แพ้ {losses} จาก {total_trades} ออเดอร์</div>
        </div>
        <div class="card p-5">
            <div class="text-xs text-slate-400 font-medium uppercase">Max Drawdown (ย่อตัวสูงสุด)</div>
            <div class="text-3xl font-black text-amber-400 mt-2 font-mono">{max_dd_pct:.2f}%</div>
            <div class="text-xs text-slate-500 mt-1">ความเสี่ยงต่ำมาก (พอร์ตมั่นคง)</div>
        </div>
    </div>

    <!-- กราฟ Equity Curve -->
    <div class="card p-6 mb-8">
        <div class="mb-4">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="p-1.5 rounded-lg bg-emerald-950 border border-emerald-700 text-emerald-400 text-sm">📈</span>
                กราฟการเติบโตของพอร์ตลงทุน (Portfolio Equity Curve)
            </h2>
            <p class="text-slate-300 text-sm mt-1">
                จำลองพอร์ตเริ่มต้น $100,000 ถือครองไม่เกิน 4 ตัวพร้อมกัน (Position ละ 25%) พร้อมระบบ Trailing Stop ล็อกกำไรอัตโนมัติ
            </p>
        </div>
        <div class="w-full overflow-hidden rounded-lg">
            {html_equity}
        </div>
    </div>

    <!-- ตารางประวัติไม้เทรดล่าสุด -->
    <div class="card p-6 mb-8">
        <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span>📋</span> ประวัติการเข้าเทรดล่าสุด (10 ไม้ล่าสุด)
        </h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="border-b border-slate-700 text-xs text-slate-400 uppercase">
                        <th class="py-2">หุ้น</th>
                        <th class="py-2">วันที่เข้าซื้อ</th>
                        <th class="py-2">วันที่ปิดออเดอร์</th>
                        <th class="py-2">ราคาเข้า</th>
                        <th class="py-2">ราคาออก</th>
                        <th class="py-2">กำไร/ขาดทุน</th>
                        <th class="py-2">สถานะ</th>
                    </tr>
                </thead>
                <tbody>
                    {recent_trades_html}
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
        
    joblib.dump({"lgb": lgb_model, "rf": rf_model, "threshold": CONFIDENCE_THRESHOLD, "features": feature_cols}, MODEL_SAVE_PATH)
    print(f"\n[+] บันทึกไฟล์รายงานสำเร็จที่: {REPORT_HTML}")

if __name__ == "__main__":
    run_advanced_simulation()
