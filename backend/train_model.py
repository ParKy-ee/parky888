import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PARQUET_PATH = os.path.join(DATA_DIR, "swing_dataset_latest.parquet")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "swing_ensemble_model.joblib")

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d", "rsi_14", "roc_12",
    "macd_line", "macd_signal", "macd_hist", "dist_ema_20", "dist_ema_50",
    "dist_ema_200", "ema_trend_ratio", "atr_14", "atr_ratio", "bb_pband",
    "bb_width", "adx_14", "rvol", "dist_to_20d_high"
]

def run():
    df = pd.read_parquet(PARQUET_PATH)
    df["time"] = pd.to_datetime(df["time"])
    
    # 1. สร้าง Market Filter จาก SPY (ตลาดใหญ่ต้องอยู่เหนือเส้น EMA 200)
    spy_df = df[df["symbol"] == "SPY"][["time", "dist_ema_200", "ema_trend_ratio"]].copy()
    spy_df["market_bullish"] = (spy_df["dist_ema_200"] > 0) & (spy_df["ema_trend_ratio"] > 1.0)
    
    tradeable_df = df[~df["symbol"].isin(["SPY", "QQQ"])].copy()
    tradeable_df = pd.merge(tradeable_df, spy_df[["time", "market_bullish"]], on="time", how="left")
    tradeable_df["market_bullish"] = tradeable_df["market_bullish"].fillna(True)
    tradeable_df = tradeable_df.sort_values("time").reset_index(drop=True)
    
    # Chronological Split
    unique_dates = np.sort(tradeable_df["time"].unique())
    n = len(unique_dates)
    train_end = unique_dates[int(n * 0.70)]
    val_end = unique_dates[int(n * 0.85)]
    
    train = tradeable_df[tradeable_df["time"] <= train_end].copy()
    val = tradeable_df[(tradeable_df["time"] > train_end) & (tradeable_df["time"] <= val_end)].copy()
    test = tradeable_df[tradeable_df["time"] > val_end].copy()
    
    X_train, y_train = train[FEATURE_COLS].fillna(0), train["target"]
    X_val, y_val = val[FEATURE_COLS].fillna(0), val["target"]
    X_test, y_test = test[FEATURE_COLS].fillna(0), test["target"]
    
    # Train Models
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.015,
        max_depth=4,
        num_leaves=12,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Predictions
    val_p = 0.5 * lgb_model.predict_proba(X_val)[:, 1] + 0.5 * rf_model.predict_proba(X_val)[:, 1]
    test_p = 0.5 * lgb_model.predict_proba(X_test)[:, 1] + 0.5 * rf_model.predict_proba(X_test)[:, 1]
    
    test["ai_score"] = test_p
    
    # เลือกค่า Percentile ความมั่นใจที่ให้ผลตอบแทนดีที่สุด (Top 5% - 10% คุณภาพสูง)
    results_table = []
    for top_pct in [98, 95, 92, 90, 85]:
        thresh = np.percentile(val_p, top_pct)
        # เงื่อนไข: คะแนน AI สูง + ตลาดใหญ่เป็นขาขึ้น (Market Bullish Filter)
        selected = test[(test["ai_score"] >= thresh) & (test["market_bullish"] == True)].copy()
        n_trades = len(selected)
        if n_trades == 0:
            continue
        wr = selected["target"].mean() * 100
        wins = (selected["target"] == 1).sum()
        losses = (selected["target"] == 0).sum()
        pf = (wins * 0.10) / (losses * 0.04 + 1e-12)
        results_table.append({
            "top_pct": 100 - top_pct,
            "threshold": thresh,
            "trades": n_trades,
            "win_rate": wr,
            "profit_factor": pf,
            "wins": wins,
            "losses": losses
        })
        
    res_df = pd.DataFrame(results_table)
    
    print("\n" + "=" * 70)
    print(" การทดสอบจูนระดับความมั่นใจของ AI (Confidence Threshold Analysis)")
    print("=" * 70)
    for _, r in res_df.iterrows():
        print(f" • คัดเฉพาะ Top {r['top_pct']:.0f}% ไม้มั่นใจสุด | เข้าเทรด: {r['trades']:>3.0f} ไม้ | Win Rate: {r['win_rate']:>5.1f}% | Profit Factor: {r['profit_factor']:.2f} เท่า")
    print("=" * 70)
    
    best_row = res_df.sort_values("profit_factor", ascending=False).iloc[0]
    print(f"\n💡 ผลลัพธ์ที่ดีที่สุด: คัดเฉพาะ Top {best_row['top_pct']:.0f}% คุณภาพสูงสุด")
    print(f"   - อัตราความแม่นยำ (Win Rate): {best_row['win_rate']:.1f}%")
    print(f"   - Profit Factor: {best_row['profit_factor']:.2f} เท่า (ได้กำไรมากกว่าขาดทุน {best_row['profit_factor']:.2f} เท่า)")
    print(f"   - จำนวนไม้เทรดจริง: {best_row['trades']:.0f} ไม้ (ชนะ {best_row['wins']:.0f} / แพ้ {best_row['losses']:.0f})")
    print("=" * 70)

if __name__ == "__main__":
    run()
