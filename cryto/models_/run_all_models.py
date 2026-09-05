import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODEL_FILES = [
    ("BTCUSD", "model_BTCUSD.py"),
    ("ETHUSD", "model_ETHUSD.py"),
    ("SOLUSD", "model_SOLUSD.py"),
    ("BNBUSD", "model_BNBUSD.py"),
    ("XRPUSD", "model_XRPUSD.py"),
]


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
HUMAN_LOG_PATH = LOGS_DIR / "crypto_models_human_log.md"
SUMMARY_CSV_PATH = LOGS_DIR / "crypto_models_summary.csv"


def run_model(symbol, filename):
    script_path = MODELS_DIR / filename
    started_at = datetime.now()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    finished_at = datetime.now()
    return {
        "symbol": symbol,
        "filename": filename,
        "returncode": result.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def load_latest(symbol):
    path = LOGS_DIR / f"{symbol.lower()}_forecast_latest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def short_float(value, digits=4):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value):
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def first_value(row, keys):
    for key in keys:
        value = row.get(key)
        if value not in [None, ""]:
            return value
    return None


def summarize_latest(symbol, payload, run_status):
    if payload is None:
        return {
            "symbol": symbol,
            "status": "ERROR",
            "final_bias": "NO_LOG",
            "decision": "NO_LOG",
            "final_score": "",
            "confidence": "",
            "alignment": "",
            "price": "",
            "best_timeframe": "",
            "entry_price": "",
            "entry_zone": "",
            "stop_loss": "",
            "take_profit_1": "",
            "reason": "Latest forecast json not found.",
            "returncode": run_status["returncode"],
        }

    summary = payload.get("summary", {})
    timeframes = payload.get("timeframes", [])
    best = max(timeframes, key=lambda row: abs(float(row.get("score", 0)))) if timeframes else None

    reasons = []
    if best:
        reasons.append(f"{best.get('interval')} strongest: {best.get('bias')} score={best.get('score')}")
        if best.get("reasons"):
            reasons.append(str(best.get("reasons")))
    for reason in summary.get("decision_reasons", []):
        reasons.append(str(reason))

    decision = summary.get("decision", "")
    entry_price = ""
    entry_zone = "WAIT"
    if best and "TRADE_READY" in decision:
        current_price = best.get("price")
        support = first_value(best, ["support"])
        resistance = first_value(best, ["resistance"])
        entry_price = short_float(current_price, 5)
        if "BULLISH" in decision:
            entry_zone = f"{short_float(current_price, 5)} to {short_float(resistance, 5)}"
        elif "BEARISH" in decision:
            entry_zone = f"{short_float(support, 5)} to {short_float(current_price, 5)}"

    return {
        "symbol": symbol,
        "status": "OK" if run_status["returncode"] == 0 else "ERROR",
        "final_bias": summary.get("final_bias", ""),
        "decision": decision,
        "final_score": short_float(summary.get("final_score"), 2),
        "confidence": pct(summary.get("confidence")),
        "alignment": json.dumps(summary.get("alignment", {}), ensure_ascii=False),
        "price": short_float(best.get("price"), 5) if best else "",
        "best_timeframe": best.get("interval", "") if best else "",
        "entry_price": entry_price,
        "entry_zone": entry_zone,
        "stop_loss": short_float(best.get("stop_loss"), 5) if best else "",
        "take_profit_1": short_float(best.get("take_profit_1"), 5) if best else "",
        "reason": " | ".join(reasons),
        "returncode": run_status["returncode"],
    }


def write_summary_csv(rows):
    fieldnames = [
        "symbol",
        "status",
        "final_bias",
        "decision",
        "final_score",
        "confidence",
        "alignment",
        "price",
        "best_timeframe",
        "entry_price",
        "entry_zone",
        "stop_loss",
        "take_profit_1",
        "reason",
        "returncode",
    ]
    temp_path = SUMMARY_CSV_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(SUMMARY_CSV_PATH)


def write_human_log(rows, run_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Crypto Model Forecast Human Log",
        "",
        f"- Generated at: {now}",
        f"- Models run: {len(run_results)}",
        "",
        "## Final Summary",
        "",
        "| Symbol | Status | Bias | Decision | Score | Confidence | Price | Strongest TF |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['symbol']} | {row['status']} | {row['final_bias']} | "
            f"{row['decision']} | {row['final_score']} | {row['confidence']} | "
            f"{row['price']} | {row['best_timeframe']} |"
        )

    lines.extend(
        [
            "",
            "## Entry And Trade Levels",
            "",
            "| Symbol | Decision | Reference TF | Current Price | Entry Price | Entry Zone | Stop Loss | Take Profit 1 |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['symbol']} | {row['decision']} | {row['best_timeframe']} | "
            f"{row['price']} | {row['entry_price']} | {row['entry_zone']} | "
            f"{row['stop_loss']} | {row['take_profit_1']} |"
        )

    lines.extend(["", "## Reasons", ""])
    for row in rows:
        lines.append(f"### {row['symbol']}")
        lines.append(f"- Alignment: {row['alignment']}")
        lines.append(f"- Reason: {row['reason'] or 'No reason available.'}")
        lines.append("")

    errors = [item for item in run_results if item["returncode"] != 0]
    if errors:
        lines.extend(["## Errors", ""])
        for item in errors:
            lines.append(f"### {item['symbol']} ({item['filename']})")
            lines.append(f"- Return code: {item['returncode']}")
            lines.append("```text")
            lines.append(item["stderr"] or item["stdout"] or "No output.")
            lines.append("```")
            lines.append("")

    temp_path = HUMAN_LOG_PATH.with_suffix(".md.tmp")
    temp_path.write_text("\n".join(lines), encoding="utf-8")
    temp_path.replace(HUMAN_LOG_PATH)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_results = []
    summary_rows = []

    print("Running all crypto forecast models...")
    for symbol, filename in MODEL_FILES:
        result = run_model(symbol, filename)
        run_results.append(result)
        payload = load_latest(symbol)
        summary_rows.append(summarize_latest(symbol, payload, result))
        status = "OK" if result["returncode"] == 0 else "ERROR"
        print(f"{symbol}: {status}")

    write_summary_csv(summary_rows)
    write_human_log(summary_rows, run_results)

    print("")
    print(f"Saved human log: {HUMAN_LOG_PATH}")
    print(f"Saved summary csv: {SUMMARY_CSV_PATH}")
    print(f"Human log size: {HUMAN_LOG_PATH.stat().st_size} bytes")
    print(f"Summary csv size: {SUMMARY_CSV_PATH.stat().st_size} bytes")
    print("")
    print("Final decisions:")
    for row in summary_rows:
        print(
            f"{row['symbol']}: {row['decision']} | "
            f"score={row['final_score']} | confidence={row['confidence']}"
        )


if __name__ == "__main__":
    main()
