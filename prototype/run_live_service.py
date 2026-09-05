import time
import sys
import logging
import subprocess
from datetime import datetime, timezone
from src.ai_dataset import export_ai_dataset
from src.paper_trade_tracker import open_from_signals, update_open_trades, write_summary
from src.signal_ingest import ingest_all
from src.paper_trade_replay import run_replay_sync
from src.paths import PROTOTYPE_RUN_HISTORY_PATH, ensure_dirs, FOREX_MODEL_RUNNER, CRYPTO_MODEL_RUNNER
from src.settings import load_settings
from src.csv_store import append_jsonl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prototype/live_service.log')
    ]
)
logger = logging.getLogger(__name__)

def run_external_models():
    logger.info("Executing external models before prototype...")
    python_cmd = [sys.executable]
    if FOREX_MODEL_RUNNER.exists():
        logger.info(f"Running Forex Models: {FOREX_MODEL_RUNNER}")
        subprocess.run(
            [*python_cmd, str(FOREX_MODEL_RUNNER)],
            cwd=FOREX_MODEL_RUNNER.parent.parent.parent,
            check=False,
        )

    if CRYPTO_MODEL_RUNNER.exists():
        logger.info(f"Running Crypto Models: {CRYPTO_MODEL_RUNNER}")
        subprocess.run(
            [*python_cmd, str(CRYPTO_MODEL_RUNNER)],
            cwd=CRYPTO_MODEL_RUNNER.parent.parent,
            check=False,
        )

def run_cycle():
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("Starting prototype cycle...")
    steps = []
    status = "OK"
    failed_step = None
    stderr_text = ""
    ingested = 0
    opened = 0
    closed = 0
    dataset_stats = {}
    try:
        # 0. Run models to get fresh signals
        run_external_models()
        steps.append({"step": "run_models", "status": "OK", "result": None})

        # 1. Sync any missed trades
        run_replay_sync()
        steps.append({"step": "replay", "status": "OK", "result": None})

        # 2. Ingest new signals from model logs
        ingested = ingest_all()
        steps.append({"step": "ingest", "status": "OK", "result": ingested})
        if ingested > 0:
            logger.info(f"Ingested {ingested} new signals")

        # 3. Open new paper trades
        opened = open_from_signals()
        steps.append({"step": "open", "status": "OK", "result": opened})
        if opened > 0:
            logger.info(f"Opened {opened} new paper trades")

        # 4. Update open trades (Check TP/SL)
        # Note: This calls fetch_candles which can be heavy.
        closed = update_open_trades()
        steps.append({"step": "update", "status": "OK", "result": closed})
        if closed > 0:
            logger.info(f"Closed {closed} paper trades")

        # 5. Export dataset for AI
        dataset_stats = export_ai_dataset()
        steps.append({"step": "export_ai", "status": "OK", "result": dataset_stats})
        write_summary()
        steps.append({"step": "summary", "status": "OK", "result": None})
        
        logger.info(f"Cycle completed. AI Dataset Rows: {dataset_stats.get('training_rows', 'N/A')}")
    except Exception as e:
        status = "ERROR"
        failed_step = "cycle"
        stderr_text = str(e)
        logger.error(f"Error in cycle: {e}", exc_info=True)
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        stdout_text = "\n".join([
            "Live cycle completed" if status == "OK" else "Live cycle completed with error(s)",
            f"Signals ingested: {ingested}",
            f"Paper trades opened: {opened}",
            f"Paper trades closed: {closed}",
            f"AI training rows: {dataset_stats.get('training_rows', 'N/A')}",
            f"AI observation rows: {dataset_stats.get('observation_rows', 'N/A')}",
        ])
        append_jsonl(
            PROTOTYPE_RUN_HISTORY_PATH,
            {
                "market": "prototype",
                "runSource": "live_cycle",
                "startedAt": started_at,
                "finishedAt": finished_at,
                "status": status,
                "runModels": False,
                "failedStep": failed_step,
                "steps": steps,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "code": 0 if status == "OK" else 1,
                "signal": None,
            },
        )

def main():
    ensure_dirs()
    settings = load_settings()
    # Default poll every 5 minutes (300 seconds)
    # We use a relatively long interval to respect yfinance limits
    poll_interval = int(settings.get("live_service_poll_seconds", 300))
    
    logger.info(f"Starting Live Service Mode (Polling every {poll_interval}s)")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            start_time = time.time()
            run_cycle()
            
            # Calculate sleep time to maintain consistent interval
            elapsed = time.time() - start_time
            sleep_time = max(1, poll_interval - elapsed)
            
            logger.info(f"Sleeping for {round(sleep_time, 1)}s...")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")

if __name__ == "__main__":
    main()
