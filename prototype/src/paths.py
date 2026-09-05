from pathlib import Path

from .csv_store import cleanup_stale_temp_files


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROTOTYPE_ROOT.parent
DATA_DIR = PROTOTYPE_ROOT / "data"
CONFIG_DIR = PROTOTYPE_ROOT / "config"

SIGNALS_DIR = DATA_DIR / "signals"
PAPER_DIR = DATA_DIR / "paper_trades"
AI_DIR = DATA_DIR / "ai"
AUDIT_DIR = DATA_DIR / "audit"

DECISION_PATH = PAPER_DIR / "paper_trade_decisions.csv"
SYSTEM_ERRORS_PATH = AUDIT_DIR / "system_errors.jsonl"
PROTOTYPE_RUN_HISTORY_PATH = AUDIT_DIR / "prototype_run_history.jsonl"

# External model runners
FOREX_MODEL_RUNNER = PROJECT_ROOT / "ai_chatbot" / "forex" / "models_" / "run_all_models.py"
CRYPTO_MODEL_RUNNER = PROJECT_ROOT / "cryto" / "models_" / "run_all_models.py"

FOREX_LOG_DIR = PROJECT_ROOT / "ai_chatbot" / "logs"
CRYPTO_LOG_DIR = PROJECT_ROOT / "cryto" / "logs"


def ensure_dirs():
    for path in [SIGNALS_DIR, PAPER_DIR, AI_DIR, AUDIT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    cleanup_stale_temp_files(DATA_DIR)
