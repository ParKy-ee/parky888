import sys
import subprocess
import traceback
import csv
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from src when run or imported from parent directories
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.ai_dataset import export_ai_dataset
from src.audit import log_system_error
from src.csv_store import append_jsonl, write_csv_atomic
from src.paper_trade_tracker import (
    open_from_signals,
    update_open_trades,
    write_summary,
    OPEN_PATH,
    CLOSED_PATH,
    DECISION_PATH,
    OPEN_FIELDS,
    CLOSED_FIELDS,
    DECISION_FIELDS,
)
from src.paths import (
    CRYPTO_MODEL_RUNNER,
    FOREX_MODEL_RUNNER,
    PROTOTYPE_RUN_HISTORY_PATH,
    CONFIG_DIR,
    AI_DIR,
    ensure_dirs,
)
from src.paper_trade_replay import run_replay_sync
from src.signal_ingest import ingest_all
from src.paper_trade_blocked import BLOCKED_PATH, BLOCKED_FIELDS
from src.settings import load_settings


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_csv_header(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            return next(reader)
    except Exception:
        return []


def get_config_hash():
    config_path = CONFIG_DIR / "settings.json"
    if not config_path.exists():
        return ""
    try:
        content = config_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:8]
    except Exception:
        return ""


def migrate_row(row, fields):
    new_row = {}
    for f in fields:
        val = row.get(f, "")
        if val is None:
            val = ""
        new_row[f] = val
        
    # Apply specific policy updates
    if "schema_version" in fields:
        new_row["schema_version"] = "2.0.0"
        
    if "initial_stop_loss" in fields and not new_row["initial_stop_loss"]:
        new_row["initial_stop_loss"] = row.get("stop_loss", "")
        
    if "initial_risk_distance" in fields and not new_row["initial_risk_distance"]:
        new_row["initial_risk_distance"] = row.get("risk_distance", "")
        
    return new_row


def migrate_and_merge_file(src_path: Path, dest_path: Path, fields, key_field, rename_source=True):
    if not src_path.exists():
        return
    
    rows = []
    try:
        with src_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for r in reader:
                rows.append(r)
    except Exception as e:
        print(f"[MIGRATION] Error reading {src_path.name}: {e}")
        return
        
    if not rows:
        return
        
    migrated_rows = [migrate_row(r, fields) for r in rows]
    
    existing_rows = []
    if dest_path.exists():
        try:
            with dest_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for r in reader:
                    existing_rows.append(r)
        except Exception:
            pass
            
    merged = []
    seen = set()
    
    def get_key(r):
        if key_field == "signal_id" and "decision_time" in r:
            return (r.get("signal_id", ""), r.get("decision_time", ""))
        return r.get(key_field, "")
        
    # Add existing rows first
    for r in existing_rows:
        k = get_key(r)
        if k not in seen:
            seen.add(k)
            merged.append(r)
            
    # Add migrated rows if not already present
    added_count = 0
    for r in migrated_rows:
        k = get_key(r)
        if k not in seen:
            seen.add(k)
            merged.append(r)
            added_count += 1
            
    write_csv_atomic(dest_path, merged, fields)
    if added_count > 0:
        print(f"[MIGRATION] Restored/Migrated {added_count} rows from {src_path.name} to {dest_path.name}")
        
    # Mark snapshot as migrated to avoid double processing
    if rename_source and src_path != dest_path:
        try:
            migrated_path = src_path.with_suffix(".csv.migrated")
            if migrated_path.exists():
                migrated_path.unlink()
            shutil.move(str(src_path), str(migrated_path))
        except Exception as e:
            print(f"[MIGRATION] Error marking snapshot as migrated: {e}")


def scan_and_migrate_archive_snapshots():
    legacy_dir = OPEN_PATH.parent / "legacy_snapshots"
    if not legacy_dir.exists():
        return
        
    mapping = [
        ("open_trades_*.csv", OPEN_PATH, OPEN_FIELDS, "trade_id"),
        ("closed_trades_*.csv", CLOSED_PATH, CLOSED_FIELDS, "trade_id"),
        ("paper_trade_decisions_*.csv", DECISION_PATH, DECISION_FIELDS, "signal_id"),
        ("paper_trade_blocked_*.csv", BLOCKED_PATH, BLOCKED_FIELDS, "signal_id")
    ]
    
    for pattern, dest_path, fields, key_field in mapping:
        for src_path in legacy_dir.glob(pattern):
            if src_path.name.endswith(".csv"):
                migrate_and_merge_file(src_path, dest_path, fields, key_field, rename_source=True)


def check_and_migrate_all_states():
    files_to_check = [
        (OPEN_PATH, OPEN_FIELDS, "trade_id"),
        (CLOSED_PATH, CLOSED_FIELDS, "trade_id"),
        (DECISION_PATH, DECISION_FIELDS, "signal_id"),
        (BLOCKED_PATH, BLOCKED_FIELDS, "signal_id")
    ]
    
    legacy_detected = False
    state_initialized = False
    
    for path, fields, key_field in files_to_check:
        if path.exists():
            header = get_csv_header(path)
            if header != fields:
                print(f"[VALIDATION] Schema mismatch on {path.name}. Expected {fields}, got {header}")
                legacy_detected = True
                break
        else:
            state_initialized = True
            
    if legacy_detected:
        print("[MIGRATION] Migrating legacy active files to new schema and creating snapshot backups.")
        for path, fields, key_field in files_to_check:
            if path.exists():
                header = get_csv_header(path)
                if header != fields:
                    # Move to snapshot backup first, then migrate from it
                    legacy_dir = path.parent / "legacy_snapshots"
                    legacy_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    archived_name = f"{path.stem}_{timestamp}{path.suffix}"
                    archived_path = legacy_dir / archived_name
                    
                    print(f"[ARCHIVE] Backup legacy file {path.name} to {archived_path}")
                    try:
                        shutil.move(str(path), str(archived_path))
                    except Exception:
                        shutil.copy(str(path), str(archived_path))
                        path.unlink()
                        
                    # Migrate the backup snapshot to the active path
                    migrate_and_merge_file(archived_path, path, fields, key_field, rename_source=False)
            else:
                write_csv_atomic(path, [], fields)
                state_initialized = True
        return True, True
    else:
        any_initialized = False
        for path, fields, key_field in files_to_check:
            if not path.exists():
                print(f"[INIT] Initializing missing state file: {path.name}")
                write_csv_atomic(path, [], fields)
                any_initialized = True
        return False, (state_initialized or any_initialized)


def verify_file_existence(*paths: Path):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected output file not found: {path}")


def run_external_models():
    print("Executing external models before prototype...")
    python_cmd = [sys.executable]
    print(f"Using Python interpreter for external models: {sys.executable}")
    if FOREX_MODEL_RUNNER.exists():
        print(f"Running Forex Models: {FOREX_MODEL_RUNNER}")
        subprocess.run(
            [*python_cmd, str(FOREX_MODEL_RUNNER)],
            cwd=FOREX_MODEL_RUNNER.parent.parent.parent,
            check=False,
        )

    if CRYPTO_MODEL_RUNNER.exists():
        print(f"Running Crypto Models: {CRYPTO_MODEL_RUNNER}")
        subprocess.run(
            [*python_cmd, str(CRYPTO_MODEL_RUNNER)],
            cwd=CRYPTO_MODEL_RUNNER.parent.parent,
            check=False,
        )


def run_step(name, fn):
    try:
        result = fn()
        print(f"[OK] {name}: {result}")
        return {"step": name, "status": "OK", "result": result}
    except Exception as error:
        traceback.print_exc()
        log_system_error("run_prototype", error, step=name)
        print(f"[ERROR] {name}: {error}", file=sys.stderr)
        return {"step": name, "status": "ERROR", "error": str(error)}


def append_run_history(entry):
    append_jsonl(PROTOTYPE_RUN_HISTORY_PATH, entry)


def main():
    ensure_dirs()
    
    # 1. Run startup schema validation & migration
    legacy_state_detected, state_initialized = check_and_migrate_all_states()
    
    # 1b. Restore/migrate any archived snapshots in legacy_snapshots
    scan_and_migrate_archive_snapshots()
    
    # 2. Check settings and warn if regime filters are empty in realistic mode
    settings = load_settings()
    is_realistic = settings.get("realistic_paper_mode", False)
    allowed_regimes = settings.get("allowed_market_regimes", [])
    blocked_regimes = settings.get("blocked_market_regimes", [])
    if is_realistic and not allowed_regimes and not blocked_regimes:
        print(
            "[WARNING] realistic_paper_mode is True, but allowed_market_regimes and "
            "blocked_market_regimes are both empty! All regimes (including UNKNOWN) will be allowed.",
            file=sys.stderr
        )

    run_models = "--run-models" in sys.argv
    started_at = utc_now_iso()
    steps = []

    if run_models:
        steps.append(run_step("run_models", run_external_models))

    steps.append(run_step("replay", run_replay_sync))
    steps.append(run_step("ingest", ingest_all))

    # Wrapped steps for file validation assertions
    def open_and_verify():
        res = open_from_signals()
        verify_file_existence(OPEN_PATH, CLOSED_PATH, DECISION_PATH, BLOCKED_PATH)
        return res

    steps.append(run_step("open", open_and_verify))
    steps.append(run_step("update", update_open_trades))

    def export_and_verify():
        res = export_ai_dataset()
        verify_file_existence(
            AI_DIR / "ai_training_dataset.csv",
            AI_DIR / "blocked_trade_dataset.csv",
            AI_DIR / "blocked_trade_analytics.csv",
            AI_DIR / "signal_observation_dataset.csv",
            AI_DIR / "cluster_opportunity_dataset.csv"
        )
        return res

    steps.append(run_step("export_ai", export_and_verify))
    steps.append(run_step("summary", write_summary))

    failed = [step for step in steps if step["status"] == "ERROR"]
    finished_at = utc_now_iso()
    status = "ERROR" if failed else "OK"

    ingested = next((s.get("result") for s in steps if s["step"] == "ingest"), None)
    opened = next((s.get("result") for s in steps if s["step"] == "open"), None)
    closed = next((s.get("result") for s in steps if s["step"] == "update"), None)
    dataset_stats = next((s.get("result") for s in steps if s["step"] == "export_ai"), None) or {}

    summary_lines = [
        "Prototype run completed" if not failed else f"Prototype run completed with {len(failed)} error(s)",
        f"Signals ingested: {ingested}",
        f"Paper trades opened: {opened}",
        f"Paper trades closed: {closed}",
        f"AI training rows: {dataset_stats.get('training_rows', 'N/A')}",
        f"AI observation rows: {dataset_stats.get('observation_rows', 'N/A')}",
    ]
    stdout_text = "\n".join(summary_lines)
    print(stdout_text)

    if failed:
        for step in failed:
            print(f"  - {step['step']}: {step.get('error')}", file=sys.stderr)

    append_run_history(
        {
            "market": "prototype",
            "runSource": "manual",
            "startedAt": started_at,
            "finishedAt": finished_at,
            "status": status,
            "runModels": run_models,
            "failedStep": failed[0]["step"] if failed else None,
            "steps": steps,
            "stdout": stdout_text,
            "stderr": "\n".join(f"{s['step']}: {s.get('error')}" for s in failed),
            "code": 1 if failed else 0,
            "signal": None,
            "schemaVersion": "2.0.0",
            "configHash": get_config_hash(),
            "stateInitialized": state_initialized,
            "legacyStateDetected": legacy_state_detected,
        }
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
