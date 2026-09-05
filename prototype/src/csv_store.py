import csv
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path: Path, timeout_seconds=10, stale_lock_seconds=300):
    lock_path = path.with_suffix(path.suffix + ".lock")
    started = time.time()
    handle = None
    while handle is None:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except (FileNotFoundError, PermissionError):
                if time.time() - started > timeout_seconds:
                    raise TimeoutError(f"Could not acquire file lock (stat failed): {lock_path}")
                time.sleep(0.1)
                continue

            # Recover from orphaned lock files left behind by interrupted runs.
            if lock_age > stale_lock_seconds:
                try:
                    lock_path.unlink()
                    continue
                except (FileNotFoundError, PermissionError):
                    continue

            if time.time() - started > timeout_seconds:
                raise TimeoutError(f"Could not acquire file lock: {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        if handle is not None:
            os.close(handle)
        try:
            lock_path.unlink()
        except (FileNotFoundError, PermissionError):
            pass


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_atomic(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    if "schema_version" in fieldnames:
        for row in rows:
            if "schema_version" not in row or not row["schema_version"]:
                row["schema_version"] = "2.0.0"
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    with file_lock(path):
        try:
            with temp_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

            last_error = None
            for _ in range(20):
                try:
                    temp_path.replace(path)
                    return
                except PermissionError as error:
                    last_error = error
                    time.sleep(0.1)

            raise PermissionError(
                f"Could not replace {path}. Close any app that has this CSV open, then retry."
            ) from last_error
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def append_unique_csv(path: Path, new_rows, fieldnames, key_field):
    existing = read_csv(path)
    seen = {row.get(key_field) for row in existing}
    accepted = [row for row in new_rows if row.get(key_field) not in seen]
    if accepted:
        write_csv_atomic(path, existing + accepted, fieldnames)
    return len(accepted)


def append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def cleanup_stale_temp_files(root: Path, max_age_seconds=3600):
    """Remove orphaned .tmp files left by interrupted atomic writes."""
    if not root.exists():
        return 0
    now = time.time()
    removed = 0
    for tmp in root.rglob("*.tmp"):
        try:
            if now - tmp.stat().st_mtime > max_age_seconds:
                tmp.unlink()
                removed += 1
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return removed
