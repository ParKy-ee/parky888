from datetime import datetime, timezone

from .csv_store import append_jsonl
from .paths import AUDIT_DIR, SYSTEM_ERRORS_PATH


DECISION_AUDIT_PATH = AUDIT_DIR / "decision_audit_log.jsonl"
EXECUTION_AUDIT_PATH = AUDIT_DIR / "execution_audit_log.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def audit_decision(event_type, payload):
    append_jsonl(
        DECISION_AUDIT_PATH,
        {
            "logged_at": utc_now(),
            "event_type": event_type,
            **payload,
        },
    )


def audit_execution(event_type, payload):
    append_jsonl(
        EXECUTION_AUDIT_PATH,
        {
            "logged_at": utc_now(),
            "event_type": event_type,
            **payload,
        },
    )


def log_system_error(component, error, **extra):
    try:
        append_jsonl(
            SYSTEM_ERRORS_PATH,
            {
                "logged_at": utc_now(),
                "component": component,
                "error": str(error),
                **extra,
            },
        )
    except Exception:
        pass
