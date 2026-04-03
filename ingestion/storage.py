from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOGS_DIR / "ingestion_log.jsonl"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_raw_snapshot(source_type: str, source_name: str, payload: dict[str, Any]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = RAW_DIR / source_type / source_name
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "source_type": source_type,
        "source_name": source_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }

    file_path = target_dir / f"{timestamp_utc()}.json"
    file_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    save_to_mongo(snapshot)
    append_log("ingestion", "success", f"Сохранён raw-снимок {source_name}", {"path": str(file_path)})
    return file_path


def append_log(event_type: str, status: str, message: str, details: dict[str, Any] | None = None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "status": status,
        "message": message,
        "details": details or {},
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def recent_snapshots(limit: int = 10) -> list[dict[str, Any]]:
    files = sorted(RAW_DIR.glob("**/*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload = data.get("payload", {})
        records = payload.get("records", [])
        items.append(
            {
                "source_type": data.get("source_type"),
                "source_name": data.get("source_name"),
                "fetched_at": data.get("fetched_at"),
                "records_count": len(records) if isinstance(records, list) else 0,
                "path": str(path),
            }
        )
    return items


def recent_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    result: list[dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def save_to_mongo(snapshot: dict[str, Any]) -> bool:
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        return False

    db_name = os.getenv("MONGODB_DB", "reading_analytics")
    collection_name = os.getenv("MONGODB_RAW_COLLECTION", "raw_ingestion")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    try:
        client.admin.command("ping")
        client[db_name][collection_name].insert_one(snapshot)
        return True
    except Exception:
        return False
    finally:
        client.close()
