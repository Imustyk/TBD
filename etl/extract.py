from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_SQLITE_URL = f"sqlite:///{DATA_DIR / 'reading_sources.db'}"
DEFAULT_REVIEWS_JSON = DATA_DIR / "mongo_reviews.json"
DEFAULT_RECOMMENDATIONS_JSON = DATA_DIR / "mongo_recommendations.json"


def extract_sqlite_data(sqlite_url: str = DEFAULT_SQLITE_URL) -> dict[str, pd.DataFrame]:
    engine = create_engine(sqlite_url)
    tables = ["users", "books", "reading_history", "ratings"]
    result: dict[str, pd.DataFrame] = {}
    with engine.connect() as connection:
        for table in tables:
            result[table] = pd.read_sql_table(table, connection)
    return result


def _load_json_documents(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def extract_mongo_data(
    mongo_uri: str | None = None,
    db_name: str | None = None,
    fallback_reviews_path: Path = DEFAULT_REVIEWS_JSON,
    fallback_recommendations_path: Path = DEFAULT_RECOMMENDATIONS_JSON,
) -> dict[str, pd.DataFrame]:
    mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
    db_name = db_name or os.getenv("MONGODB_DB", "reading_analytics")

    if not mongo_uri:
        return {
            "reviews": _load_json_documents(fallback_reviews_path),
            "recommendations": _load_json_documents(fallback_recommendations_path),
        }

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    try:
        database = client[db_name]
        reviews = pd.DataFrame(list(database.reviews.find({}, {"_id": 0})))
        recommendations = pd.DataFrame(list(database.recommendations.find({}, {"_id": 0})))
        return {"reviews": reviews, "recommendations": recommendations}
    finally:
        client.close()


def extract_raw_ingestion_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    snapshots: list[dict] = []
    records: list[dict] = []

    if raw_dir.exists():
        for path in sorted(raw_dir.glob("**/*.json")):
            with path.open("r", encoding="utf-8") as file:
                snapshot = json.load(file)

            payload = snapshot.get("payload", {})
            snapshot_record = {
                "source_type": snapshot.get("source_type"),
                "source_name": snapshot.get("source_name"),
                "fetched_at": snapshot.get("fetched_at"),
                "raw_file_path": str(path),
                "source_url": payload.get("source_url"),
                "query": payload.get("query"),
                "records_count": len(payload.get("records", [])),
            }
            snapshots.append(snapshot_record)

            for index, item in enumerate(payload.get("records", []), start=1):
                records.append(
                    {
                        **snapshot_record,
                        "record_index": index,
                        "record_payload": json.dumps(item, ensure_ascii=False),
                    }
                )

    return {
        "raw_ingestion_snapshots": pd.DataFrame(snapshots),
        "raw_ingestion_records": pd.DataFrame(records),
    }


def extract_all(sqlite_url: str = DEFAULT_SQLITE_URL) -> dict[str, pd.DataFrame]:
    sql_data = extract_sqlite_data(sqlite_url)
    mongo_data = extract_mongo_data()
    raw_ingestion_data = extract_raw_ingestion_data()
    return {**sql_data, **mongo_data, **raw_ingestion_data}


if __name__ == "__main__":
    extracted = extract_all()
    for name, frame in extracted.items():
        print(f"{name}: {frame.shape}")
