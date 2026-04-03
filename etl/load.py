from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from etl.extract import extract_all
from etl.transform import save_staging_files, transform_to_star_schema


WAREHOUSE_DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "reading_dwh.db"
SCHEMA_PATH = PROJECT_ROOT / "dwh" / "schema.sql"


def _apply_schema(connection: sqlite3.Connection) -> None:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    connection.executescript(schema)


def load_into_warehouse(transformed: dict) -> Path:
    WAREHOUSE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(WAREHOUSE_DB_PATH)
    try:
        _apply_schema(connection)
        for table_name in [
            "dim_user",
            "dim_category",
            "dim_book",
            "dim_source",
            "dim_time",
            "fact_reading_activity",
            "fact_external_book_observation",
        ]:
            transformed[table_name].to_sql(table_name, connection, if_exists="append", index=False)
    finally:
        connection.close()
    return WAREHOUSE_DB_PATH


def run_pipeline() -> Path:
    extracted = extract_all()
    transformed = transform_to_star_schema(extracted)
    save_staging_files(transformed)
    warehouse_path = load_into_warehouse(transformed)
    return warehouse_path


if __name__ == "__main__":
    path = run_pipeline()
    print(f"Warehouse loaded successfully: {path}")
