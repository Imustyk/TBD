from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.api_ingestion import fetch_openlibrary_books
from ingestion.scraper_ingestion import scrape_books_catalog
from ingestion.storage import RAW_DIR, recent_logs, recent_snapshots


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_api_ingestion(query: str, limit: int) -> dict[str, Any]:
    return fetch_openlibrary_books(query=query, limit=limit)


def run_scraping_ingestion(url: str) -> dict[str, Any]:
    return scrape_books_catalog(url=url)


def ingestion_status() -> dict[str, Any]:
    raw_files = list(RAW_DIR.glob("**/*.json"))
    return {
        "raw_snapshots": len(raw_files),
        "recent_snapshots": recent_snapshots(),
        "recent_logs": recent_logs(),
    }
