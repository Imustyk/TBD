from __future__ import annotations

from typing import Any

import requests

from ingestion.storage import append_log, save_raw_snapshot


OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"


def fetch_openlibrary_books(query: str = "machine learning", limit: int = 10) -> dict[str, Any]:
    params = {"q": query, "limit": limit}
    response = requests.get(OPEN_LIBRARY_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    docs = data.get("docs", [])
    records = []
    for item in docs:
        records.append(
            {
                "title": item.get("title"),
                "author_name": item.get("author_name", []),
                "first_publish_year": item.get("first_publish_year"),
                "edition_count": item.get("edition_count"),
                "subject": item.get("subject", [])[:5],
                "language": item.get("language", [])[:3],
                "isbn": item.get("isbn", [])[:3],
                "ratings_average": item.get("ratings_average"),
                "ratings_count": item.get("ratings_count"),
            }
        )

    payload = {
        "query": query,
        "source_url": response.url,
        "records": records,
        "num_found": data.get("numFound", len(records)),
    }
    snapshot_path = save_raw_snapshot("api", "openlibrary", payload)
    append_log("api_ingestion", "success", "Загрузка данных из Open Library завершена", {"query": query, "count": len(records)})
    return {
        "source": "Open Library API",
        "count": len(records),
        "path": str(snapshot_path),
        "records": records,
    }


if __name__ == "__main__":
    result = fetch_openlibrary_books()
    print(result)
