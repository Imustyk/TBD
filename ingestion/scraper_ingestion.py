from __future__ import annotations

from typing import Any

import requests
from bs4 import BeautifulSoup

from ingestion.storage import append_log, save_raw_snapshot


BOOKS_TOSCRAPE_URL = "https://books.toscrape.com/catalogue/page-1.html"


def scrape_books_catalog(url: str = BOOKS_TOSCRAPE_URL) -> dict[str, Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    items = soup.select("article.product_pod")
    records = []
    for item in items:
        title_node = item.select_one("h3 a")
        price_node = item.select_one(".price_color")
        stock_node = item.select_one(".availability")
        rating_node = item.select_one("p.star-rating")

        records.append(
            {
                "title": title_node["title"] if title_node and title_node.has_attr("title") else None,
                "book_url": title_node["href"] if title_node and title_node.has_attr("href") else None,
                "price": price_node.get_text(strip=True) if price_node else None,
                "availability": stock_node.get_text(" ", strip=True) if stock_node else None,
                "rating_class": " ".join(rating_node.get("class", [])) if rating_node else None,
            }
        )

    payload = {
        "source_url": url,
        "records": records,
        "html_length": len(response.text),
    }
    snapshot_path = save_raw_snapshot("scraping", "books_toscrape", payload)
    append_log("scraping", "success", "Скраппинг каталога книг завершён", {"count": len(records)})
    return {
        "source": "Books to Scrape",
        "count": len(records),
        "path": str(snapshot_path),
        "records": records,
    }


if __name__ == "__main__":
    result = scrape_books_catalog()
    print(result)
