from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pymongo import MongoClient


BASE_DIR = Path(__file__).resolve().parent
SQLITE_PATH = BASE_DIR / "reading_sources.db"
REVIEWS_JSON_PATH = BASE_DIR / "mongo_reviews.json"
RECOMMENDATIONS_JSON_PATH = BASE_DIR / "mongo_recommendations.json"


def _build_users() -> pd.DataFrame:
    countries = ["Romania", "Moldova", "Italy", "France", "Germany"]
    favorite_categories = ["Fiction", "Science", "History", "Fantasy", "Business"]
    names = [
        "Ana Popescu",
        "Mihai Ionescu",
        "Elena Rusu",
        "Victor Marin",
        "Sofia Luca",
        "Daria Pavel",
        "Ion Sandu",
        "Maria Negru",
        "Adrian Ceban",
        "Irina Munteanu",
        "Andrei Toma",
        "Natalia Costin",
    ]
    rows = []
    for idx, name in enumerate(names, start=1):
        rows.append(
            {
                "user_id": idx,
                "name": name,
                "age": random.randint(16, 62),
                "country": random.choice(countries),
                "registration_date": str(date(2024, 1, 1) + timedelta(days=random.randint(0, 420))),
                "favorite_category": random.choice(favorite_categories),
            }
        )
    return pd.DataFrame(rows)


def _build_books() -> pd.DataFrame:
    books = [
        ("Atomic Habits", "James Clear", "Business", 2018),
        ("Sapiens", "Yuval Noah Harari", "History", 2011),
        ("The Hobbit", "J. R. R. Tolkien", "Fantasy", 1937),
        ("Clean Code", "Robert C. Martin", "Science", 2008),
        ("1984", "George Orwell", "Fiction", 1949),
        ("Deep Work", "Cal Newport", "Business", 2016),
        ("A Brief History of Time", "Stephen Hawking", "Science", 1988),
        ("The Book Thief", "Markus Zusak", "Fiction", 2005),
        ("The Silk Roads", "Peter Frankopan", "History", 2015),
        ("Mistborn", "Brandon Sanderson", "Fantasy", 2006),
    ]
    rows = []
    for idx, (title, author, category, year) in enumerate(books, start=1):
        rows.append(
            {
                "book_id": idx,
                "title": title,
                "author": author,
                "category": category,
                "publication_year": year,
            }
        )
    return pd.DataFrame(rows)


def _build_reading_history(users_df: pd.DataFrame, books_df: pd.DataFrame) -> pd.DataFrame:
    start_day = date(2025, 1, 1)
    rows = []
    reading_id = 1
    for user_id in users_df["user_id"]:
        for _ in range(random.randint(8, 16)):
            book = books_df.sample(1).iloc[0]
            minutes = random.randint(20, 240)
            days_offset = random.randint(0, 360)
            completed = random.choice([0, 1, 1, 1])
            rows.append(
                {
                    "id": reading_id,
                    "user_id": user_id,
                    "book_id": int(book["book_id"]),
                    "reading_time": minutes,
                    "date": str(start_day + timedelta(days=days_offset)),
                    "completed": completed,
                }
            )
            reading_id += 1
    return pd.DataFrame(rows)


def _build_ratings(history_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rating_id = 1
    review_templates = [
        "Very engaging and easy to follow.",
        "Solid choice for weekend reading.",
        "Useful ideas and memorable examples.",
        "Interesting themes, but pacing is uneven.",
        "A book I would recommend to friends.",
    ]
    for _, row in history_df.iterrows():
        if random.random() < 0.8:
            rows.append(
                {
                    "id": rating_id,
                    "user_id": int(row["user_id"]),
                    "book_id": int(row["book_id"]),
                    "rating": random.randint(2, 5),
                    "review": random.choice(review_templates),
                }
            )
            rating_id += 1
    return pd.DataFrame(rows)


def _build_review_documents(books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> list[dict]:
    sentiment_map = {2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    documents = []
    for idx, row in ratings_df.iterrows():
        book_row = books_df.loc[books_df["book_id"] == row["book_id"]].iloc[0]
        documents.append(
            {
                "review_id": idx + 1,
                "user_id": int(row["user_id"]),
                "book_id": int(row["book_id"]),
                "book_title": book_row["title"],
                "comment": row["review"],
                "sentiment": sentiment_map.get(int(row["rating"]), "neutral"),
                "keywords": [book_row["category"].lower(), "reading", "engagement"],
            }
        )
    return documents


def _build_recommendation_documents(users_df: pd.DataFrame, books_df: pd.DataFrame) -> list[dict]:
    documents = []
    for _, user in users_df.iterrows():
        favorite = user["favorite_category"]
        candidates = books_df.loc[books_df["category"] == favorite]["title"].tolist()
        documents.append(
            {
                "user_id": int(user["user_id"]),
                "favorite_category": favorite,
                "recommended_books": candidates[:3],
                "reason": f"Readers with strong interest in {favorite} tend to finish these books.",
            }
        )
    return documents


def _write_sqlite(users_df: pd.DataFrame, books_df: pd.DataFrame, history_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    connection = sqlite3.connect(SQLITE_PATH)
    try:
        users_df.to_sql("users", connection, if_exists="replace", index=False)
        books_df.to_sql("books", connection, if_exists="replace", index=False)
        history_df.to_sql("reading_history", connection, if_exists="replace", index=False)
        ratings_df.to_sql("ratings", connection, if_exists="replace", index=False)
    finally:
        connection.close()


def _write_json(path: Path, documents: list[dict]) -> None:
    path.write_text(json.dumps(documents, indent=2), encoding="utf-8")


def _write_mongodb(documents: list[dict], recommendations: list[dict]) -> None:
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        return

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    try:
        db = client[os.getenv("MONGODB_DB", "reading_analytics")]
        db.reviews.delete_many({})
        db.recommendations.delete_many({})
        if documents:
            db.reviews.insert_many(documents)
        if recommendations:
            db.recommendations.insert_many(recommendations)
    finally:
        client.close()


def main() -> None:
    random.seed(42)
    users_df = _build_users()
    books_df = _build_books()
    history_df = _build_reading_history(users_df, books_df)
    ratings_df = _build_ratings(history_df)

    review_documents = _build_review_documents(books_df, ratings_df)
    recommendation_documents = _build_recommendation_documents(users_df, books_df)

    _write_sqlite(users_df, books_df, history_df, ratings_df)
    _write_json(REVIEWS_JSON_PATH, review_documents)
    _write_json(RECOMMENDATIONS_JSON_PATH, recommendation_documents)
    _write_mongodb(review_documents, recommendation_documents)

    print(f"SQLite source created at: {SQLITE_PATH}")
    print(f"Review documents exported to: {REVIEWS_JSON_PATH}")
    print(f"Recommendation documents exported to: {RECOMMENDATIONS_JSON_PATH}")


if __name__ == "__main__":
    main()
