from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "data" / "staging"


def _safe_divide(left: pd.Series, right: pd.Series | int | float) -> pd.Series:
    if isinstance(right, pd.Series):
        return left.div(right.where(right != 0, 1))
    safe_right = right if right != 0 else 1
    return left.div(safe_right)


def _future_event_count_within_days(group: pd.DataFrame, days: int) -> pd.Series:
    ordered = group.sort_values("date").copy()
    dates = ordered["date"].to_numpy(dtype="datetime64[ns]")
    result = []
    for idx, current_date in enumerate(dates):
        upper_bound = current_date + pd.Timedelta(days=days)
        future_mask = (dates > current_date) & (dates <= upper_bound)
        result.append(int(future_mask.sum()))
    return pd.Series(result, index=ordered.index)


def _future_repeat_flag_within_days(group: pd.DataFrame, days: int) -> pd.Series:
    return _future_event_count_within_days(group, days).gt(0).astype(float)


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _parse_price(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(\d+[.,]\d+|\d+)", str(value).replace(",", "."))
    return float(match.group(1)) if match else None


def _rating_from_class(value: object) -> float | None:
    rating_map = {"One": 1.0, "Two": 2.0, "Three": 3.0, "Four": 4.0, "Five": 5.0}
    if value is None or pd.isna(value):
        return None
    for token in str(value).split():
        if token in rating_map:
            return rating_map[token]
    return None


def _prepare_external_records(raw_records: pd.DataFrame) -> pd.DataFrame:
    if raw_records.empty:
        return pd.DataFrame(
            columns=[
                "source_type",
                "source_name",
                "fetched_at",
                "raw_file_path",
                "source_url",
                "query",
                "external_title",
                "external_author",
                "publication_year",
                "external_category",
                "external_rating",
                "price",
                "availability",
                "popularity_signal",
                "title_key",
            ]
        )

    rows: list[dict] = []
    for _, row in raw_records.iterrows():
        payload = json.loads(row["record_payload"])
        base = {
            "source_type": row["source_type"],
            "source_name": row["source_name"],
            "fetched_at": row["fetched_at"],
            "raw_file_path": row["raw_file_path"],
            "source_url": row["source_url"],
            "query": row["query"],
        }

        if row["source_name"] == "openlibrary":
            title = payload.get("title")
            author = ", ".join(payload.get("author_name", [])[:3]) or "Не указан"
            subjects = payload.get("subject", [])
            category = subjects[0] if subjects else "API-книги"
            external_rating = payload.get("ratings_average")
            ratings_count = payload.get("ratings_count") or 0
            edition_count = payload.get("edition_count") or 0
            popularity_signal = round(edition_count * 0.7 + ratings_count * 0.2 + (external_rating or 0) * 0.1, 3)
            rows.append(
                {
                    **base,
                    "external_title": title,
                    "external_author": author,
                    "publication_year": payload.get("first_publish_year"),
                    "external_category": category,
                    "external_rating": external_rating,
                    "price": None,
                    "availability": None,
                    "popularity_signal": popularity_signal,
                    "title_key": _normalize_text(title),
                }
            )
        elif row["source_name"] == "books_toscrape":
            title = payload.get("title")
            external_rating = _rating_from_class(payload.get("rating_class"))
            price = _parse_price(payload.get("price"))
            availability = payload.get("availability")
            in_stock_bonus = 3 if availability and "In stock" in str(availability) else 0
            affordability_bonus = max(0, 60 - (price or 0)) * 0.05
            popularity_signal = round((external_rating or 0) * 8 + in_stock_bonus + affordability_bonus, 3)
            rows.append(
                {
                    **base,
                    "external_title": title,
                    "external_author": "Не указан",
                    "publication_year": None,
                    "external_category": "Скраппинг-каталог",
                    "external_rating": external_rating,
                    "price": price,
                    "availability": availability,
                    "popularity_signal": popularity_signal,
                    "title_key": _normalize_text(title),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["fetched_at"] = pd.to_datetime(frame["fetched_at"])
        frame["date_id"] = frame["fetched_at"].dt.strftime("%Y%m%d").astype(int)
    return frame


def _attach_external_book_matches(external_records: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    if external_records.empty:
        return external_records.assign(matched_book_id=pd.Series(dtype="Int64"))

    title_lookup = books[["book_id", "title_key", "category"]].copy()
    matches: list[int | None] = []
    for _, row in external_records.iterrows():
        title_key = row["title_key"]
        match = title_lookup.loc[title_lookup["title_key"] == title_key]
        if match.empty:
            match = title_lookup.loc[
                title_lookup["title_key"].apply(
                    lambda book_key: bool(book_key)
                    and bool(title_key)
                    and (book_key in title_key or title_key in book_key)
                    and min(len(book_key), len(title_key)) >= 5
                )
            ]
        matches.append(int(match.iloc[0]["book_id"]) if not match.empty else None)

    result = external_records.copy()
    result["matched_book_id"] = pd.Series(matches, dtype="Int64")
    return result


def transform_to_star_schema(extracted: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    users = extracted["users"].copy()
    books = extracted["books"].copy()
    history = extracted["reading_history"].copy()
    ratings = extracted["ratings"].copy()
    reviews = extracted.get("reviews", pd.DataFrame()).copy()
    raw_ingestion_records = extracted.get("raw_ingestion_records", pd.DataFrame()).copy()

    history["date"] = pd.to_datetime(history["date"])
    external_records = _prepare_external_records(raw_ingestion_records)

    books["title_key"] = books["title"].map(_normalize_text)
    external_records = _attach_external_book_matches(external_records, books)

    if not ratings.empty:
        ratings_agg = (
            ratings.groupby("book_id", as_index=False)
            .agg(average_rating=("rating", "mean"), ratings_count=("rating", "count"))
        )
    else:
        ratings_agg = pd.DataFrame(columns=["book_id", "average_rating", "ratings_count"])

    reads_per_book = history.groupby("book_id", as_index=False).agg(total_reads=("id", "count"))
    user_activity = history.groupby("user_id", as_index=False).agg(
        total_sessions=("id", "count"),
        total_reading_time=("reading_time", "sum"),
        avg_reading_time=("reading_time", "mean"),
    )

    if external_records.empty:
        external_book_signals = pd.DataFrame(columns=["title_key", "external_signal_score", "external_observation_count"])
        dim_source = pd.DataFrame(columns=["source_id", "source_name", "source_type", "base_url"])
        fact_external_book_observation = pd.DataFrame(
            columns=[
                "source_id",
                "date_id",
                "book_id",
                "external_title",
                "external_author",
                "external_category",
                "publication_year",
                "external_rating",
                "price",
                "availability",
                "popularity_signal",
                "raw_file_path",
                "fetched_at",
            ]
        )
    else:
        external_book_signals = (
            external_records.dropna(subset=["matched_book_id"])
            .groupby("matched_book_id", as_index=False)
            .agg(
                external_signal_score=("popularity_signal", "mean"),
                external_observation_count=("external_title", "count"),
            )
            .round({"external_signal_score": 3})
            .rename(columns={"matched_book_id": "book_id"})
        )

        dim_source = (
            external_records[["source_name", "source_type", "source_url"]]
            .sort_values(["source_type", "source_name", "source_url"])
            .drop_duplicates(subset=["source_name", "source_type"], keep="first")
            .reset_index(drop=True)
            .assign(source_id=lambda df: df.index + 1)
            .rename(columns={"source_url": "base_url"})
        )[["source_id", "source_name", "source_type", "base_url"]]

    category_candidates = pd.concat(
        [
            books.rename(columns={"category": "category_name"})[["category_name"]],
            external_records[["external_category"]].rename(columns={"external_category": "category_name"}),
        ],
        ignore_index=True,
    )
    category_map = (
        category_candidates.dropna()
        .drop_duplicates()
        .sort_values("category_name")
        .reset_index(drop=True)
        .assign(category_id=lambda df: df.index + 1)
    )

    dim_category = category_map[["category_id", "category_name"]]
    books = books.merge(category_map, left_on="category", right_on="category_name", how="left")
    books = books.merge(ratings_agg, on="book_id", how="left").merge(reads_per_book, on="book_id", how="left")
    books = books.merge(external_book_signals, on="book_id", how="left")
    books["average_rating"] = books["average_rating"].fillna(0.0).round(2)
    books["total_reads"] = books["total_reads"].fillna(0).astype(int)
    books["external_signal_score"] = books["external_signal_score"].fillna(0.0).round(3)
    books["external_observation_count"] = books["external_observation_count"].fillna(0).astype(int)

    users = users.merge(user_activity, on="user_id", how="left")
    users["total_sessions"] = users["total_sessions"].fillna(0).astype(int)
    users["total_reading_time"] = users["total_reading_time"].fillna(0).astype(int)
    users["avg_reading_time"] = users["avg_reading_time"].fillna(0.0)
    users["activity_segment"] = pd.cut(
        users["total_sessions"],
        bins=[-1, 5, 10, 1000],
        labels=["low", "medium", "high"],
    ).astype(str)

    dim_user = users[
        [
            "user_id",
            "name",
            "age",
            "country",
            "registration_date",
            "favorite_category",
            "activity_segment",
        ]
    ].rename(columns={"name": "full_name"})

    dim_book = books[
        [
            "book_id",
            "title",
            "author",
            "category_id",
            "publication_year",
            "average_rating",
            "total_reads",
            "external_signal_score",
            "external_observation_count",
        ]
    ]

    time_source = pd.concat(
        [
            history.rename(columns={"date": "time_value"})[["time_value"]],
            external_records.rename(columns={"fetched_at": "time_value"})[["time_value"]],
        ],
        ignore_index=True,
    ).dropna()
    time_source["time_value"] = pd.to_datetime(time_source["time_value"], utc=True).dt.tz_localize(None)

    dim_time = (
        time_source.sort_values("time_value")
        .assign(
            full_date=lambda df: df["time_value"].dt.strftime("%Y-%m-%d"),
            date_id=lambda df: df["time_value"].dt.strftime("%Y%m%d").astype(int),
            year=lambda df: df["time_value"].dt.year,
            quarter=lambda df: df["time_value"].dt.quarter,
            month=lambda df: df["time_value"].dt.month,
            month_name=lambda df: df["time_value"].dt.strftime("%B"),
            week=lambda df: df["time_value"].dt.isocalendar().week.astype(int),
            day=lambda df: df["time_value"].dt.day,
            weekday=lambda df: df["time_value"].dt.strftime("%A"),
        )
        .drop_duplicates(subset=["date_id"])
        [["date_id", "full_date", "year", "quarter", "month", "month_name", "week", "day", "weekday"]]
        .reset_index(drop=True)
    )

    latest_rating = (
        ratings.sort_values("id")
        .groupby(["user_id", "book_id"], as_index=False)
        .tail(1)[["user_id", "book_id", "rating"]]
        if not ratings.empty
        else pd.DataFrame(columns=["user_id", "book_id", "rating"])
    )

    review_counts = (
        reviews.groupby("book_id", as_index=False).agg(review_count=("review_id", "count"))
        if not reviews.empty and "review_id" in reviews.columns
        else pd.DataFrame(columns=["book_id", "review_count"])
    )

    history = history.sort_values(["date", "id"]).reset_index(drop=True)
    history["reads_count"] = history.groupby("book_id").cumcount()
    history["user_past_sessions"] = history.groupby("user_id").cumcount()
    history["past_avg_reading_time"] = (
        history.groupby("user_id")["reading_time"]
        .transform(lambda s: s.shift().expanding().mean())
        .fillna(history["reading_time"].median())
    )
    history["book_popularity_target"] = (
        history.groupby("book_id", group_keys=False).apply(_future_event_count_within_days, days=30).astype(float)
    )
    history["user_read_probability_target"] = (
        history.groupby(["user_id", "book_id"], group_keys=False)
        .apply(_future_repeat_flag_within_days, days=60)
        .astype(float)
    )

    fact = history.merge(latest_rating, on=["user_id", "book_id"], how="left")
    fact = fact.merge(
        books[
            [
                "book_id",
                "category_id",
                "average_rating",
                "total_reads",
                "external_signal_score",
                "external_observation_count",
            ]
        ],
        on="book_id",
        how="left",
    )
    fact = fact.merge(review_counts, on="book_id", how="left")
    fact["review_count"] = fact["review_count"].fillna(0).astype(int)
    fact["date_id"] = fact["date"].dt.strftime("%Y%m%d").astype(int)
    fact["external_signal_score"] = fact["external_signal_score"].fillna(0.0)
    fact["user_activity_score"] = (
        fact["user_past_sessions"].fillna(0) * 0.55
        + _safe_divide(fact["past_avg_reading_time"].fillna(0), 60) * 0.35
        + _safe_divide(fact["external_signal_score"], 100) * 0.10
    ).round(3)
    fact["book_popularity_target"] = fact["book_popularity_target"].astype(float).round(3)
    fact["user_read_probability_target"] = fact["user_read_probability_target"].astype(float).round(3)
    fact["rating"] = fact["rating"].fillna(fact["average_rating"]).fillna(0).round(2)

    fact_reading_activity = fact[
        [
            "user_id",
            "book_id",
            "category_id",
            "date_id",
            "reading_time",
            "rating",
            "reads_count",
            "user_activity_score",
            "book_popularity_target",
            "user_read_probability_target",
        ]
    ].copy()

    if not external_records.empty:
        book_match_map = books[["book_id", "title_key"]].copy()
        fact_external_book_observation = (
            external_records.merge(dim_source, on=["source_name", "source_type"], how="left")
        )[
            [
                "source_id",
                "date_id",
                "matched_book_id",
                "external_title",
                "external_author",
                "external_category",
                "publication_year",
                "external_rating",
                "price",
                "availability",
                "popularity_signal",
                "raw_file_path",
                "fetched_at",
            ]
        ]
        fact_external_book_observation = fact_external_book_observation.rename(columns={"matched_book_id": "book_id"})
        fact_external_book_observation["book_id"] = fact_external_book_observation["book_id"].astype("Int64")

    feature_store = (
        fact_reading_activity.merge(
            dim_user[["user_id", "age", "country", "favorite_category", "activity_segment"]],
            on="user_id",
            how="left",
        )
        .merge(
            dim_book[
                [
                    "book_id",
                    "title",
                    "author",
                    "publication_year",
                    "average_rating",
                    "total_reads",
                    "external_signal_score",
                    "external_observation_count",
                ]
            ],
            on="book_id",
            how="left",
        )
        .merge(dim_category, on="category_id", how="left")
    )

    feature_store["category_name"] = feature_store["category_name"].fillna("Unknown")
    feature_store["country"] = feature_store["country"].fillna("Unknown")
    feature_store["favorite_category"] = feature_store["favorite_category"].fillna("Unknown")

    return {
        "dim_user": dim_user,
        "dim_book": dim_book,
        "dim_time": dim_time,
        "dim_category": dim_category,
        "dim_source": dim_source,
        "fact_reading_activity": fact_reading_activity,
        "fact_external_book_observation": fact_external_book_observation,
        "feature_store": feature_store,
        "external_book_catalog": external_records,
    }


def save_staging_files(transformed: dict[str, pd.DataFrame], output_dir: Path = STAGING_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in transformed.items():
        frame.to_csv(output_dir / f"{table_name}.csv", index=False)


if __name__ == "__main__":
    from etl.extract import extract_all

    data = extract_all()
    transformed = transform_to_star_schema(data)
    save_staging_files(transformed)
    for name, frame in transformed.items():
        print(f"{name}: {frame.shape}")
