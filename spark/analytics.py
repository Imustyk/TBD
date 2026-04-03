from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spark.session import build_spark_session


STAGING_DIR = PROJECT_ROOT / "data" / "staging"
SPARK_OUTPUT_DIR = PROJECT_ROOT / "data" / "warehouse" / "spark_marts"


def _read_staging_csv(spark: object, file_name: str) -> DataFrame:
    path = STAGING_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Staging file not found: {path}")

    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )


def _prepare_feature_store() -> tuple[object, DataFrame]:
    spark = build_spark_session()
    feature_store = _read_staging_csv(spark, "feature_store.csv")
    dim_time = _read_staging_csv(spark, "dim_time.csv")

    df = (
        feature_store.join(dim_time, on="date_id", how="left")
        .withColumn("full_date", F.to_date("full_date"))
        .fillna({"category_name": "Unknown", "country": "Unknown", "favorite_category": "Unknown"})
    )
    df.createOrReplaceTempView("feature_store_enriched")
    return spark, df


def build_top_books_mart(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("title", "category_name")
        .agg(
            F.count("*").alias("sessions_count"),
            F.sum("reading_time").alias("total_reading_time"),
            F.avg("rating").alias("avg_rating"),
            F.avg("external_signal_score").alias("avg_external_signal"),
        )
        .orderBy(F.desc("sessions_count"), F.desc("total_reading_time"))
    )


def build_user_activity_mart(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("user_id", "country", "activity_segment")
        .agg(
            F.count("*").alias("sessions_count"),
            F.sum("reading_time").alias("total_reading_time"),
            F.avg("user_activity_score").alias("avg_activity_score"),
            F.avg("user_read_probability_target").alias("avg_future_read_probability"),
        )
        .orderBy(F.desc("sessions_count"), F.desc("avg_activity_score"))
    )


def build_category_mart(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("category_name")
        .agg(
            F.count("*").alias("sessions_count"),
            F.avg("book_popularity_target").alias("avg_future_popularity"),
            F.avg("rating").alias("avg_rating"),
            F.avg("external_signal_score").alias("avg_external_signal"),
        )
        .orderBy(F.desc("sessions_count"))
    )


def build_monthly_trends_sql(spark: object) -> DataFrame:
    return spark.sql(
        """
        SELECT
            year,
            month,
            month_name,
            COUNT(*) AS sessions_count,
            ROUND(SUM(reading_time), 2) AS total_reading_time,
            ROUND(AVG(rating), 3) AS avg_rating,
            ROUND(AVG(book_popularity_target), 3) AS avg_future_popularity,
            ROUND(AVG(user_read_probability_target), 3) AS avg_future_read_probability
        FROM feature_store_enriched
        GROUP BY year, month, month_name
        ORDER BY year, month
        """
    )


def build_weekday_activity_sql(spark: object) -> DataFrame:
    return spark.sql(
        """
        SELECT
            weekday,
            COUNT(*) AS sessions_count,
            ROUND(SUM(reading_time), 2) AS total_reading_time,
            ROUND(AVG(rating), 3) AS avg_rating,
            ROUND(AVG(user_activity_score), 3) AS avg_user_activity_score
        FROM feature_store_enriched
        GROUP BY weekday
        ORDER BY sessions_count DESC, total_reading_time DESC
        """
    )


def build_correlation_mart(spark: object, df: DataFrame) -> DataFrame:
    correlation_pairs = [
        ("reads_count", "book_popularity_target"),
        ("rating", "book_popularity_target"),
        ("reading_time", "rating"),
        ("user_activity_score", "user_read_probability_target"),
        ("external_signal_score", "book_popularity_target"),
        ("average_rating", "rating"),
    ]

    rows: list[tuple[str, str, float | None, float | None]] = []
    for left, right in correlation_pairs:
        correlation = df.stat.corr(left, right)
        correlation_value = round(float(correlation), 4) if correlation is not None else None
        rows.append(
            (
                left,
                right,
                correlation_value,
                round(abs(correlation_value), 4) if correlation_value is not None else None,
            )
        )

    correlation_df = spark.createDataFrame(
        rows,
        schema=["feature_x", "feature_y", "correlation", "abs_correlation"],
    )
    return correlation_df.orderBy(F.desc("abs_correlation"), F.asc("feature_x"))


def build_ml_feature_mart(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("category_name", "activity_segment")
        .agg(
            F.count("*").alias("samples_count"),
            F.avg("reads_count").alias("avg_reads_count"),
            F.avg("user_activity_score").alias("avg_user_activity_score"),
            F.avg("external_signal_score").alias("avg_external_signal"),
            F.avg("book_popularity_target").alias("avg_future_popularity"),
            F.avg("user_read_probability_target").alias("avg_future_read_probability"),
        )
        .orderBy(F.desc("samples_count"), F.desc("avg_future_popularity"))
    )


def build_time_window_mart(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("year", "month", "month_name", "category_name")
        .agg(
            F.count("*").alias("sessions_count"),
            F.sum("reading_time").alias("total_reading_time"),
            F.avg("rating").alias("avg_rating"),
        )
        .orderBy("year", "month", F.desc("sessions_count"))
    )


def save_mart(df: DataFrame, mart_name: str) -> str:
    target = SPARK_OUTPUT_DIR / mart_name
    target.parent.mkdir(parents=True, exist_ok=True)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(target))
    return str(target)


def run_spark_analytics() -> dict[str, str]:
    spark, df = _prepare_feature_store()
    try:
        top_books = build_top_books_mart(df)
        user_activity = build_user_activity_mart(df)
        categories = build_category_mart(df)
        monthly_trends = build_monthly_trends_sql(spark)
        weekday_activity = build_weekday_activity_sql(spark)
        correlations = build_correlation_mart(spark, df)
        ml_feature_mart = build_ml_feature_mart(df)
        time_window_mart = build_time_window_mart(df)

        return {
            "top_books_mart": save_mart(top_books, "top_books_mart"),
            "user_activity_mart": save_mart(user_activity, "user_activity_mart"),
            "category_mart": save_mart(categories, "category_mart"),
            "monthly_trends_sql_mart": save_mart(monthly_trends, "monthly_trends_sql_mart"),
            "weekday_activity_sql_mart": save_mart(weekday_activity, "weekday_activity_sql_mart"),
            "correlation_mart": save_mart(correlations, "correlation_mart"),
            "ml_feature_mart": save_mart(ml_feature_mart, "ml_feature_mart"),
            "time_window_mart": save_mart(time_window_mart, "time_window_mart"),
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    result = run_spark_analytics()
    for key, value in result.items():
        print(f"{key}: {value}")
