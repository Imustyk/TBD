from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "logs" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient


if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from ingestion.pipeline import ingestion_status, run_api_ingestion, run_scraping_ingestion
from ingestion.storage import append_log
from ml.predict import ReadingPredictor


WAREHOUSE_DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "reading_dwh.db"
SOURCE_DB_PATH = PROJECT_ROOT / "data" / "reading_sources.db"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PAGE_DASHBOARD = "Панель"
PAGE_ANALYTICS = "Аналитика"
PAGE_SPARK = "Spark"
PAGE_PREDICTION = "Прогнозирование"
PAGE_MANAGEMENT = "Управление"
SPARK_MARTS_DIR = PROJECT_ROOT / "data" / "warehouse" / "spark_marts"
SPARK_MART_NAMES = [
    "top_books_mart",
    "user_activity_mart",
    "category_mart",
    "monthly_trends_sql_mart",
    "weekday_activity_sql_mart",
    "correlation_mart",
    "ml_feature_mart",
    "time_window_mart",
]


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    if not WAREHOUSE_DB_PATH.exists():
        return {}

    connection = sqlite3.connect(WAREHOUSE_DB_PATH)
    try:
        tables = {
            "fact": pd.read_sql_query("SELECT * FROM fact_reading_activity", connection),
            "dim_user": pd.read_sql_query("SELECT * FROM dim_user", connection),
            "dim_book": pd.read_sql_query("SELECT * FROM dim_book", connection),
            "dim_time": pd.read_sql_query("SELECT * FROM dim_time", connection),
            "dim_category": pd.read_sql_query("SELECT * FROM dim_category", connection),
        }
        for optional_table in ["dim_source", "fact_external_book_observation"]:
            try:
                tables[optional_table] = pd.read_sql_query(f"SELECT * FROM {optional_table}", connection)
            except Exception:
                tables[optional_table] = pd.DataFrame()
        return tables
    finally:
        connection.close()


def load_metrics() -> dict[str, dict]:
    metrics = {}
    for name in ["xgboost_popularity_metrics.json", "lightgbm_reader_activity_metrics.json"]:
        path = MODELS_DIR / name
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def read_spark_mart(mart_name: str) -> pd.DataFrame:
    mart_dir = SPARK_MARTS_DIR / mart_name
    if not mart_dir.exists():
        return pd.DataFrame()

    csv_candidates = sorted(mart_dir.glob("part-*.csv"))
    if not csv_candidates:
        csv_candidates = sorted(mart_dir.glob("*.csv"))
    if not csv_candidates:
        return pd.DataFrame()

    return pd.read_csv(csv_candidates[0])


@st.cache_data
def load_spark_marts() -> dict[str, pd.DataFrame]:
    if not SPARK_MARTS_DIR.exists():
        return {}

    marts = {name: read_spark_mart(name) for name in SPARK_MART_NAMES}
    return {name: frame for name, frame in marts.items() if not frame.empty}


def build_semantic_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fact = tables["fact"]
    books = tables["dim_book"].drop(columns=["category_id"], errors="ignore")
    users = tables["dim_user"]
    categories = tables["dim_category"]
    time_dim = tables["dim_time"]

    merged = (
        fact.merge(books, on="book_id", how="left")
        .merge(users, on="user_id", how="left")
        .merge(categories, on="category_id", how="left")
        .merge(time_dim, on="date_id", how="left")
    )
    merged["full_date"] = pd.to_datetime(merged["full_date"])
    return merged


def build_external_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fact_external = tables.get("fact_external_book_observation", pd.DataFrame())
    dim_source = tables.get("dim_source", pd.DataFrame())
    dim_time = tables["dim_time"]
    if fact_external.empty or dim_source.empty:
        return pd.DataFrame()

    merged = fact_external.merge(dim_source, on="source_id", how="left").merge(dim_time, on="date_id", how="left")
    merged["full_date"] = pd.to_datetime(merged["full_date"])
    return merged


def mongo_connection_status() -> tuple[str, str]:
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        return "Не настроено", "warning"

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
        return "Подключено", "success"
    except Exception:
        return "Недоступно", "error"
    finally:
        client.close()


def execute_python_script(script_relative_path: str) -> dict[str, str | int | bool]:
    script_path = PROJECT_ROOT / script_relative_path
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    success = result.returncode == 0
    append_log(
        "script",
        "success" if success else "error",
        f"Запуск {script_relative_path}",
        {"returncode": result.returncode},
    )
    return {
        "success": success,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def execute_script_chain(scripts: list[str]) -> dict[str, str | int | bool]:
    outputs = []
    for script in scripts:
        result = execute_python_script(script)
        outputs.append(f"## {script}\n{result['stdout']}\n{result['stderr']}".strip())
        if not result["success"]:
            return {
                "success": False,
                "returncode": result["returncode"],
                "stdout": "\n\n".join(outputs),
                "stderr": "",
            }
    return {
        "success": True,
        "returncode": 0,
        "stdout": "\n\n".join(outputs),
        "stderr": "",
    }


def show_action_result() -> None:
    action = st.session_state.get("last_action_result")
    if not action:
        return
    if action["success"]:
        st.success(action["message"])
    else:
        st.error(action["message"])
    if action.get("output"):
        st.code(action["output"], language="bash")


def save_action_result(message: str, success: bool, output: str = "") -> None:
    st.session_state["last_action_result"] = {
        "message": message,
        "success": success,
        "output": output,
    }
    st.cache_data.clear()


def render_dashboard(df: pd.DataFrame, external_df: pd.DataFrame) -> None:
    st.subheader("Панель мониторинга")
    total_users = int(df["user_id"].nunique())
    total_books = int(df["book_id"].nunique())
    total_sessions = int(len(df))
    avg_rating = round(float(df["rating"].mean()), 2)
    external_observations = int(len(external_df)) if not external_df.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Пользователи", total_users)
    col2.metric("Книги", total_books)
    col3.metric("Сессии чтения", total_sessions)
    col4.metric("Средний рейтинг", avg_rating)
    col5.metric("Внешние наблюдения", external_observations)

    top_books = (
        df.groupby("title", as_index=False)
        .agg(total_reads=("reads_count", "max"), reading_time=("reading_time", "sum"))
        .sort_values(["total_reads", "reading_time"], ascending=False)
        .head(10)
    )
    st.plotly_chart(
        px.bar(top_books, x="title", y="reading_time", color="total_reads", title="Топ книг"),
        width="stretch",
    )

    trends = (
        df.groupby("full_date", as_index=False)
        .agg(reading_time=("reading_time", "sum"), ratings=("rating", "mean"))
        .sort_values("full_date")
    )
    st.plotly_chart(
        px.line(trends, x="full_date", y="reading_time", title="Тренды чтения"),
        width="stretch",
    )

    user_activity = (
        df.groupby("full_name", as_index=False)
        .agg(total_time=("reading_time", "sum"), sessions=("user_id", "count"))
        .sort_values("total_time", ascending=False)
        .head(10)
    )
    st.plotly_chart(
        px.bar(user_activity, x="full_name", y="total_time", color="sessions", title="Активность пользователей"),
        width="stretch",
    )

    if not external_df.empty:
        external_top = (
            external_df.groupby("external_title", as_index=False)
            .agg(popularity_signal=("popularity_signal", "mean"))
            .sort_values("popularity_signal", ascending=False)
            .head(10)
        )
        st.plotly_chart(
            px.bar(
                external_top,
                x="external_title",
                y="popularity_signal",
                title="Топ внешних книг по сигналу популярности",
            ),
            width="stretch",
        )


def render_analytics(df: pd.DataFrame, external_df: pd.DataFrame) -> None:
    st.subheader(PAGE_ANALYTICS)

    category_stats = (
        df.groupby("category_name", as_index=False)
        .agg(
            total_reading_time=("reading_time", "sum"),
            avg_rating=("rating", "mean"),
            total_sessions=("user_id", "count"),
        )
        .sort_values("total_reading_time", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.pie(category_stats, names="category_name", values="total_sessions", title="Популярные категории"),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            px.histogram(df, x="rating", nbins=8, title="Распределение рейтингов"),
            width="stretch",
        )

    monthly = (
        df.groupby(["year", "month_name"], as_index=False)
        .agg(total_reading_time=("reading_time", "sum"))
        .sort_values("total_reading_time", ascending=False)
    )
    st.plotly_chart(
        px.bar(monthly, x="month_name", y="total_reading_time", color="year", title="Время чтения по месяцам"),
        width="stretch",
    )

    st.dataframe(
        category_stats.rename(
            columns={
                "category_name": "Категория",
                "total_reading_time": "Общее время чтения",
                "avg_rating": "Средний рейтинг",
                "total_sessions": "Сессии",
            }
        ),
        width="stretch",
    )

    if not external_df.empty:
        st.markdown("### Внешние источники")
        source_stats = (
            external_df.groupby(["source_name", "source_type"], as_index=False)
            .agg(
                observations=("external_title", "count"),
                avg_signal=("popularity_signal", "mean"),
                avg_rating=("external_rating", "mean"),
            )
            .sort_values("observations", ascending=False)
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.bar(source_stats, x="source_name", y="observations", color="source_type", title="Наблюдения по источникам"),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                px.scatter(
                    source_stats,
                    x="observations",
                    y="avg_signal",
                    color="source_name",
                    size="observations",
                    title="Сигнал популярности по источникам",
                ),
                width="stretch",
            )

        external_top = (
            external_df.groupby(["external_title", "source_name"], as_index=False)
            .agg(
                popularity_signal=("popularity_signal", "mean"),
                external_rating=("external_rating", "mean"),
            )
            .sort_values("popularity_signal", ascending=False)
            .head(15)
        )
        st.dataframe(
            external_top.rename(
                columns={
                    "external_title": "Книга",
                    "source_name": "Источник",
                    "popularity_signal": "Сигнал популярности",
                    "external_rating": "Внешний рейтинг",
                }
            ),
            width="stretch",
        )


def render_prediction(df: pd.DataFrame, predictor: ReadingPredictor) -> None:
    st.subheader("Прогнозирование")
    categories = sorted(df["category_name"].dropna().unique().tolist())
    countries = sorted(df["country"].dropna().unique().tolist())

    with st.form("prediction_form"):
        category_name = st.selectbox("Категория книги", categories)
        favorite_category = st.selectbox("Любимая категория пользователя", categories, index=0)
        country = st.selectbox("Страна", countries)
        reads_count = st.slider("Количество прочтений", 0, 100, 30)
        rating = st.slider("Наблюдаемый рейтинг", 1.0, 5.0, 4.2, 0.1)
        average_rating = st.slider("Средний рейтинг книги", 1.0, 5.0, 4.0, 0.1)
        external_signal_score = st.slider("Внешний сигнал популярности", 0.0, 300.0, 10.0, 0.5)
        reading_time = st.slider("Время чтения (минуты)", 10, 300, 120)
        user_activity_score = st.slider("Оценка активности пользователя", 0.0, 12.0, 5.0, 0.1)
        publication_year = st.slider("Год публикации", 1950, 2026, 2018)
        age = st.slider("Возраст пользователя", 12, 80, 24)

        submitted = st.form_submit_button("Построить прогноз")

    if not submitted:
        return

    if user_activity_score >= 7:
        activity_segment = "high"
    elif user_activity_score >= 3:
        activity_segment = "medium"
    else:
        activity_segment = "low"

    payload = {
        "reads_count": reads_count,
        "rating": rating,
        "user_activity_score": user_activity_score,
        "category_name": category_name,
        "reading_time": reading_time,
        "average_rating": average_rating,
        "external_signal_score": external_signal_score,
        "publication_year": publication_year,
        "country": country,
        "favorite_category": favorite_category,
        "age": age,
        "activity_segment": activity_segment,
    }

    predictions = predictor.predict(payload)

    col1, col2 = st.columns(2)
    col1.metric("Прогноз популярности книги", predictions["book_popularity"])
    col2.metric("Вероятность читательской активности", f"{predictions['user_read_probability']:.1%}")

    prediction_df = pd.DataFrame(
        [
            {"metric": "Популярность книги", "value": predictions["book_popularity"]},
            {"metric": "Читательская активность", "value": predictions["user_read_probability"]},
        ]
    )
    st.plotly_chart(
        px.bar(prediction_df, x="metric", y="value", color="metric", title="Результаты прогноза"),
        width="stretch",
    )


def render_spark(spark_marts: dict[str, pd.DataFrame], metrics: dict[str, dict]) -> None:
    st.subheader("Spark / PySpark")

    if not spark_marts:
        st.info("Spark-витрины ещё не рассчитаны. Открой раздел `Управление` и запусти Spark-аналитику.")
        st.code("python spark/analytics.py", language="bash")
        return

    top_books = spark_marts.get("top_books_mart", pd.DataFrame())
    monthly_trends = spark_marts.get("monthly_trends_sql_mart", pd.DataFrame())
    weekday_activity = spark_marts.get("weekday_activity_sql_mart", pd.DataFrame())
    correlations = spark_marts.get("correlation_mart", pd.DataFrame())
    ml_feature_mart = spark_marts.get("ml_feature_mart", pd.DataFrame())
    time_window_mart = spark_marts.get("time_window_mart", pd.DataFrame())

    strongest_correlation = 0.0
    if not correlations.empty:
        strongest_correlation = float(correlations["abs_correlation"].max())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spark-витрины", len(spark_marts))
    col2.metric("Книг в витрине", int(top_books["title"].nunique()) if not top_books.empty else 0)
    col3.metric("Месяцы в трендах", int(len(monthly_trends)) if not monthly_trends.empty else 0)
    col4.metric("Макс. |корреляция|", round(strongest_correlation, 3))

    if not top_books.empty:
        st.plotly_chart(
            px.bar(
                top_books.head(10),
                x="title",
                y="total_reading_time",
                color="sessions_count",
                title="Spark DataFrame: топ книг по времени чтения",
            ),
            width="stretch",
        )

    if not monthly_trends.empty:
        monthly_trends = monthly_trends.copy()
        monthly_trends["period"] = monthly_trends["year"].astype(str) + "-" + monthly_trends["month"].astype(str).str.zfill(2)
        st.plotly_chart(
            px.line(
                monthly_trends,
                x="period",
                y="total_reading_time",
                markers=True,
                title="Spark SQL: временной анализ по месяцам",
            ),
            width="stretch",
        )

    col1, col2 = st.columns(2)
    with col1:
        if not weekday_activity.empty:
            st.plotly_chart(
                px.bar(
                    weekday_activity,
                    x="weekday",
                    y="sessions_count",
                    color="avg_user_activity_score",
                    title="Spark SQL: активность по дням недели",
                ),
                width="stretch",
            )
    with col2:
        if not correlations.empty:
            correlation_chart = correlations.copy()
            correlation_chart["pair"] = correlation_chart["feature_x"] + " vs " + correlation_chart["feature_y"]
            st.plotly_chart(
                px.bar(
                    correlation_chart,
                    x="pair",
                    y="correlation",
                    color="abs_correlation",
                    title="Корреляции признаков в Spark",
                ),
                width="stretch",
            )

    st.markdown("### Связь Spark с ML")
    if not ml_feature_mart.empty:
        st.plotly_chart(
            px.scatter(
                ml_feature_mart,
                x="avg_user_activity_score",
                y="avg_future_read_probability",
                color="activity_segment",
                size="samples_count",
                hover_name="category_name",
                title="Spark-витрина признаков для ML-сегментов",
            ),
            width="stretch",
        )
        st.dataframe(ml_feature_mart, width="stretch")
    else:
        st.info("ML-ориентированная Spark-витрина пока не рассчитана.")

    st.markdown("### Spark и визуальная аналитика")
    if not time_window_mart.empty:
        st.dataframe(time_window_mart.head(20), width="stretch")

    if metrics:
        with st.expander("Метрики моделей и связь со Spark-витринами", expanded=False):
            st.json(metrics)


def render_management() -> None:
    st.subheader("Управление процессами")
    show_action_result()

    mongo_text, mongo_state = mongo_connection_status()
    status = ingestion_status()
    model_files = list(MODELS_DIR.glob("*.joblib"))
    spark_marts = load_spark_marts()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Источник SQLite", "Готов" if SOURCE_DB_PATH.exists() else "Нет")
    col2.metric("DWH", "Готов" if WAREHOUSE_DB_PATH.exists() else "Нет")
    col3.metric("Raw-снимки", status["raw_snapshots"])
    col4.metric("Модели", len(model_files))
    col5.metric("MongoDB", mongo_text)
    col6.metric("Spark-витрины", len(spark_marts))

    if mongo_state == "warning":
        st.warning("MongoDB ещё не настроена. Используй `.env` и скрипт `scripts/setup_project.command`.")
    elif mongo_state == "error":
        st.error("Не удалось подключиться к MongoDB. Проверь Docker-контейнер и значение `MONGODB_URI`.")
    else:
        st.success("Подключение к MongoDB активно.")

    left, right = st.columns(2)

    with left:
        st.markdown("### Реальные источники данных")
        with st.form("api_form"):
            api_query = st.text_input("Поисковый запрос для API", value="data science")
            api_limit = st.slider("Количество записей из API", 1, 30, 10)
            api_submit = st.form_submit_button("Загрузить из Open Library API")

        if api_submit:
            with st.spinner("Загружаю данные из API..."):
                try:
                    result = run_api_ingestion(api_query, api_limit)
                    save_action_result(
                        f"API-загрузка завершена: получено {result['count']} записей.",
                        True,
                        f"Источник: {result['source']}\nRaw-файл: {result['path']}",
                    )
                    st.rerun()
                except Exception as error:
                    append_log("api_ingestion", "error", "Ошибка загрузки из API", {"error": str(error)})
                    save_action_result(f"Ошибка API-загрузки: {error}", False)
                    st.rerun()

        with st.form("scraping_form"):
            scraping_url = st.text_input(
                "URL для скраппинга",
                value="https://books.toscrape.com/catalogue/page-1.html",
            )
            scraping_submit = st.form_submit_button("Запустить скраппинг")

        if scraping_submit:
            with st.spinner("Выполняю скраппинг сайта..."):
                try:
                    result = run_scraping_ingestion(scraping_url)
                    save_action_result(
                        f"Скраппинг завершён: получено {result['count']} записей.",
                        True,
                        f"Источник: {result['source']}\nRaw-файл: {result['path']}",
                    )
                    st.rerun()
                except Exception as error:
                    append_log("scraping", "error", "Ошибка скраппинга", {"error": str(error)})
                    save_action_result(f"Ошибка скраппинга: {error}", False)
                    st.rerun()

    with right:
        st.markdown("### Управление пайплайном")
        if st.button("Сгенерировать демо-данные", width="stretch"):
            with st.spinner("Генерирую демонстрационные источники..."):
                result = execute_python_script("data/generate_sample_data.py")
                save_action_result("Генерация демо-данных завершена." if result["success"] else "Генерация демо-данных завершилась с ошибкой.", bool(result["success"]), f"{result['stdout']}\n{result['stderr']}")
                st.rerun()

        if st.button("Запустить ETL и загрузить DWH", width="stretch"):
            with st.spinner("Запускаю ETL-пайплайн..."):
                result = execute_python_script("etl/load.py")
                save_action_result("ETL успешно выполнен." if result["success"] else "ETL завершился с ошибкой.", bool(result["success"]), f"{result['stdout']}\n{result['stderr']}")
                st.rerun()

        if st.button("Обучить XGBoost", width="stretch"):
            with st.spinner("Обучаю XGBoost..."):
                result = execute_python_script("ml/train_xgboost.py")
                save_action_result("Модель XGBoost обучена." if result["success"] else "Ошибка обучения XGBoost.", bool(result["success"]), f"{result['stdout']}\n{result['stderr']}")
                st.rerun()

        if st.button("Обучить LightGBM", width="stretch"):
            with st.spinner("Обучаю LightGBM..."):
                result = execute_python_script("ml/train_lightgbm.py")
                save_action_result("Модель LightGBM обучена." if result["success"] else "Ошибка обучения LightGBM.", bool(result["success"]), f"{result['stdout']}\n{result['stderr']}")
                st.rerun()

        if st.button("Запустить Spark-аналитику", width="stretch"):
            with st.spinner("Строю Spark-витрины и временной анализ..."):
                result = execute_python_script("spark/analytics.py")
                save_action_result("Spark-аналитика выполнена." if result["success"] else "Ошибка Spark-аналитики.", bool(result["success"]), f"{result['stdout']}\n{result['stderr']}")
                st.rerun()

        if st.button("Полный цикл: данные -> ETL -> модели", width="stretch"):
            with st.spinner("Выполняю полный цикл обработки..."):
                result = execute_script_chain(
                    [
                        "data/generate_sample_data.py",
                        "etl/load.py",
                        "spark/analytics.py",
                        "ml/train_xgboost.py",
                        "ml/train_lightgbm.py",
                    ]
                )
                save_action_result("Полный цикл выполнен успешно." if result["success"] else "Полный цикл завершился с ошибкой.", bool(result["success"]), str(result["stdout"]))
                st.rerun()

        if st.button("Обновить данные интерфейса", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("### Последние raw-снимки")
    snapshots = status["recent_snapshots"]
    if snapshots:
        st.dataframe(pd.DataFrame(snapshots), width="stretch")
    else:
        st.info("Пока нет сохранённых raw-снимков.")

    st.markdown("### Журнал загрузок")
    logs = status["recent_logs"]
    if logs:
        st.dataframe(pd.DataFrame(logs), width="stretch")
    else:
        st.info("Журнал событий пока пуст.")


def main() -> None:
    st.set_page_config(page_title="Читай и заставь их читать", layout="wide")
    st.title("Читай и заставь их читать")
    st.caption("Локальная система анализа, визуализации и прогнозирования читательской активности")

    metrics = load_metrics()
    if metrics:
        with st.expander("Метрики моделей", expanded=False):
            st.json(metrics)

    page = st.sidebar.radio(
        "Раздел",
        [PAGE_DASHBOARD, PAGE_ANALYTICS, PAGE_SPARK, PAGE_PREDICTION, PAGE_MANAGEMENT],
    )

    if page == PAGE_MANAGEMENT:
        render_management()
        return

    tables = load_tables()
    if not tables:
        st.warning("Хранилище данных ещё не подготовлено. Открой раздел `Управление` и запусти генерацию данных, ETL и обучение моделей.")
        st.code(
            "python data/generate_sample_data.py\npython etl/load.py\npython ml/train_xgboost.py\npython ml/train_lightgbm.py",
            language="bash",
        )
        return

    semantic_df = build_semantic_dataset(tables)
    external_df = build_external_dataset(tables)
    spark_marts = load_spark_marts()
    predictor = ReadingPredictor()

    if page == PAGE_DASHBOARD:
        render_dashboard(semantic_df, external_df)
    elif page == PAGE_ANALYTICS:
        render_analytics(semantic_df, external_df)
    elif page == PAGE_SPARK:
        render_spark(spark_marts, metrics)
    else:
        render_prediction(semantic_df, predictor)


if __name__ == "__main__":
    main()
