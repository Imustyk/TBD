# Читай и заставь их читать

Информационная система анализа и прогнозирования читательской активности. Проект демонстрирует полный поток данных: от операционных источников в `SQLite` и `MongoDB` и реального `ingestion` через `API` и скраппинг до `Data Warehouse`, ML-прогнозов на `XGBoost` и `LightGBM`, а также веб-аналитики и управления пайплайном в `Streamlit`.

Расширенное описание проекта, требований и плана развития находится в `PROJECT_PLAN.md`.

## Архитектура

```text
Data Sources
  |- books
  |- users
  |- reading history
  |- ratings and reviews
        |
        v
Operational Storage
  |- SQLite: users, books, reading_history, ratings
  |- MongoDB: reviews, comments, recommendations
        |
        v
ETL Pipeline
  |- Extract: pandas + sqlalchemy + pymongo
  |- Transform: cleansing, aggregation, feature engineering
  |- Load: star schema in Data Warehouse
        |
        v
Spark Analytics Layer
  |- PySpark local processing
  |- marts over feature_store
        |
        v
Data Warehouse
  |- fact_reading_activity
  |- dim_user
  |- dim_book
  |- dim_time
  |- dim_category
        |
        v
Machine Learning
  |- XGBoost Regressor -> book popularity
  |- LightGBM Regressor -> reader activity probability
        |
        v
Streamlit Dashboard
  |- Dashboard
  |- Analytics
  |- Prediction
```

## Источники данных

- `SQLite` хранит структурированные сущности: пользователи, книги, история чтения, рейтинги.
- `MongoDB` хранит неструктурированные данные: отзывы, комментарии, рекомендации.
- Для локального демо предусмотрен генератор `data/generate_sample_data.py`, который:
  - создаёт `SQLite` базу `data/reading_sources.db`
  - формирует JSON-экспорт Mongo-коллекций
  - при наличии `MONGODB_URI` заполняет настоящую MongoDB

## Многомерная модель DWH

### Fact table

`fact_reading_activity`

- `user_id`
- `book_id`
- `reading_time`
- `rating`
- `date_id`
- `category_id`
- `reads_count`
- `user_activity_score`
- `book_popularity_target`
- `user_read_probability_target`

### Dimension tables

- `dim_user`
- `dim_book`
- `dim_time`
- `dim_category`

Схема хранения находится в `dwh/schema.sql`.

## Машинное обучение

### Цели

- прогноз популярности книги
- прогноз вероятности того, что пользователь прочитает книгу

### Модели

- `ml/train_xgboost.py` обучает `XGBoost Regressor`
- `ml/train_lightgbm.py` обучает `LightGBM Regressor`

### Признаки

- количество прочтений
- рейтинг книги
- активность пользователя
- категория книги
- время чтения
- средний рейтинг
- возраст пользователя
- поведенческий сегмент

### Метрики

- `RMSE`
- `MAE`
- `R²`

Метрики сохраняются в каталог `models/` рядом с обученными артефактами.

## Streamlit dashboard

Приложение `app/streamlit_app.py` содержит пять страниц:

- `Панель`: пользователи, книги, сессии чтения, топ книг, тренды, активность пользователей
- `Аналитика`: статистика чтения, распределение рейтингов, топ категорий
- `Spark`: Spark SQL-витрины, корреляции, временной анализ и ML-ориентированные агрегаты
- `Прогнозирование`: форма ввода параметров книги и пользователя, прогноз популярности и читательской активности
- `Управление`: запуск ingestion, ETL, обучения моделей, просмотр raw-снимков и журнала событий

## Ingestion

Каталог `ingestion/` отвечает за получение реальных данных и сохранение `raw`-снимков.

- `ingestion/api_ingestion.py`: загрузка данных через `Open Library API`
- `ingestion/scraper_ingestion.py`: скраппинг тестового каталога `Books to Scrape`
- `ingestion/storage.py`: сохранение raw-данных в `data/raw/`, логирование и запись в `MongoDB`
- `ingestion/pipeline.py`: единая точка запуска и получения статуса ingestion

## Apache Spark

В проект добавлен отдельный слой `Apache Spark` для локальной аналитической обработки staging-данных.

- `spark/session.py`: создание локального `SparkSession`
- `spark/analytics.py`: построение витрин `top_books_mart`, `user_activity_mart`, `category_mart`
- `Spark SQL`: формирование витрин `monthly_trends_sql_mart` и `weekday_activity_sql_mart`
- `корреляционный анализ`: витрина `correlation_mart` по ключевым признакам и целям ML
- `временной анализ`: витрина `time_window_mart` по категориям и месяцам
- `связь с ML`: витрина `ml_feature_mart` с агрегированными признаками и таргетами
- результаты Spark-обработки сохраняются в `data/warehouse/spark_marts/`

## Структура проекта

```text
project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── generate_sample_data.py
│   ├── raw/
│   ├── staging/
│   ├── snapshots/
│   └── warehouse/
├── dwh/
│   └── schema.sql
├── etl/
│   ├── extract.py
│   ├── transform.py
│   └── load.py
├── ingestion/
│   ├── api_ingestion.py
│   ├── pipeline.py
│   ├── scraper_ingestion.py
│   └── storage.py
├── logs/
├── ml/
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   └── predict.py
├── models/
├── scripts/
│   ├── setup_project.bat
│   └── setup_project.command
├── spark/
│   ├── analytics.py
│   └── session.py
├── README.md
└── requirements.txt
```

## Запуск

Быстрый автоматический вариант:

```bash
./scripts/setup_project.command
```

Для Windows подготовлен аналогичный файл:

```bat
scripts\setup_project.bat
```

Ручной вариант:

Создать и заполнить демонстрационные источники:

```bash
python data/generate_sample_data.py
```

Запустить ETL и загрузить DWH:

```bash
python etl/load.py
```

Обучить модели:

```bash
python ml/train_xgboost.py
python ml/train_lightgbm.py
```

Запустить Spark-витрины:

```bash
python spark/analytics.py
```

Запустить веб-интерфейс:

```bash
streamlit run app/streamlit_app.py
```

## Итог

Система:

- собирает данные о чтении
- хранит данные в `SQLite` и `MongoDB`
- умеет получать данные через `API` и веб-скраппинг
- сохраняет `raw`-снимки в `data/raw/`
- обрабатывает их через `DWH pipeline`
- поддерживает локальную аналитическую обработку через `Apache Spark`
- использует `Spark SQL`, корреляции и временные Spark-витрины
- строит прогнозы с `XGBoost` и `LightGBM`
- отображает аналитику и управление процессами через `Streamlit dashboard`
