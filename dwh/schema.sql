PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fact_reading_activity;
DROP TABLE IF EXISTS fact_external_book_observation;
DROP TABLE IF EXISTS dim_time;
DROP TABLE IF EXISTS dim_source;
DROP TABLE IF EXISTS dim_book;
DROP TABLE IF EXISTS dim_user;
DROP TABLE IF EXISTS dim_category;

CREATE TABLE dim_user (
    user_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    age INTEGER,
    country TEXT,
    registration_date TEXT,
    favorite_category TEXT,
    activity_segment TEXT
);

CREATE TABLE dim_category (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_book (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    publication_year INTEGER,
    average_rating REAL,
    total_reads INTEGER,
    external_signal_score REAL DEFAULT 0,
    external_observation_count INTEGER DEFAULT 0,
    FOREIGN KEY (category_id) REFERENCES dim_category(category_id)
);

CREATE TABLE dim_source (
    source_id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    base_url TEXT
);

CREATE TABLE dim_time (
    date_id INTEGER PRIMARY KEY,
    full_date TEXT NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    week INTEGER NOT NULL,
    day INTEGER NOT NULL,
    weekday TEXT NOT NULL
);

CREATE TABLE fact_reading_activity (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    reading_time INTEGER NOT NULL,
    rating REAL,
    reads_count INTEGER NOT NULL,
    user_activity_score REAL NOT NULL,
    book_popularity_target REAL NOT NULL,
    user_read_probability_target REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES dim_user(user_id),
    FOREIGN KEY (book_id) REFERENCES dim_book(book_id),
    FOREIGN KEY (category_id) REFERENCES dim_category(category_id),
    FOREIGN KEY (date_id) REFERENCES dim_time(date_id)
);

CREATE TABLE fact_external_book_observation (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    book_id INTEGER,
    external_title TEXT NOT NULL,
    external_author TEXT,
    external_category TEXT,
    publication_year INTEGER,
    external_rating REAL,
    price REAL,
    availability TEXT,
    popularity_signal REAL NOT NULL,
    raw_file_path TEXT,
    fetched_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES dim_source(source_id),
    FOREIGN KEY (date_id) REFERENCES dim_time(date_id),
    FOREIGN KEY (book_id) REFERENCES dim_book(book_id)
);
