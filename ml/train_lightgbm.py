from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "staging" / "feature_store.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_reader_activity.joblib"
METRICS_PATH = MODELS_DIR / "lightgbm_reader_activity_metrics.json"

FEATURE_COLUMNS = [
    "reads_count",
    "rating",
    "user_activity_score",
    "category_name",
    "reading_time",
    "average_rating",
    "external_signal_score",
    "favorite_category",
    "age",
    "activity_segment",
]
TARGET_COLUMN = "user_read_probability_target"


def _time_based_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("date_id").reset_index(drop=True)
    unique_dates = ordered["date_id"].drop_duplicates().sort_values().to_list()
    split_index = max(1, int(len(unique_dates) * (1 - test_ratio)))
    cutoff_date = unique_dates[split_index - 1]
    train_df = ordered.loc[ordered["date_id"] <= cutoff_date].copy()
    test_df = ordered.loc[ordered["date_id"] > cutoff_date].copy()
    if test_df.empty:
        fallback_index = max(1, len(ordered) - max(1, int(len(ordered) * test_ratio)))
        train_df = ordered.iloc[:fallback_index].copy()
        test_df = ordered.iloc[fallback_index:].copy()
    return train_df, test_df


def train_model() -> dict[str, float]:
    df = pd.read_csv(FEATURE_STORE_PATH)
    train_df, test_df = _time_based_split(df)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    numeric_features = [
        "reads_count",
        "rating",
        "user_activity_score",
        "reading_time",
        "average_rating",
        "external_signal_score",
        "age",
    ]
    categorical_features = ["category_name", "favorite_category", "activity_segment"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        memory=None,
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                LGBMRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test).clip(0, 1)

    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(train_model())
