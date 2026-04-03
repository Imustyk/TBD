from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_popularity.joblib"
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_reader_activity.joblib"


class ReadingPredictor:
    def __init__(self) -> None:
        self.popularity_model = joblib.load(XGBOOST_MODEL_PATH) if XGBOOST_MODEL_PATH.exists() else None
        self.activity_model = joblib.load(LIGHTGBM_MODEL_PATH) if LIGHTGBM_MODEL_PATH.exists() else None

    def is_ready(self) -> bool:
        return self.popularity_model is not None and self.activity_model is not None

    @staticmethod
    def _align_features(model, frame: pd.DataFrame) -> pd.DataFrame:
        expected_columns = getattr(model, "feature_names_in_", None)
        if expected_columns is None:
            preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
            expected_columns = getattr(preprocessor, "feature_names_in_", None)

        if expected_columns is None:
            return frame

        aligned = frame.copy()
        for column in expected_columns:
            if column not in aligned.columns:
                aligned[column] = 0.0

        return aligned[list(expected_columns)]

    def predict(self, payload: dict) -> dict[str, float]:
        popularity_frame = pd.DataFrame(
            [
                {
                    "reads_count": payload["reads_count"],
                    "rating": payload["rating"],
                    "user_activity_score": payload["user_activity_score"],
                    "category_name": payload["category_name"],
                    "reading_time": payload["reading_time"],
                    "average_rating": payload["average_rating"],
                    "external_signal_score": payload["external_signal_score"],
                    "publication_year": payload["publication_year"],
                    "country": payload["country"],
                    "activity_segment": payload["activity_segment"],
                }
            ]
        )

        activity_frame = pd.DataFrame(
            [
                {
                    "reads_count": payload["reads_count"],
                    "rating": payload["rating"],
                    "user_activity_score": payload["user_activity_score"],
                    "category_name": payload["category_name"],
                    "reading_time": payload["reading_time"],
                    "average_rating": payload["average_rating"],
                    "external_signal_score": payload["external_signal_score"],
                    "favorite_category": payload["favorite_category"],
                    "age": payload["age"],
                    "activity_segment": payload["activity_segment"],
                }
            ]
        )

        if not self.is_ready():
            heuristic_popularity = round(
                payload["reads_count"] * 0.50
                + payload["average_rating"] * 0.25
                + payload["user_activity_score"] * 0.15
                + payload["external_signal_score"] * 0.10,
                3,
            )
            heuristic_probability = round(
                min(
                    1.0,
                    payload["user_activity_score"] * 0.08
                    + payload["average_rating"] * 0.08
                    + payload["reading_time"] / 400,
                    + payload["external_signal_score"] / 500,
                ),
                3,
            )
            return {
                "book_popularity": heuristic_popularity,
                "user_read_probability": heuristic_probability,
            }

        popularity_input = self._align_features(self.popularity_model, popularity_frame)
        activity_input = self._align_features(self.activity_model, activity_frame)

        popularity = float(self.popularity_model.predict(popularity_input)[0])
        probability = float(self.activity_model.predict(activity_input)[0])
        return {
            "book_popularity": round(popularity, 3),
            "user_read_probability": round(max(0.0, min(1.0, probability)), 3),
        }


def _parse_cli_args(argv: list[str]) -> dict:
    if len(argv) != 11:
        raise SystemExit(
            "Usage: python ml/predict.py <reads_count> <rating> <user_activity_score> <category_name> "
            "<reading_time> <average_rating> <external_signal_score> <publication_year> <country> <favorite_category> <age>"
        )

    (
        reads_count,
        rating,
        activity_score,
        category_name,
        reading_time,
        avg_rating,
        external_signal_score,
        publication_year,
        country,
        favorite,
        age,
    ) = argv
    activity_segment = "high" if float(activity_score) >= 7 else "medium" if float(activity_score) >= 3 else "low"
    return {
        "reads_count": float(reads_count),
        "rating": float(rating),
        "user_activity_score": float(activity_score),
        "category_name": category_name,
        "reading_time": float(reading_time),
        "average_rating": float(avg_rating),
        "external_signal_score": float(external_signal_score),
        "publication_year": int(publication_year),
        "country": country,
        "favorite_category": favorite,
        "age": int(age),
        "activity_segment": activity_segment,
    }


if __name__ == "__main__":
    predictor = ReadingPredictor()
    result = predictor.predict(_parse_cli_args(sys.argv[1:]))
    print(result)
