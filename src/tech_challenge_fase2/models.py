"""Pipelines dos tres modelos usados na Fase 1."""

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            )
        ]
    )


def knn_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        ]
    )


MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": logistic_regression_pipeline,
    "random_forest": random_forest_pipeline,
    "knn": knn_pipeline,
}


def build_models() -> dict[str, BaseEstimator]:
    """Cria instancias novas para impedir reuso acidental de modelos ajustados."""

    return {name: factory() for name, factory in MODEL_FACTORIES.items()}
