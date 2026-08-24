from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tech_challenge_fase2.models import build_models


def test_scaled_models_keep_scaler_inside_pipeline() -> None:
    models = build_models()

    for name in ("logistic_regression", "knn"):
        assert isinstance(models[name], Pipeline)
        assert isinstance(models[name].named_steps["scaler"], StandardScaler)

    assert isinstance(models["knn"].named_steps["model"], KNeighborsClassifier)
    assert models["knn"].named_steps["model"].n_jobs == 1


def test_random_forest_is_deterministic_and_not_scaled() -> None:
    model = build_models()["random_forest"]

    assert "scaler" not in model.named_steps
    estimator = model.named_steps["model"]
    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.random_state == 42
    assert estimator.n_estimators == 200
    assert estimator.class_weight == "balanced"


def test_model_factories_return_fresh_instances() -> None:
    first = build_models()
    second = build_models()

    assert all(first[name] is not second[name] for name in first)
