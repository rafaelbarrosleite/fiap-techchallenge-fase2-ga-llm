import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tech_challenge_fase2.genetic.genomes import (
    KNNGenome,
    LogisticRegressionGenome,
    RandomForestGenome,
)
from tech_challenge_fase2.genetic.search_spaces import SPACES
from tech_challenge_fase2.genetic.serialization import (
    genome_from_dict,
    genome_to_dict,
)


@pytest.mark.parametrize("model_name", sorted(SPACES))
def test_sampled_genomes_are_valid_and_round_trip(model_name: str) -> None:
    space = SPACES[model_name]
    rng = np.random.default_rng(42)

    for _ in range(30):
        genome = space.sample(rng)
        assert space.is_valid(genome)
        assert genome_from_dict(genome_to_dict(genome)) == genome


def test_logistic_repairs_limits_and_uses_compatible_solver() -> None:
    space = SPACES["logistic_regression"]
    repaired = space.repair(
        LogisticRegressionGenome(
            log10_c=99.0,
            penalty="invalid",  # type: ignore[arg-type]
            class_weight="invalid",  # type: ignore[arg-type]
        )
    )
    estimator = space.build_estimator(repaired, random_state=42)

    assert repaired == LogisticRegressionGenome(3.0, "l2", None)
    assert isinstance(estimator, Pipeline)
    assert isinstance(estimator.named_steps["scaler"], StandardScaler)
    model = estimator.named_steps["model"]
    assert model.solver == "liblinear"
    assert model.max_iter == 2000
    assert model.penalty in {"l1", "l2"}


def test_random_forest_repairs_optional_and_numeric_genes() -> None:
    space = SPACES["random_forest"]
    repaired = space.repair(
        RandomForestGenome(
            n_estimators=999,
            max_depth=1,
            min_samples_split=99,
            min_samples_leaf=-2,
            max_features="invalid",  # type: ignore[arg-type]
            class_weight="invalid",  # type: ignore[arg-type]
        )
    )
    estimator = space.build_estimator(repaired, random_state=77)

    assert repaired.n_estimators == 500
    assert repaired.max_depth == 3
    assert repaired.min_samples_split == 20
    assert repaired.min_samples_leaf == 1
    assert "scaler" not in estimator.named_steps
    assert estimator.named_steps["model"].random_state == 77
    assert space.is_valid(repaired)


@pytest.mark.parametrize(
    ("metric", "p", "valid"),
    [
        ("minkowski", 1, True),
        ("minkowski", None, False),
        ("euclidean", None, True),
        ("euclidean", 2, False),
        ("manhattan", None, True),
        ("manhattan", 1, False),
    ],
)
def test_knn_metric_and_p_compatibility(
    metric: str,
    p: int | None,
    valid: bool,
) -> None:
    space = SPACES["knn"]
    genome = KNNGenome(
        n_neighbors=5,
        weights="uniform",
        metric=metric,  # type: ignore[arg-type]
        p=p,  # type: ignore[arg-type]
    )

    assert space.is_valid(genome) is valid
    repaired = space.repair(genome)
    assert space.is_valid(repaired)
    assert repaired.p in (1, 2) if repaired.metric == "minkowski" else repaired.p is None


def test_knn_pipeline_scales_inside_each_cv_fit() -> None:
    space = SPACES["knn"]
    genome = KNNGenome(5, "distance", "euclidean", None)
    estimator = space.build_estimator(genome, random_state=42)

    assert isinstance(estimator.named_steps["scaler"], StandardScaler)
    assert estimator.named_steps["model"].metric == "euclidean"
