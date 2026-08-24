import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning

from tech_challenge_fase2.config import DEFAULT_DATA_PATH
from tech_challenge_fase2.data import load_dataset, split_development_test
from tech_challenge_fase2.genetic.fitness import (
    GeneticFitnessEvaluator,
    calculate_fitness,
)
from tech_challenge_fase2.genetic.search_spaces import SPACES


class WarningClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        warnings.warn("limite de iteracoes", ConvergenceWarning)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class FailingClassifier(WarningClassifier):
    def fit(self, X, y):
        raise RuntimeError("falha controlada")


def _development_evaluator(model_name: str = "logistic_regression"):
    X, y = load_dataset(DEFAULT_DATA_PATH)
    split = split_development_test(X, y)
    evaluator = GeneticFitnessEvaluator(
        space=SPACES[model_name],
        X_development=split.X_development,
        y_development=split.y_development,
    )
    return evaluator, split


def test_fitness_formula_is_explicit_and_correct() -> None:
    final, base = calculate_fitness(
        mean_recall_malignant=0.90,
        std_recall_malignant=0.10,
        mean_f1_malignant=0.80,
        mean_roc_auc=0.95,
        instability_weight=0.10,
    )

    assert base == 0.60 * 0.90 + 0.25 * 0.80 + 0.15 * 0.95
    assert final == base - 0.10 * 0.10


def test_more_instability_reduces_fitness_when_means_are_equal() -> None:
    stable, stable_base = calculate_fitness(
        mean_recall_malignant=0.80,
        std_recall_malignant=0.00,
        mean_f1_malignant=0.82,
        mean_roc_auc=0.90,
    )
    unstable, unstable_base = calculate_fitness(
        mean_recall_malignant=0.80,
        std_recall_malignant=0.20,
        mean_f1_malignant=0.82,
        mean_roc_auc=0.90,
    )

    assert stable_base == unstable_base
    assert stable > unstable
    assert stable - unstable == pytest.approx(0.02)


def test_cv_has_five_stratified_folds_and_never_contains_holdout_indices() -> None:
    evaluator, split = _development_evaluator()
    holdout_indices = set(split.X_test.index)
    validation_indices = [
        index for fold in evaluator.validation_fold_indices for index in fold
    ]

    assert len(evaluator.validation_fold_indices) == 5
    assert set(evaluator.development_indices).isdisjoint(holdout_indices)
    assert set(validation_indices).isdisjoint(holdout_indices)
    assert set(validation_indices) == set(evaluator.development_indices)
    assert len(validation_indices) == len(set(validation_indices))
    for fold_indices in evaluator.validation_fold_indices:
        labels = split.y_development.loc[list(fold_indices)]
        assert set(labels.unique()) == {0, 1}


def test_evaluator_records_all_metrics_per_fold() -> None:
    evaluator, _ = _development_evaluator()
    genome = SPACES["logistic_regression"].sample(np.random.default_rng(42))
    result = evaluator.evaluate(genome)

    assert result.succeeded
    assert len(result.fold_metrics) == 5
    assert 0.0 <= result.fitness <= result.base_fitness <= 1.0
    assert result.evaluation_seconds > 0
    assert result.failure is None


def test_convergence_warning_is_recorded_without_discarding_metrics() -> None:
    evaluator, split = _development_evaluator()
    evaluator = GeneticFitnessEvaluator(
        space=SPACES["logistic_regression"],
        X_development=split.X_development,
        y_development=split.y_development,
        estimator_builder=lambda genome, seed: WarningClassifier(),
    )
    genome = SPACES["logistic_regression"].sample(np.random.default_rng(42))
    result = evaluator.evaluate(genome)

    assert result.succeeded
    assert len(result.fold_metrics) == 5
    assert len(result.issues) == 5
    assert all("type=convergence" in issue for issue in result.issues)


def test_evaluation_failure_receives_explicit_penalty() -> None:
    evaluator, split = _development_evaluator()
    evaluator = GeneticFitnessEvaluator(
        space=SPACES["logistic_regression"],
        X_development=split.X_development,
        y_development=split.y_development,
        estimator_builder=lambda genome, seed: FailingClassifier(),
    )
    genome = SPACES["logistic_regression"].sample(np.random.default_rng(42))
    result = evaluator.evaluate(genome)

    assert not result.succeeded
    assert result.fitness == -1.0
    assert "falha controlada" in result.failure
