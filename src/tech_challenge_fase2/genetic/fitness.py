"""Fitness isolado, calculado somente com CV no desenvolvimento."""

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .genomes import Genome
from .search_spaces import GenomeSpace


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    recall_malignant: float
    f1_malignant: float
    roc_auc: float
    train_rows: int
    validation_rows: int


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    base_fitness: float
    mean_recall_malignant: float
    std_recall_malignant: float
    mean_f1_malignant: float
    mean_roc_auc: float
    evaluation_seconds: float
    fold_metrics: tuple[FoldMetrics, ...]
    issues: tuple[str, ...] = ()
    failure: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fold_metrics"] = [asdict(fold) for fold in self.fold_metrics]
        payload["issues"] = list(self.issues)
        return payload

    def deterministic_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evaluation_seconds")
        return payload


def calculate_fitness(
    *,
    mean_recall_malignant: float,
    std_recall_malignant: float,
    mean_f1_malignant: float,
    mean_roc_auc: float,
    instability_weight: float = 0.10,
) -> tuple[float, float]:
    """Retorna (fitness final, fitness antes da penalidade)."""

    for name, value in (
        ("mean_recall_malignant", mean_recall_malignant),
        ("mean_f1_malignant", mean_f1_malignant),
        ("mean_roc_auc", mean_roc_auc),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} deve estar entre 0 e 1.")
    if not 0.0 <= std_recall_malignant <= 0.5:
        raise ValueError("std_recall_malignant deve estar entre 0 e 0,5.")
    if not 0.0 <= instability_weight <= 1.0:
        raise ValueError("instability_weight deve estar entre 0 e 1.")

    base_fitness = (
        0.60 * mean_recall_malignant
        + 0.25 * mean_f1_malignant
        + 0.15 * mean_roc_auc
    )
    final_fitness = base_fitness - instability_weight * std_recall_malignant
    return float(final_fitness), float(base_fitness)


EstimatorBuilder = Callable[[Genome, int], BaseEstimator]


class GeneticFitnessEvaluator:
    """Mantem folds fixos e recebe apenas o conjunto de desenvolvimento."""

    def __init__(
        self,
        *,
        space: GenomeSpace[Genome],
        X_development: pd.DataFrame,
        y_development: pd.Series,
        cv_splits: int = 5,
        cv_seed: int = 42,
        estimator_seed: int = 42,
        instability_weight: float = 0.10,
        estimator_builder: EstimatorBuilder | None = None,
    ) -> None:
        if cv_splits != 5:
            raise ValueError("A avaliacao genetica exige cinco dobras.")
        if len(X_development) != len(y_development):
            raise ValueError("X e y de desenvolvimento devem ter o mesmo tamanho.")
        if len(X_development) == 0:
            raise ValueError("O conjunto de desenvolvimento nao pode ser vazio.")
        if set(y_development.unique()) != {0, 1}:
            raise ValueError("A avaliacao exige alvo binario codificado como 0 e 1.")
        self.space = space
        self.X_development = X_development.copy(deep=False)
        self.y_development = y_development.copy(deep=False)
        self.cv_splits = cv_splits
        self.cv_seed = cv_seed
        self.estimator_seed = estimator_seed
        self.instability_weight = instability_weight
        self._estimator_builder = estimator_builder
        splitter = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=cv_seed,
        )
        self._folds = tuple(
            (train_positions, validation_positions)
            for train_positions, validation_positions in splitter.split(
                self.X_development, self.y_development
            )
        )

    @property
    def development_indices(self) -> tuple[Any, ...]:
        return tuple(self.X_development.index.tolist())

    @property
    def validation_fold_indices(self) -> tuple[tuple[Any, ...], ...]:
        index = self.X_development.index
        return tuple(
            tuple(index[validation_positions].tolist())
            for _, validation_positions in self._folds
        )

    def _build_estimator(self, genome: Genome) -> BaseEstimator:
        if self._estimator_builder is not None:
            return self._estimator_builder(genome, self.estimator_seed)
        return self.space.build_estimator(
            genome,
            random_state=self.estimator_seed,
        )

    def evaluate(self, genome: Genome) -> FitnessResult:
        started = perf_counter()
        issues: list[str] = []
        folds: list[FoldMetrics] = []
        try:
            self.space.require_valid(genome)
            for fold_number, (train_positions, validation_positions) in enumerate(
                self._folds,
                start=1,
            ):
                estimator = self._build_estimator(genome)
                X_train = self.X_development.iloc[train_positions]
                y_train = self.y_development.iloc[train_positions]
                X_validation = self.X_development.iloc[validation_positions]
                y_validation = self.y_development.iloc[validation_positions]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(X_train, y_train)
                for warning in caught:
                    category = warning.category.__name__
                    message = str(warning.message)
                    prefix = "convergence" if issubclass(
                        warning.category, ConvergenceWarning
                    ) else "warning"
                    issues.append(
                        f"fold={fold_number};type={prefix};category={category};"
                        f"message={message}"
                    )
                prediction = estimator.predict(X_validation)
                if not hasattr(estimator, "predict_proba"):
                    raise TypeError("O estimador precisa expor predict_proba.")
                probability = estimator.predict_proba(X_validation)[:, 1]
                folds.append(
                    FoldMetrics(
                        fold=fold_number,
                        recall_malignant=float(
                            recall_score(
                                y_validation,
                                prediction,
                                pos_label=1,
                                zero_division=0,
                            )
                        ),
                        f1_malignant=float(
                            f1_score(
                                y_validation,
                                prediction,
                                pos_label=1,
                                zero_division=0,
                            )
                        ),
                        roc_auc=float(roc_auc_score(y_validation, probability)),
                        train_rows=len(train_positions),
                        validation_rows=len(validation_positions),
                    )
                )
        except Exception as error:
            return FitnessResult(
                fitness=-1.0,
                base_fitness=-1.0,
                mean_recall_malignant=0.0,
                std_recall_malignant=0.0,
                mean_f1_malignant=0.0,
                mean_roc_auc=0.0,
                evaluation_seconds=perf_counter() - started,
                fold_metrics=tuple(folds),
                issues=tuple(issues),
                failure=f"{type(error).__name__}: {error}",
            )

        recalls = np.array([fold.recall_malignant for fold in folds])
        mean_recall = float(recalls.mean())
        std_recall = float(recalls.std(ddof=0))
        mean_f1 = float(np.mean([fold.f1_malignant for fold in folds]))
        mean_auc = float(np.mean([fold.roc_auc for fold in folds]))
        final_fitness, base_fitness = calculate_fitness(
            mean_recall_malignant=mean_recall,
            std_recall_malignant=std_recall,
            mean_f1_malignant=mean_f1,
            mean_roc_auc=mean_auc,
            instability_weight=self.instability_weight,
        )
        return FitnessResult(
            fitness=final_fitness,
            base_fitness=base_fitness,
            mean_recall_malignant=mean_recall,
            std_recall_malignant=std_recall,
            mean_f1_malignant=mean_f1,
            mean_roc_auc=mean_auc,
            evaluation_seconds=perf_counter() - started,
            fold_metrics=tuple(folds),
            issues=tuple(issues),
        )

