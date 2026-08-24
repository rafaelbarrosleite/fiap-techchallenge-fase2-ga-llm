"""Amostragem, validacao, reparacao e decodificacao dos genomas."""

from dataclasses import fields, replace
from typing import Generic, TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .genomes import (
    Genome,
    KNNGenome,
    LogisticRegressionGenome,
    RandomForestGenome,
)

G = TypeVar("G", bound=Genome)


def _choice(rng: np.random.Generator, values: tuple[object, ...]) -> object:
    return values[int(rng.integers(0, len(values)))]


def _clip_int(value: int, minimum: int, maximum: int) -> int:
    return int(min(max(int(value), minimum), maximum))


def _clip_float(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(float(value), minimum), maximum))


class GenomeSpace(Generic[G]):
    model_name: str
    genome_type: type[G]

    def sample(self, rng: np.random.Generator) -> G:
        raise NotImplementedError

    def repair(self, genome: G) -> G:
        raise NotImplementedError

    def validation_errors(self, genome: G) -> list[str]:
        raise NotImplementedError

    def is_valid(self, genome: G) -> bool:
        return not self.validation_errors(genome)

    def require_valid(self, genome: G) -> None:
        errors = self.validation_errors(genome)
        if errors:
            raise ValueError("; ".join(errors))

    def build_estimator(self, genome: G, *, random_state: int) -> BaseEstimator:
        raise NotImplementedError

    @property
    def gene_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self.genome_type))


class LogisticRegressionSpace(GenomeSpace[LogisticRegressionGenome]):
    model_name = LogisticRegressionGenome.model_name
    genome_type = LogisticRegressionGenome
    penalties = ("l1", "l2")
    class_weights = (None, "balanced")

    def sample(self, rng: np.random.Generator) -> LogisticRegressionGenome:
        return LogisticRegressionGenome(
            log10_c=float(rng.uniform(-4.0, 3.0)),
            penalty=_choice(rng, self.penalties),  # type: ignore[arg-type]
            class_weight=_choice(rng, self.class_weights),  # type: ignore[arg-type]
        )

    def repair(self, genome: LogisticRegressionGenome) -> LogisticRegressionGenome:
        penalty = genome.penalty if genome.penalty in self.penalties else "l2"
        class_weight = (
            genome.class_weight
            if genome.class_weight in self.class_weights
            else None
        )
        return replace(
            genome,
            log10_c=_clip_float(genome.log10_c, -4.0, 3.0),
            penalty=penalty,
            class_weight=class_weight,
        )

    def validation_errors(self, genome: LogisticRegressionGenome) -> list[str]:
        errors: list[str] = []
        if not isinstance(genome, self.genome_type):
            return ["Tipo de genoma incompativel com Regressao Logistica."]
        if not -4.0 <= genome.log10_c <= 3.0:
            errors.append("log10_c fora de [-4, 3].")
        if genome.penalty not in self.penalties:
            errors.append("penalty invalida.")
        if genome.class_weight not in self.class_weights:
            errors.append("class_weight invalido.")
        return errors

    def build_estimator(
        self,
        genome: LogisticRegressionGenome,
        *,
        random_state: int,
    ) -> Pipeline:
        self.require_valid(genome)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=10.0 ** genome.log10_c,
                        penalty=genome.penalty,
                        solver="liblinear",
                        class_weight=genome.class_weight,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )


class RandomForestSpace(GenomeSpace[RandomForestGenome]):
    model_name = RandomForestGenome.model_name
    genome_type = RandomForestGenome
    max_features_values = ("sqrt", "log2", 0.5, 1.0)
    class_weights = (None, "balanced", "balanced_subsample")

    def sample(self, rng: np.random.Generator) -> RandomForestGenome:
        max_depth = None if rng.random() < 0.20 else int(rng.integers(3, 21))
        return RandomForestGenome(
            n_estimators=int(rng.integers(100, 501)),
            max_depth=max_depth,
            min_samples_split=int(rng.integers(2, 21)),
            min_samples_leaf=int(rng.integers(1, 11)),
            max_features=_choice(rng, self.max_features_values),  # type: ignore[arg-type]
            class_weight=_choice(rng, self.class_weights),  # type: ignore[arg-type]
        )

    def repair(self, genome: RandomForestGenome) -> RandomForestGenome:
        max_depth = (
            None
            if genome.max_depth is None
            else _clip_int(genome.max_depth, 3, 20)
        )
        max_features = (
            genome.max_features
            if genome.max_features in self.max_features_values
            else "sqrt"
        )
        class_weight = (
            genome.class_weight
            if genome.class_weight in self.class_weights
            else None
        )
        return replace(
            genome,
            n_estimators=_clip_int(genome.n_estimators, 100, 500),
            max_depth=max_depth,
            min_samples_split=_clip_int(genome.min_samples_split, 2, 20),
            min_samples_leaf=_clip_int(genome.min_samples_leaf, 1, 10),
            max_features=max_features,
            class_weight=class_weight,
        )

    def validation_errors(self, genome: RandomForestGenome) -> list[str]:
        if not isinstance(genome, self.genome_type):
            return ["Tipo de genoma incompativel com Random Forest."]
        errors: list[str] = []
        if not 100 <= genome.n_estimators <= 500:
            errors.append("n_estimators fora de [100, 500].")
        if genome.max_depth is not None and not 3 <= genome.max_depth <= 20:
            errors.append("max_depth fora de {None} U [3, 20].")
        if not 2 <= genome.min_samples_split <= 20:
            errors.append("min_samples_split fora de [2, 20].")
        if not 1 <= genome.min_samples_leaf <= 10:
            errors.append("min_samples_leaf fora de [1, 10].")
        if genome.max_features not in self.max_features_values:
            errors.append("max_features invalido.")
        if genome.class_weight not in self.class_weights:
            errors.append("class_weight invalido.")
        return errors

    def build_estimator(
        self,
        genome: RandomForestGenome,
        *,
        random_state: int,
    ) -> Pipeline:
        self.require_valid(genome)
        return Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=genome.n_estimators,
                        max_depth=genome.max_depth,
                        min_samples_split=genome.min_samples_split,
                        min_samples_leaf=genome.min_samples_leaf,
                        max_features=genome.max_features,
                        class_weight=genome.class_weight,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                )
            ]
        )


class KNNSpace(GenomeSpace[KNNGenome]):
    model_name = KNNGenome.model_name
    genome_type = KNNGenome
    weights_values = ("uniform", "distance")
    metric_values = ("minkowski", "euclidean", "manhattan")

    def sample(self, rng: np.random.Generator) -> KNNGenome:
        metric = _choice(rng, self.metric_values)
        return KNNGenome(
            n_neighbors=int(_choice(rng, tuple(range(3, 32, 2)))),
            weights=_choice(rng, self.weights_values),  # type: ignore[arg-type]
            metric=metric,  # type: ignore[arg-type]
            p=int(_choice(rng, (1, 2))) if metric == "minkowski" else None,
        )

    def repair(self, genome: KNNGenome) -> KNNGenome:
        n_neighbors = _clip_int(genome.n_neighbors, 3, 31)
        if n_neighbors % 2 == 0:
            n_neighbors = n_neighbors + 1 if n_neighbors < 31 else 29
        weights = (
            genome.weights if genome.weights in self.weights_values else "uniform"
        )
        metric = (
            genome.metric if genome.metric in self.metric_values else "minkowski"
        )
        p = genome.p if metric == "minkowski" and genome.p in (1, 2) else None
        if metric == "minkowski" and p is None:
            p = 2
        return replace(
            genome,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
        )

    def validation_errors(self, genome: KNNGenome) -> list[str]:
        if not isinstance(genome, self.genome_type):
            return ["Tipo de genoma incompativel com KNN."]
        errors: list[str] = []
        if not 3 <= genome.n_neighbors <= 31 or genome.n_neighbors % 2 == 0:
            errors.append("n_neighbors deve ser impar em [3, 31].")
        if genome.weights not in self.weights_values:
            errors.append("weights invalido.")
        if genome.metric not in self.metric_values:
            errors.append("metric invalida.")
        if genome.metric == "minkowski" and genome.p not in (1, 2):
            errors.append("p deve ser 1 ou 2 para minkowski.")
        if genome.metric != "minkowski" and genome.p is not None:
            errors.append("p deve ser None quando metric nao e minkowski.")
        return errors

    def build_estimator(
        self,
        genome: KNNGenome,
        *,
        random_state: int,
    ) -> Pipeline:
        del random_state  # KNN nao possui random_state.
        self.require_valid(genome)
        parameters: dict[str, object] = {
            "n_neighbors": genome.n_neighbors,
            "weights": genome.weights,
            "metric": genome.metric,
            "n_jobs": 1,
        }
        if genome.metric == "minkowski":
            parameters["p"] = genome.p
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(**parameters)),
            ]
        )


SPACES: dict[str, GenomeSpace[Genome]] = {
    "logistic_regression": LogisticRegressionSpace(),
    "random_forest": RandomForestSpace(),
    "knn": KNNSpace(),
}


def get_search_space(model_name: str) -> GenomeSpace[Genome]:
    try:
        return SPACES[model_name]
    except KeyError as error:
        raise ValueError(f"Modelo genetico desconhecido: {model_name}") from error
