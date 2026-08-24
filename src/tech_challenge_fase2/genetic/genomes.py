"""Genomas tipados e legiveis para cada familia de modelo."""

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

Penalty = Literal["l1", "l2"]
ClassWeight = Literal["balanced"] | None
ForestClassWeight = Literal["balanced", "balanced_subsample"] | None
KNNWeights = Literal["uniform", "distance"]
KNNMetric = Literal["minkowski", "euclidean", "manhattan"]


@dataclass(frozen=True)
class LogisticRegressionGenome:
    """`C` e representado em log10 para cobrir ordens de grandeza."""

    model_name: ClassVar[str] = "logistic_regression"
    log10_c: float
    penalty: Penalty
    class_weight: ClassWeight


@dataclass(frozen=True)
class RandomForestGenome:
    model_name: ClassVar[str] = "random_forest"
    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    max_features: Literal["sqrt", "log2"] | float
    class_weight: ForestClassWeight


@dataclass(frozen=True)
class KNNGenome:
    model_name: ClassVar[str] = "knn"
    n_neighbors: int
    weights: KNNWeights
    metric: KNNMetric
    p: Literal[1, 2] | None


Genome: TypeAlias = LogisticRegressionGenome | RandomForestGenome | KNNGenome
