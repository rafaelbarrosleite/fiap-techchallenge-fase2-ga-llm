"""Configuracoes reproduziveis do algoritmo genetico."""

from dataclasses import asdict, dataclass, replace
from typing import Any

from ..config import RANDOM_STATE


@dataclass(frozen=True)
class GAConfig:
    """Parametros da evolucao e da avaliacao por validacao cruzada."""

    name: str
    population_size: int
    max_generations: int
    crossover_rate: float
    mutation_rate: float
    elite_count: int
    tournament_size: int
    seed: int = RANDOM_STATE
    cv_splits: int = 5
    cv_seed: int = RANDOM_STATE
    estimator_seed: int = RANDOM_STATE
    instability_weight: float = 0.10
    stagnation_generations: int | None = None
    min_improvement: float = 1e-12

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size deve ser pelo menos 2.")
        if self.max_generations < 1:
            raise ValueError("max_generations deve ser pelo menos 1.")
        for field_name in ("crossover_rate", "mutation_rate"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} deve estar entre 0 e 1.")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count deve estar entre 1 e population_size - 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size deve estar entre 2 e population_size.")
        if self.cv_splits != 5:
            raise ValueError("O protocolo aprovado exige exatamente cinco dobras.")
        if not 0.0 <= self.instability_weight <= 1.0:
            raise ValueError("instability_weight deve estar entre 0 e 1.")
        if self.stagnation_generations is not None and self.stagnation_generations < 1:
            raise ValueError("stagnation_generations deve ser positivo ou None.")
        if self.min_improvement < 0:
            raise ValueError("min_improvement nao pode ser negativo.")

    @property
    def maximum_candidate_evaluations(self) -> int:
        """Teto sem considerar cache: populacao inicial + geracoes."""

        return self.population_size * (self.max_generations + 1)

    @property
    def maximum_model_fits(self) -> int:
        return self.maximum_candidate_evaluations * self.cv_splits

    def with_seed(self, seed: int) -> "GAConfig":
        return replace(self, seed=seed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXPERIMENT_CONFIGS: dict[str, GAConfig] = {
    "A_small": GAConfig(
        name="A_small",
        population_size=20,
        max_generations=10,
        crossover_rate=0.70,
        mutation_rate=0.10,
        elite_count=2,
        tournament_size=3,
    ),
    "B_balanced": GAConfig(
        name="B_balanced",
        population_size=40,
        max_generations=20,
        crossover_rate=0.80,
        mutation_rate=0.20,
        elite_count=2,
        tournament_size=3,
    ),
    "C_exploratory": GAConfig(
        name="C_exploratory",
        population_size=60,
        max_generations=30,
        crossover_rate=0.75,
        mutation_rate=0.30,
        elite_count=4,
        tournament_size=4,
    ),
}


def smoke_config(seed: int = RANDOM_STATE) -> GAConfig:
    """Configuracao barata para integracao; nao e experimento oficial."""

    return GAConfig(
        name="smoke",
        population_size=4,
        max_generations=2,
        crossover_rate=0.80,
        mutation_rate=0.35,
        elite_count=1,
        tournament_size=2,
        seed=seed,
        stagnation_generations=None,
    )

