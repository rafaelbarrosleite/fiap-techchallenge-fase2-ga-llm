"""Estruturas auditaveis de individuos avaliados e geracoes."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .fitness import FitnessResult
from .genomes import Genome
from .serialization import genome_key, genome_to_dict


@dataclass(frozen=True)
class EvaluatedIndividual:
    genome: Genome
    fitness: FitnessResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": genome_to_dict(self.genome),
            "fitness": self.fitness.to_dict(),
        }

    def deterministic_dict(self) -> dict[str, Any]:
        return {
            "genome": genome_to_dict(self.genome),
            "fitness": self.fitness.deterministic_dict(),
        }


def rank_key(individual: EvaluatedIndividual) -> tuple[float, float, float, float, float, str]:
    """Ordenacao deterministica; tempo nao decide empates."""

    metrics = individual.fitness
    return (
        metrics.fitness,
        metrics.mean_recall_malignant,
        -metrics.std_recall_malignant,
        metrics.mean_f1_malignant,
        metrics.mean_roc_auc,
        genome_key(individual.genome),
    )


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    global_best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity_ratio: float
    unique_individuals: int
    cache_size: int
    failure_count: int
    issue_count: int
    best_genome: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_generation(
    *,
    generation: int,
    population: list[EvaluatedIndividual],
    global_best: EvaluatedIndividual,
    cache_size: int,
) -> GenerationRecord:
    if not population:
        raise ValueError("A populacao avaliada nao pode ser vazia.")
    best = max(population, key=rank_key)
    values = [individual.fitness.fitness for individual in population]
    unique = len({genome_key(individual.genome) for individual in population})
    return GenerationRecord(
        generation=generation,
        best_fitness=best.fitness.fitness,
        global_best_fitness=global_best.fitness.fitness,
        mean_fitness=float(np.mean(values)),
        worst_fitness=float(np.min(values)),
        diversity_ratio=unique / len(population),
        unique_individuals=unique,
        cache_size=cache_size,
        failure_count=sum(not individual.fitness.succeeded for individual in population),
        issue_count=sum(len(individual.fitness.issues) for individual in population),
        best_genome=genome_to_dict(best.genome),
    )

