"""Operadores geneticos implementados explicitamente."""

from dataclasses import dataclass, fields, replace

import numpy as np

from .genomes import Genome, KNNGenome, LogisticRegressionGenome, RandomForestGenome
from .history import EvaluatedIndividual, rank_key
from .search_spaces import GenomeSpace
from .serialization import genome_key


@dataclass
class OperatorAudit:
    """Contadores observacionais; nao interferem nas decisoes aleatorias."""

    invalid_before_repair: int = 0
    repaired_individuals: int = 0


def _repair_with_audit(
    genome: Genome,
    space: GenomeSpace[Genome],
    audit: OperatorAudit | None,
) -> Genome:
    if audit is not None and space.validation_errors(genome):
        audit.invalid_before_repair += 1
    repaired = space.repair(genome)
    if audit is not None and repaired != genome:
        audit.repaired_individuals += 1
    return repaired


def create_initial_population(
    *,
    space: GenomeSpace[Genome],
    population_size: int,
    rng: np.random.Generator,
    audit: OperatorAudit | None = None,
) -> list[Genome]:
    """Amostra uma populacao valida e sem duplicatas."""

    population: list[Genome] = []
    keys: set[str] = set()
    maximum_attempts = population_size * 200
    for _ in range(maximum_attempts):
        genome = _repair_with_audit(space.sample(rng), space, audit)
        space.require_valid(genome)
        key = genome_key(genome)
        if key not in keys:
            population.append(genome)
            keys.add(key)
            if len(population) == population_size:
                return population
    raise RuntimeError("Nao foi possivel criar uma populacao inicial diversa.")


def tournament_select(
    population: list[EvaluatedIndividual],
    *,
    tournament_size: int,
    rng: np.random.Generator,
) -> EvaluatedIndividual:
    if not 2 <= tournament_size <= len(population):
        raise ValueError("Tamanho do torneio incompativel com a populacao.")
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    competitors = [population[int(index)] for index in indices]
    return max(competitors, key=rank_key)


def uniform_crossover(
    parent_a: Genome,
    parent_b: Genome,
    *,
    space: GenomeSpace[Genome],
    rng: np.random.Generator,
    audit: OperatorAudit | None = None,
) -> tuple[Genome, Genome]:
    if type(parent_a) is not type(parent_b):
        raise TypeError("O crossover exige pais do mesmo tipo.")
    if not isinstance(parent_a, space.genome_type):
        raise TypeError("Os pais nao pertencem ao espaco informado.")
    values_a: dict[str, object] = {}
    values_b: dict[str, object] = {}
    for field in fields(parent_a):
        if rng.random() < 0.5:
            values_a[field.name] = getattr(parent_a, field.name)
            values_b[field.name] = getattr(parent_b, field.name)
        else:
            values_a[field.name] = getattr(parent_b, field.name)
            values_b[field.name] = getattr(parent_a, field.name)
    child_a = _repair_with_audit(type(parent_a)(**values_a), space, audit)
    child_b = _repair_with_audit(type(parent_b)(**values_b), space, audit)
    space.require_valid(child_a)
    space.require_valid(child_b)
    return child_a, child_b


def _other_choice(
    rng: np.random.Generator,
    choices: tuple[object, ...],
    current: object,
) -> object:
    alternatives = tuple(value for value in choices if value != current)
    return alternatives[int(rng.integers(0, len(alternatives)))]


def _different_int(
    rng: np.random.Generator,
    minimum: int,
    maximum: int,
    current: int,
    *,
    step: int = 1,
) -> int:
    choices = tuple(value for value in range(minimum, maximum + 1, step) if value != current)
    return int(_other_choice(rng, choices, current))


def _mutate_logistic(
    genome: LogisticRegressionGenome,
    gene: str,
    rng: np.random.Generator,
) -> LogisticRegressionGenome:
    if gene == "log10_c":
        candidate = float(np.clip(genome.log10_c + rng.normal(0.0, 0.75), -4.0, 3.0))
        if candidate == genome.log10_c:
            candidate = -3.999999 if genome.log10_c == -4.0 else 2.999999
        return replace(genome, log10_c=candidate)
    if gene == "penalty":
        return replace(genome, penalty="l2" if genome.penalty == "l1" else "l1")
    if gene == "class_weight":
        return replace(
            genome,
            class_weight=None if genome.class_weight == "balanced" else "balanced",
        )
    raise ValueError(f"Gene desconhecido da Regressao Logistica: {gene}")


def _mutate_forest(
    genome: RandomForestGenome,
    gene: str,
    rng: np.random.Generator,
) -> RandomForestGenome:
    if gene == "n_estimators":
        return replace(
            genome,
            n_estimators=_different_int(rng, 100, 500, genome.n_estimators),
        )
    if gene == "max_depth":
        choices: tuple[object, ...] = (None, *range(3, 21))
        return replace(
            genome,
            max_depth=_other_choice(rng, choices, genome.max_depth),  # type: ignore[arg-type]
        )
    if gene == "min_samples_split":
        return replace(
            genome,
            min_samples_split=_different_int(
                rng, 2, 20, genome.min_samples_split
            ),
        )
    if gene == "min_samples_leaf":
        return replace(
            genome,
            min_samples_leaf=_different_int(rng, 1, 10, genome.min_samples_leaf),
        )
    if gene == "max_features":
        return replace(
            genome,
            max_features=_other_choice(
                rng,
                ("sqrt", "log2", 0.5, 1.0),
                genome.max_features,
            ),  # type: ignore[arg-type]
        )
    if gene == "class_weight":
        return replace(
            genome,
            class_weight=_other_choice(
                rng,
                (None, "balanced", "balanced_subsample"),
                genome.class_weight,
            ),  # type: ignore[arg-type]
        )
    raise ValueError(f"Gene desconhecido da Random Forest: {gene}")


def _mutate_knn(
    genome: KNNGenome,
    gene: str,
    rng: np.random.Generator,
) -> KNNGenome:
    if gene == "n_neighbors":
        return replace(
            genome,
            n_neighbors=_different_int(
                rng, 3, 31, genome.n_neighbors, step=2
            ),
        )
    if gene == "weights":
        return replace(
            genome,
            weights="distance" if genome.weights == "uniform" else "uniform",
        )
    if gene == "metric":
        metric = _other_choice(
            rng,
            ("minkowski", "euclidean", "manhattan"),
            genome.metric,
        )
        return replace(
            genome,
            metric=metric,  # type: ignore[arg-type]
            p=2 if metric == "minkowski" else None,
        )
    if gene == "p":
        if genome.metric != "minkowski":
            raise ValueError("p so pode sofrer mutacao com metrica minkowski.")
        return replace(genome, p=1 if genome.p == 2 else 2)
    raise ValueError(f"Gene desconhecido do KNN: {gene}")


def _active_gene_names(genome: Genome, space: GenomeSpace[Genome]) -> tuple[str, ...]:
    names = space.gene_names
    if isinstance(genome, KNNGenome) and genome.metric != "minkowski":
        return tuple(name for name in names if name != "p")
    return names


def mutate_genome(
    genome: Genome,
    *,
    space: GenomeSpace[Genome],
    mutation_rate: float,
    rng: np.random.Generator,
    force_change: bool = False,
    audit: OperatorAudit | None = None,
) -> Genome:
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate deve estar entre 0 e 1.")
    space.require_valid(genome)
    mutated = genome
    active_genes = _active_gene_names(mutated, space)
    selected = [gene for gene in active_genes if rng.random() < mutation_rate]
    if force_change and not selected:
        selected = [active_genes[int(rng.integers(0, len(active_genes)))]]
    for gene in selected:
        if isinstance(mutated, LogisticRegressionGenome):
            mutated = _mutate_logistic(mutated, gene, rng)
        elif isinstance(mutated, RandomForestGenome):
            mutated = _mutate_forest(mutated, gene, rng)
        elif isinstance(mutated, KNNGenome):
            if gene == "p" and mutated.metric != "minkowski":
                continue
            mutated = _mutate_knn(mutated, gene, rng)
        else:
            raise TypeError(f"Tipo de genoma nao suportado: {type(mutated)}")
        mutated = _repair_with_audit(mutated, space, audit)
    if force_change and mutated == genome:
        gene = active_genes[int(rng.integers(0, len(active_genes)))]
        return _force_single_mutation(genome, gene, space, rng, audit)
    space.require_valid(mutated)
    return mutated


def _force_single_mutation(
    genome: Genome,
    gene: str,
    space: GenomeSpace[Genome],
    rng: np.random.Generator,
    audit: OperatorAudit | None = None,
) -> Genome:
    if isinstance(genome, LogisticRegressionGenome):
        mutated: Genome = _mutate_logistic(genome, gene, rng)
    elif isinstance(genome, RandomForestGenome):
        mutated = _mutate_forest(genome, gene, rng)
    elif isinstance(genome, KNNGenome):
        mutated = _mutate_knn(genome, gene, rng)
    else:
        raise TypeError(f"Tipo de genoma nao suportado: {type(genome)}")
    mutated = _repair_with_audit(mutated, space, audit)
    if mutated == genome:
        raise RuntimeError("A mutacao forcada nao alterou o genoma.")
    space.require_valid(mutated)
    return mutated


def select_elites(
    population: list[EvaluatedIndividual],
    elite_count: int,
) -> list[EvaluatedIndividual]:
    if not 1 <= elite_count < len(population):
        raise ValueError("elite_count incompativel com a populacao.")
    return sorted(population, key=rank_key, reverse=True)[:elite_count]
