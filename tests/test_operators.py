import numpy as np
import pytest

from tech_challenge_fase2.genetic.fitness import FitnessResult
from tech_challenge_fase2.genetic.history import EvaluatedIndividual
from tech_challenge_fase2.genetic.operators import (
    create_initial_population,
    mutate_genome,
    select_elites,
    tournament_select,
    uniform_crossover,
)
from tech_challenge_fase2.genetic.search_spaces import SPACES
from tech_challenge_fase2.genetic.serialization import genome_key


def _fitness(value: float) -> FitnessResult:
    return FitnessResult(
        fitness=value,
        base_fitness=value,
        mean_recall_malignant=value,
        std_recall_malignant=0.0,
        mean_f1_malignant=value,
        mean_roc_auc=value,
        evaluation_seconds=0.0,
        fold_metrics=(),
    )


@pytest.mark.parametrize("model_name", sorted(SPACES))
def test_initial_population_is_valid_and_unique(model_name: str) -> None:
    space = SPACES[model_name]
    population = create_initial_population(
        space=space,
        population_size=12,
        rng=np.random.default_rng(42),
    )

    assert len(population) == 12
    assert len({genome_key(genome) for genome in population}) == 12
    assert all(space.is_valid(genome) for genome in population)


def test_tournament_selects_the_best_when_all_compete() -> None:
    space = SPACES["logistic_regression"]
    genomes = create_initial_population(
        space=space,
        population_size=4,
        rng=np.random.default_rng(42),
    )
    population = [
        EvaluatedIndividual(genome, _fitness(float(index)))
        for index, genome in enumerate(genomes)
    ]

    selected = tournament_select(
        population,
        tournament_size=len(population),
        rng=np.random.default_rng(5),
    )

    assert selected.fitness.fitness == 3.0


@pytest.mark.parametrize("model_name", sorted(SPACES))
def test_uniform_crossover_preserves_types_and_validity(model_name: str) -> None:
    space = SPACES[model_name]
    parents = create_initial_population(
        space=space,
        population_size=2,
        rng=np.random.default_rng(7),
    )

    children = uniform_crossover(
        parents[0],
        parents[1],
        space=space,
        rng=np.random.default_rng(8),
    )

    assert all(type(child) is type(parents[0]) for child in children)
    assert all(space.is_valid(child) for child in children)


@pytest.mark.parametrize("model_name", sorted(SPACES))
def test_typed_mutation_changes_genome_and_keeps_limits(model_name: str) -> None:
    space = SPACES[model_name]
    rng = np.random.default_rng(11)
    genome = space.sample(rng)

    for _ in range(30):
        mutated = mutate_genome(
            genome,
            space=space,
            mutation_rate=0.0,
            rng=rng,
            force_change=True,
        )
        assert mutated != genome
        assert type(mutated) is type(genome)
        assert space.is_valid(mutated)
        genome = mutated


def test_mutation_can_leave_and_enter_optional_none() -> None:
    space = SPACES["random_forest"]
    rng = np.random.default_rng(12)
    genome = space.sample(rng)
    genome = type(genome)(
        n_estimators=genome.n_estimators,
        max_depth=None,
        min_samples_split=genome.min_samples_split,
        min_samples_leaf=genome.min_samples_leaf,
        max_features=genome.max_features,
        class_weight=genome.class_weight,
    )
    mutated = mutate_genome(
        genome,
        space=space,
        mutation_rate=1.0,
        rng=rng,
    )

    assert mutated.max_depth is not None
    assert space.is_valid(mutated)

    with_depth = type(genome)(
        n_estimators=genome.n_estimators,
        max_depth=10,
        min_samples_split=genome.min_samples_split,
        min_samples_leaf=genome.min_samples_leaf,
        max_features=genome.max_features,
        class_weight=genome.class_weight,
    )
    entered_none = any(
        mutate_genome(
            with_depth,
            space=space,
            mutation_rate=1.0,
            rng=np.random.default_rng(seed),
        ).max_depth
        is None
        for seed in range(100)
    )
    assert entered_none


def test_elitism_preserves_highest_ranked_individuals() -> None:
    space = SPACES["knn"]
    genomes = create_initial_population(
        space=space,
        population_size=5,
        rng=np.random.default_rng(42),
    )
    population = [
        EvaluatedIndividual(genome, _fitness(float(index)))
        for index, genome in enumerate(genomes)
    ]

    elites = select_elites(population, 2)

    assert [elite.fitness.fitness for elite in elites] == [4.0, 3.0]
