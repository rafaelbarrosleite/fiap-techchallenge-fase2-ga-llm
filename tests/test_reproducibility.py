import numpy as np

from tech_challenge_fase2.config import DEFAULT_DATA_PATH
from tech_challenge_fase2.data import load_dataset, split_development_test
from tech_challenge_fase2.genetic.config import GAConfig
from tech_challenge_fase2.genetic.engine import GeneticAlgorithm
from tech_challenge_fase2.genetic.fitness import GeneticFitnessEvaluator
from tech_challenge_fase2.genetic.operators import create_initial_population
from tech_challenge_fase2.genetic.search_spaces import SPACES
from tech_challenge_fase2.genetic.serialization import genome_key


def _run(seed: int):
    X, y = load_dataset(DEFAULT_DATA_PATH)
    split = split_development_test(X, y)
    space = SPACES["logistic_regression"]
    config = GAConfig(
        name="reproducibility_test",
        population_size=3,
        max_generations=1,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_count=1,
        tournament_size=2,
        seed=seed,
    )
    evaluator = GeneticFitnessEvaluator(
        space=space,
        X_development=split.X_development,
        y_development=split.y_development,
    )
    return GeneticAlgorithm(
        space=space,
        evaluator=evaluator,
        configuration=config,
    ).run()


def test_same_seed_produces_same_signature_and_history() -> None:
    first = _run(42)
    second = _run(42)

    assert first.reproducibility_signature == second.reproducibility_signature
    assert first.best_individual.genome == second.best_individual.genome
    assert first.deterministic_payload() == second.deterministic_payload()


def test_different_seed_changes_initial_diversity() -> None:
    space = SPACES["logistic_regression"]
    first = create_initial_population(
        space=space,
        population_size=8,
        rng=np.random.default_rng(42),
    )
    second = create_initial_population(
        space=space,
        population_size=8,
        rng=np.random.default_rng(43),
    )

    assert [genome_key(genome) for genome in first] != [
        genome_key(genome) for genome in second
    ]

