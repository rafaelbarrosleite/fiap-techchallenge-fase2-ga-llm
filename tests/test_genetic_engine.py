from dataclasses import replace

import pytest

from tech_challenge_fase2.config import DEFAULT_DATA_PATH
from tech_challenge_fase2.data import load_dataset, split_development_test
from tech_challenge_fase2.genetic.config import GAConfig, smoke_config
from tech_challenge_fase2.genetic.engine import GeneticAlgorithm
from tech_challenge_fase2.genetic.fitness import FitnessResult, GeneticFitnessEvaluator
from tech_challenge_fase2.genetic.search_spaces import SPACES


def _minimal_config(seed: int = 42) -> GAConfig:
    return GAConfig(
        name="test",
        population_size=2,
        max_generations=1,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_count=1,
        tournament_size=2,
        seed=seed,
    )


def _development_data():
    X, y = load_dataset(DEFAULT_DATA_PATH)
    return split_development_test(X, y)


@pytest.mark.parametrize("model_name", sorted(SPACES))
def test_engine_evolves_all_three_models_on_development_only(model_name: str) -> None:
    split = _development_data()
    space = SPACES[model_name]
    config = _minimal_config()
    evaluator = GeneticFitnessEvaluator(
        space=space,
        X_development=split.X_development,
        y_development=split.y_development,
    )

    result = GeneticAlgorithm(
        space=space,
        evaluator=evaluator,
        configuration=config,
    ).run()

    assert result.model == model_name
    assert len(result.history) == config.max_generations + 1
    assert result.history[-1].global_best_fitness >= result.history[0].global_best_fitness
    assert result.stop_reason == "max_generations"
    assert result.total_candidate_requests == config.maximum_candidate_evaluations
    assert result.total_unique_evaluations <= result.total_candidate_requests
    assert space.is_valid(result.best_individual.genome)
    assert len(result.best_individual.fitness.fold_metrics) == 5
    assert set(evaluator.development_indices).isdisjoint(split.X_test.index)


class ConstantEvaluator:
    def __init__(self, space):
        self.space = space

    def evaluate(self, genome):
        return FitnessResult(
            fitness=0.5,
            base_fitness=0.5,
            mean_recall_malignant=0.5,
            std_recall_malignant=0.0,
            mean_f1_malignant=0.5,
            mean_roc_auc=0.5,
            evaluation_seconds=0.0,
            fold_metrics=(),
        )


def test_optional_stagnation_stops_and_records_reason() -> None:
    space = SPACES["knn"]
    config = replace(
        smoke_config(),
        max_generations=5,
        stagnation_generations=1,
    )
    result = GeneticAlgorithm(
        space=space,
        evaluator=ConstantEvaluator(space),  # type: ignore[arg-type]
        configuration=config,
    ).run()

    assert result.stop_reason == "stagnation_1_generations"
    assert len(result.history) == 2

