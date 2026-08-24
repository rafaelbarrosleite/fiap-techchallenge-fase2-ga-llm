"""Engine geracional com cache, elitismo e parada por estagnacao."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .config import GAConfig
from .fitness import FitnessResult, GeneticFitnessEvaluator
from .genomes import Genome
from .history import (
    EvaluatedIndividual,
    GenerationRecord,
    rank_key,
    summarize_generation,
)
from .operators import (
    OperatorAudit,
    create_initial_population,
    mutate_genome,
    select_elites,
    tournament_select,
    uniform_crossover,
)
from .search_spaces import GenomeSpace
from .serialization import genome_from_dict, genome_key, stable_sha256

CheckpointCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class GeneticRunResult:
    model: str
    configuration: GAConfig
    best_individual: EvaluatedIndividual
    history: tuple[GenerationRecord, ...]
    stop_reason: str
    total_seconds: float
    total_candidate_requests: int
    total_unique_evaluations: int
    unique_individuals: int
    failure_count: int
    issue_count: int
    cache_hits: int
    invalid_before_repair: int
    repaired_individuals: int

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "configuration": self.configuration.to_dict(),
            "best_individual": self.best_individual.deterministic_dict(),
            "history": [record.to_dict() for record in self.history],
            "stop_reason": self.stop_reason,
            "total_candidate_requests": self.total_candidate_requests,
            "total_unique_evaluations": self.total_unique_evaluations,
            "unique_individuals": self.unique_individuals,
            "failure_count": self.failure_count,
            "issue_count": self.issue_count,
            "cache_hits": self.cache_hits,
            "invalid_before_repair": self.invalid_before_repair,
            "repaired_individuals": self.repaired_individuals,
        }

    @property
    def reproducibility_signature(self) -> str:
        return stable_sha256(self.deterministic_payload())

    @property
    def total_model_fits(self) -> int:
        return self.total_unique_evaluations * self.configuration.cv_splits

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.deterministic_payload(),
            "best_individual": self.best_individual.to_dict(),
            "total_seconds": self.total_seconds,
            "total_model_fits": self.total_model_fits,
            "reproducibility_signature": self.reproducibility_signature,
        }


class GeneticAlgorithm:
    """Implementa o ciclo evolutivo sem biblioteca de GA."""

    def __init__(
        self,
        *,
        space: GenomeSpace[Genome],
        evaluator: GeneticFitnessEvaluator,
        configuration: GAConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        if evaluator.space.model_name != space.model_name:
            raise ValueError("Evaluator e espaco pertencem a modelos diferentes.")
        self.space = space
        self.evaluator = evaluator
        self.configuration = configuration
        self.logger = logger or logging.getLogger("tech_challenge_fase2.genetic")
        self.rng = np.random.default_rng(configuration.seed)
        self._cache: dict[str, FitnessResult] = {}
        self._candidate_requests = 0
        self._operator_audit = OperatorAudit()

    def _evaluate(self, genome: Genome) -> EvaluatedIndividual:
        self._candidate_requests += 1
        key = genome_key(genome)
        if key not in self._cache:
            self._cache[key] = self.evaluator.evaluate(genome)
        return EvaluatedIndividual(genome=genome, fitness=self._cache[key])

    def _evaluate_population(
        self,
        genomes: list[Genome],
    ) -> list[EvaluatedIndividual]:
        return [self._evaluate(genome) for genome in genomes]

    def _append_diverse_child(
        self,
        child: Genome,
        genomes: list[Genome],
        generation_keys: set[str],
    ) -> None:
        for _ in range(8):
            key = genome_key(child)
            if key not in generation_keys:
                genomes.append(child)
                generation_keys.add(key)
                return
            child = mutate_genome(
                child,
                space=self.space,
                mutation_rate=0.0,
                rng=self.rng,
                force_change=True,
                audit=self._operator_audit,
            )
        genomes.append(child)
        generation_keys.add(genome_key(child))

    def _next_generation(
        self,
        population: list[EvaluatedIndividual],
    ) -> list[Genome]:
        config = self.configuration
        elites = select_elites(population, config.elite_count)
        next_genomes = [elite.genome for elite in elites]
        generation_keys = {genome_key(genome) for genome in next_genomes}

        while len(next_genomes) < config.population_size:
            parent_a = tournament_select(
                population,
                tournament_size=config.tournament_size,
                rng=self.rng,
            ).genome
            parent_b = tournament_select(
                population,
                tournament_size=config.tournament_size,
                rng=self.rng,
            ).genome
            if self.rng.random() < config.crossover_rate:
                child_a, child_b = uniform_crossover(
                    parent_a,
                    parent_b,
                    space=self.space,
                    rng=self.rng,
                    audit=self._operator_audit,
                )
            else:
                child_a, child_b = parent_a, parent_b
            for child in (child_a, child_b):
                child = mutate_genome(
                    child,
                    space=self.space,
                    mutation_rate=config.mutation_rate,
                    rng=self.rng,
                    audit=self._operator_audit,
                )
                self._append_diverse_child(
                    child,
                    next_genomes,
                    generation_keys,
                )
                if len(next_genomes) == config.population_size:
                    break
        return next_genomes

    @staticmethod
    def _fitness_from_dict(payload: dict[str, Any]) -> FitnessResult:
        from .fitness import FoldMetrics

        return FitnessResult(
            fitness=payload["fitness"],
            base_fitness=payload["base_fitness"],
            mean_recall_malignant=payload["mean_recall_malignant"],
            std_recall_malignant=payload["std_recall_malignant"],
            mean_f1_malignant=payload["mean_f1_malignant"],
            mean_roc_auc=payload["mean_roc_auc"],
            evaluation_seconds=payload["evaluation_seconds"],
            fold_metrics=tuple(FoldMetrics(**fold) for fold in payload["fold_metrics"]),
            issues=tuple(payload.get("issues", [])),
            failure=payload.get("failure"),
        )

    @classmethod
    def _individual_from_dict(cls, payload: dict[str, Any]) -> EvaluatedIndividual:
        return EvaluatedIndividual(
            genome=genome_from_dict(payload["genome"]),
            fitness=cls._fitness_from_dict(payload["fitness"]),
        )

    def _checkpoint_payload(
        self,
        *,
        generation: int,
        population: list[EvaluatedIndividual],
        global_best: EvaluatedIndividual,
        history: list[GenerationRecord],
        stagnation: int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": "1.0",
            "model": self.space.model_name,
            "configuration": self.configuration.to_dict(),
            "completed_generation": generation,
            "population": [individual.to_dict() for individual in population],
            "global_best": global_best.to_dict(),
            "history": [record.to_dict() for record in history],
            "cache": {key: value.to_dict() for key, value in self._cache.items()},
            "rng_state": self.rng.bit_generator.state,
            "candidate_requests": self._candidate_requests,
            "stagnation": stagnation,
            "elapsed_seconds": elapsed_seconds,
            "operator_audit": {
                "invalid_before_repair": self._operator_audit.invalid_before_repair,
                "repaired_individuals": self._operator_audit.repaired_individuals,
            },
        }

    def _restore_checkpoint(
        self, payload: dict[str, Any]
    ) -> tuple[int, list[EvaluatedIndividual], EvaluatedIndividual, list[GenerationRecord], int, float]:
        if payload.get("checkpoint_schema_version") != "1.0":
            raise ValueError("Schema de checkpoint nao suportado.")
        if payload.get("model") != self.space.model_name:
            raise ValueError("Checkpoint pertence a outro modelo.")
        if payload.get("configuration") != self.configuration.to_dict():
            raise ValueError("Checkpoint pertence a outra configuracao.")
        population = [self._individual_from_dict(item) for item in payload["population"]]
        global_best = self._individual_from_dict(payload["global_best"])
        history = [GenerationRecord(**record) for record in payload["history"]]
        self._cache = {
            key: self._fitness_from_dict(value) for key, value in payload["cache"].items()
        }
        self.rng.bit_generator.state = payload["rng_state"]
        self._candidate_requests = int(payload["candidate_requests"])
        audit = payload.get("operator_audit", {})
        self._operator_audit = OperatorAudit(
            invalid_before_repair=int(audit.get("invalid_before_repair", 0)),
            repaired_individuals=int(audit.get("repaired_individuals", 0)),
        )
        return (
            int(payload["completed_generation"]),
            population,
            global_best,
            history,
            int(payload["stagnation"]),
            float(payload.get("elapsed_seconds", 0.0)),
        )

    def run(
        self,
        *,
        resume_state: dict[str, Any] | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> GeneticRunResult:
        started = perf_counter()
        config = self.configuration
        accumulated_seconds = 0.0
        if resume_state is None:
            initial_genomes = create_initial_population(
                space=self.space,
                population_size=config.population_size,
                rng=self.rng,
                audit=self._operator_audit,
            )
            population = self._evaluate_population(initial_genomes)
            global_best = max(population, key=rank_key)
            history: list[GenerationRecord] = [
                summarize_generation(
                    generation=0,
                    population=population,
                    global_best=global_best,
                    cache_size=len(self._cache),
                )
            ]
            completed_generation = 0
            stagnation = 0
            self.logger.info(
                "GA model=%s generation=0 best=%.6f unique=%s",
                self.space.model_name,
                global_best.fitness.fitness,
                len(self._cache),
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    self._checkpoint_payload(
                        generation=0,
                        population=population,
                        global_best=global_best,
                        history=history,
                        stagnation=stagnation,
                        elapsed_seconds=perf_counter() - started,
                    )
                )
        else:
            (
                completed_generation,
                population,
                global_best,
                history,
                stagnation,
                accumulated_seconds,
            ) = self._restore_checkpoint(resume_state)
            self.logger.info(
                "GA model=%s retomado_apos_geracao=%s unique=%s",
                self.space.model_name,
                completed_generation,
                len(self._cache),
            )

        stop_reason = "max_generations"
        for generation in range(completed_generation + 1, config.max_generations + 1):
            genomes = self._next_generation(population)
            population = self._evaluate_population(genomes)
            generation_best = max(population, key=rank_key)
            previous_fitness = global_best.fitness.fitness
            if rank_key(generation_best) > rank_key(global_best):
                global_best = generation_best
            if global_best.fitness.fitness > previous_fitness + config.min_improvement:
                stagnation = 0
            else:
                stagnation += 1
            history.append(
                summarize_generation(
                    generation=generation,
                    population=population,
                    global_best=global_best,
                    cache_size=len(self._cache),
                )
            )
            self.logger.info(
                "GA model=%s generation=%s best=%.6f global=%.6f unique=%s",
                self.space.model_name,
                generation,
                generation_best.fitness.fitness,
                global_best.fitness.fitness,
                len(self._cache),
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    self._checkpoint_payload(
                        generation=generation,
                        population=population,
                        global_best=global_best,
                        history=history,
                        stagnation=stagnation,
                        elapsed_seconds=accumulated_seconds + perf_counter() - started,
                    )
                )
            patience = config.stagnation_generations
            if patience is not None and stagnation >= patience:
                stop_reason = f"stagnation_{patience}_generations"
                break

        failures = sum(not result.succeeded for result in self._cache.values())
        issues = sum(len(result.issues) for result in self._cache.values())
        return GeneticRunResult(
            model=self.space.model_name,
            configuration=config,
            best_individual=global_best,
            history=tuple(history),
            stop_reason=stop_reason,
            total_seconds=accumulated_seconds + perf_counter() - started,
            total_candidate_requests=self._candidate_requests,
            total_unique_evaluations=len(self._cache),
            unique_individuals=len(self._cache),
            failure_count=failures,
            issue_count=issues,
            cache_hits=self._candidate_requests - len(self._cache),
            invalid_before_repair=self._operator_audit.invalid_before_repair,
            repaired_individuals=self._operator_audit.repaired_individuals,
        )
