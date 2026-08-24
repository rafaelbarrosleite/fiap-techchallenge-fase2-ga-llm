"""Comparacoes por CV entre baseline, algoritmo genetico e busca aleatoria."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import DEFAULT_DATA_PATH, PROJECT_ROOT, RANDOM_STATE
from .data import file_sha256, load_dataset, split_development_test
from .genetic.fitness import calculate_fitness
from .genetic.official import (
    CONFIG_ORDER,
    MODEL_ORDER,
    experiment_paths,
    software_versions,
    validate_official_artifact,
)
from .genetic.search_spaces import SPACES
from .genetic.serialization import genome_key, genome_to_dict, save_json, stable_sha256
from .models import MODEL_FACTORIES

SELECTION_TOLERANCE = 1e-12


def canonical_candidate_key(model_name: str, parameters: dict[str, Any]) -> str:
    """Normaliza chaves equivalentes entre baseline, GA e busca aleatoria."""

    if model_name == "logistic_regression":
        if "C" in parameters:
            c_value = float(parameters["C"])
        else:
            c_value = 10.0 ** float(parameters["log10_c"])
        normalized = {
            "model": model_name,
            "C": c_value,
            "penalty": parameters.get("penalty", "l2"),
            "solver": parameters.get("solver", "liblinear"),
            "class_weight": parameters.get("class_weight"),
        }
    elif model_name == "random_forest":
        normalized = {"model": model_name}
        normalized.update(
            {
                key: parameters.get(key)
                for key in (
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "class_weight",
            )
            }
        )
    else:
        metric = parameters.get("metric", "minkowski")
        normalized = {
            "model": model_name,
            "n_neighbors": parameters.get("n_neighbors"),
            "weights": parameters.get("weights"),
            "metric": metric,
            "p": parameters.get("p") if metric == "minkowski" else None,
        }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _folds(X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return tuple(splitter.split(X, y))


def evaluate_estimator_cv(
    estimator_factory: Callable[[], BaseEstimator],
    X_development: pd.DataFrame,
    y_development: pd.Series,
    *,
    cv_seed: int = RANDOM_STATE,
    instability_weight: float = 0.10,
) -> dict[str, Any]:
    """Avalia uma Pipeline nos mesmos folds e com a mesma formula do GA."""

    started = perf_counter()
    fold_metrics: list[dict[str, Any]] = []
    issues: list[str] = []
    for fold_number, (train_positions, validation_positions) in enumerate(
        _folds(X_development, y_development, cv_seed), start=1
    ):
        estimator = estimator_factory()
        X_train = X_development.iloc[train_positions]
        y_train = y_development.iloc[train_positions]
        X_validation = X_development.iloc[validation_positions]
        y_validation = y_development.iloc[validation_positions]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)
        issues.extend(
            f"fold={fold_number};category={item.category.__name__};message={item.message}"
            for item in caught
        )
        prediction = estimator.predict(X_validation)
        probability = estimator.predict_proba(X_validation)[:, 1]
        fold_metrics.append(
            {
                "fold": fold_number,
                "recall_malignant": float(
                    recall_score(y_validation, prediction, pos_label=1, zero_division=0)
                ),
                "f1_malignant": float(
                    f1_score(y_validation, prediction, pos_label=1, zero_division=0)
                ),
                "roc_auc": float(roc_auc_score(y_validation, probability)),
                "train_rows": len(train_positions),
                "validation_rows": len(validation_positions),
            }
        )
    recalls = np.array([fold["recall_malignant"] for fold in fold_metrics])
    mean_recall = float(recalls.mean())
    std_recall = float(recalls.std(ddof=0))
    mean_f1 = float(np.mean([fold["f1_malignant"] for fold in fold_metrics]))
    mean_auc = float(np.mean([fold["roc_auc"] for fold in fold_metrics]))
    fitness, base_fitness = calculate_fitness(
        mean_recall_malignant=mean_recall,
        std_recall_malignant=std_recall,
        mean_f1_malignant=mean_f1,
        mean_roc_auc=mean_auc,
        instability_weight=instability_weight,
    )
    return {
        "fitness": fitness,
        "base_fitness": base_fitness,
        "mean_recall_malignant": mean_recall,
        "std_recall_malignant": std_recall,
        "mean_f1_malignant": mean_f1,
        "mean_roc_auc": mean_auc,
        "evaluation_seconds": perf_counter() - started,
        "fold_metrics": fold_metrics,
        "issues": issues,
        "failure": None,
    }


def run_comparable_baselines(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path | None = None,
) -> dict[str, Any]:
    X, y = load_dataset(data_path)
    split = split_development_test(X, y)
    models: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        estimator = MODEL_FACTORIES[model_name]()
        model = estimator.named_steps["model"]
        if model_name == "logistic_regression":
            parameters = {
                "C": model.C,
                "penalty": model.penalty,
                "solver": model.solver,
                "class_weight": model.class_weight,
                "max_iter": model.max_iter,
            }
        elif model_name == "random_forest":
            parameters = {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
                "max_features": model.max_features,
                "class_weight": model.class_weight,
            }
        else:
            parameters = {
                "n_neighbors": model.n_neighbors,
                "weights": model.weights,
                "metric": model.metric,
                "p": model.p,
            }
        models[model_name] = {
            "origin": "baseline_cv",
            "parameters": parameters,
            "metrics": evaluate_estimator_cv(
                MODEL_FACTORIES[model_name],
                split.X_development,
                split.y_development,
            ),
            "candidate_evaluations": 1,
            "model_fits": 5,
        }
    payload = {
        "schema_version": "1.0",
        "artifact_type": "comparable_cv_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": file_sha256(Path(data_path)),
        "data_scope": {
            "scope": "development_only",
            "holdout_used": False,
            "development_rows": len(split.X_development),
            "held_out_rows": len(split.X_test),
            "cv_splits": 5,
            "cv_seed": 42,
            "classification_threshold": 0.5,
        },
        "software_versions": software_versions(),
        "models": models,
    }
    payload["signature"] = stable_sha256(payload)
    save_json(
        payload,
        output_path
        or PROJECT_ROOT / "artifacts" / "comparison" / "baseline_cv.json",
    )
    return payload


def load_official_artifacts(
    *, artifact_root: Path | None = None
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for label in CONFIG_ORDER:
        for model_name in MODEL_ORDER:
            path = experiment_paths(
                model_name, label, artifact_root=artifact_root
            )["artifact"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_official_artifact(payload)
            artifacts.append(payload)
    if len(artifacts) != 9:
        raise ValueError("A agregacao oficial exige exatamente nove experimentos.")
    return artifacts


def _round_for_tolerance(value: float) -> float:
    return round(float(value), 12)


def _complexity_key(candidate: dict[str, Any]) -> tuple[float, float]:
    if candidate["model"] != "random_forest":
        return (0.0, 0.0)
    params = candidate.get("parameters", {})
    n_estimators = params.get("n_estimators", params.get("model__n_estimators", 10**9))
    max_depth = params.get("max_depth", params.get("model__max_depth"))
    depth = 10**9 if max_depth is None else int(max_depth)
    return (-float(n_estimators), -float(depth))


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        _round_for_tolerance(metrics["fitness"]),
        _round_for_tolerance(metrics["mean_recall_malignant"]),
        -_round_for_tolerance(metrics["std_recall_malignant"]),
        _round_for_tolerance(metrics["mean_f1_malignant"]),
        _round_for_tolerance(metrics["mean_roc_auc"]),
        *_complexity_key(candidate),
        candidate["canonical_key"],
    )


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("A selecao exige pelo menos um candidato.")
    return max(candidates, key=candidate_rank_key)


def ga_candidate(artifact: dict[str, Any]) -> dict[str, Any]:
    run = artifact["run"]
    genome = run["best_individual"]["genome"]
    parameters = {key: value for key, value in genome.items() if key != "model"}
    return {
        "model": run["model"],
        "origin": f"GA_{artifact['identity']['config_label']}",
        "parameters": parameters,
        "metrics": run["best_individual"]["fitness"],
        "canonical_key": canonical_candidate_key(run["model"], parameters),
        "candidate_evaluations": run["total_unique_evaluations"],
        "model_fits": run["total_model_fits"],
        "duration_seconds": run["total_seconds"],
        "source_signature": run["reproducibility_signature"],
    }


def best_ga_by_model(
    artifacts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        candidates = [
            ga_candidate(artifact)
            for artifact in artifacts
            if artifact["run"]["model"] == model_name
        ]
        if len(candidates) != 3:
            raise ValueError(f"Esperados tres experimentos GA para {model_name}.")
        result[model_name] = select_best_candidate(candidates)
    return result


def _unique_random_genomes(model_name: str, budget: int, seed: int):
    space = SPACES[model_name]
    rng = np.random.default_rng(seed)
    population = []
    seen: set[str] = set()
    # 15 valores impares de k x 2 pesos x (2 Minkowski + Euclidiana + Manhattan).
    maximum_unique = 120 if model_name == "knn" else None
    effective_budget = min(budget, maximum_unique or budget)
    while len(population) < effective_budget:
        genome = space.repair(space.sample(rng))
        key = genome_key(genome)
        if key not in seen:
            seen.add(key)
            population.append(genome)
    return population, effective_budget


def _pipeline_parameters(genome) -> dict[str, Any]:
    payload = genome_to_dict(genome)
    model_name = payload.pop("model")
    if model_name == "logistic_regression":
        return {
            "model__C": 10.0 ** payload["log10_c"],
            "model__penalty": payload["penalty"],
            "model__class_weight": payload["class_weight"],
        }
    if model_name == "random_forest":
        return {f"model__{key}": value for key, value in payload.items()}
    parameters = {
        "model__n_neighbors": payload["n_neighbors"],
        "model__weights": payload["weights"],
        "model__metric": payload["metric"],
    }
    if payload["metric"] == "minkowski":
        parameters["model__p"] = payload["p"]
    return parameters


def composite_cv_candidates(
    cv_results: dict[str, Any], *, model_name: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, params in enumerate(cv_results["params"]):
        mean_recall = float(cv_results["mean_test_recall_malignant"][index])
        std_recall = float(cv_results["std_test_recall_malignant"][index])
        mean_f1 = float(cv_results["mean_test_f1_malignant"][index])
        mean_auc = float(cv_results["mean_test_roc_auc"][index])
        fitness, base = calculate_fitness(
            mean_recall_malignant=mean_recall,
            std_recall_malignant=std_recall,
            mean_f1_malignant=mean_f1,
            mean_roc_auc=mean_auc,
        )
        folds = [
            {
                "fold": fold + 1,
                "recall_malignant": float(
                    cv_results[f"split{fold}_test_recall_malignant"][index]
                ),
                "f1_malignant": float(
                    cv_results[f"split{fold}_test_f1_malignant"][index]
                ),
                "roc_auc": float(cv_results[f"split{fold}_test_roc_auc"][index]),
            }
            for fold in range(5)
        ]
        normalized_params = {
            key.removeprefix("model__"): value for key, value in params.items()
        }
        candidates.append(
            {
                "model": model_name,
                "origin": "RandomizedSearchCV",
                "parameters": normalized_params,
                "metrics": {
                    "fitness": fitness,
                    "base_fitness": base,
                    "mean_recall_malignant": mean_recall,
                    "std_recall_malignant": std_recall,
                    "mean_f1_malignant": mean_f1,
                    "mean_roc_auc": mean_auc,
                    "fold_metrics": folds,
                    "issues": [],
                    "failure": None,
                },
                "canonical_key": canonical_candidate_key(
                    model_name, normalized_params
                ),
                "cv_index": index,
            }
        )
    return candidates


def run_randomized_comparisons(
    *,
    ga_winners: dict[str, dict[str, Any]],
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Usa RandomizedSearchCV sem refit final e selecao composta auditavel."""

    X, y = load_dataset(data_path)
    split = split_development_test(X, y)
    results: dict[str, Any] = {}
    scoring = {
        "recall_malignant": make_scorer(
            recall_score, pos_label=1, zero_division=0
        ),
        "f1_malignant": make_scorer(f1_score, pos_label=1, zero_division=0),
        "roc_auc": "roc_auc",
    }
    folds = _folds(split.X_development, split.y_development, seed)
    for model_name in MODEL_ORDER:
        requested_budget = int(ga_winners[model_name]["candidate_evaluations"])
        genomes, effective_budget = _unique_random_genomes(
            model_name, requested_budget, seed
        )
        parameter_distributions = [
            {key: [value] for key, value in _pipeline_parameters(genome).items()}
            for genome in genomes
        ]
        search = RandomizedSearchCV(
            estimator=SPACES[model_name].build_estimator(
                genomes[0], random_state=seed
            ),
            param_distributions=parameter_distributions,
            n_iter=effective_budget,
            scoring=scoring,
            refit=False,
            cv=folds,
            random_state=seed,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(split.X_development, split.y_development)
        duration = perf_counter() - started
        candidates = composite_cv_candidates(
            search.cv_results_, model_name=model_name
        )
        winner = select_best_candidate(candidates)
        winner.update(
            {
                "candidate_evaluations": effective_budget,
                "model_fits": effective_budget * 5,
                "duration_seconds": duration,
            }
        )
        results[model_name] = {
            "requested_budget": requested_budget,
            "effective_unique_budget": effective_budget,
            "budget_limitation": (
                "Espaco KNN possui somente 120 combinacoes unicas."
                if effective_budget < requested_budget
                else None
            ),
            "refit_performed": False,
            "selection_mechanism": (
                "RandomizedSearchCV multi-scorer com refit=False; vencedor "
                "calculado por fitness composto e desempate deterministico."
            ),
            "warnings": [
                f"{item.category.__name__}: {item.message}" for item in caught
            ],
            "winner": winner,
            "candidate_count": len(candidates),
            "candidates_signature": stable_sha256(candidates),
        }
    payload = {
        "schema_version": "1.0",
        "artifact_type": "randomized_search_cv_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": file_sha256(Path(data_path)),
        "data_scope": {
            "scope": "development_only",
            "holdout_used": False,
            "cv_splits": 5,
            "cv_seed": seed,
            "classification_threshold": 0.5,
        },
        "selection_tolerance": SELECTION_TOLERANCE,
        "software_versions": software_versions(),
        "models": results,
    }
    payload["signature"] = stable_sha256(payload)
    save_json(
        payload,
        output_path
        or PROJECT_ROOT / "artifacts" / "comparison" / "randomized_search_cv.json",
    )
    return payload
