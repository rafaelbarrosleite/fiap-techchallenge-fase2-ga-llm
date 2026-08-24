import json
from pathlib import Path

import numpy as np
import pytest

from tech_challenge_fase2.comparison import (
    canonical_candidate_key,
    candidate_rank_key,
    composite_cv_candidates,
    evaluate_estimator_cv,
    load_official_artifacts,
    select_best_candidate,
)
from tech_challenge_fase2.config import DEFAULT_DATA_PATH
from tech_challenge_fase2.data import load_dataset, split_development_test
from tech_challenge_fase2.genetic.config import GAConfig
from tech_challenge_fase2.genetic.fitness import calculate_fitness
from tech_challenge_fase2.genetic.official import (
    CONFIG_ORDER,
    MODEL_ORDER,
    completed_artifact_matches,
    completed_experiment_matches,
    create_execution_manifest,
    experiment_identity,
    official_configuration,
    run_official_battery,
    run_official_experiment,
    validate_official_artifact,
)
from tech_challenge_fase2.genetic.serialization import stable_sha256
from tech_challenge_fase2.models import logistic_regression_pipeline
from tech_challenge_fase2.reporting import freeze_candidates, generate_report_figures


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("A", (20, 10, 0.70, 0.10, 2, 3)),
        ("B", (40, 20, 0.80, 0.20, 2, 3)),
        ("C", (60, 30, 0.75, 0.30, 4, 4)),
    ],
)
def test_official_configurations_are_locked(label, expected) -> None:
    config = official_configuration(label)
    assert (
        config.population_size,
        config.max_generations,
        config.crossover_rate,
        config.mutation_rate,
        config.elite_count,
        config.tournament_size,
    ) == expected
    assert (config.seed, config.cv_splits, config.cv_seed, config.estimator_seed) == (
        42,
        5,
        42,
        42,
    )


def test_comparable_baseline_uses_same_five_fold_fitness() -> None:
    X, y = load_dataset(DEFAULT_DATA_PATH)
    split = split_development_test(X, y)
    metrics = evaluate_estimator_cv(
        logistic_regression_pipeline,
        split.X_development,
        split.y_development,
    )
    expected, base = calculate_fitness(
        mean_recall_malignant=metrics["mean_recall_malignant"],
        std_recall_malignant=metrics["std_recall_malignant"],
        mean_f1_malignant=metrics["mean_f1_malignant"],
        mean_roc_auc=metrics["mean_roc_auc"],
    )
    assert len(metrics["fold_metrics"]) == 5
    assert metrics["fitness"] == expected
    assert metrics["base_fitness"] == base


def _tiny_config() -> GAConfig:
    return GAConfig(
        name="tiny_official_test",
        population_size=2,
        max_generations=1,
        crossover_rate=0.7,
        mutation_rate=0.1,
        elite_count=1,
        tournament_size=2,
    )


def test_official_artifact_resume_and_holdout_isolation(tmp_path: Path) -> None:
    first = run_official_experiment(
        model_name="logistic_regression",
        config_label="A",
        artifact_root=tmp_path,
        configuration_override=_tiny_config(),
    )
    validate_official_artifact(first)
    assert first["data_scope"]["holdout_used"] is False
    assert first["data_scope"]["holdout_accessible_to_fitness"] is False
    assert len(first["run"]["history"]) == 2
    assert first["run"]["cache_hits"] == (
        first["run"]["total_candidate_requests"]
        - first["run"]["total_unique_evaluations"]
    )

    artifact_path = next((tmp_path / "experiments").glob("*.json"))
    status_path = next((tmp_path / "status").glob("*.json"))
    checkpoint_path = next((tmp_path / "checkpoints").glob("*.json"))
    identity = first["identity"]["sha256"]
    assert completed_experiment_matches(artifact_path, status_path, identity)

    artifact_path.unlink()
    resumed = run_official_experiment(
        model_name="logistic_regression",
        config_label="A",
        artifact_root=tmp_path,
        configuration_override=_tiny_config(),
    )
    assert checkpoint_path.is_file()
    assert resumed["run"]["reproducibility_signature"] == first["run"][
        "reproducibility_signature"
    ]


def test_partial_artifact_is_never_completed(tmp_path: Path) -> None:
    artifact = tmp_path / "partial.json"
    status = tmp_path / "status.json"
    artifact.write_text('{"schema_version":', encoding="utf-8")
    status.write_text('{"status":"completed"}', encoding="utf-8")
    assert not completed_artifact_matches(artifact, "0" * 64)
    assert not completed_experiment_matches(artifact, status, "0" * 64)


def test_configuration_change_invalidates_identity() -> None:
    common = {
        "model_name": "knn",
        "config_label": "A",
        "dataset_sha256": "d" * 64,
        "development_indices_sha256": "i" * 64,
        "code_sha256": "c" * 64,
    }
    first = experiment_identity(configuration=official_configuration("A"), **common)
    changed = experiment_identity(
        configuration=GAConfig(
            name="changed",
            population_size=20,
            max_generations=11,
            crossover_rate=0.7,
            mutation_rate=0.1,
            elite_count=2,
            tournament_size=3,
        ),
        **common,
    )
    assert first["sha256"] != changed["sha256"]


def test_battery_runs_nine_in_required_serial_order(monkeypatch) -> None:
    calls = []

    def fake_run(**kwargs):
        calls.append((kwargs["config_label"], kwargs["model_name"]))
        return {"ok": True}

    monkeypatch.setattr(
        "tech_challenge_fase2.genetic.official.run_official_experiment", fake_run
    )
    monkeypatch.setattr(
        "tech_challenge_fase2.genetic.official.validate_official_artifact",
        lambda payload: None,
    )
    results = run_official_battery()
    assert calls == [
        (label, model) for label in CONFIG_ORDER for model in MODEL_ORDER
    ]
    assert len(results) == 9


def test_aggregation_requires_and_loads_nine(monkeypatch, tmp_path: Path) -> None:
    paths = {}
    for label in CONFIG_ORDER:
        for model in MODEL_ORDER:
            path = tmp_path / f"{label}_{model}.json"
            path.write_text(json.dumps({"label": label, "model": model}), encoding="utf-8")
            paths[(model, label)] = path

    monkeypatch.setattr(
        "tech_challenge_fase2.comparison.experiment_paths",
        lambda model, label, artifact_root=None: {"artifact": paths[(model, label)]},
    )
    monkeypatch.setattr(
        "tech_challenge_fase2.comparison.validate_official_artifact",
        lambda payload: None,
    )
    assert len(load_official_artifacts(artifact_root=tmp_path)) == 9


def test_composite_randomized_fitness_and_deterministic_tie_break() -> None:
    cv_results = {
        "params": [{"model__n_neighbors": 3}, {"model__n_neighbors": 5}],
        "mean_test_recall_malignant": np.array([0.9, 0.9]),
        "std_test_recall_malignant": np.array([0.1, 0.0]),
        "mean_test_f1_malignant": np.array([0.8, 0.8]),
        "mean_test_roc_auc": np.array([0.95, 0.95]),
    }
    for fold in range(5):
        cv_results[f"split{fold}_test_recall_malignant"] = np.array([0.9, 0.9])
        cv_results[f"split{fold}_test_f1_malignant"] = np.array([0.8, 0.8])
        cv_results[f"split{fold}_test_roc_auc"] = np.array([0.95, 0.95])
    candidates = composite_cv_candidates(cv_results, model_name="knn")
    assert candidates[1]["metrics"]["fitness"] > candidates[0]["metrics"]["fitness"]
    assert select_best_candidate(candidates) == candidates[1]

    tied = [dict(candidates[1]), dict(candidates[1])]
    tied[0]["canonical_key"] = "a"
    tied[1]["canonical_key"] = "b"
    assert candidate_rank_key(tied[1]) > candidate_rank_key(tied[0])


def test_canonical_key_is_equal_for_equivalent_cross_method_parameters() -> None:
    ga_parameters = {
        "n_neighbors": 3,
        "weights": "uniform",
        "metric": "minkowski",
        "p": 1,
    }
    randomized_parameters = {
        "weights": "uniform",
        "p": 1,
        "n_neighbors": 3,
        "metric": "minkowski",
    }
    assert canonical_candidate_key("knn", ga_parameters) == canonical_candidate_key(
        "knn", randomized_parameters
    )


def _synthetic_inputs():
    artifacts = []
    for label_index, label in enumerate(CONFIG_ORDER):
        for model_index, model in enumerate(MODEL_ORDER):
            fitness = 0.90 + label_index * 0.01 + model_index * 0.001
            genome = (
                {"model": model, "log10_c": 0.0, "penalty": "l2", "class_weight": None}
                if model == "logistic_regression"
                else {
                    "model": model,
                    "n_estimators": 100,
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "class_weight": None,
                }
                if model == "random_forest"
                else {
                    "model": model,
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "metric": "minkowski",
                    "p": 2,
                }
            )
            metrics = {
                "fitness": fitness,
                "base_fitness": fitness + 0.002,
                "mean_recall_malignant": fitness,
                "std_recall_malignant": 0.02,
                "mean_f1_malignant": fitness,
                "mean_roc_auc": 0.98,
                "fold_metrics": [
                    {
                        "fold": fold,
                        "recall_malignant": fitness,
                        "f1_malignant": fitness,
                        "roc_auc": 0.98,
                    }
                    for fold in range(1, 6)
                ],
                "issues": [],
                "failure": None,
            }
            history = [
                {
                    "generation": generation,
                    "best_fitness": fitness,
                    "global_best_fitness": fitness,
                    "mean_fitness": fitness - 0.01,
                    "worst_fitness": fitness - 0.02,
                    "diversity_ratio": 1.0 - generation * 0.1,
                    "unique_individuals": 10,
                    "cache_size": 10,
                    "failure_count": 0,
                    "issue_count": 0,
                    "best_genome": genome,
                }
                for generation in range(3)
            ]
            artifacts.append(
                {
                    "identity": {"config_label": label},
                    "run": {
                        "model": model,
                        "best_individual": {"genome": genome, "fitness": metrics},
                        "history": history,
                        "total_unique_evaluations": 10,
                        "total_model_fits": 50,
                        "total_seconds": 12.0,
                        "reproducibility_signature": f"{label}{model}",
                    },
                }
            )
    baseline_models = {}
    randomized_models = {}
    for index, model in enumerate(MODEL_ORDER):
        parameters = (
            {"C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None}
            if model == "logistic_regression"
            else {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": None,
            }
            if model == "random_forest"
            else {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski",
                "p": 2,
            }
        )
        metrics = {
            "fitness": 0.88 + index * 0.001,
            "base_fitness": 0.882 + index * 0.001,
            "mean_recall_malignant": 0.88,
            "std_recall_malignant": 0.02,
            "mean_f1_malignant": 0.89,
            "mean_roc_auc": 0.97,
            "evaluation_seconds": 1.0,
            "fold_metrics": [],
        }
        baseline_models[model] = {
            "parameters": parameters,
            "metrics": metrics,
            "candidate_evaluations": 1,
            "model_fits": 5,
        }
        random_winner = {
            "model": model,
            "origin": "RandomizedSearchCV",
            "parameters": parameters,
            "metrics": {**metrics, "fitness": 0.89 + index * 0.001},
            "canonical_key": model,
            "candidate_evaluations": 10,
            "model_fits": 50,
            "duration_seconds": 2.0,
        }
        randomized_models[model] = {"winner": random_winner}
    baseline = {
        "dataset_sha256": "d" * 64,
        "signature": "b" * 64,
        "models": baseline_models,
    }
    randomized = {"signature": "r" * 64, "models": randomized_models}
    return artifacts, baseline, randomized


def test_freezing_and_synthetic_chart_generation(tmp_path: Path) -> None:
    artifacts, baseline, randomized = _synthetic_inputs()
    frozen, manifest = freeze_candidates(
        artifacts,
        baseline,
        randomized,
        output_path=tmp_path / "frozen.json",
        manifest_path=tmp_path / "manifest.json",
    )
    assert set(frozen["winners_by_model"]) == set(MODEL_ORDER)
    assert frozen["protocol"]["holdout_used"] is False
    assert manifest["frozen_candidates_signature"] == frozen["signature"]
    figures = generate_report_figures(
        artifacts, baseline, randomized, output_dir=tmp_path / "figures"
    )
    assert len(figures) == 7
    assert all(path.stat().st_size > 0 for path in figures)


def test_manifest_records_hashes_versions_and_cost(tmp_path: Path) -> None:
    manifest = create_execution_manifest(output_path=tmp_path / "execution.json")
    signature = manifest.pop("manifest_sha256")
    assert signature == stable_sha256(manifest)
    assert manifest["data_scope"]["holdout_used"] is False
    assert manifest["dataset"]["sha256"] == (
        "1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33"
    )
    assert manifest["cost_estimate"]["maximum_model_fits_all_nine"] == 43800
    assert manifest["source_signature"]["sha256"]
