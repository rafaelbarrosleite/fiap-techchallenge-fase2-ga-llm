import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

import tech_challenge_fase2.final_evaluation as final
from tech_challenge_fase2.final_evaluation import (
    FinalEvaluationError,
    ManualInterventionRequired,
    build_candidate_pipeline,
    build_final_evaluation_plan,
    classification_metrics,
    exact_mcnemar,
    paired_bootstrap_differences,
    prepare_final_evaluation,
    run_final_evaluation,
    validate_plan,
    wilson_interval,
)
from tech_challenge_fase2.final_reporting import generate_final_figures
from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256


def _params(model: str, method: str):
    if model == "logistic_regression":
        if method == "baseline":
            return {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None, "max_iter": 1000}
        return {"C": 0.2 if method == "ga" else 0.3, "penalty": "l2", "class_weight": "balanced"}
    if model == "random_forest":
        return {
            "n_estimators": 200 if method == "baseline" else 100,
            "max_depth": None if method == "baseline" else 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced" if method != "ga" else None,
        }
    return {
        "n_neighbors": 5 if method == "baseline" else 3,
        "weights": "uniform",
        "metric": "minkowski",
        "p": 2 if method == "baseline" else 1,
    }


def _synthetic_candidates():
    candidates = []
    for model in final.MODEL_ORDER:
        for method in final.METHOD_ORDER:
            parameters = _params(model, method)
            canonical = final.canonical_candidate_key(model, parameters)
            group = stable_sha256({"model": model, "canonical_key": canonical})[:16]
            if model == "knn" and method == "random_search":
                ga = next(item for item in candidates if item["candidate_id"] == "knn__ga")
                canonical, group, parameters = ga["canonical_key"], ga["training_group_id"], ga["parameters"]
            candidates.append(
                {
                    "candidate_id": f"{model}__{method}",
                    "model": model,
                    "method": method,
                    "origin": "baseline_cv" if method == "baseline" else "GA_A" if method == "ga" else "RandomizedSearchCV",
                    "parameters": parameters,
                    "canonical_key": canonical,
                    "training_group_id": group,
                    "cv_metrics": {
                        "mean_recall_malignant": 0.8,
                        "mean_f1_malignant": 0.8,
                        "mean_roc_auc": 0.9,
                    },
                    "optimization_cost": {"candidate_evaluations": 1, "model_fits": 5, "duration_seconds": 1.0},
                }
            )
    return candidates


def _split_evidence():
    return {
        "development_rows": 455,
        "test_rows": 114,
        "development_class_counts": {"0": 285, "1": 170},
        "test_class_counts": {"0": 72, "1": 42},
        "development_indices_sha256": "d" * 64,
        "test_indices_sha256": "t" * 64,
        "combined_split_sha256": "s" * 64,
        "overlap_count": 0,
        "source_rows": 569,
        "feature_count": 30,
    }


def _plan(candidates=None):
    return build_final_evaluation_plan(
        candidates=candidates or _synthetic_candidates(),
        dataset_sha256="a" * 64,
        split_evidence=_split_evidence(),
        source_signature={"sha256": "c" * 64, "files": {}},
        frozen_signature="f" * 64,
        selection_signature="m" * 64,
        created_at_utc="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.parametrize("candidate", _synthetic_candidates())
def test_reconstructs_all_frozen_pipeline_types(candidate) -> None:
    pipeline = build_candidate_pipeline(candidate)
    assert isinstance(pipeline, Pipeline)
    assert not hasattr(pipeline.named_steps["model"], "best_estimator_")
    if candidate["model"] in {"logistic_regression", "knn"}:
        assert "scaler" in pipeline.named_steps
    else:
        assert "scaler" not in pipeline.named_steps


def test_plan_is_signed_frozen_and_rejects_tampering() -> None:
    plan = _plan()
    validate_plan(plan)
    assert len(plan["candidates"]) == 9
    assert len(plan["unique_training_groups"]) == 8
    plan["classification_threshold"] = 0.4
    with pytest.raises(FinalEvaluationError, match="Assinatura"):
        validate_plan(plan)


def test_metrics_specificity_and_confusion_matrix() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.7])
    metrics, prediction = classification_metrics(y_true, probabilities)
    assert prediction.tolist() == [0, 1, 0, 1, 0, 1]
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["false_negatives"] == 1


def test_wilson_interval_contains_observed_proportion() -> None:
    interval = wilson_interval(39, 42)
    assert interval["lower"] < 39 / 42 < interval["upper"]
    assert 0 <= interval["lower"] <= interval["upper"] <= 1
    assert interval["total"] == 42


def test_paired_comparison_and_bootstrap_are_deterministic() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    a = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    b = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    pa = np.where(a == 1, 0.8, 0.2)
    pb = np.where(b == 1, 0.8, 0.2)
    mcnemar = exact_mcnemar(y, a, b)
    first = paired_bootstrap_differences(y, a, pa, b, pb, replicates=200, seed=7)
    second = paired_bootstrap_differences(y, a, pa, b, pb, replicates=200, seed=7)
    assert mcnemar["discordant_total"] == 3
    assert first == second
    assert first["valid_replicates"] <= 200
    assert "recall_malignant" in first["intervals"]


def test_preflight_uses_only_injected_synthetic_evidence(monkeypatch, tmp_path: Path) -> None:
    candidates = _synthetic_candidates()
    signed_frozen = final._sign({"signature_source": "frozen"})
    signed_selection = final._sign({"signature_source": "selection"})
    monkeypatch.setattr(final, "load_frozen_candidate_records", lambda: (candidates, {"frozen": signed_frozen, "selection": signed_selection}))
    monkeypatch.setattr(final, "file_sha256", lambda path: final.EXPECTED_DATASET_SHA256)
    monkeypatch.setattr(final, "_split_evidence", lambda path: (object(), _split_evidence()))
    monkeypatch.setattr(final, "final_code_signature", lambda: {"sha256": "c" * 64, "files": {}})
    monkeypatch.setattr(
        final,
        "_validate_lineage",
        lambda artifacts: {
            "selection_lineage_has_no_test_metrics": True,
            "official_ga_lineage_is_development_only": True,
            "official_ga_artifacts_validated": 9,
            "source_artifacts_not_newer_than_freeze": True,
            "candidate_count_after_freeze": 0,
            "historical_baseline_exception": {"present": True},
        },
    )
    report, plan = prepare_final_evaluation(
        data_path=tmp_path / "synthetic.csv",
        final_root=tmp_path / "final",
        pytest_result={"command": "synthetic", "exit_code": 0, "reported_test_count": 70, "summary_tail": "70 passed"},
    )
    assert report["approved"] is True
    assert report["fit_calls"] == report["predict_calls_on_holdout"] == 0
    assert report["search_calls"] == 0
    assert plan["classification_threshold"] == 0.5
    assert json.loads((tmp_path / "final" / "final_evaluation_status.json").read_text())["status"] == "prepared"


def test_atomic_json_write_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    save_json({"value": 1}, path)
    save_json({"value": 2, "complete": True}, path)
    assert json.loads(path.read_text()) == {"value": 2, "complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_joblib_round_trip_for_trusted_local_pipeline(tmp_path: Path) -> None:
    pipeline = build_candidate_pipeline(_synthetic_candidates()[0])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    assert type(loaded.named_steps["model"]) is type(pipeline.named_steps["model"])


def test_started_status_blocks_automatic_restart(tmp_path: Path) -> None:
    save_json({"artifact_type": "final_evaluation_status", "status": "started"}, tmp_path / "final_evaluation_status.json")
    with pytest.raises(ManualInterventionRequired, match="started"):
        run_final_evaluation(final_root=tmp_path, figure_root=tmp_path / "figures")


def test_partial_result_blocks_overwrite_before_any_data_access(tmp_path: Path) -> None:
    save_json({"artifact_type": "final_evaluation_status", "status": "prepared"}, tmp_path / "final_evaluation_status.json")
    save_json({"partial": True}, tmp_path / "final_test_results.json")
    with pytest.raises(FinalEvaluationError, match="sobrescrita"):
        run_final_evaluation(final_root=tmp_path, figure_root=tmp_path / "figures")


def test_completed_execution_is_loaded_without_fit_or_predict(monkeypatch, tmp_path: Path) -> None:
    expected = {"artifact_type": "final_test_results", "candidate_origins": 9}
    save_json(expected, tmp_path / "final_test_results.json")
    monkeypatch.setattr(final, "_completed_manifest_valid", lambda root: True)
    monkeypatch.setattr(final, "_validate_execution_inputs", lambda *args, **kwargs: pytest.fail("Nao deve abrir dados"))
    assert run_final_evaluation(final_root=tmp_path, figure_root=tmp_path / "figures") == expected


def _synthetic_figure_payloads():
    results = []
    intervals = {}
    for candidate in _synthetic_candidates():
        candidate_id = candidate["candidate_id"]
        method = candidate["method"]
        recall = {"baseline": 0.75, "ga": 0.80, "random_search": 0.80}[method]
        metrics = {
            "accuracy": 0.80,
            "precision_malignant": 0.75,
            "recall_malignant": recall,
            "f1_malignant": 0.77,
            "roc_auc": 0.88,
            "specificity": 0.82,
            "balanced_accuracy": 0.81,
            "true_positives": int(round(recall * 40)),
            "true_negatives": 59,
            "false_positives": 13,
            "false_negatives": 40 - int(round(recall * 40)),
            "confusion_matrix": [[59, 13], [40 - int(round(recall * 40)), int(round(recall * 40))]],
        }
        results.append(
            {
                **candidate,
                "metrics": metrics,
                "roc_curve": {"false_positive_rate": [0, 0.2, 1], "true_positive_rate": [0, 0.8, 1], "thresholds": [float("inf"), 0.5, 0]},
                "cv_metrics": candidate["cv_metrics"],
            }
        )
        intervals[candidate_id] = {"recall_malignant": {"estimate": recall, "lower": max(0, recall - 0.1), "upper": min(1, recall + 0.1)}}
    return (
        {"candidate_results": results},
        {"records": []},
        {"comparisons": []},
        {"candidate_intervals": intervals},
    )


def test_generates_six_synthetic_figures(tmp_path: Path) -> None:
    figures = generate_final_figures(*_synthetic_figure_payloads(), output_dir=tmp_path)
    assert len(figures) == 6
    assert all(path.stat().st_size > 0 for path in figures)


def test_final_module_does_not_import_search_engines() -> None:
    source = inspect.getsource(final)
    assert "from sklearn.model_selection import RandomizedSearchCV" not in source
    assert "GeneticAlgorithm(" not in source
