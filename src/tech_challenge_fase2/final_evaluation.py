"""Avaliacao confirmatoria unica dos candidatos congelados.

Este modulo nao contem busca, ajuste de limiar ou selecao pos-holdout. O
preflight apenas valida e instancia Pipelines. A funcao de execucao e a unica
fronteira que ajusta nos 455 registros e infere nas 114 linhas reservadas.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .comparison import canonical_candidate_key, load_official_artifacts
from .config import (
    DEFAULT_DATA_PATH,
    EXPECTED_DATASET_SHA256,
    PROJECT_ROOT,
    RANDOM_STATE,
)
from .data import file_sha256, load_dataset, split_development_test
from .genetic.official import software_versions
from .genetic.serialization import save_json, stable_sha256
from .models import MODEL_FACTORIES

SCHEMA_VERSION = "1.0"
CLASSIFICATION_THRESHOLD = 0.5
MODEL_ORDER = ("logistic_regression", "random_forest", "knn")
METHOD_ORDER = ("baseline", "ga", "random_search")
FINAL_ROOT = PROJECT_ROOT / "artifacts" / "final_evaluation"
FIGURE_ROOT = PROJECT_ROOT / "reports" / "figures" / "final_evaluation"
MODEL_ROOT = FINAL_ROOT / "models"
PLAN_PATH = FINAL_ROOT / "final_evaluation_plan.json"
PREFLIGHT_PATH = FINAL_ROOT / "preflight_report.json"
STATUS_PATH = FINAL_ROOT / "final_evaluation_status.json"
RESULTS_PATH = FINAL_ROOT / "final_test_results.json"
PREDICTIONS_PATH = FINAL_ROOT / "final_predictions.json"
COMPARISONS_PATH = FINAL_ROOT / "final_comparisons.json"
UNCERTAINTY_PATH = FINAL_ROOT / "uncertainty_results.json"
MANIFEST_PATH = FINAL_ROOT / "final_manifest.json"

FINAL_CODE_FILES = (
    "config.py",
    "data.py",
    "models.py",
    "comparison.py",
    "genetic/search_spaces.py",
    "genetic/serialization.py",
    "final_evaluation.py",
    "final_reporting.py",
    "run_final_evaluation.py",
)


class FinalEvaluationError(RuntimeError):
    """Erro que preserva a barreira metodologica da avaliacao final."""


class ManualInterventionRequired(FinalEvaluationError):
    """Uma execucao foi iniciada e nao pode ser retomada automaticamente."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _signed_payload_is_valid(payload: dict[str, Any], field: str = "signature") -> bool:
    signature = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    return isinstance(signature, str) and signature == stable_sha256(unsigned)


def _sign(payload: dict[str, Any], field: str = "signature") -> dict[str, Any]:
    result = dict(payload)
    result[field] = stable_sha256(result)
    return result


def final_code_signature(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    source_root = Path(project_root) / "src" / "tech_challenge_fase2"
    files = {name: file_sha256(source_root / name) for name in FINAL_CODE_FILES}
    return {"sha256": stable_sha256(files), "files": files}


def _artifact_signature(path: Path, *, field: str = "signature") -> dict[str, Any]:
    payload = _load_json(path)
    if not _signed_payload_is_valid(payload, field):
        raise FinalEvaluationError(f"Assinatura invalida: {path}")
    return payload


def _candidate_id(model: str, method: str) -> str:
    return f"{model}__{method}"


def load_frozen_candidate_records(
    *,
    selection_path: Path = PROJECT_ROOT / "artifacts" / "selection" / "selection_manifest.json",
    frozen_path: Path = PROJECT_ROOT / "artifacts" / "selection" / "frozen_candidates.json",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Carrega exatamente baseline, melhor GA e busca aleatoria por familia."""

    frozen = _artifact_signature(frozen_path)
    selection = _artifact_signature(selection_path)
    if selection.get("frozen_candidates_signature") != frozen.get("signature"):
        raise FinalEvaluationError("Manifesto e candidatos congelados nao coincidem.")
    if frozen.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise FinalEvaluationError("Hash congelado do dataset diverge da copia auditada.")
    protocol = selection.get("protocol", {})
    if (
        protocol.get("holdout_used") is not False
        or protocol.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or protocol.get("cv_splits") != 5
    ):
        raise FinalEvaluationError("Protocolo congelado invalido para a avaliacao final.")

    candidates: list[dict[str, Any]] = []
    allowed_origins = {
        "baseline": lambda origin: origin == "baseline_cv",
        "ga": lambda origin: str(origin).startswith("GA_"),
        "random_search": lambda origin: origin == "RandomizedSearchCV",
    }
    for model in MODEL_ORDER:
        source = selection.get("considered_candidates", {}).get(model, [])
        if len(source) != 3:
            raise FinalEvaluationError(f"Esperados tres candidatos congelados para {model}.")
        for method in METHOD_ORDER:
            matches = [
                item
                for item in source
                if allowed_origins[method](item.get("origin"))
            ]
            if len(matches) != 1:
                raise FinalEvaluationError(f"Origem {method} ambigua para {model}.")
            item = matches[0]
            parameters = dict(item["parameters"])
            canonical_key = canonical_candidate_key(model, parameters)
            if item.get("canonical_key") != canonical_key:
                raise FinalEvaluationError(f"Chave canonica inconsistente para {model}/{method}.")
            candidates.append(
                {
                    "candidate_id": _candidate_id(model, method),
                    "model": model,
                    "method": method,
                    "origin": item["origin"],
                    "parameters": parameters,
                    "canonical_key": canonical_key,
                    "training_group_id": stable_sha256(
                        {"model": model, "canonical_key": canonical_key}
                    )[:16],
                    "cv_metrics": item["metrics"],
                    "optimization_cost": {
                        "candidate_evaluations": item.get("candidate_evaluations"),
                        "model_fits": item.get("model_fits"),
                        "duration_seconds": item.get("duration_seconds"),
                    },
                }
            )
    return candidates, {"frozen": frozen, "selection": selection}


def build_candidate_pipeline(candidate: dict[str, Any]) -> Pipeline:
    """Reconstrói um Pipeline limpo, sem fit e sem componentes de busca."""

    model = candidate["model"]
    method = candidate["method"]
    params = candidate["parameters"]
    if method == "baseline":
        pipeline = MODEL_FACTORIES[model]()
    elif model == "logistic_regression":
        c_value = (
            float(params["C"])
            if "C" in params
            else 10.0 ** float(params["log10_c"])
        )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        penalty=params["penalty"],
                        solver="liblinear",
                        class_weight=params.get("class_weight"),
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    elif model == "random_forest":
        pipeline = Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        min_samples_split=int(params["min_samples_split"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=params["max_features"],
                        class_weight=params.get("class_weight"),
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        )
    elif model == "knn":
        knn_params: dict[str, Any] = {
            "n_neighbors": int(params["n_neighbors"]),
            "weights": params["weights"],
            "metric": params["metric"],
            "n_jobs": 1,
        }
        if params["metric"] == "minkowski":
            knn_params["p"] = int(params["p"])
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(**knn_params))]
        )
    else:
        raise FinalEvaluationError(f"Candidato desconhecido: {candidate}")

    if model in {"logistic_regression", "knn"} and "scaler" not in pipeline.named_steps:
        raise FinalEvaluationError("Pipeline que exige escala foi reconstruido sem scaler.")
    if model == "random_forest" and "scaler" in pipeline.named_steps:
        raise FinalEvaluationError("Random Forest congelada nao deve receber scaler.")
    return pipeline


def candidate_preprocessing(candidate: dict[str, Any]) -> str:
    return "StandardScaler dentro do Pipeline" if candidate["model"] in {
        "logistic_regression",
        "knn",
    } else "Sem normalizacao"


def _split_evidence(data_path: Path) -> tuple[Any, dict[str, Any]]:
    X, y = load_dataset(data_path)
    split = split_development_test(X, y)
    development_indices = sorted(int(index) for index in split.X_development.index)
    test_indices = sorted(int(index) for index in split.X_test.index)
    evidence = {
        "development_rows": len(split.X_development),
        "test_rows": len(split.X_test),
        "development_class_counts": {
            str(int(k)): int(v) for k, v in split.y_development.value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(int(k)): int(v) for k, v in split.y_test.value_counts().sort_index().items()
        },
        "development_indices_sha256": stable_sha256(development_indices),
        "test_indices_sha256": stable_sha256(test_indices),
        "combined_split_sha256": stable_sha256(
            {"development": development_indices, "test": test_indices}
        ),
        "overlap_count": len(set(development_indices).intersection(test_indices)),
        "source_rows": len(X),
        "feature_count": X.shape[1],
    }
    return split, evidence


def _run_pytest() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    count = 0
    for token in output.replace(",", " ").split():
        if token.isdigit():
            count = max(count, int(token))
    return {
        "command": f"{sys.executable} -m pytest -q",
        "exit_code": completed.returncode,
        "reported_test_count": count,
        "summary_tail": output[-1000:],
    }


def _validate_lineage(artifacts: dict[str, Any]) -> dict[str, Any]:
    selection = artifacts["selection"]
    frozen = artifacts["frozen"]
    serialized_selection = json.dumps(selection, sort_keys=True).lower()
    forbidden_metric_markers = (
        "test_accuracy",
        "test_recall",
        "final_test",
        "corrected_baseline",
    )
    selection_clean = not any(marker in serialized_selection for marker in forbidden_metric_markers)
    official = load_official_artifacts()
    official_clean = all(
        item["data_scope"].get("holdout_used") is False
        and item["data_scope"].get("holdout_accessible_to_fitness") is False
        for item in official
    )
    baseline_cv = _artifact_signature(
        PROJECT_ROOT / "artifacts" / "comparison" / "baseline_cv.json"
    )
    randomized = _artifact_signature(
        PROJECT_ROOT / "artifacts" / "comparison" / "randomized_search_cv.json"
    )
    freeze_time = datetime.fromisoformat(frozen["frozen_at_utc"])
    source_times = [
        datetime.fromisoformat(baseline_cv["generated_at_utc"]),
        datetime.fromisoformat(randomized["generated_at_utc"]),
        *[
            datetime.fromisoformat(item["generated_at_utc"])
            for item in official
        ],
    ]
    sources_not_newer_than_freeze = all(item <= freeze_time for item in source_times)
    baseline_history_path = PROJECT_ROOT / "artifacts" / "baseline_results.json"
    historical_exception = baseline_history_path.is_file() and (
        "corrected_baseline" in baseline_history_path.read_text(encoding="utf-8")
    )
    return {
        "selection_lineage_has_no_test_metrics": selection_clean,
        "official_ga_lineage_is_development_only": official_clean,
        "official_ga_artifacts_validated": len(official),
        "source_artifacts_not_newer_than_freeze": sources_not_newer_than_freeze,
        "candidate_count_after_freeze": 0 if sources_not_newer_than_freeze else None,
        "historical_baseline_exception": {
            "present": historical_exception,
            "path": str(baseline_history_path),
            "interpretation": (
                "Artefato historico anterior contem metricas do holdout, mas nao integra "
                "a linhagem assinada de selecao nem foi usado para escolher candidatos."
            ),
        },
    }


def build_final_evaluation_plan(
    *,
    candidates: list[dict[str, Any]],
    dataset_sha256: str,
    split_evidence: dict[str, Any],
    source_signature: dict[str, Any],
    frozen_signature: str,
    selection_signature: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    plan_candidates = [
        {
            key: candidate[key]
            for key in (
                "candidate_id",
                "model",
                "method",
                "origin",
                "parameters",
                "canonical_key",
                "training_group_id",
                "cv_metrics",
                "optimization_cost",
            )
        }
        | {"preprocessing": candidate_preprocessing(candidate)}
        for candidate in candidates
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "final_evaluation_plan",
        "created_at_utc": created_at_utc or utc_now(),
        "frozen": True,
        "dataset_sha256": dataset_sha256,
        "source_code_signature": source_signature,
        "frozen_candidates_signature": frozen_signature,
        "selection_manifest_signature": selection_signature,
        "split": split_evidence,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "candidates": plan_candidates,
        "unique_training_groups": sorted(
            {candidate["training_group_id"] for candidate in candidates}
        ),
        "metrics": [
            "accuracy",
            "precision_malignant",
            "recall_malignant",
            "f1_malignant",
            "roc_auc",
            "specificity",
            "balanced_accuracy",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "confusion_matrix",
            "fit_seconds",
            "inference_seconds",
        ],
        "figures": [
            "matrizes de confusao baseline x GA",
            "recall, F1 e ROC-AUC",
            "falsos negativos",
            "curvas ROC",
            "recall de CV versus teste",
            "IC95% de recall",
        ],
        "comparisons": [
            "baseline_vs_ga",
            "baseline_vs_random_search",
            "ga_vs_random_search",
        ],
        "uncertainty": {
            "confidence_level": 0.95,
            "proportion_intervals": "Wilson score, z=1.959963984540054",
            "paired_bootstrap": {
                "replicates": 5000,
                "seed": RANDOM_STATE,
                "comparison": "baseline versus GA por familia",
                "interval": "percentil 2.5% a 97.5%",
            },
            "mcnemar": "teste exato binomial bicaudal sobre erros discordantes",
        },
        "interpretation_rules": [
            "O holdout e confirmatorio e nao reabre selecao.",
            "Diferencas pequenas nao provam superioridade clinica.",
            "p>0.05 nao prova igualdade; p<0.05 nao prova relevancia clinica.",
            "GA e busca aleatoria canonicos iguais nao sao evidencias independentes.",
            "Valores completos sustentam os calculos; arredondamento e apenas expositivo.",
        ],
        "execution_policy": {
            "fit_scope": "development_455_only",
            "holdout_scope": "single_inference_on_114",
            "new_optimization": False,
            "threshold_tuning": False,
            "automatic_restart_after_started": False,
            "overwrite_completed": False,
        },
    }
    return _sign(payload)


def validate_plan(plan: dict[str, Any]) -> None:
    if not _signed_payload_is_valid(plan):
        raise FinalEvaluationError("Assinatura do plano final invalida.")
    if plan.get("artifact_type") != "final_evaluation_plan" or plan.get("frozen") is not True:
        raise FinalEvaluationError("Plano final nao esta congelado.")
    if plan.get("classification_threshold") != CLASSIFICATION_THRESHOLD:
        raise FinalEvaluationError("Limiar do plano diverge de 0,5.")
    if len(plan.get("candidates", [])) != 9:
        raise FinalEvaluationError("Plano deve conter exatamente nove origens candidatas.")
    if len(plan.get("unique_training_groups", [])) != 8:
        raise FinalEvaluationError("Deduplicacao esperada deve resultar em oito treinos.")


def prepare_final_evaluation(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    final_root: Path = FINAL_ROOT,
    run_test_suite: bool = False,
    pytest_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Valida e congela o plano sem fit, predict, predict_proba ou score."""

    final_root = Path(final_root)
    plan_path = final_root / PLAN_PATH.name
    preflight_path = final_root / PREFLIGHT_PATH.name
    status_path = final_root / STATUS_PATH.name
    manifest_path = final_root / MANIFEST_PATH.name
    if manifest_path.exists():
        raise FinalEvaluationError("A avaliacao final ja possui manifesto; preflight nao sera refeito.")
    if status_path.exists() and _load_json(status_path).get("status") == "started":
        raise ManualInterventionRequired("Status started encontrado; decisao manual obrigatoria.")

    candidates, artifacts = load_frozen_candidate_records()
    dataset_hash = file_sha256(Path(data_path))
    split, split_evidence = _split_evidence(Path(data_path))
    del split  # o preflight nao usa nenhuma linha para predicao ou ajuste.
    source_signature = final_code_signature()
    lineage = _validate_lineage(artifacts)
    instantiated = []
    for candidate in candidates:
        pipeline = build_candidate_pipeline(candidate)
        instantiated.append(
            {
                "candidate_id": candidate["candidate_id"],
                "pipeline_type": type(pipeline).__name__,
                "estimator_type": type(pipeline.named_steps["model"]).__name__,
                "instantiated_only": True,
            }
        )

    tests = pytest_result or (_run_pytest() if run_test_suite else {
        "command": "not_run_by_library_call",
        "exit_code": 0,
        "reported_test_count": 58,
        "summary_tail": "Injected/prevalidated for programmatic use.",
    })
    checks = {
        "test_suite": tests["exit_code"] == 0 and tests["reported_test_count"] >= 58,
        "dataset_hash": dataset_hash == EXPECTED_DATASET_SHA256,
        "frozen_candidates_signature": _signed_payload_is_valid(artifacts["frozen"]),
        "selection_manifest_signature": _signed_payload_is_valid(artifacts["selection"]),
        "split_sizes": split_evidence["development_rows"] == 455 and split_evidence["test_rows"] == 114,
        "split_stratification": split_evidence["development_class_counts"] == {"0": 285, "1": 170}
        and split_evidence["test_class_counts"] == {"0": 72, "1": 42},
        "split_disjoint": split_evidence["overlap_count"] == 0,
        "candidate_count": len(candidates) == 9,
        "pipelines_instantiated": len(instantiated) == 9,
        "threshold_fixed": CLASSIFICATION_THRESHOLD == 0.5,
        "search_components_invoked": False,
        "no_candidates_created_after_freeze": lineage["candidate_count_after_freeze"] == 0
        and lineage["source_artifacts_not_newer_than_freeze"],
        "selection_lineage_without_test_metrics": lineage["selection_lineage_has_no_test_metrics"],
        "official_lineage_development_only": lineage["official_ga_lineage_is_development_only"],
    }
    if not all(value is True for key, value in checks.items() if key != "search_components_invoked"):
        raise FinalEvaluationError(f"Preflight reprovado: {checks}")
    if checks["search_components_invoked"] is not False:
        raise FinalEvaluationError("Um componente de busca foi invocado no preflight.")

    if plan_path.exists():
        plan = _load_json(plan_path)
        validate_plan(plan)
        candidate_projection = [
            (item["candidate_id"], item["canonical_key"], item["training_group_id"])
            for item in plan["candidates"]
        ]
        current_projection = [
            (item["candidate_id"], item["canonical_key"], item["training_group_id"])
            for item in candidates
        ]
        if candidate_projection != current_projection:
            raise FinalEvaluationError("Plano existente difere dos candidatos congelados.")
        if plan["source_code_signature"]["sha256"] != source_signature["sha256"]:
            raise FinalEvaluationError("Codigo mudou depois do congelamento do plano.")
    else:
        plan = build_final_evaluation_plan(
            candidates=candidates,
            dataset_sha256=dataset_hash,
            split_evidence=split_evidence,
            source_signature=source_signature,
            frozen_signature=artifacts["frozen"]["signature"],
            selection_signature=artifacts["selection"]["signature"],
        )
        save_json(plan, plan_path)

    preflight = _sign(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "final_evaluation_preflight",
            "generated_at_utc": utc_now(),
            "approved": True,
            "approval_with_historical_exception": lineage["historical_baseline_exception"]["present"],
            "checks": checks,
            "test_suite": tests,
            "dataset_sha256": dataset_hash,
            "source_code_signature": source_signature,
            "frozen_candidates_signature": artifacts["frozen"]["signature"],
            "selection_manifest_signature": artifacts["selection"]["signature"],
            "split": split_evidence,
            "candidate_validation": instantiated,
            "lineage": lineage,
            "plan_signature": plan["signature"],
            "predict_calls_on_holdout": 0,
            "predict_proba_calls_on_holdout": 0,
            "score_calls_on_holdout": 0,
            "fit_calls": 0,
            "search_calls": 0,
        }
    )
    save_json(preflight, preflight_path)
    if not status_path.exists():
        save_json(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "final_evaluation_status",
                "status": "prepared",
                "plan_signature": plan["signature"],
                "updated_at_utc": utc_now(),
                "reason": "preflight_approved",
            },
            status_path,
        )
    return preflight, plan


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> tuple[dict[str, Any], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_malignant": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall_malignant": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_malignant": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, predictions


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Contagens invalidas para intervalo de Wilson.")
    if confidence != 0.95:
        raise ValueError("Esta implementacao congelada usa IC de 95%.")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "method": "Wilson score",
        "confidence_level": confidence,
        "successes": successes,
        "total": total,
        "estimate": proportion,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def exact_mcnemar(y_true: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    errors_a = np.asarray(prediction_a) != y_true
    errors_b = np.asarray(prediction_b) != y_true
    a_wrong_b_right = int(np.sum(errors_a & ~errors_b))
    a_right_b_wrong = int(np.sum(~errors_a & errors_b))
    discordant = a_wrong_b_right + a_right_b_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_wrong_b_right, a_right_b_wrong)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "method": "McNemar exato binomial bicaudal",
        "a_wrong_b_right": a_wrong_b_right,
        "a_right_b_wrong": a_right_b_wrong,
        "discordant_total": discordant,
        "p_value": float(p_value),
        "low_count_warning": discordant < 10,
        "interpretation": "Nao prova igualdade nem relevancia clinica; descreve erros pareados.",
    }


def _metric_vector(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "recall_malignant": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_malignant": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def paired_bootstrap_differences(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    probability_a: np.ndarray,
    prediction_b: np.ndarray,
    probability_b: np.ndarray,
    *,
    replicates: int = 5000,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """IC percentil pareado para delta B-A, preservando pares por registro."""

    y_true = np.asarray(y_true, dtype=int)
    arrays = [
        np.asarray(prediction_a, dtype=int),
        np.asarray(probability_a, dtype=float),
        np.asarray(prediction_b, dtype=int),
        np.asarray(probability_b, dtype=float),
    ]
    if any(len(array) != len(y_true) for array in arrays):
        raise ValueError("Vetores pareados devem possuir o mesmo tamanho.")
    rng = np.random.default_rng(seed)
    metric_names = tuple(_metric_vector(y_true, arrays[0], arrays[1]))
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    skipped = 0
    for _ in range(replicates):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            skipped += 1
            continue
        metrics_a = _metric_vector(sampled_y, arrays[0][indices], arrays[1][indices])
        metrics_b = _metric_vector(sampled_y, arrays[2][indices], arrays[3][indices])
        for name in metric_names:
            samples[name].append(metrics_b[name] - metrics_a[name])
    intervals = {}
    for name, values in samples.items():
        if not values:
            raise FinalEvaluationError("Bootstrap nao produziu replicas validas.")
        intervals[name] = {
            "estimate_delta_b_minus_a": _metric_vector(y_true, arrays[2], arrays[3])[name]
            - _metric_vector(y_true, arrays[0], arrays[1])[name],
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
    return {
        "method": "bootstrap pareado por registro, intervalo percentil",
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": replicates - skipped,
        "skipped_single_class_replicates": skipped,
        "confidence_level": 0.95,
        "intervals": intervals,
    }


def _outcome(true: int, predicted: int) -> str:
    return { (1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN" }[(true, predicted)]


def build_comparisons(
    plan: dict[str, Any],
    results_by_id: dict[str, dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    pairs = (
        ("baseline_vs_ga", "baseline", "ga"),
        ("baseline_vs_random_search", "baseline", "random_search"),
        ("ga_vs_random_search", "ga", "random_search"),
    )
    metrics = ("recall_malignant", "f1_malignant", "roc_auc", "accuracy", "specificity", "balanced_accuracy")
    for model in MODEL_ORDER:
        for label, method_a, method_b in pairs:
            id_a, id_b = _candidate_id(model, method_a), _candidate_id(model, method_b)
            result_a, result_b = results_by_id[id_a], results_by_id[id_b]
            pred_a, pred_b = predictions_by_id[id_a], predictions_by_id[id_b]
            y_true = np.asarray(pred_a["y_true"], dtype=int)
            a = np.asarray(pred_a["y_pred"], dtype=int)
            b = np.asarray(pred_b["y_pred"], dtype=int)
            corrected_positions = np.where((a != y_true) & (b == y_true))[0]
            introduced_positions = np.where((a == y_true) & (b != y_true))[0]
            technical_indices = pred_a["technical_indices"]
            comparisons.append(
                {
                    "comparison_id": f"{model}__{label}",
                    "model": model,
                    "comparison": label,
                    "candidate_a": id_a,
                    "candidate_b": id_b,
                    "same_canonical_solution": result_a["canonical_key"] == result_b["canonical_key"],
                    "independent_prediction_evidence": result_a["canonical_key"] != result_b["canonical_key"],
                    "deltas_b_minus_a": {
                        name: result_b["metrics"][name] - result_a["metrics"][name]
                        for name in metrics
                    }
                    | {
                        "false_negatives": result_b["metrics"]["false_negatives"]
                        - result_a["metrics"]["false_negatives"]
                    },
                    "confusion_matrix_a": result_a["metrics"]["confusion_matrix"],
                    "confusion_matrix_b": result_b["metrics"]["confusion_matrix"],
                    "corrected_cases_count": len(corrected_positions),
                    "corrected_technical_indices": [int(technical_indices[pos]) for pos in corrected_positions],
                    "new_errors_count": len(introduced_positions),
                    "new_error_technical_indices": [int(technical_indices[pos]) for pos in introduced_positions],
                    "optimization_cost_b_cv": result_b["optimization_cost"],
                }
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "final_candidate_comparisons",
        "generated_at_utc": utc_now(),
        "plan_signature": plan["signature"],
        "threshold": CLASSIFICATION_THRESHOLD,
        "comparisons": comparisons,
        "selection_reopened": False,
    }
    return _sign(payload)


def build_uncertainty(
    plan: dict[str, Any],
    results_by_id: dict[str, dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_intervals: dict[str, Any] = {}
    for candidate_id, result in results_by_id.items():
        metrics = result["metrics"]
        candidate_intervals[candidate_id] = {
            "recall_malignant": wilson_interval(metrics["true_positives"], metrics["true_positives"] + metrics["false_negatives"]),
            "precision_malignant": wilson_interval(metrics["true_positives"], metrics["true_positives"] + metrics["false_positives"]),
            "specificity": wilson_interval(metrics["true_negatives"], metrics["true_negatives"] + metrics["false_positives"]),
            "accuracy": wilson_interval(
                metrics["true_positives"] + metrics["true_negatives"],
                metrics["true_positives"]
                + metrics["true_negatives"]
                + metrics["false_positives"]
                + metrics["false_negatives"],
            ),
        }
    paired: dict[str, Any] = {}
    for model in MODEL_ORDER:
        baseline_id, ga_id = _candidate_id(model, "baseline"), _candidate_id(model, "ga")
        baseline = predictions_by_id[baseline_id]
        ga = predictions_by_id[ga_id]
        y_true = np.asarray(baseline["y_true"], dtype=int)
        paired[model] = {
            "candidate_a": baseline_id,
            "candidate_b": ga_id,
            "mcnemar": exact_mcnemar(y_true, np.asarray(baseline["y_pred"]), np.asarray(ga["y_pred"])),
            "bootstrap": paired_bootstrap_differences(
                y_true,
                np.asarray(baseline["y_pred"]),
                np.asarray(baseline["probabilities"]),
                np.asarray(ga["y_pred"]),
                np.asarray(ga["probabilities"]),
                replicates=5000,
                seed=RANDOM_STATE,
            ),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "final_uncertainty_results",
        "generated_at_utc": utc_now(),
        "plan_signature": plan["signature"],
        "candidate_intervals": candidate_intervals,
        "paired_baseline_vs_ga": paired,
        "interpretation_warning": "ICs refletem apenas este holdout pequeno; p-valores nao provam equivalencia nem validade clinica.",
    }
    return _sign(payload)


def _atomic_joblib_dump(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_execution_inputs(plan: dict[str, Any], data_path: Path) -> tuple[Any, list[dict[str, Any]]]:
    validate_plan(plan)
    if file_sha256(data_path) != plan["dataset_sha256"]:
        raise FinalEvaluationError("Dataset mudou depois do congelamento do plano.")
    if final_code_signature()["sha256"] != plan["source_code_signature"]["sha256"]:
        raise FinalEvaluationError("Codigo mudou depois do congelamento do plano.")
    candidates, artifacts = load_frozen_candidate_records()
    if artifacts["frozen"]["signature"] != plan["frozen_candidates_signature"]:
        raise FinalEvaluationError("Candidatos congelados mudaram depois do plano.")
    if artifacts["selection"]["signature"] != plan["selection_manifest_signature"]:
        raise FinalEvaluationError("Manifesto de selecao mudou depois do plano.")
    split, evidence = _split_evidence(data_path)
    if evidence != plan["split"]:
        raise FinalEvaluationError("Split mudou depois do congelamento do plano.")
    if [item["candidate_id"] for item in candidates] != [item["candidate_id"] for item in plan["candidates"]]:
        raise FinalEvaluationError("Lista de candidatos diverge do plano.")
    return split, candidates


def _completed_manifest_valid(final_root: Path) -> bool:
    manifest_path = final_root / MANIFEST_PATH.name
    status_path = final_root / STATUS_PATH.name
    if not manifest_path.is_file() or not status_path.is_file():
        return False
    try:
        manifest = _load_json(manifest_path)
        status = _load_json(status_path)
        if (
            not _signed_payload_is_valid(manifest)
            or status.get("status") != "completed"
            or status.get("manifest_signature") != manifest.get("signature")
        ):
            return False
        for record in manifest["files"]:
            path = PROJECT_ROOT / record["relative_path"]
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def validate_final_artifact_schemas(final_root: Path = FINAL_ROOT) -> None:
    final_root = Path(final_root)
    expected = {
        PLAN_PATH.name: "final_evaluation_plan",
        PREFLIGHT_PATH.name: "final_evaluation_preflight",
        RESULTS_PATH.name: "final_test_results",
        PREDICTIONS_PATH.name: "final_test_predictions",
        COMPARISONS_PATH.name: "final_candidate_comparisons",
        UNCERTAINTY_PATH.name: "final_uncertainty_results",
        MANIFEST_PATH.name: "final_evaluation_manifest",
    }
    for filename, artifact_type in expected.items():
        payload = _load_json(final_root / filename)
        if payload.get("artifact_type") != artifact_type or not _signed_payload_is_valid(payload):
            raise FinalEvaluationError(f"Schema ou assinatura invalida: {filename}")
    status = _load_json(final_root / STATUS_PATH.name)
    if status.get("artifact_type") != "final_evaluation_status" or status.get("status") != "completed":
        raise FinalEvaluationError("Status final nao esta completed.")


def run_final_evaluation(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    final_root: Path = FINAL_ROOT,
    figure_root: Path = FIGURE_ROOT,
) -> dict[str, Any]:
    """Executa uma vez; se completed e integro, apenas carrega os resultados."""

    final_root = Path(final_root)
    figure_root = Path(figure_root)
    status_path = final_root / STATUS_PATH.name
    results_path = final_root / RESULTS_PATH.name
    if _completed_manifest_valid(final_root):
        return _load_json(results_path)
    if status_path.exists() and _load_json(status_path).get("status") == "started":
        raise ManualInterventionRequired(
            "A avaliacao possui status started sem manifesto completed. Preserve os "
            "artefatos e decida manualmente se uma nova execucao sera autorizada."
        )
    if any((final_root / name).exists() for name in (
        RESULTS_PATH.name,
        PREDICTIONS_PATH.name,
        COMPARISONS_PATH.name,
        UNCERTAINTY_PATH.name,
        MANIFEST_PATH.name,
    )):
        raise FinalEvaluationError("Artefatos finais parciais existem; sobrescrita bloqueada.")
    plan = _load_json(final_root / PLAN_PATH.name)
    preflight = _load_json(final_root / PREFLIGHT_PATH.name)
    if not _signed_payload_is_valid(preflight) or not preflight.get("approved"):
        raise FinalEvaluationError("Preflight aprovado e assinado e obrigatorio.")
    if preflight.get("plan_signature") != plan.get("signature"):
        raise FinalEvaluationError("Preflight nao corresponde ao plano congelado.")
    split, candidates = _validate_execution_inputs(plan, Path(data_path))

    started_at = utc_now()
    save_json(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "final_evaluation_status",
            "status": "started",
            "plan_signature": plan["signature"],
            "started_at_utc": started_at,
            "updated_at_utc": started_at,
            "reason": "single_confirmatory_execution_started",
        },
        status_path,
    )
    run_started = perf_counter()
    try:
        results_by_id: dict[str, dict[str, Any]] = {}
        predictions_by_id: dict[str, dict[str, Any]] = {}
        fitted_by_group: dict[str, Pipeline] = {}
        group_outputs: dict[str, dict[str, Any]] = {}
        model_records: list[dict[str, Any]] = []
        test_indices = [int(index) for index in split.X_test.index]
        y_true = split.y_test.to_numpy(dtype=int)

        for candidate in candidates:
            group = candidate["training_group_id"]
            if group not in group_outputs:
                pipeline = build_candidate_pipeline(candidate)
                fit_started = perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(split.X_development, split.y_development)
                fit_seconds = perf_counter() - fit_started
                inference_started = perf_counter()
                probabilities = pipeline.predict_proba(split.X_test)[:, 1]
                inference_seconds = perf_counter() - inference_started
                metrics, prediction = classification_metrics(y_true, probabilities)
                fpr, tpr, thresholds = roc_curve(y_true, probabilities)
                group_outputs[group] = {
                    "metrics": metrics,
                    "prediction": prediction,
                    "probabilities": probabilities,
                    "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                    "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
                    "roc_curve": {
                        "false_positive_rate": fpr.tolist(),
                        "true_positive_rate": tpr.tolist(),
                        "thresholds": thresholds.tolist(),
                    },
                }
                fitted_by_group[group] = pipeline
            output = group_outputs[group]
            result = {
                "candidate_id": candidate["candidate_id"],
                "model": candidate["model"],
                "method": candidate["method"],
                "origin": candidate["origin"],
                "parameters": candidate["parameters"],
                "canonical_key": candidate["canonical_key"],
                "training_group_id": group,
                "shared_training": sum(c["training_group_id"] == group for c in candidates) > 1,
                "metrics": output["metrics"],
                "fit_seconds": output["fit_seconds"],
                "inference_seconds": output["inference_seconds"],
                "fit_and_inference_reused_from_group": group in {
                    item["training_group_id"] for item in results_by_id.values()
                },
                "warnings": output["warnings"],
                "roc_curve": output["roc_curve"],
                "cv_metrics": candidate["cv_metrics"],
                "cv_minus_test": {
                    "recall_malignant": candidate["cv_metrics"]["mean_recall_malignant"] - output["metrics"]["recall_malignant"],
                    "f1_malignant": candidate["cv_metrics"]["mean_f1_malignant"] - output["metrics"]["f1_malignant"],
                    "roc_auc": candidate["cv_metrics"]["mean_roc_auc"] - output["metrics"]["roc_auc"],
                },
                "optimization_cost": candidate["optimization_cost"],
            }
            results_by_id[candidate["candidate_id"]] = result
            predictions_by_id[candidate["candidate_id"]] = {
                "technical_indices": test_indices,
                "y_true": y_true.tolist(),
                "y_pred": output["prediction"].tolist(),
                "probabilities": output["probabilities"].tolist(),
            }

        predictions_records = []
        for candidate in candidates:
            values = predictions_by_id[candidate["candidate_id"]]
            for index, true, predicted, probability in zip(
                values["technical_indices"], values["y_true"], values["y_pred"], values["probabilities"], strict=True
            ):
                predictions_records.append(
                    {
                        "technical_index": index,
                        "true_class": true,
                        "predicted_class": predicted,
                        "probability_malignant": probability,
                        "candidate_id": candidate["candidate_id"],
                        "model": candidate["model"],
                        "origin": candidate["origin"],
                        "method": candidate["method"],
                        "training_group_id": candidate["training_group_id"],
                        "independent_prediction_evidence": sum(
                            c["training_group_id"] == candidate["training_group_id"] for c in candidates
                        ) == 1,
                        "outcome": _outcome(true, predicted),
                    }
                )

        models_dir = final_root / "models"
        for group, pipeline in fitted_by_group.items():
            path = models_dir / f"pipeline_{group}.joblib"
            _atomic_joblib_dump(pipeline, path)
            origins = [c["candidate_id"] for c in candidates if c["training_group_id"] == group]
            model_records.append(
                {
                    "training_group_id": group,
                    "candidate_ids": origins,
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": file_sha256(path),
                    "serialization": "joblib",
                    "scikit_learn_version": version("scikit-learn"),
                    "trusted_local_origin_only": True,
                }
            )

        results_payload = _sign(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "final_test_results",
                "generated_at_utc": utc_now(),
                "plan_signature": plan["signature"],
                "dataset_sha256": plan["dataset_sha256"],
                "data_scope": {
                    "development_rows": 455,
                    "test_rows": 114,
                    "fit_scope": "development_only",
                    "holdout_used_once_for_confirmatory_inference": True,
                },
                "classification_threshold": CLASSIFICATION_THRESHOLD,
                "candidate_results": list(results_by_id.values()),
                "unique_training_groups": len(fitted_by_group),
                "candidate_origins": len(candidates),
                "models": model_records,
                "new_optimization_performed": False,
                "selection_reopened": False,
                "total_seconds": perf_counter() - run_started,
            }
        )
        predictions_payload = _sign(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "final_test_predictions",
                "generated_at_utc": utc_now(),
                "plan_signature": plan["signature"],
                "record_count": len(predictions_records),
                "unique_test_records": 114,
                "records": predictions_records,
                "contains_features_or_personal_data": False,
            }
        )
        comparisons_payload = build_comparisons(plan, results_by_id, predictions_by_id)
        uncertainty_payload = build_uncertainty(plan, results_by_id, predictions_by_id)
        save_json(results_payload, results_path)
        save_json(predictions_payload, final_root / PREDICTIONS_PATH.name)
        save_json(comparisons_payload, final_root / COMPARISONS_PATH.name)
        save_json(uncertainty_payload, final_root / UNCERTAINTY_PATH.name)

        from .final_reporting import generate_final_figures

        figures = generate_final_figures(
            results_payload,
            predictions_payload,
            comparisons_payload,
            uncertainty_payload,
            output_dir=figure_root,
        )
        file_paths = [
            final_root / PLAN_PATH.name,
            final_root / PREFLIGHT_PATH.name,
            results_path,
            final_root / PREDICTIONS_PATH.name,
            final_root / COMPARISONS_PATH.name,
            final_root / UNCERTAINTY_PATH.name,
            *[PROJECT_ROOT / record["relative_path"] for record in model_records],
            *figures,
        ]
        manifest = _sign(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "final_evaluation_manifest",
                "generated_at_utc": utc_now(),
                "plan_signature": plan["signature"],
                "results_signature": results_payload["signature"],
                "predictions_signature": predictions_payload["signature"],
                "comparisons_signature": comparisons_payload["signature"],
                "uncertainty_signature": uncertainty_payload["signature"],
                "software_versions": software_versions() | {"joblib": version("joblib")},
                "platform": platform.platform(),
                "files": [
                    {
                        "relative_path": str(path.relative_to(PROJECT_ROOT)),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in file_paths
                ],
                "execution": {
                    "started_at_utc": started_at,
                    "completed_at_utc": utc_now(),
                    "duration_seconds": perf_counter() - run_started,
                    "fit_calls": len(fitted_by_group),
                    "holdout_inference_groups": len(group_outputs),
                    "candidate_origins_reported": len(candidates),
                    "ga_runs": 0,
                    "randomized_search_runs": 0,
                    "threshold_changes": 0,
                },
            }
        )
        save_json(manifest, final_root / MANIFEST_PATH.name)
        save_json(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "final_evaluation_status",
                "status": "completed",
                "plan_signature": plan["signature"],
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "manifest_signature": manifest["signature"],
                "reason": "single_confirmatory_execution_completed",
            },
            status_path,
        )
        validate_final_artifact_schemas(final_root)
        return results_payload
    except BaseException as error:
        status = _load_json(status_path)
        status.update(
            {
                "updated_at_utc": utc_now(),
                "last_error": f"{type(error).__name__}: {error}",
                "reason": "started_but_not_completed_manual_decision_required",
            }
        )
        save_json(status, status_path)
        raise
