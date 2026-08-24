"""Constroi somente o recorte agregado autorizado dos artefatos congelados."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from tech_challenge_fase2.genetic.serialization import stable_sha256

from .privacy import validate_sanitized_input
from .schemas import DISCLAIMER, METHOD_NAMES, MODEL_NAMES, SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_ROOT = PROJECT_ROOT / "artifacts" / "final_evaluation"
SOURCE_FILENAMES = (
    "final_test_results.json", "uncertainty_results.json",
    "final_manifest.json", "final_evaluation_plan.json",
)
SELECTION_DOCUMENT = PROJECT_ROOT / "docs" / "decisoes_tecnicas.md"
FROZEN_SELECTED_CANDIDATE = "logistic_regression__random_search"


class InputBuildError(RuntimeError):
    """Os artefatos nao permitem construir uma entrada segura."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_signed(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("artifact_type") != artifact_type or signature != stable_sha256(unsigned):
        raise InputBuildError(f"Schema ou assinatura invalida: {path.name}.")
    return payload


def load_validated_sources(final_root: Path = FINAL_ROOT) -> dict[str, dict[str, Any]]:
    final_root = Path(final_root)
    results = _load_signed(final_root / "final_test_results.json", "final_test_results")
    uncertainty = _load_signed(final_root / "uncertainty_results.json", "final_uncertainty_results")
    manifest = _load_signed(final_root / "final_manifest.json", "final_evaluation_manifest")
    plan = _load_signed(final_root / "final_evaluation_plan.json", "final_evaluation_plan")
    plan_signature = plan["signature"]
    if results.get("plan_signature") != plan_signature or uncertainty.get("plan_signature") != plan_signature:
        raise InputBuildError("Resultados e incerteza nao pertencem ao plano final.")
    if manifest.get("plan_signature") != plan_signature:
        raise InputBuildError("Manifesto e plano final divergem.")
    manifest_hashes = {Path(item["relative_path"]).name: item["sha256"] for item in manifest.get("files", [])}
    for filename in ("final_test_results.json", "uncertainty_results.json", "final_evaluation_plan.json"):
        path = final_root / filename
        if manifest_hashes.get(filename) != file_sha256(path):
            raise InputBuildError(f"Hash do manifesto diverge para {filename}.")
    if results.get("selection_reopened") is not False or results.get("new_optimization_performed") is not False:
        raise InputBuildError("A Missao 4 nao preserva a selecao/otimizacao esperada.")
    if results.get("classification_threshold") != plan.get("classification_threshold"):
        raise InputBuildError("Threshold diverge entre plano e resultado.")
    return {"results": results, "uncertainty": uncertainty, "manifest": manifest, "plan": plan}


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate["metrics"]
    return {
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "origin": candidate["origin"],
        "metrics": {
            "recall_malignant": metrics["recall_malignant"],
            "f1_malignant": metrics["f1_malignant"],
            "roc_auc": metrics["roc_auc"],
            "true_positives": metrics["true_positives"],
            "true_negatives": metrics["true_negatives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
        },
    }


def build_llm_input(
    *, final_root: Path = FINAL_ROOT, selection_document: Path = SELECTION_DOCUMENT,
) -> dict[str, Any]:
    sources = load_validated_sources(final_root)
    results, uncertainty, plan = sources["results"], sources["uncertainty"], sources["plan"]
    by_id = {item["candidate_id"]: item for item in results["candidate_results"]}
    plan_by_id = {item["candidate_id"]: item for item in plan["candidates"]}
    if set(by_id) != set(plan_by_id) or len(by_id) != 9:
        raise InputBuildError("Candidatos do plano e dos resultados divergem.")
    selected = plan_by_id.get(FROZEN_SELECTED_CANDIDATE)
    if selected is None:
        raise InputBuildError("Candidato global congelado nao existe no plano.")
    max_fitness = max(float(item["cv_metrics"]["fitness"]) for item in plan["candidates"])
    if abs(float(selected["cv_metrics"]["fitness"]) - max_fitness) > 1e-12:
        raise InputBuildError("Fonte auxiliar aponta candidato sem fitness maximo de CV.")
    selection_document = Path(selection_document)
    document_text = selection_document.read_text(encoding="utf-8")
    expected_phrase = "Regressão Logística da busca aleatória"
    if expected_phrase not in document_text:
        raise InputBuildError("Documento auxiliar nao confirma o vencedor congelado.")

    comparisons: list[dict[str, Any]] = []
    uncertainty_items: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        entries = {method: by_id[f"{model}__{method}"] for method in METHOD_NAMES}
        comparisons.append({
            "model": model,
            "baseline": _candidate_summary(entries["baseline"]),
            "ga": _candidate_summary(entries["ga"]),
            "random_search": _candidate_summary(entries["random_search"]),
            "cv_recall_baseline": entries["baseline"]["cv_metrics"]["mean_recall_malignant"],
            "cv_recall_ga": entries["ga"]["cv_metrics"]["mean_recall_malignant"],
        })
        paired = uncertainty["paired_baseline_vs_ga"][model]
        delta = paired["bootstrap"]["intervals"]["recall_malignant"]
        baseline_ci = uncertainty["candidate_intervals"][f"{model}__baseline"]["recall_malignant"]
        ga_ci = uncertainty["candidate_intervals"][f"{model}__ga"]["recall_malignant"]
        uncertainty_items.append({
            "model": model,
            "baseline_recall_ci": {key: baseline_ci[key] for key in ("lower", "upper", "confidence_level")},
            "ga_recall_ci": {key: ga_ci[key] for key in ("lower", "upper", "confidence_level")},
            "delta_recall": delta["estimate_delta_b_minus_a"],
            "delta_recall_ci": {
                "lower": delta["lower"], "upper": delta["upper"],
                "confidence_level": paired["bootstrap"]["confidence_level"],
            },
            "delta_ci_includes_zero": delta["lower"] <= 0.0 <= delta["upper"],
            "mcnemar_p_value": paired["mcnemar"]["p_value"],
        })

    paths = {name: Path(final_root) / name for name in SOURCE_FILENAMES}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_summary": {
            "development_rows": results["data_scope"]["development_rows"],
            "test_rows": results["data_scope"]["test_rows"],
            "malignant_test_cases": plan["split"]["test_class_counts"]["1"],
            "classification_threshold": results["classification_threshold"],
            "candidate_origins": results["candidate_origins"],
            "unique_training_groups": results["unique_training_groups"],
            "selection_reopened": results["selection_reopened"],
            "new_optimization_performed": results["new_optimization_performed"],
            "holdout_role": "confirmatory_only; it did not reopen model selection",
        },
        "model_comparison": comparisons,
        "uncertainty_summary": uncertainty_items,
        "selected_model": {
            "candidate_id": FROZEN_SELECTED_CANDIDATE,
            "model": selected["model"], "method": selected["method"], "origin": selected["origin"],
            "frozen_before_holdout": True,
            "authority": "Mission 4 frozen documentation, checked against maximum CV fitness in the signed plan",
            "authority_limitation": (
                "The four priority structured artifacts do not expose an explicit global-selected-model field; "
                "the documented frozen decision is auxiliary and was not inferred from holdout performance."
            ),
        },
        "selection_rationale": (
            "The candidate was selected from development cross-validation before the holdout. "
            "The confirmatory holdout did not reopen selection, even if another candidate tied there."
        ),
        "limitations": [
            "Only 42 malignant cases are present in the confirmatory holdout, so uncertainty remains broad.",
            "Intervals that include zero do not provide enough evidence for statistical superiority.",
            "This is an academic experiment and not a clinical validation.",
            "The structured Mission 4 artifacts omit an explicit global-selected-model field; documented frozen provenance is auxiliary.",
        ],
        "safety_context": {
            "academic_experimental_only": True, "individual_data_included": False,
            "clinical_use_authorized": False, "diagnosis_allowed": False,
            "medical_recommendation_allowed": False, "required_disclaimer": DISCLAIMER,
        },
        "source_provenance": {
            "artifacts": [
                {"filename": name, "sha256": file_sha256(path), "signature": sources[{"final_test_results.json": "results", "uncertainty_results.json": "uncertainty", "final_manifest.json": "manifest", "final_evaluation_plan.json": "plan"}[name]]["signature"]}
                for name, path in paths.items()
            ],
            "documentation_auxiliary": {"filename": str(selection_document.relative_to(PROJECT_ROOT)), "sha256": file_sha256(selection_document)},
            "all_sources_aggregate": True, "mission4_artifacts_unchanged": True,
        },
    }
    validate_sanitized_input(payload)
    return payload

