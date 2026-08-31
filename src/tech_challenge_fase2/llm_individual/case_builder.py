"""Constroi um caso individual explicavel sem expor uma linha do dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tech_challenge_fase2.config import DEFAULT_DATA_PATH, EXPECTED_DATASET_SHA256
from tech_challenge_fase2.data import file_sha256, load_dataset, split_development_test
from tech_challenge_fase2.genetic.serialization import stable_sha256
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT
from tech_challenge_fase2.llm.schemas import DISCLAIMER

from .schemas import validate_input

FINAL_ROOT = PROJECT_ROOT / "artifacts" / "final_evaluation"
PLAN_PATH = FINAL_ROOT / "final_evaluation_plan.json"
MANIFEST_PATH = FINAL_ROOT / "final_manifest.json"
SELECTED_CANDIDATE = "logistic_regression__random_search"

FEATURE_LABELS = {
    "radius_mean": "raio médio", "texture_mean": "textura média",
    "perimeter_mean": "perímetro médio", "area_mean": "área média",
    "smoothness_mean": "suavidade média", "compactness_mean": "compacidade média",
    "concavity_mean": "concavidade média", "concave points_mean": "pontos côncavos médios",
    "symmetry_mean": "simetria média", "fractal_dimension_mean": "dimensão fractal média",
    "radius_se": "variação do raio", "texture_se": "variação da textura",
    "perimeter_se": "variação do perímetro", "area_se": "variação da área",
    "smoothness_se": "variação da suavidade", "compactness_se": "variação da compacidade",
    "concavity_se": "variação da concavidade", "concave points_se": "variação dos pontos côncavos",
    "symmetry_se": "variação da simetria", "fractal_dimension_se": "variação da dimensão fractal",
    "radius_worst": "maior raio", "texture_worst": "maior textura",
    "perimeter_worst": "maior perímetro", "area_worst": "maior área",
    "smoothness_worst": "maior suavidade", "compactness_worst": "maior compacidade",
    "concavity_worst": "maior concavidade", "concave points_worst": "maiores pontos côncavos",
    "symmetry_worst": "maior simetria", "fractal_dimension_worst": "maior dimensão fractal",
}


class IndividualInputError(RuntimeError):
    """Uma protecao do caso individual foi violada."""


def _load_signed(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("artifact_type") != artifact_type or signature != stable_sha256(unsigned):
        raise IndividualInputError(f"Artefato invalido: {path}.")
    return payload


def _selected_model() -> tuple[Any, dict[str, Any], str, str]:
    plan = _load_signed(PLAN_PATH, "final_evaluation_plan")
    manifest = _load_signed(MANIFEST_PATH, "final_evaluation_manifest")
    candidate = next(
        (item for item in plan["candidates"] if item["candidate_id"] == SELECTED_CANDIDATE),
        None,
    )
    if candidate is None:
        raise IndividualInputError("Candidato congelado nao encontrado no plano.")
    group = candidate["training_group_id"]
    relative_path = f"artifacts/final_evaluation/models/pipeline_{group}.joblib"
    record = next((item for item in manifest["files"] if item["relative_path"] == relative_path), None)
    if record is None:
        raise IndividualInputError("Modelo congelado nao esta no manifesto final.")
    model_path = PROJECT_ROOT / relative_path
    if file_sha256(model_path) != record["sha256"]:
        raise IndividualInputError("Hash do modelo congelado divergiu do manifesto.")
    pipeline = joblib.load(model_path)
    if list(pipeline.named_steps) != ["scaler", "model"]:
        raise IndividualInputError("Pipeline selecionado possui estrutura inesperada.")
    model = pipeline.named_steps["model"]
    if model.__class__.__name__ != "LogisticRegression" or not hasattr(model, "coef_"):
        raise IndividualInputError("Explicador local exige a Regressao Logistica congelada.")
    return pipeline, candidate, relative_path, record["sha256"]


def _representative_development_case(pipeline: Any, X_development: Any) -> tuple[Any, float]:
    probabilities = np.asarray(pipeline.predict_proba(X_development))[:, 1]
    eligible = np.flatnonzero(probabilities >= 0.5)
    if not len(eligible):
        raise IndividualInputError("Modelo nao produziu caso demonstrativo da classe maligna.")
    ordered = eligible[np.argsort(np.abs(probabilities[eligible] - 0.75), kind="stable")]
    position = int(ordered[0])
    return X_development.iloc[[position]], float(probabilities[position])


def _band(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "low"
    if value > upper:
        return "high"
    return "typical"


def _signals(pipeline: Any, row: Any, X_development: Any) -> list[dict[str, Any]]:
    scaler, model = pipeline.named_steps["scaler"], pipeline.named_steps["model"]
    transformed = np.asarray(scaler.transform(row))[0]
    coefficients = np.asarray(model.coef_)[0]
    contributions = transformed * coefficients
    order = np.argsort(-np.abs(contributions), kind="stable")[:5]
    magnitudes = np.abs(contributions[order])
    weights = magnitudes / magnitudes.sum() * 100.0
    rounded = [round(float(value), 2) for value in weights]
    rounded[-1] = round(100.0 - sum(rounded[:-1]), 2)
    result: list[dict[str, Any]] = []
    for rank, (position, importance) in enumerate(zip(order, rounded, strict=True), start=1):
        feature = str(row.columns[int(position)])
        lower, upper = X_development[feature].quantile([0.25, 0.75]).tolist()
        contribution = float(contributions[int(position)])
        result.append({
            "rank": rank,
            "feature": feature,
            "display_name": FEATURE_LABELS[feature],
            "observed_band": _band(float(row.iloc[0, int(position)]), float(lower), float(upper)),
            "influence_direction": "toward_malignant" if contribution >= 0 else "toward_benign",
            "relative_importance_percent": importance,
        })
    return result


def _aggregate_context() -> dict[str, Any]:
    results = _load_signed(FINAL_ROOT / "final_test_results.json", "final_test_results")
    item = next(
        value for value in results["candidate_results"]
        if value["candidate_id"] == SELECTED_CANDIDATE
    )
    metrics = item["metrics"]
    return {
        "recall_malignant": metrics["recall_malignant"],
        "f1_malignant": metrics["f1_malignant"],
        "roc_auc": metrics["roc_auc"],
        "false_negatives": metrics["false_negatives"],
        "scope": "confirmatory_holdout_aggregate_only",
    }


def build_individual_input(*, data_path: Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Gera um unico caso desidentificado sem fit e sem acessar o holdout como caso."""

    pipeline, candidate, relative_model_path, model_hash = _selected_model()
    X, y = load_dataset(Path(data_path))
    split = split_development_test(X, y)
    row, probability = _representative_development_case(pipeline, split.X_development)
    distance = abs(probability - 0.5)
    distance_band = (
        "close_to_threshold" if distance < 0.1
        else "moderate_distance" if distance < 0.3
        else "far_from_threshold"
    )
    payload = {
        "schema_version": "3.0",
        "contract_version": "individual_v1",
        "case_context": {
            "case_reference": "demo_case_001",
            "source_scope": "development_only",
            "deidentified": True,
            "source_record_reconstructible": False,
            "raw_feature_values_included": False,
            "ground_truth_included": False,
            "predicted_pattern": "malignant_pattern" if probability >= 0.5 else "benign_pattern",
            "probability_malignant": round(probability, 6),
            "classification_threshold": 0.5,
            "distance_from_threshold_band": distance_band,
        },
        "model_context": {
            "candidate_id": SELECTED_CANDIDATE,
            "model_family": "logistic_regression",
            "method": "random_search",
            "training_group_id": candidate["training_group_id"],
            "trained_on_development_rows": 455,
            "model_artifact_sha256": model_hash,
            "new_training_performed": False,
            "holdout_inference_performed": False,
            "aggregate_validation_context": _aggregate_context(),
        },
        "explanation_signals": _signals(pipeline, row, split.X_development),
        "medical_context": {
            "intended_audience": "health_professionals_and_non_specialists",
            "action_scope": "human_review_and_model_interpretation_only",
            "clinical_use_authorized": False,
            "diagnosis_claim_allowed": False,
            "treatment_recommendation_allowed": False,
            "required_disclaimer": DISCLAIMER,
        },
        "future_text_integration": {
            "contract_ready": True,
            "text_data_included": False,
            "planned_fields": ["clinical_note_summary", "exam_report_summary"],
            "required_safeguards": [
                "explicit_authorization", "deidentification", "source_provenance",
                "schema_validation", "clinical_safety_review",
            ],
        },
        "source_provenance": {
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "model_artifact_relative_path": relative_model_path,
            "development_only": True,
            "test_or_holdout_case_used": False,
            "patient_identifiers_included": False,
            "original_row_index_included": False,
        },
    }
    validate_input(payload)
    return payload
