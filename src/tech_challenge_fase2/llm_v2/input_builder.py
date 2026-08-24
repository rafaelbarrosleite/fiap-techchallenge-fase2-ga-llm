"""Constroi o contrato V2 somente a partir de agregados autoritativos congelados."""

from __future__ import annotations

from typing import Any

from tech_challenge_fase2.genetic.serialization import stable_sha256
from tech_challenge_fase2.llm.input_builder import build_llm_input, load_validated_sources
from tech_challenge_fase2.llm.schemas import METHOD_NAMES, MODEL_NAMES

from .schemas import CONTRACT_VERSION_V2, PAIR_METHODS, SCHEMA_VERSION_V2, comparison_id, validate_input_v2


def _relation(left: float, right: float, tolerance: float = 1e-12) -> str:
    if left > right + tolerance:
        return "left_higher"
    if right > left + tolerance:
        return "right_higher"
    return "equal"


def _candidate_map(v1_model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {method: v1_model[method] for method in METHOD_NAMES}


def _pair(model: str, candidates: dict[str, dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_candidate, right_candidate = candidates[left], candidates[right]
    left_metrics, right_metrics = left_candidate["metrics"], right_candidate["metrics"]
    threshold_keys = ("true_positives", "true_negatives", "false_positives", "false_negatives")
    return {
        "comparison_id": comparison_id(model, left, right), "model": model,
        "evaluation_scope": "confirmatory_holdout", "left_method": left, "right_method": right,
        "left_candidate_id": left_candidate["candidate_id"],
        "right_candidate_id": right_candidate["candidate_id"],
        "same_confusion_matrix": all(left_metrics[key] == right_metrics[key] for key in threshold_keys),
        "different_roc_auc": abs(left_metrics["roc_auc"] - right_metrics["roc_auc"]) > 1e-12,
        "recall_relation": _relation(left_metrics["recall_malignant"], right_metrics["recall_malignant"]),
        "f1_relation": _relation(left_metrics["f1_malignant"], right_metrics["f1_malignant"]),
        "roc_auc_relation": _relation(left_metrics["roc_auc"], right_metrics["roc_auc"]),
        "metric_delta": {
            "direction": "right_minus_left",
            "recall_malignant": right_metrics["recall_malignant"] - left_metrics["recall_malignant"],
            "f1_malignant": right_metrics["f1_malignant"] - left_metrics["f1_malignant"],
            "roc_auc": right_metrics["roc_auc"] - left_metrics["roc_auc"],
        },
    }


def build_llm_input_v2() -> dict[str, Any]:
    """Adiciona pares e McNemar agregados sem acessar dataset ou previsoes."""

    v1 = build_llm_input()
    sources = load_validated_sources()
    uncertainty_source = sources["uncertainty"]
    by_model = {item["model"]: item for item in v1["model_comparison"]}
    model_results = []
    pairs = []
    uncertainty_comparisons = []
    for model in MODEL_NAMES:
        candidates = _candidate_map(by_model[model])
        model_results.append({
            "model": model,
            "candidates": [candidates[method] for method in METHOD_NAMES],
        })
        pairs.extend(_pair(model, candidates, left, right) for left, right in PAIR_METHODS)
        v1_uncertainty = next(item for item in v1["uncertainty_summary"] if item["model"] == model)
        authoritative = uncertainty_source["paired_baseline_vs_ga"][model]["mcnemar"]
        uncertainty_comparisons.append({
            "comparison_id": comparison_id(model, "baseline", "ga"), "model": model,
            "left_method": "baseline", "right_method": "ga",
            "left_recall_interval": v1_uncertainty["baseline_recall_ci"],
            "right_recall_interval": v1_uncertainty["ga_recall_ci"],
            "delta_recall": {
                "direction": "right_minus_left", "estimate": v1_uncertainty["delta_recall"],
                "interval": v1_uncertainty["delta_recall_ci"],
                "includes_zero": v1_uncertainty["delta_ci_includes_zero"],
            },
            "mcnemar": {
                "method": authoritative["method"],
                "left_wrong_right_correct": authoritative["a_wrong_b_right"],
                "left_correct_right_wrong": authoritative["a_right_b_wrong"],
                "discordant_total": authoritative["discordant_total"],
                "p_value": authoritative["p_value"],
                "low_count_warning": authoritative["low_count_warning"],
                "evidence_source": "authoritative_aggregate_artifact",
            },
        })
    selected = dict(v1["selected_model"])
    payload = {
        "schema_version": SCHEMA_VERSION_V2, "contract_version": CONTRACT_VERSION_V2,
        "experiment_summary": v1["experiment_summary"], "model_results": model_results,
        "comparison_pairs": pairs, "uncertainty_comparisons": uncertainty_comparisons,
        "selected_model": selected, "selection_rationale": v1["selection_rationale"],
        "limitations": list(v1["limitations"]) + [
            "Pairwise findings are limited to the explicit comparison_id values supplied in this contract.",
            "McNemar low-count statements are allowed only because signed aggregate discordance counts are supplied.",
        ],
        "safety_context": v1["safety_context"],
        "source_provenance": {
            "v1_input_sha256": stable_sha256(v1),
            "authoritative_artifacts": v1["source_provenance"]["artifacts"],
            "mcnemar_source": {
                "filename": "uncertainty_results.json",
                "artifact_signature": uncertainty_source["signature"],
                "fields_reused": [
                    "a_wrong_b_right", "a_right_b_wrong", "discordant_total",
                    "p_value", "low_count_warning", "method",
                ],
                "individual_predictions_used": False, "statistic_recomputed": False,
            },
            "all_sources_aggregate": True, "historical_artifacts_unchanged": True,
        },
    }
    validate_input_v2(payload)
    return payload
