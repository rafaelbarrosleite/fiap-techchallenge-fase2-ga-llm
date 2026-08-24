"""Avaliacao offline V2 nas mesmas cinco dimensoes da V1."""

from __future__ import annotations

import re
from typing import Any

from tech_challenge_fase2.llm.safety import output_text, validate_safety

from .factuality import validate_factuality_v2


def evaluate_output_v2(output: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    factuality = validate_factuality_v2(output, source)
    safety = validate_safety(output)
    text = output_text(output)
    normalized = text.lower()
    expected_pairs = {item["comparison_id"] for item in source["comparison_pairs"]}
    actual_pairs = {item.get("comparison_id") for item in output.get("comparison_findings", [])}
    uncertainty_pairs = {item.get("comparison_id") for item in output.get("uncertainty_findings", [])}
    completeness_checks = {
        "selected_model": bool(output.get("modelo_selecionado")),
        "three_model_families": len(output.get("model_results", [])) == 3,
        "nine_explicit_pairs": actual_pairs == expected_pairs,
        "three_uncertainty_pairs": len(uncertainty_pairs) == 3,
        "mcnemar_evidence": all(item.get("mcnemar", {}).get("evidence_source") == "authoritative_aggregate_artifact" for item in output.get("uncertainty_findings", [])),
        "limitations": bool(output.get("limitacoes")),
        "selection_rationale": output.get("holdout_nao_reabriu_selecao") is True,
        "non_clinical_warning": output.get("uso_clinico_autorizado") is False and safety["disclaimer_valid"],
    }
    completeness_score = sum(completeness_checks.values()) / len(completeness_checks)
    words = re.findall(r"\b[\wÀ-ÿ-]+\b", text)
    sentences = [item for item in re.split(r"[.!?]+", text) if item.strip()]
    average_sentence_words = len(words) / max(1, len(sentences))
    clarity_checks = {
        "reasonable_length": 120 <= len(words) <= 1600,
        "sentence_length": average_sentence_words <= 32,
        "structured_sections": all(output.get(key) for key in ("resumo_executivo", "comparison_findings", "limitacoes", "conclusao")),
        "pair_labels_present": all(pair_id in text for pair_id in expected_pairs),
    }
    clarity_score = sum(clarity_checks.values()) / len(clarity_checks)
    calibration_checks = {
        "observation_language": any(phrase in normalized for phrase in ("foi observado", "resultados sugerem", "ganhos observados")),
        "statistical_inference_distinguished": "nao ha evidencia suficiente" in normalized or "não há evidência suficiente" in normalized,
        "clinical_meaning_distinguished": "nao representa validacao clinica" in normalized or "não representa validação clínica" in normalized,
        "no_forbidden_certainty": not any(word in normalized for word in (" provou ", " comprovou ", " garantiu ", " e superior ", " é superior ")),
        "zero_interval_not_superior": not any(item["delta_recall"]["includes_zero"] and "estatisticamente superior" in normalized for item in source["uncertainty_comparisons"]),
        "mcnemar_not_equality": "nao prova igualdade" in normalized or "não prova igualdade" in normalized,
        "low_count_supported": all(
            (not item["limited_by_few_discordances"])
            or item["mcnemar"]["discordant_total"] is not None
            for item in output["uncertainty_findings"]
        ),
    }
    calibration_score = sum(calibration_checks.values()) / len(calibration_checks)
    dimensions = {
        "factuality": {"score": 1.0 if factuality["passed"] else 0.0, "passed": factuality["passed"]},
        "completeness": {"score": completeness_score, "passed": completeness_score == 1.0, "checks": completeness_checks},
        "clarity": {"score": clarity_score, "passed": clarity_score >= 0.75, "checks": clarity_checks, "word_count": len(words), "average_sentence_words": average_sentence_words},
        "safety": {"score": 1.0 if safety["passed"] else 0.0, "passed": safety["passed"]},
        "scientific_calibration": {"score": calibration_score, "passed": calibration_score == 1.0, "checks": calibration_checks},
    }
    return {
        "approved": all(item["passed"] for item in dimensions.values()),
        "overall_score": sum(item["score"] for item in dimensions.values()) / len(dimensions),
        "dimensions": dimensions, "factuality": factuality, "safety": safety,
        "evaluation_is_deterministic": True, "llm_judge_used": False,
    }
