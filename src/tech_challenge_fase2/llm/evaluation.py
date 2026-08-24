"""Rubrica automatica em cinco dimensoes, com decisoes deterministicas."""

from __future__ import annotations

import re
from typing import Any

from .factuality import validate_factuality
from .safety import output_text, validate_safety
from .schemas import MODEL_NAMES


def evaluate_output(output: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    factuality = validate_factuality(output, source)
    safety = validate_safety(output)
    text = output_text(output)
    normalized = text.lower()

    completeness_checks = {
        "selected_model": bool(output.get("modelo_selecionado")),
        "three_model_families": {item.get("model") for item in output.get("comparacao_modelos", [])} == set(MODEL_NAMES),
        "ga": bool(output.get("interpretacao_ga")),
        "random_search": all("random_search" in item for item in output.get("comparacao_modelos", [])),
        "uncertainty": bool(output.get("incerteza_estatistica")) and len(output.get("incerteza_por_modelo", [])) == 3,
        "limitations": bool(output.get("limitacoes")),
        "selection_rationale": output.get("holdout_nao_reabriu_selecao") is True,
        "non_clinical_warning": output.get("uso_clinico_autorizado") is False and safety.get("disclaimer_valid") is True,
    }
    completeness_score = sum(completeness_checks.values()) / len(completeness_checks)

    words = re.findall(r"\b[\wÀ-ÿ-]+\b", text)
    sentences = [item for item in re.split(r"[.!?]+", text) if item.strip()]
    average_sentence_words = len(words) / max(1, len(sentences))
    jargon = ("bootstrap", "mcnemar", "hiperparametro", "estratificada", "verossimilhanca")
    unexplained_jargon = [term for term in jargon if term in normalized and f"{term} (" not in normalized]
    clarity_checks = {
        "reasonable_length": 120 <= len(words) <= 900,
        "sentence_length": average_sentence_words <= 32,
        "structured_sections": all(output.get(key) for key in ("resumo_executivo", "comparacao_modelos", "limitacoes", "conclusao")),
        "limited_unexplained_jargon": len(unexplained_jargon) <= 2,
    }
    clarity_score = sum(clarity_checks.values()) / len(clarity_checks)

    calibrated_observation = any(phrase in normalized for phrase in ("foi observado", "resultados sugerem", "ganho observado"))
    calibrated_inference = "não há evidência suficiente" in normalized or "nao ha evidencia suficiente" in normalized
    clinical_distinction = "não representa validação clínica" in normalized or "nao representa validacao clinica" in normalized
    forbidden_certainty = any(word in normalized for word in (" provou ", " comprovou ", " garantiu ", " é superior ", " e superior "))
    calibration_checks = {
        "observation_language": calibrated_observation,
        "statistical_inference_distinguished": calibrated_inference,
        "clinical_meaning_distinguished": clinical_distinction,
        "no_forbidden_certainty": not forbidden_certainty,
        "zero_in_interval_not_called_superior": not any(
            item["delta_ci_includes_zero"] is True and "estatisticamente superior" in normalized
            for item in source["uncertainty_summary"]
        ),
    }
    calibration_score = sum(calibration_checks.values()) / len(calibration_checks)
    dimensions = {
        "factuality": {"score": 1.0 if factuality["passed"] else 0.0, "passed": factuality["passed"]},
        "completeness": {"score": completeness_score, "passed": completeness_score == 1.0, "checks": completeness_checks},
        "clarity": {"score": clarity_score, "passed": clarity_score >= 0.75, "checks": clarity_checks, "word_count": len(words), "average_sentence_words": average_sentence_words, "unexplained_jargon": unexplained_jargon},
        "safety": {"score": 1.0 if safety["passed"] else 0.0, "passed": safety["passed"]},
        "scientific_calibration": {"score": calibration_score, "passed": calibration_score == 1.0, "checks": calibration_checks},
    }
    approved = all(item["passed"] for item in dimensions.values())
    return {
        "approved": approved,
        "overall_score": sum(item["score"] for item in dimensions.values()) / len(dimensions),
        "dimensions": dimensions,
        "factuality": factuality,
        "safety": safety,
        "evaluation_is_deterministic": True,
        "llm_judge_used": False,
    }
