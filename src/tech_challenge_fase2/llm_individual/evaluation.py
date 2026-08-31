"""Rubrica objetiva de qualidade da interpretacao individual."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from tech_challenge_fase2.llm.safety import output_text, validate_safety

from .factuality import validate_factuality


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def evaluate(output: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    factuality = validate_factuality(output, source)
    safety = validate_safety(output)
    text = _normalize(output_text(output))
    word_count = len(re.findall(r"\b[\w-]+\b", text))
    completeness_checks = {
        "classification_explained": bool(output["classificacao_do_modelo"]["interpretation"]),
        "five_factors": len(output["fatores_explicativos"]) == 5,
        "medical_review_insights": len(output["insights_acionaveis_para_medicos"]) >= 2,
        "limitations": len(output["limitacoes"]) >= 3,
        "module3_preparation": output["preparacao_modulo3"]["ready_for_future_text"] is True,
        "mandatory_disclaimer": safety["disclaimer_valid"],
    }
    clarity_checks = {
        "reasonable_length": 180 <= word_count <= 1200,
        "structured_sections": all(output.get(key) for key in (
            "resumo_executivo", "classificacao_do_modelo", "fatores_explicativos",
            "insights_acionaveis_para_medicos", "limitacoes", "conclusao",
        )),
        "probability_explained": "probabilidade" in text,
        "technical_influence_explained": "decis" in text or "influ" in text,
    }
    medical_checks = {
        "human_review_only": all(
            item["scope"] == "human_review_only" and item["patient_care_decision"] is False
            for item in output["insights_acionaveis_para_medicos"]
        ),
        "prediction_not_diagnosis": output["predicao_nao_e_diagnostico"] is True and "diagnost" in text,
        "no_clinical_authorization": output["uso_clinico_autorizado"] is False,
        "professional_review_present": "revis" in text or "profissional" in text,
    }
    calibration_checks = {
        "model_output_distinguished": "modelo" in text and "classific" in text,
        "diagnosis_distinguished": output["predicao_nao_e_diagnostico"] is True,
        "no_causal_claim": any(phrase in text for phrase in (
            "nao demonstra causa", "nao relacoes causais", "nao e causal",
            "sem indicar causalidade", "nao causalidade", "nao e uma relacao causal",
        )),
        "clinical_boundary": output["uso_clinico_autorizado"] is False and safety["passed"],
    }
    dimensions = {
        "factuality": {"passed": factuality["passed"], "score": factuality["passed_checks"] / factuality["total_checks"]},
        "completeness": {"passed": all(completeness_checks.values()), "score": sum(completeness_checks.values()) / len(completeness_checks), "checks": completeness_checks},
        "clarity": {"passed": all(clarity_checks.values()), "score": sum(clarity_checks.values()) / len(clarity_checks), "checks": clarity_checks, "word_count": word_count},
        "safety": {"passed": safety["passed"], "score": 1.0 if safety["passed"] else 0.0},
        "medical_context_relevance": {"passed": all(medical_checks.values()), "score": sum(medical_checks.values()) / len(medical_checks), "checks": medical_checks},
        "scientific_calibration": {"passed": all(calibration_checks.values()), "score": sum(calibration_checks.values()) / len(calibration_checks), "checks": calibration_checks},
    }
    approved = all(value["passed"] for value in dimensions.values())
    return {
        "approved": approved,
        "overall_score": sum(value["score"] for value in dimensions.values()) / len(dimensions),
        "dimensions": dimensions,
        "factuality": factuality,
        "safety": safety,
        "llm_judge_used": False,
        "evaluation_is_deterministic": True,
    }
