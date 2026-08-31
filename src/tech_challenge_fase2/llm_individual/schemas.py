"""Contratos fechados da explicacao individual desidentificada."""

from __future__ import annotations

import math
import re
from typing import Any

from tech_challenge_fase2.llm.schemas import DISCLAIMER

SCHEMA_VERSION = "3.0"
CONTRACT_VERSION = "individual_v1"
PREDICTED_PATTERNS = ("benign_pattern", "malignant_pattern")
OBSERVED_BANDS = ("low", "typical", "high")
INFLUENCE_DIRECTIONS = ("toward_benign", "toward_malignant")
DISTANCE_BANDS = ("close_to_threshold", "moderate_distance", "far_from_threshold")


class IndividualSchemaError(ValueError):
    """O contrato individual nao foi respeitado."""


def _object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IndividualSchemaError(f"{name} deve ser objeto.")
    missing, extra = keys.difference(value), set(value).difference(keys)
    if missing or extra:
        raise IndividualSchemaError(
            f"Campos invalidos em {name}; ausentes={sorted(missing)}, extras={sorted(extra)}."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndividualSchemaError(f"{name} deve ser texto nao vazio.")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndividualSchemaError(f"{name} deve ser numerico.")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise IndividualSchemaError(f"{name} deve estar entre zero e um.")
    return number


CASE_KEYS = {
    "case_reference", "source_scope", "deidentified", "source_record_reconstructible",
    "raw_feature_values_included", "ground_truth_included", "predicted_pattern",
    "probability_malignant", "classification_threshold", "distance_from_threshold_band",
}
MODEL_KEYS = {
    "candidate_id", "model_family", "method", "training_group_id", "trained_on_development_rows",
    "model_artifact_sha256", "new_training_performed", "holdout_inference_performed",
    "aggregate_validation_context",
}
AGGREGATE_KEYS = {
    "recall_malignant", "f1_malignant", "roc_auc", "false_negatives", "scope",
}
SIGNAL_KEYS = {
    "rank", "feature", "display_name", "observed_band", "influence_direction",
    "relative_importance_percent",
}
MEDICAL_KEYS = {
    "intended_audience", "action_scope", "clinical_use_authorized", "diagnosis_claim_allowed",
    "treatment_recommendation_allowed", "required_disclaimer",
}
TEXT_KEYS = {
    "contract_ready", "text_data_included", "planned_fields", "required_safeguards",
}
PROVENANCE_KEYS = {
    "dataset_sha256", "model_artifact_relative_path", "development_only",
    "test_or_holdout_case_used", "patient_identifiers_included", "original_row_index_included",
}
INPUT_KEYS = {
    "schema_version", "contract_version", "case_context", "model_context",
    "explanation_signals", "medical_context", "future_text_integration", "source_provenance",
}


def _validate_signals(value: Any, *, output: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 5:
        raise IndividualSchemaError("Devem existir exatamente cinco sinais explicativos.")
    seen: set[str] = set()
    total = 0.0
    expected_keys = SIGNAL_KEYS | ({"explanation"} if output else set())
    for index, raw in enumerate(value, start=1):
        item = _object(raw, f"signals[{index}]", expected_keys)
        if item["rank"] != index:
            raise IndividualSchemaError("Ranks dos sinais devem ser sequenciais de 1 a 5.")
        feature = _text(item["feature"], "feature")
        if not re.fullmatch(r"[a-z][a-z0-9_ ]*", feature) or feature in seen:
            raise IndividualSchemaError("Feature invalida ou duplicada.")
        seen.add(feature)
        _text(item["display_name"], "display_name")
        if item["observed_band"] not in OBSERVED_BANDS:
            raise IndividualSchemaError("Faixa observada invalida.")
        if item["influence_direction"] not in INFLUENCE_DIRECTIONS:
            raise IndividualSchemaError("Direcao de influencia invalida.")
        importance = _probability(float(item["relative_importance_percent"]) / 100.0, "relative_importance_percent")
        total += importance * 100.0
        if output:
            _text(item["explanation"], "explanation")
    if not math.isclose(total, 100.0, abs_tol=0.05):
        raise IndividualSchemaError("Importancias relativas devem somar 100%.")
    return value


def validate_input(payload: Any) -> dict[str, Any]:
    top = _object(payload, "individual_input", INPUT_KEYS)
    if top["schema_version"] != SCHEMA_VERSION or top["contract_version"] != CONTRACT_VERSION:
        raise IndividualSchemaError("Versao individual deve ser selecionada explicitamente.")
    case = _object(top["case_context"], "case_context", CASE_KEYS)
    if not re.fullmatch(r"demo_case_[0-9]{3}", _text(case["case_reference"], "case_reference")):
        raise IndividualSchemaError("Referencia do caso deve ser opaca e demonstrativa.")
    if case["source_scope"] != "development_only":
        raise IndividualSchemaError("Somente casos do desenvolvimento sao autorizados.")
    required_case_flags = {
        "deidentified": True,
        "source_record_reconstructible": False,
        "raw_feature_values_included": False,
        "ground_truth_included": False,
    }
    if any(case[key] is not expected for key, expected in required_case_flags.items()):
        raise IndividualSchemaError("Protecoes do caso desidentificado foram violadas.")
    if case["predicted_pattern"] not in PREDICTED_PATTERNS:
        raise IndividualSchemaError("Classe predita invalida.")
    _probability(case["probability_malignant"], "probability_malignant")
    threshold = _probability(case["classification_threshold"], "classification_threshold")
    if threshold != 0.5 or case["distance_from_threshold_band"] not in DISTANCE_BANDS:
        raise IndividualSchemaError("Threshold ou faixa de distancia invalidos.")

    model = _object(top["model_context"], "model_context", MODEL_KEYS)
    if (
        model["candidate_id"] != "logistic_regression__random_search"
        or model["model_family"] != "logistic_regression"
        or model["method"] != "random_search"
        or model["trained_on_development_rows"] != 455
        or model["new_training_performed"] is not False
        or model["holdout_inference_performed"] is not False
    ):
        raise IndividualSchemaError("Contexto do modelo congelado invalido.")
    for key in ("training_group_id", "model_artifact_sha256"):
        _text(model[key], key)
    aggregate = _object(model["aggregate_validation_context"], "aggregate_validation_context", AGGREGATE_KEYS)
    for key in ("recall_malignant", "f1_malignant", "roc_auc"):
        _probability(aggregate[key], key)
    if not isinstance(aggregate["false_negatives"], int) or aggregate["false_negatives"] < 0:
        raise IndividualSchemaError("false_negatives invalido.")
    if aggregate["scope"] != "confirmatory_holdout_aggregate_only":
        raise IndividualSchemaError("Contexto agregado invalido.")
    _validate_signals(top["explanation_signals"])

    medical = _object(top["medical_context"], "medical_context", MEDICAL_KEYS)
    if medical != {
        "intended_audience": "health_professionals_and_non_specialists",
        "action_scope": "human_review_and_model_interpretation_only",
        "clinical_use_authorized": False,
        "diagnosis_claim_allowed": False,
        "treatment_recommendation_allowed": False,
        "required_disclaimer": DISCLAIMER,
    }:
        raise IndividualSchemaError("Contexto medico seguro invalido.")
    future = _object(top["future_text_integration"], "future_text_integration", TEXT_KEYS)
    if future["contract_ready"] is not True or future["text_data_included"] is not False:
        raise IndividualSchemaError("Integracao textual futura nao pode incluir texto nesta etapa.")
    if future["planned_fields"] != ["clinical_note_summary", "exam_report_summary"]:
        raise IndividualSchemaError("Campos textuais futuros invalidos.")
    if not isinstance(future["required_safeguards"], list) or not future["required_safeguards"]:
        raise IndividualSchemaError("Salvaguardas textuais futuras ausentes.")
    provenance = _object(top["source_provenance"], "source_provenance", PROVENANCE_KEYS)
    if not (
        provenance["development_only"] is True
        and provenance["test_or_holdout_case_used"] is False
        and provenance["patient_identifiers_included"] is False
        and provenance["original_row_index_included"] is False
    ):
        raise IndividualSchemaError("Proveniencia individual invalida.")
    for key in ("dataset_sha256", "model_artifact_relative_path"):
        _text(provenance[key], key)
    return top


OUTPUT_CLASSIFICATION_KEYS = {
    "candidate_id", "predicted_pattern", "probability_malignant", "classification_threshold",
    "interpretation",
}
INSIGHT_KEYS = {"action", "rationale", "scope", "patient_care_decision"}
MODULE3_KEYS = {"ready_for_future_text", "current_text_data_used", "explanation"}
OUTPUT_KEYS = {
    "schema_version", "contract_version", "case_reference", "resumo_executivo",
    "classificacao_do_modelo", "fatores_explicativos", "insights_acionaveis_para_medicos",
    "limitacoes", "preparacao_modulo3", "conclusao", "disclaimer",
    "predicao_nao_e_diagnostico", "uso_clinico_autorizado",
}


def validate_output(payload: Any) -> dict[str, Any]:
    top = _object(payload, "individual_output", OUTPUT_KEYS)
    if top["schema_version"] != SCHEMA_VERSION or top["contract_version"] != CONTRACT_VERSION:
        raise IndividualSchemaError("Saida individual deve declarar versao 3.0.")
    if not re.fullmatch(r"demo_case_[0-9]{3}", _text(top["case_reference"], "case_reference")):
        raise IndividualSchemaError("Referencia de caso invalida.")
    for key in ("resumo_executivo", "conclusao"):
        _text(top[key], key)
    classification = _object(top["classificacao_do_modelo"], "classificacao_do_modelo", OUTPUT_CLASSIFICATION_KEYS)
    if classification["candidate_id"] != "logistic_regression__random_search":
        raise IndividualSchemaError("Candidato explicado invalido.")
    if classification["predicted_pattern"] not in PREDICTED_PATTERNS:
        raise IndividualSchemaError("Classe explicada invalida.")
    _probability(classification["probability_malignant"], "probability_malignant")
    if _probability(classification["classification_threshold"], "classification_threshold") != 0.5:
        raise IndividualSchemaError("Threshold explicado foi alterado.")
    _text(classification["interpretation"], "interpretation")
    _validate_signals(top["fatores_explicativos"], output=True)
    insights = top["insights_acionaveis_para_medicos"]
    if not isinstance(insights, list) or not 2 <= len(insights) <= 4:
        raise IndividualSchemaError("Devem existir de dois a quatro insights seguros.")
    for index, raw in enumerate(insights):
        item = _object(raw, f"insights[{index}]", INSIGHT_KEYS)
        _text(item["action"], "action")
        _text(item["rationale"], "rationale")
        if item["scope"] != "human_review_only" or item["patient_care_decision"] is not False:
            raise IndividualSchemaError("Insight excedeu o escopo de revisao humana.")
    if not isinstance(top["limitacoes"], list) or len(top["limitacoes"]) < 3:
        raise IndividualSchemaError("A saida deve apresentar ao menos tres limitacoes.")
    for item in top["limitacoes"]:
        _text(item, "limitacao")
    module3 = _object(top["preparacao_modulo3"], "preparacao_modulo3", MODULE3_KEYS)
    if module3["ready_for_future_text"] is not True or module3["current_text_data_used"] is not False:
        raise IndividualSchemaError("Estado da futura integracao textual invalido.")
    _text(module3["explanation"], "preparacao_modulo3.explanation")
    if (
        top["disclaimer"] != DISCLAIMER
        or top["predicao_nao_e_diagnostico"] is not True
        or top["uso_clinico_autorizado"] is not False
    ):
        raise IndividualSchemaError("Disclaimer ou limites clinicos invalidos.")
    return top


def output_json_schema() -> dict[str, Any]:
    signal_properties = {
        "rank": {"type": "integer", "minimum": 1, "maximum": 5},
        "feature": {"type": "string"},
        "display_name": {"type": "string"},
        "observed_band": {"type": "string", "enum": list(OBSERVED_BANDS)},
        "influence_direction": {"type": "string", "enum": list(INFLUENCE_DIRECTIONS)},
        "relative_importance_percent": {"type": "number", "minimum": 0, "maximum": 100},
        "explanation": {"type": "string"},
    }
    insight_properties = {
        "action": {"type": "string"}, "rationale": {"type": "string"},
        "scope": {"type": "string", "const": "human_review_only"},
        "patient_care_decision": {"type": "boolean", "const": False},
    }
    classification_properties = {
        "candidate_id": {"type": "string", "const": "logistic_regression__random_search"},
        "predicted_pattern": {"type": "string", "enum": list(PREDICTED_PATTERNS)},
        "probability_malignant": {"type": "number", "minimum": 0, "maximum": 1},
        "classification_threshold": {"type": "number", "const": 0.5},
        "interpretation": {"type": "string"},
    }
    module3_properties = {
        "ready_for_future_text": {"type": "boolean", "const": True},
        "current_text_data_used": {"type": "boolean", "const": False},
        "explanation": {"type": "string"},
    }
    properties = {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "case_reference": {"type": "string", "pattern": "^demo_case_[0-9]{3}$"},
        "resumo_executivo": {"type": "string"},
        "classificacao_do_modelo": {
            "type": "object", "additionalProperties": False,
            "properties": classification_properties, "required": list(classification_properties),
        },
        "fatores_explicativos": {
            "type": "array", "minItems": 5, "maxItems": 5,
            "items": {"type": "object", "additionalProperties": False, "properties": signal_properties, "required": list(signal_properties)},
        },
        "insights_acionaveis_para_medicos": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": {"type": "object", "additionalProperties": False, "properties": insight_properties, "required": list(insight_properties)},
        },
        "limitacoes": {"type": "array", "minItems": 3, "items": {"type": "string"}},
        "preparacao_modulo3": {
            "type": "object", "additionalProperties": False,
            "properties": module3_properties, "required": list(module3_properties),
        },
        "conclusao": {"type": "string"},
        "disclaimer": {"type": "string", "const": DISCLAIMER},
        "predicao_nao_e_diagnostico": {"type": "boolean", "const": True},
        "uso_clinico_autorizado": {"type": "boolean", "const": False},
    }
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}
