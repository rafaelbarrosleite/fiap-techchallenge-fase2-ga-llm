"""Schemas fechados do contrato LLM V2 com pares comparativos explicitos."""

from __future__ import annotations

from typing import Any

from tech_challenge_fase2.llm.schemas import DISCLAIMER, METHOD_NAMES, MODEL_NAMES, validate_metrics

SCHEMA_VERSION_V2 = "2.0"
CONTRACT_VERSION_V2 = "v2"
PAIR_METHODS = (
    ("baseline", "ga"),
    ("ga", "random_search"),
    ("baseline", "random_search"),
)
RELATIONS = ("left_higher", "right_higher", "equal")
CHANGE_VALUES = ("improved", "worsened", "unchanged")
THRESHOLD_KEYS = ("true_positives", "true_negatives", "false_positives", "false_negatives")
METRIC_KEYS = {
    "recall_malignant", "f1_malignant", "roc_auc", "true_positives",
    "true_negatives", "false_positives", "false_negatives",
}


class SchemaV2Error(ValueError):
    """O payload nao satisfaz o contrato fechado 2.0."""


def _object(value: Any, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaV2Error(f"{name} deve ser objeto.")
    missing = required.difference(value)
    extra = set(value).difference(required)
    if missing or extra:
        raise SchemaV2Error(f"{name}: ausentes={sorted(missing)}, extras={sorted(extra)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaV2Error(f"{name} deve ser texto nao vazio.")
    return value


def _number(value: Any, name: str, *, probability: bool = False, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaV2Error(f"{name} deve ser numerico.")
    result = float(value)
    if probability and not 0.0 <= result <= 1.0:
        raise SchemaV2Error(f"{name} deve estar entre zero e um.")
    if integer and (result < 0 or not result.is_integer()):
        raise SchemaV2Error(f"{name} deve ser inteiro nao negativo.")
    return result


def comparison_id(model: str, left_method: str, right_method: str) -> str:
    return f"{model}__{left_method}_vs_{right_method}"


def _validate_candidate(value: Any, name: str, *, model: str) -> None:
    candidate = _object(value, name, {"candidate_id", "method", "origin", "metrics"})
    method = candidate["method"]
    if method not in METHOD_NAMES or candidate["candidate_id"] != f"{model}__{method}":
        raise SchemaV2Error(f"{name} possui identidade inconsistente.")
    _text(candidate["origin"], f"{name}.origin")
    validate_metrics(candidate["metrics"], f"{name}.metrics")


def _validate_model_results(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise SchemaV2Error(f"{name} deve conter tres familias.")
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item = _object(raw, f"{name}[{index}]", {"model", "candidates"})
        model = item["model"]
        if model not in MODEL_NAMES or model in seen:
            raise SchemaV2Error(f"{name} contem modelo invalido ou duplicado.")
        seen.add(model)
        candidates = item["candidates"]
        if not isinstance(candidates, list) or len(candidates) != 3:
            raise SchemaV2Error(f"{name}.{model}.candidates deve conter tres metodos.")
        methods: set[str] = set()
        for candidate_index, candidate in enumerate(candidates):
            _validate_candidate(candidate, f"{name}.{model}.candidates[{candidate_index}]", model=model)
            method = candidate["method"]
            if method in methods:
                raise SchemaV2Error(f"Metodo duplicado em {model}.")
            methods.add(method)
        if methods != set(METHOD_NAMES):
            raise SchemaV2Error(f"Metodos incompletos em {model}.")


PAIR_KEYS = {
    "comparison_id", "model", "evaluation_scope", "left_method", "right_method",
    "left_candidate_id", "right_candidate_id", "same_confusion_matrix",
    "different_roc_auc", "recall_relation", "f1_relation", "roc_auc_relation",
    "metric_delta",
}
DELTA_KEYS = {"direction", "recall_malignant", "f1_malignant", "roc_auc"}


def _validate_pairs(value: Any, name: str, *, output: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 9:
        raise SchemaV2Error(f"{name} deve conter exatamente nove pares.")
    expected_ids = {
        comparison_id(model, left, right)
        for model in MODEL_NAMES for left, right in PAIR_METHODS
    }
    seen: set[str] = set()
    for index, raw in enumerate(value):
        keys = PAIR_KEYS | ({"tradeoff_present", "interpretation"} if output else set())
        item = _object(raw, f"{name}[{index}]", keys)
        model, left, right = item["model"], item["left_method"], item["right_method"]
        if model not in MODEL_NAMES or (left, right) not in PAIR_METHODS:
            raise SchemaV2Error(f"Par nao autorizado em {name}[{index}].")
        expected_id = comparison_id(model, left, right)
        if item["comparison_id"] != expected_id or expected_id in seen:
            raise SchemaV2Error(f"comparison_id inconsistente ou duplicado: {expected_id}.")
        seen.add(expected_id)
        if item["evaluation_scope"] != "confirmatory_holdout":
            raise SchemaV2Error("O escopo do par deve ser confirmatory_holdout.")
        if item["left_candidate_id"] != f"{model}__{left}" or item["right_candidate_id"] != f"{model}__{right}":
            raise SchemaV2Error(f"Candidatos nao correspondem ao par {expected_id}.")
        for boolean_field in ("same_confusion_matrix", "different_roc_auc"):
            if not isinstance(item[boolean_field], bool):
                raise SchemaV2Error(f"{name}.{boolean_field} deve ser booleano.")
        for relation_field in ("recall_relation", "f1_relation", "roc_auc_relation"):
            if item[relation_field] not in RELATIONS:
                raise SchemaV2Error(f"Relacao invalida em {name}.{relation_field}.")
        delta = _object(item["metric_delta"], f"{name}.metric_delta", DELTA_KEYS)
        if delta["direction"] != "right_minus_left":
            raise SchemaV2Error("A direcao do delta deve ser right_minus_left.")
        for metric in ("recall_malignant", "f1_malignant", "roc_auc"):
            _number(delta[metric], f"{name}.metric_delta.{metric}")
        if output:
            if not isinstance(item["tradeoff_present"], bool):
                raise SchemaV2Error("tradeoff_present deve ser booleano e pertence ao par explicito.")
            _text(item["interpretation"], f"{name}.interpretation")
    if seen != expected_ids:
        raise SchemaV2Error(f"{name} nao cobre os nove pares autorizados.")


INTERVAL_KEYS = {"lower", "upper", "confidence_level"}
DELTA_RECALL_KEYS = {"direction", "estimate", "interval", "includes_zero"}
MCNEMAR_KEYS = {
    "method", "left_wrong_right_correct", "left_correct_right_wrong",
    "discordant_total", "p_value", "low_count_warning", "evidence_source",
}
UNCERTAINTY_KEYS = {
    "comparison_id", "model", "left_method", "right_method", "left_recall_interval",
    "right_recall_interval", "delta_recall", "mcnemar",
}


def _validate_interval(value: Any, name: str) -> None:
    interval = _object(value, name, INTERVAL_KEYS)
    lower = _number(interval["lower"], f"{name}.lower")
    upper = _number(interval["upper"], f"{name}.upper")
    _number(interval["confidence_level"], f"{name}.confidence_level", probability=True)
    if lower > upper:
        raise SchemaV2Error(f"Intervalo invertido em {name}.")


def _validate_uncertainty(value: Any, name: str, *, output: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise SchemaV2Error(f"{name} deve conter tres comparacoes baseline_vs_ga.")
    seen: set[str] = set()
    for index, raw in enumerate(value):
        keys = UNCERTAINTY_KEYS | ({"limited_by_few_discordances", "interpretation"} if output else set())
        item = _object(raw, f"{name}[{index}]", keys)
        model = item["model"]
        expected_id = comparison_id(model, "baseline", "ga")
        if (
            model not in MODEL_NAMES or item["comparison_id"] != expected_id
            or item["left_method"] != "baseline" or item["right_method"] != "ga"
            or expected_id in seen
        ):
            raise SchemaV2Error(f"Par de incerteza inconsistente em {name}[{index}].")
        seen.add(expected_id)
        _validate_interval(item["left_recall_interval"], f"{name}.left_recall_interval")
        _validate_interval(item["right_recall_interval"], f"{name}.right_recall_interval")
        delta = _object(item["delta_recall"], f"{name}.delta_recall", DELTA_RECALL_KEYS)
        if delta["direction"] != "right_minus_left":
            raise SchemaV2Error("Direcao do delta de recall deve ser right_minus_left.")
        _number(delta["estimate"], f"{name}.delta_recall.estimate")
        _validate_interval(delta["interval"], f"{name}.delta_recall.interval")
        if not isinstance(delta["includes_zero"], bool):
            raise SchemaV2Error("includes_zero deve ser booleano.")
        mcnemar = _object(item["mcnemar"], f"{name}.mcnemar", MCNEMAR_KEYS)
        _text(mcnemar["method"], f"{name}.mcnemar.method")
        for field in ("left_wrong_right_correct", "left_correct_right_wrong", "discordant_total"):
            _number(mcnemar[field], f"{name}.mcnemar.{field}", integer=True)
        if mcnemar["discordant_total"] != mcnemar["left_wrong_right_correct"] + mcnemar["left_correct_right_wrong"]:
            raise SchemaV2Error("discordant_total diverge das contagens direcionais.")
        _number(mcnemar["p_value"], f"{name}.mcnemar.p_value", probability=True)
        if not isinstance(mcnemar["low_count_warning"], bool):
            raise SchemaV2Error("low_count_warning deve ser booleano.")
        if mcnemar["evidence_source"] != "authoritative_aggregate_artifact":
            raise SchemaV2Error("Fonte de McNemar nao autorizada.")
        if output:
            if item["limited_by_few_discordances"] is not mcnemar["low_count_warning"]:
                raise SchemaV2Error("Conclusao de baixa contagem diverge da evidencia agregada.")
            _text(item["interpretation"], f"{name}.interpretation")
    if len(seen) != 3:
        raise SchemaV2Error(f"{name} nao cobre as tres familias.")


def _validate_selected(value: Any, name: str) -> None:
    selected = _object(value, name, {
        "candidate_id", "model", "method", "origin", "frozen_before_holdout",
        "authority", "authority_limitation", "explanation",
    })
    if selected["model"] not in MODEL_NAMES or selected["method"] not in METHOD_NAMES:
        raise SchemaV2Error("Modelo selecionado invalido.")
    if selected["candidate_id"] != f"{selected['model']}__{selected['method']}":
        raise SchemaV2Error("candidate_id selecionado inconsistente.")
    if selected["frozen_before_holdout"] is not True:
        raise SchemaV2Error("Selecao deve estar congelada antes do holdout.")
    for key in ("origin", "authority", "authority_limitation", "explanation"):
        _text(selected[key], f"{name}.{key}")


INPUT_KEYS = {
    "schema_version", "contract_version", "experiment_summary", "model_results",
    "comparison_pairs", "uncertainty_comparisons", "selected_model",
    "selection_rationale", "limitations", "safety_context", "source_provenance",
}
OUTPUT_KEYS = {
    "schema_version", "contract_version", "resumo_executivo", "modelo_selecionado",
    "model_results", "comparison_findings", "interpretacao_comparacoes",
    "incerteza_estatistica", "uncertainty_findings", "limitacoes", "conclusao",
    "disclaimer", "holdout_nao_reabriu_selecao", "uso_clinico_autorizado",
}


def validate_input_v2(payload: Any) -> dict[str, Any]:
    top = _object(payload, "entrada_v2", INPUT_KEYS)
    if top["schema_version"] != SCHEMA_VERSION_V2 or top["contract_version"] != CONTRACT_VERSION_V2:
        raise SchemaV2Error("Versao V2 deve ser selecionada explicitamente.")
    summary = _object(top["experiment_summary"], "experiment_summary", {
        "development_rows", "test_rows", "malignant_test_cases", "classification_threshold",
        "candidate_origins", "unique_training_groups", "selection_reopened",
        "new_optimization_performed", "holdout_role",
    })
    if summary["selection_reopened"] is not False or summary["new_optimization_performed"] is not False:
        raise SchemaV2Error("V2 nao pode reabrir selecao ou registrar nova otimizacao.")
    _validate_model_results(top["model_results"], "model_results")
    _validate_pairs(top["comparison_pairs"], "comparison_pairs")
    _validate_uncertainty(top["uncertainty_comparisons"], "uncertainty_comparisons")
    selected = dict(top["selected_model"])
    selected["explanation"] = top["selection_rationale"]
    _validate_selected(selected, "selected_model")
    if not isinstance(top["limitations"], list) or not top["limitations"]:
        raise SchemaV2Error("limitations deve ser lista nao vazia.")
    safety = _object(top["safety_context"], "safety_context", {
        "academic_experimental_only", "individual_data_included", "clinical_use_authorized",
        "diagnosis_allowed", "medical_recommendation_allowed", "required_disclaimer",
    })
    if safety != {
        "academic_experimental_only": True, "individual_data_included": False,
        "clinical_use_authorized": False, "diagnosis_allowed": False,
        "medical_recommendation_allowed": False, "required_disclaimer": DISCLAIMER,
    }:
        raise SchemaV2Error("Contexto de seguranca V2 invalido.")
    provenance = _object(top["source_provenance"], "source_provenance", {
        "v1_input_sha256", "authoritative_artifacts", "mcnemar_source",
        "all_sources_aggregate", "historical_artifacts_unchanged",
    })
    if provenance["all_sources_aggregate"] is not True or provenance["historical_artifacts_unchanged"] is not True:
        raise SchemaV2Error("Proveniencia V2 invalida.")
    _text(provenance["v1_input_sha256"], "v1_input_sha256")
    return top


def validate_output_v2(payload: Any) -> dict[str, Any]:
    top = _object(payload, "saida_v2", OUTPUT_KEYS)
    if top["schema_version"] != SCHEMA_VERSION_V2 or top["contract_version"] != CONTRACT_VERSION_V2:
        raise SchemaV2Error("Saida deve declarar V2 explicitamente.")
    for key in ("resumo_executivo", "interpretacao_comparacoes", "incerteza_estatistica", "conclusao"):
        _text(top[key], key)
    _validate_selected(top["modelo_selecionado"], "modelo_selecionado")
    _validate_model_results(top["model_results"], "model_results")
    _validate_pairs(top["comparison_findings"], "comparison_findings", output=True)
    _validate_uncertainty(top["uncertainty_findings"], "uncertainty_findings", output=True)
    if not isinstance(top["limitacoes"], list) or not top["limitacoes"]:
        raise SchemaV2Error("Saida V2 deve conter limitacoes.")
    if top["disclaimer"] != DISCLAIMER or top["uso_clinico_autorizado"] is not False:
        raise SchemaV2Error("Saida V2 nao pode alterar o disclaimer ou autorizar uso clinico.")
    if top["holdout_nao_reabriu_selecao"] is not True:
        raise SchemaV2Error("Saida V2 deve preservar a selecao congelada.")
    return top


def output_json_schema_v2() -> dict[str, Any]:
    """JSON Schema fechado para uma futura chamada real, sem selecionar V2 implicitamente."""

    metric_properties = {
        "recall_malignant": {"type": "number", "minimum": 0, "maximum": 1},
        "f1_malignant": {"type": "number", "minimum": 0, "maximum": 1},
        "roc_auc": {"type": "number", "minimum": 0, "maximum": 1},
        "true_positives": {"type": "integer", "minimum": 0},
        "true_negatives": {"type": "integer", "minimum": 0},
        "false_positives": {"type": "integer", "minimum": 0},
        "false_negatives": {"type": "integer", "minimum": 0},
    }
    metric_schema = {"type": "object", "additionalProperties": False, "properties": metric_properties, "required": list(metric_properties)}
    candidate_properties = {
        "candidate_id": {"type": "string"}, "method": {"type": "string", "enum": list(METHOD_NAMES)},
        "origin": {"type": "string"}, "metrics": metric_schema,
    }
    candidate_schema = {"type": "object", "additionalProperties": False, "properties": candidate_properties, "required": list(candidate_properties)}
    model_result_properties = {
        "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "candidates": {"type": "array", "minItems": 3, "maxItems": 3, "items": candidate_schema},
    }
    delta_properties = {
        "direction": {"type": "string", "const": "right_minus_left"},
        "recall_malignant": {"type": "number"}, "f1_malignant": {"type": "number"},
        "roc_auc": {"type": "number"},
    }
    delta_schema = {"type": "object", "additionalProperties": False, "properties": delta_properties, "required": list(delta_properties)}
    pair_properties = {
        "comparison_id": {"type": "string"}, "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "evaluation_scope": {"type": "string", "const": "confirmatory_holdout"},
        "left_method": {"type": "string", "enum": list(METHOD_NAMES)},
        "right_method": {"type": "string", "enum": list(METHOD_NAMES)},
        "left_candidate_id": {"type": "string"}, "right_candidate_id": {"type": "string"},
        "same_confusion_matrix": {"type": "boolean"}, "different_roc_auc": {"type": "boolean"},
        "recall_relation": {"type": "string", "enum": list(RELATIONS)},
        "f1_relation": {"type": "string", "enum": list(RELATIONS)},
        "roc_auc_relation": {"type": "string", "enum": list(RELATIONS)},
        "metric_delta": delta_schema, "tradeoff_present": {"type": "boolean"},
        "interpretation": {"type": "string", "minLength": 1},
    }
    pair_schema = {"type": "object", "additionalProperties": False, "properties": pair_properties, "required": list(pair_properties)}
    interval_properties = {
        "lower": {"type": "number"}, "upper": {"type": "number"},
        "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
    }
    interval_schema = {"type": "object", "additionalProperties": False, "properties": interval_properties, "required": list(interval_properties)}
    delta_recall_properties = {
        "direction": {"type": "string", "const": "right_minus_left"},
        "estimate": {"type": "number"}, "interval": interval_schema,
        "includes_zero": {"type": "boolean"},
    }
    mcnemar_properties = {
        "method": {"type": "string"}, "left_wrong_right_correct": {"type": "integer", "minimum": 0},
        "left_correct_right_wrong": {"type": "integer", "minimum": 0},
        "discordant_total": {"type": "integer", "minimum": 0},
        "p_value": {"type": "number", "minimum": 0, "maximum": 1},
        "low_count_warning": {"type": "boolean"},
        "evidence_source": {"type": "string", "const": "authoritative_aggregate_artifact"},
    }
    mcnemar_schema = {"type": "object", "additionalProperties": False, "properties": mcnemar_properties, "required": list(mcnemar_properties)}
    uncertainty_properties = {
        "comparison_id": {"type": "string"}, "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "left_method": {"type": "string", "const": "baseline"},
        "right_method": {"type": "string", "const": "ga"},
        "left_recall_interval": interval_schema, "right_recall_interval": interval_schema,
        "delta_recall": {"type": "object", "additionalProperties": False, "properties": delta_recall_properties, "required": list(delta_recall_properties)},
        "mcnemar": mcnemar_schema, "limited_by_few_discordances": {"type": "boolean"},
        "interpretation": {"type": "string", "minLength": 1},
    }
    uncertainty_schema = {"type": "object", "additionalProperties": False, "properties": uncertainty_properties, "required": list(uncertainty_properties)}
    selected_properties = {
        "candidate_id": {"type": "string"}, "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "method": {"type": "string", "enum": list(METHOD_NAMES)}, "origin": {"type": "string"},
        "frozen_before_holdout": {"type": "boolean", "const": True}, "authority": {"type": "string"},
        "authority_limitation": {"type": "string"}, "explanation": {"type": "string"},
    }
    properties = {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION_V2},
        "contract_version": {"type": "string", "const": CONTRACT_VERSION_V2},
        "resumo_executivo": {"type": "string", "minLength": 1},
        "modelo_selecionado": {"type": "object", "additionalProperties": False, "properties": selected_properties, "required": list(selected_properties)},
        "model_results": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False, "properties": model_result_properties, "required": list(model_result_properties)}},
        "comparison_findings": {"type": "array", "minItems": 9, "maxItems": 9, "items": pair_schema},
        "interpretacao_comparacoes": {"type": "string", "minLength": 1},
        "incerteza_estatistica": {"type": "string", "minLength": 1},
        "uncertainty_findings": {"type": "array", "minItems": 3, "maxItems": 3, "items": uncertainty_schema},
        "limitacoes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "conclusao": {"type": "string", "minLength": 1},
        "disclaimer": {"type": "string", "const": DISCLAIMER},
        "holdout_nao_reabriu_selecao": {"type": "boolean", "const": True},
        "uso_clinico_autorizado": {"type": "boolean", "const": False},
    }
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}
