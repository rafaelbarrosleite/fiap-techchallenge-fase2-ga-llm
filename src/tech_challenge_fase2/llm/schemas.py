"""Contratos estritos, sem dependencia de validacao em servico externo."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"
MODEL_NAMES = ("logistic_regression", "random_forest", "knn")
METHOD_NAMES = ("baseline", "ga", "random_search")
CHANGE_VALUES = ("improved", "worsened", "unchanged")

DISCLAIMER = (
    "Este resultado possui finalidade exclusivamente acadêmica e experimental. "
    "Os modelos avaliados não foram validados para uso clínico e não devem ser "
    "utilizados para diagnóstico, tratamento ou tomada de decisão médica."
)


class SchemaError(ValueError):
    """O payload nao satisfaz o contrato fechado."""


def _object(value: Any, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{name} deve ser um objeto.")
    missing = required.difference(value)
    extra = set(value).difference(required)
    if missing or extra:
        raise SchemaError(f"{name}: ausentes={sorted(missing)}, extras={sorted(extra)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} deve ser texto nao vazio.")
    return value


def _number(value: Any, name: str, *, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} deve ser numerico.")
    result = float(value)
    if probability and not 0.0 <= result <= 1.0:
        raise SchemaError(f"{name} deve estar entre zero e um.")
    return result


METRIC_KEYS = {
    "recall_malignant", "f1_malignant", "roc_auc", "true_positives",
    "true_negatives", "false_positives", "false_negatives",
}
CV_KEYS = {"fitness", "recall_malignant", "f1_malignant", "roc_auc"}


def validate_metrics(value: Any, name: str) -> None:
    data = _object(value, name, METRIC_KEYS)
    for key in ("recall_malignant", "f1_malignant", "roc_auc"):
        _number(data[key], f"{name}.{key}", probability=True)
    for key in ("true_positives", "true_negatives", "false_positives", "false_negatives"):
        number = _number(data[key], f"{name}.{key}")
        if number < 0 or not number.is_integer():
            raise SchemaError(f"{name}.{key} deve ser inteiro nao negativo.")


def validate_input(payload: Any) -> dict[str, Any]:
    top = _object(payload, "entrada", {
        "schema_version", "experiment_summary", "model_comparison",
        "uncertainty_summary", "selected_model", "selection_rationale",
        "limitations", "safety_context", "source_provenance",
    })
    if top["schema_version"] != SCHEMA_VERSION:
        raise SchemaError("Versao do contrato de entrada nao suportada.")
    summary = _object(top["experiment_summary"], "experiment_summary", {
        "development_rows", "test_rows", "malignant_test_cases", "classification_threshold",
        "candidate_origins", "unique_training_groups", "selection_reopened",
        "new_optimization_performed", "holdout_role",
    })
    for key in ("development_rows", "test_rows", "malignant_test_cases", "candidate_origins", "unique_training_groups"):
        _number(summary[key], f"experiment_summary.{key}")
    _number(summary["classification_threshold"], "classification_threshold", probability=True)
    if summary["selection_reopened"] is not False or summary["new_optimization_performed"] is not False:
        raise SchemaError("A entrada nao pode reabrir selecao nem registrar nova otimizacao.")
    _text(summary["holdout_role"], "holdout_role")

    comparisons = top["model_comparison"]
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise SchemaError("model_comparison deve conter exatamente tres familias.")
    seen: set[str] = set()
    comparison_keys = {"model", "baseline", "ga", "random_search", "cv_recall_baseline", "cv_recall_ga"}
    candidate_keys = {"candidate_id", "method", "origin", "metrics"}
    for index, item in enumerate(comparisons):
        item = _object(item, f"model_comparison[{index}]", comparison_keys)
        model = item["model"]
        if model not in MODEL_NAMES or model in seen:
            raise SchemaError("Familia de modelo ausente, duplicada ou invalida.")
        seen.add(model)
        for method in METHOD_NAMES:
            candidate = _object(item[method], f"{model}.{method}", candidate_keys)
            if candidate["method"] != method:
                raise SchemaError(f"Metodo inconsistente em {model}.{method}.")
            if candidate["candidate_id"] != f"{model}__{method}":
                raise SchemaError(f"candidate_id inconsistente em {model}.{method}.")
            _text(candidate["candidate_id"], "candidate_id")
            _text(candidate["origin"], "origin")
            validate_metrics(candidate["metrics"], f"{model}.{method}.metrics")
        _number(item["cv_recall_baseline"], "cv_recall_baseline", probability=True)
        _number(item["cv_recall_ga"], "cv_recall_ga", probability=True)
    if seen != set(MODEL_NAMES):
        raise SchemaError("As tres familias obrigatorias devem estar presentes.")

    uncertainty = top["uncertainty_summary"]
    if not isinstance(uncertainty, list) or len(uncertainty) != 3:
        raise SchemaError("uncertainty_summary deve conter tres itens.")
    uncertainty_keys = {
        "model", "baseline_recall_ci", "ga_recall_ci", "delta_recall",
        "delta_recall_ci", "delta_ci_includes_zero", "mcnemar_p_value",
    }
    interval_keys = {"lower", "upper", "confidence_level"}
    uncertainty_seen: set[str] = set()
    for index, item in enumerate(uncertainty):
        item = _object(item, f"uncertainty_summary[{index}]", uncertainty_keys)
        if item["model"] not in MODEL_NAMES or item["model"] in uncertainty_seen:
            raise SchemaError("Modelo invalido na incerteza.")
        uncertainty_seen.add(item["model"])
        for field in ("baseline_recall_ci", "ga_recall_ci", "delta_recall_ci"):
            interval = _object(item[field], field, interval_keys)
            _number(interval["lower"], f"{field}.lower")
            _number(interval["upper"], f"{field}.upper")
            _number(interval["confidence_level"], f"{field}.confidence_level", probability=True)
            if interval["lower"] > interval["upper"]:
                raise SchemaError(f"Intervalo invertido em {field}.")
        _number(item["delta_recall"], "delta_recall")
        if not isinstance(item["delta_ci_includes_zero"], bool):
            raise SchemaError("delta_ci_includes_zero deve ser booleano.")
        _number(item["mcnemar_p_value"], "mcnemar_p_value", probability=True)

    selected = _object(top["selected_model"], "selected_model", {
        "candidate_id", "model", "method", "origin", "frozen_before_holdout",
        "authority", "authority_limitation",
    })
    if selected["model"] not in MODEL_NAMES or selected["method"] not in METHOD_NAMES:
        raise SchemaError("Modelo selecionado invalido.")
    if selected["candidate_id"] != f"{selected['model']}__{selected['method']}":
        raise SchemaError("candidate_id selecionado inconsistente.")
    if selected["frozen_before_holdout"] is not True:
        raise SchemaError("O modelo deve ter sido congelado antes do holdout.")
    for key in ("candidate_id", "origin", "authority", "authority_limitation"):
        _text(selected[key], f"selected_model.{key}")
    _text(top["selection_rationale"], "selection_rationale")
    if not isinstance(top["limitations"], list) or not top["limitations"]:
        raise SchemaError("limitations deve ser lista nao vazia.")
    for item in top["limitations"]:
        _text(item, "limitation")
    safety = _object(top["safety_context"], "safety_context", {
        "academic_experimental_only", "individual_data_included", "clinical_use_authorized",
        "diagnosis_allowed", "medical_recommendation_allowed", "required_disclaimer",
    })
    expected = {
        "academic_experimental_only": True, "individual_data_included": False,
        "clinical_use_authorized": False, "diagnosis_allowed": False,
        "medical_recommendation_allowed": False,
    }
    for key, value in expected.items():
        if safety[key] is not value:
            raise SchemaError(f"Contexto de seguranca invalido em {key}.")
    if safety["required_disclaimer"] != DISCLAIMER:
        raise SchemaError("Disclaimer de entrada divergente.")
    provenance = _object(top["source_provenance"], "source_provenance", {
        "artifacts", "documentation_auxiliary", "all_sources_aggregate",
        "mission4_artifacts_unchanged",
    })
    if not isinstance(provenance["artifacts"], list) or len(provenance["artifacts"]) != 4:
        raise SchemaError("Quatro artefatos estruturados sao obrigatorios.")
    source_names = {
        "final_test_results.json", "uncertainty_results.json",
        "final_manifest.json", "final_evaluation_plan.json",
    }
    artifact_names: set[str] = set()
    for index, artifact in enumerate(provenance["artifacts"]):
        artifact = _object(artifact, f"source_provenance.artifacts[{index}]", {"filename", "sha256", "signature"})
        if artifact["filename"] not in source_names or artifact["filename"] in artifact_names:
            raise SchemaError("Fonte estruturada ausente, duplicada ou nao autorizada.")
        artifact_names.add(artifact["filename"])
        for field in ("sha256", "signature"):
            value = _text(artifact[field], f"artifact.{field}")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise SchemaError(f"Hash invalido em artifact.{field}.")
    auxiliary = _object(provenance["documentation_auxiliary"], "documentation_auxiliary", {"filename", "sha256"})
    _text(auxiliary["filename"], "documentation_auxiliary.filename")
    auxiliary_hash = _text(auxiliary["sha256"], "documentation_auxiliary.sha256")
    if len(auxiliary_hash) != 64 or any(character not in "0123456789abcdef" for character in auxiliary_hash):
        raise SchemaError("Hash auxiliar invalido.")
    if provenance["all_sources_aggregate"] is not True or provenance["mission4_artifacts_unchanged"] is not True:
        raise SchemaError("Proveniencia agregada ou imutabilidade invalida.")
    return top


OUTPUT_KEYS = {
    "schema_version", "resumo_executivo", "modelo_selecionado", "comparacao_modelos",
    "interpretacao_ga", "incerteza_estatistica", "incerteza_por_modelo", "limitacoes", "conclusao", "disclaimer",
    "holdout_nao_reabriu_selecao", "uso_clinico_autorizado",
}
OUTPUT_SELECTED_KEYS = {"candidate_id", "model", "method", "explanation"}
OUTPUT_COMPARISON_KEYS = {
    "model", "baseline", "ga", "random_search", "ga_recall_change",
    "ga_f1_change", "ga_auc_change", "tradeoff_present",
    "cv_gain_confirmed_on_holdout", "same_threshold_outcomes_different_auc", "interpretation",
}


def validate_output(payload: Any) -> dict[str, Any]:
    top = _object(payload, "saida", OUTPUT_KEYS)
    if top["schema_version"] != SCHEMA_VERSION:
        raise SchemaError("Versao do contrato de saida nao suportada.")
    for key in ("resumo_executivo", "interpretacao_ga", "incerteza_estatistica", "conclusao"):
        _text(top[key], key)
    selected = _object(top["modelo_selecionado"], "modelo_selecionado", OUTPUT_SELECTED_KEYS)
    if selected["model"] not in MODEL_NAMES or selected["method"] not in METHOD_NAMES:
        raise SchemaError("Modelo selecionado invalido na saida.")
    if selected["candidate_id"] != f"{selected['model']}__{selected['method']}":
        raise SchemaError("candidate_id selecionado inconsistente na saida.")
    _text(selected["candidate_id"], "candidate_id")
    _text(selected["explanation"], "explanation")
    comparisons = top["comparacao_modelos"]
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise SchemaError("comparacao_modelos deve conter tres itens.")
    seen: set[str] = set()
    metric_output_keys = METRIC_KEYS | {"method"}
    for index, item in enumerate(comparisons):
        item = _object(item, f"comparacao_modelos[{index}]", OUTPUT_COMPARISON_KEYS)
        if item["model"] not in MODEL_NAMES or item["model"] in seen:
            raise SchemaError("Modelo duplicado ou invalido na saida.")
        seen.add(item["model"])
        for method in METHOD_NAMES:
            metrics = _object(item[method], f"saida.{item['model']}.{method}", metric_output_keys)
            if metrics["method"] != method:
                raise SchemaError("Metodo numerico inconsistente na saida.")
            validate_metrics({key: metrics[key] for key in METRIC_KEYS}, f"saida.{method}")
        for field in ("ga_recall_change", "ga_f1_change", "ga_auc_change"):
            if item[field] not in CHANGE_VALUES:
                raise SchemaError(f"Mudanca invalida em {field}.")
        for field in ("tradeoff_present", "cv_gain_confirmed_on_holdout", "same_threshold_outcomes_different_auc"):
            if not isinstance(item[field], bool):
                raise SchemaError(f"{field} deve ser booleano.")
        _text(item["interpretation"], "interpretation")
    if seen != set(MODEL_NAMES):
        raise SchemaError("A saida deve cobrir as tres familias.")
    uncertainty = top["incerteza_por_modelo"]
    if not isinstance(uncertainty, list) or len(uncertainty) != 3:
        raise SchemaError("incerteza_por_modelo deve conter tres itens.")
    uncertainty_keys = {
        "model", "baseline_recall_ci", "ga_recall_ci", "delta_recall",
        "delta_recall_ci", "delta_ci_includes_zero", "mcnemar_p_value",
    }
    interval_keys = {"lower", "upper", "confidence_level"}
    uncertainty_seen: set[str] = set()
    for index, item in enumerate(uncertainty):
        item = _object(item, f"incerteza_por_modelo[{index}]", uncertainty_keys)
        if item["model"] not in MODEL_NAMES or item["model"] in uncertainty_seen:
            raise SchemaError("Modelo invalido na incerteza de saida.")
        uncertainty_seen.add(item["model"])
        for field in ("baseline_recall_ci", "ga_recall_ci", "delta_recall_ci"):
            interval = _object(item[field], field, interval_keys)
            _number(interval["lower"], f"{field}.lower")
            _number(interval["upper"], f"{field}.upper")
            _number(interval["confidence_level"], f"{field}.confidence_level", probability=True)
        _number(item["delta_recall"], "delta_recall")
        if not isinstance(item["delta_ci_includes_zero"], bool):
            raise SchemaError("delta_ci_includes_zero deve ser booleano.")
        _number(item["mcnemar_p_value"], "mcnemar_p_value", probability=True)
    if not isinstance(top["limitacoes"], list) or not top["limitacoes"]:
        raise SchemaError("A saida deve conter limitacoes.")
    for item in top["limitacoes"]:
        _text(item, "limitacao")
    if top["holdout_nao_reabriu_selecao"] is not True:
        raise SchemaError("A saida deve preservar a selecao congelada.")
    if top["uso_clinico_autorizado"] is not False:
        raise SchemaError("A saida nao pode autorizar uso clinico.")
    if top["disclaimer"] != DISCLAIMER:
        raise SchemaError("Disclaimer obrigatorio ausente ou alterado.")
    return top


def output_json_schema() -> dict[str, Any]:
    """JSON Schema enviado ao provider real para Structured Outputs."""

    metric_properties = {
        "method": {"type": "string", "enum": list(METHOD_NAMES)},
        "recall_malignant": {"type": "number", "minimum": 0, "maximum": 1},
        "f1_malignant": {"type": "number", "minimum": 0, "maximum": 1},
        "roc_auc": {"type": "number", "minimum": 0, "maximum": 1},
        "true_positives": {"type": "integer", "minimum": 0},
        "true_negatives": {"type": "integer", "minimum": 0},
        "false_positives": {"type": "integer", "minimum": 0},
        "false_negatives": {"type": "integer", "minimum": 0},
    }
    metric_schema = {
        "type": "object", "additionalProperties": False,
        "properties": metric_properties, "required": list(metric_properties),
    }
    comparison_properties = {
        "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "baseline": metric_schema, "ga": metric_schema, "random_search": metric_schema,
        "ga_recall_change": {"type": "string", "enum": list(CHANGE_VALUES)},
        "ga_f1_change": {"type": "string", "enum": list(CHANGE_VALUES)},
        "ga_auc_change": {"type": "string", "enum": list(CHANGE_VALUES)},
        "tradeoff_present": {"type": "boolean"},
        "cv_gain_confirmed_on_holdout": {"type": "boolean"},
        "same_threshold_outcomes_different_auc": {"type": "boolean"},
        "interpretation": {"type": "string", "minLength": 1},
    }
    selected_properties = {
        "candidate_id": {"type": "string"}, "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "method": {"type": "string", "enum": list(METHOD_NAMES)}, "explanation": {"type": "string"},
    }
    interval_properties = {
        "lower": {"type": "number"}, "upper": {"type": "number"},
        "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
    }
    interval_schema = {"type": "object", "additionalProperties": False, "properties": interval_properties, "required": list(interval_properties)}
    uncertainty_properties = {
        "model": {"type": "string", "enum": list(MODEL_NAMES)},
        "baseline_recall_ci": interval_schema, "ga_recall_ci": interval_schema,
        "delta_recall": {"type": "number"}, "delta_recall_ci": interval_schema,
        "delta_ci_includes_zero": {"type": "boolean"},
        "mcnemar_p_value": {"type": "number", "minimum": 0, "maximum": 1},
    }
    properties = {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "resumo_executivo": {"type": "string", "minLength": 1},
        "modelo_selecionado": {"type": "object", "additionalProperties": False, "properties": selected_properties, "required": list(selected_properties)},
        "comparacao_modelos": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False, "properties": comparison_properties, "required": list(comparison_properties)}},
        "interpretacao_ga": {"type": "string", "minLength": 1},
        "incerteza_estatistica": {"type": "string", "minLength": 1},
        "incerteza_por_modelo": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False, "properties": uncertainty_properties, "required": list(uncertainty_properties)}},
        "limitacoes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "conclusao": {"type": "string", "minLength": 1},
        "disclaimer": {"type": "string", "const": DISCLAIMER},
        "holdout_nao_reabriu_selecao": {"type": "boolean", "const": True},
        "uso_clinico_autorizado": {"type": "boolean", "const": False},
    }
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}
