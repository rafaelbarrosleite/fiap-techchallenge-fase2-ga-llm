"""Validacao factual independente da geracao do provider."""

from __future__ import annotations

import math
import re
from typing import Any

from .providers import _change
from .schemas import METHOD_NAMES, MODEL_NAMES, validate_output


def _equal_number(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12,
    )


def _all_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _all_text(child)]
    if isinstance(value, list):
        return [text for child in value for text in _all_text(child)]
    return []


def _all_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for child in value.values() for number in _all_numbers(child)]
    if isinstance(value, list):
        return [number for child in value for number in _all_numbers(child)]
    return []


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?%?")


def _unexpected_text_numbers(output: dict[str, Any], source: dict[str, Any]) -> list[str]:
    allowed = _all_numbers(source) + [0.05, 95.0]
    unexpected: list[str] = []
    for text in _all_text(output):
        for token in NUMBER_PATTERN.findall(text):
            percent = token.endswith("%")
            raw = token.rstrip("%").replace(",", ".")
            value = float(raw) / 100.0 if percent else float(raw)
            if not any(math.isclose(value, item, rel_tol=0.0, abs_tol=5e-7) for item in allowed):
                unexpected.append(token)
    return sorted(set(unexpected))


def validate_factuality(output: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    try:
        validate_output(output)
        check("output_schema", True, "Contrato fechado satisfeito.")
    except ValueError as error:
        check("output_schema", False, str(error))
        return {"passed": False, "checks": checks, "unexpected_text_numbers": []}

    selected_out = output["modelo_selecionado"]
    selected_in = source["selected_model"]
    for key in ("candidate_id", "model", "method"):
        check(f"selected_model.{key}", selected_out[key] == selected_in[key], f"esperado={selected_in[key]}")

    out_by_model = {item["model"]: item for item in output["comparacao_modelos"]}
    source_by_model = {item["model"]: item for item in source["model_comparison"]}
    for model in MODEL_NAMES:
        actual = out_by_model.get(model)
        expected = source_by_model[model]
        check(f"{model}.present", actual is not None, "Familia obrigatoria.")
        if actual is None:
            continue
        for method in METHOD_NAMES:
            source_metrics = expected[method]["metrics"]
            output_metrics = actual[method]
            check(f"{model}.{method}.method", output_metrics["method"] == method, f"esperado={method}")
            for metric, expected_value in source_metrics.items():
                check(
                    f"{model}.{method}.{metric}", _equal_number(output_metrics[metric], expected_value),
                    f"esperado={expected_value}",
                )
        baseline, ga = expected["baseline"]["metrics"], expected["ga"]["metrics"]
        expected_changes = {
            "ga_recall_change": _change(ga["recall_malignant"], baseline["recall_malignant"]),
            "ga_f1_change": _change(ga["f1_malignant"], baseline["f1_malignant"]),
            "ga_auc_change": _change(ga["roc_auc"], baseline["roc_auc"]),
        }
        for field, value in expected_changes.items():
            check(f"{model}.{field}", actual[field] == value, f"esperado={value}")
        directions = set(expected_changes.values()).difference({"unchanged"})
        expected_tradeoff = len(directions) > 1
        check(f"{model}.tradeoff", actual["tradeoff_present"] == expected_tradeoff, f"esperado={expected_tradeoff}")
        threshold_keys = ("true_positives", "true_negatives", "false_positives", "false_negatives")
        same_threshold = all(ga[key] == baseline[key] for key in threshold_keys)
        expected_same_auc = same_threshold and not _equal_number(ga["roc_auc"], baseline["roc_auc"])
        check(f"{model}.same_threshold_different_auc", actual["same_threshold_outcomes_different_auc"] == expected_same_auc, f"esperado={expected_same_auc}")
        expected_confirmation = (
            expected["cv_recall_ga"] > expected["cv_recall_baseline"] + 1e-12
            and ga["recall_malignant"] > baseline["recall_malignant"] + 1e-12
        )
        check(f"{model}.cv_gain_confirmed", actual["cv_gain_confirmed_on_holdout"] == expected_confirmation, f"esperado={expected_confirmation}")

    output_uncertainty = {item["model"]: item for item in output["incerteza_por_modelo"]}
    for expected in source["uncertainty_summary"]:
        model = expected["model"]
        actual = output_uncertainty.get(model)
        check(f"{model}.uncertainty_present", actual is not None, "Intervalos obrigatorios.")
        if actual is None:
            continue
        for key in ("delta_recall", "mcnemar_p_value"):
            check(f"{model}.{key}", _equal_number(actual[key], expected[key]), f"esperado={expected[key]}")
        check(f"{model}.delta_ci_includes_zero", actual["delta_ci_includes_zero"] == expected["delta_ci_includes_zero"], f"esperado={expected['delta_ci_includes_zero']}")
        for interval_name in ("baseline_recall_ci", "ga_recall_ci", "delta_recall_ci"):
            for key in ("lower", "upper", "confidence_level"):
                check(
                    f"{model}.{interval_name}.{key}",
                    _equal_number(actual[interval_name][key], expected[interval_name][key]),
                    f"esperado={expected[interval_name][key]}",
                )

    check("holdout_selection_preserved", output["holdout_nao_reabriu_selecao"] is True, "Holdout confirmatorio.")
    check("clinical_use_not_authorized", output["uso_clinico_autorizado"] is False, "Uso clinico proibido.")
    unexpected = _unexpected_text_numbers(output, source)
    check("no_unexpected_narrative_numbers", not unexpected, f"numeros_nao_autorizados={unexpected}")
    return {"passed": all(item["passed"] for item in checks), "checks": checks, "unexpected_text_numbers": unexpected}

