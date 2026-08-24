"""Factualidade V2 independente, indexada por comparison_id explicito."""

from __future__ import annotations

import math
import re
from typing import Any

from tech_challenge_fase2.llm.safety import output_text

from .schemas import validate_output_v2


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _all_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_numbers(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_numbers(child)]
    return []


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?%?")


def validate_factuality_v2(output: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, expected: Any = None, actual: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "expected": expected, "actual": actual})

    try:
        validate_output_v2(output)
        check("output_schema_v2", True, "schema_2.0", "schema_2.0")
    except ValueError as error:
        check("output_schema_v2", False, "schema_2.0", str(error))
        return {"passed": False, "checks": checks, "unexpected_text_numbers": []}

    selected_expected, selected_actual = source["selected_model"], output["modelo_selecionado"]
    for key in ("candidate_id", "model", "method", "origin", "frozen_before_holdout"):
        check(f"selected_model.{key}", _equal(selected_actual[key], selected_expected[key]), selected_expected[key], selected_actual[key])

    source_models = {item["model"]: item for item in source["model_results"]}
    output_models = {item["model"]: item for item in output["model_results"]}
    for model, expected_model in source_models.items():
        actual_model = output_models.get(model)
        check(f"model_results.{model}.present", actual_model is not None, True, actual_model is not None)
        if actual_model is None:
            continue
        expected_candidates = {item["method"]: item for item in expected_model["candidates"]}
        actual_candidates = {item["method"]: item for item in actual_model["candidates"]}
        for method, expected_candidate in expected_candidates.items():
            actual_candidate = actual_candidates.get(method)
            check(f"{model}.{method}.present", actual_candidate is not None, True, actual_candidate is not None)
            if actual_candidate is None:
                continue
            for key in ("candidate_id", "method", "origin"):
                check(f"{model}.{method}.{key}", _equal(actual_candidate[key], expected_candidate[key]), expected_candidate[key], actual_candidate[key])
            for metric, expected_value in expected_candidate["metrics"].items():
                actual_value = actual_candidate["metrics"][metric]
                check(f"{model}.{method}.{metric}", _equal(actual_value, expected_value), expected_value, actual_value)

    source_pairs = {item["comparison_id"]: item for item in source["comparison_pairs"]}
    output_pairs = {item["comparison_id"]: item for item in output["comparison_findings"]}
    pair_fields = (
        "model", "evaluation_scope", "left_method", "right_method", "left_candidate_id",
        "right_candidate_id", "same_confusion_matrix", "different_roc_auc",
        "recall_relation", "f1_relation", "roc_auc_relation",
    )
    for pair_id, expected_pair in source_pairs.items():
        actual_pair = output_pairs.get(pair_id)
        check(f"pair.{pair_id}.present", actual_pair is not None, True, actual_pair is not None)
        if actual_pair is None:
            continue
        for field in pair_fields:
            check(f"pair.{pair_id}.{field}", _equal(actual_pair[field], expected_pair[field]), expected_pair[field], actual_pair[field])
        for metric, expected_value in expected_pair["metric_delta"].items():
            actual_value = actual_pair["metric_delta"][metric]
            check(f"pair.{pair_id}.metric_delta.{metric}", _equal(actual_value, expected_value), expected_value, actual_value)

    source_uncertainty = {item["comparison_id"]: item for item in source["uncertainty_comparisons"]}
    output_uncertainty = {item["comparison_id"]: item for item in output["uncertainty_findings"]}
    for pair_id, expected_item in source_uncertainty.items():
        actual_item = output_uncertainty.get(pair_id)
        check(f"uncertainty.{pair_id}.present", actual_item is not None, True, actual_item is not None)
        if actual_item is None:
            continue
        for field in ("model", "left_method", "right_method"):
            check(f"uncertainty.{pair_id}.{field}", _equal(actual_item[field], expected_item[field]), expected_item[field], actual_item[field])
        for interval_name in ("left_recall_interval", "right_recall_interval"):
            for field, expected_value in expected_item[interval_name].items():
                actual_value = actual_item[interval_name][field]
                check(f"uncertainty.{pair_id}.{interval_name}.{field}", _equal(actual_value, expected_value), expected_value, actual_value)
        for field, expected_value in expected_item["delta_recall"].items():
            if field == "interval":
                for interval_field, interval_expected in expected_value.items():
                    actual_value = actual_item["delta_recall"]["interval"][interval_field]
                    check(f"uncertainty.{pair_id}.delta_recall.interval.{interval_field}", _equal(actual_value, interval_expected), interval_expected, actual_value)
            else:
                actual_value = actual_item["delta_recall"][field]
                check(f"uncertainty.{pair_id}.delta_recall.{field}", _equal(actual_value, expected_value), expected_value, actual_value)
        for field, expected_value in expected_item["mcnemar"].items():
            actual_value = actual_item["mcnemar"][field]
            check(f"uncertainty.{pair_id}.mcnemar.{field}", _equal(actual_value, expected_value), expected_value, actual_value)
        check(
            f"uncertainty.{pair_id}.limited_by_few_discordances",
            actual_item["limited_by_few_discordances"] is expected_item["mcnemar"]["low_count_warning"],
            expected_item["mcnemar"]["low_count_warning"], actual_item["limited_by_few_discordances"],
        )

    check("holdout_selection_preserved", output["holdout_nao_reabriu_selecao"] is True, True, output["holdout_nao_reabriu_selecao"])
    check("clinical_use_not_authorized", output["uso_clinico_autorizado"] is False, False, output["uso_clinico_autorizado"])
    allowed = _all_numbers(source) + [0.05, 95.0]
    unexpected: list[str] = []
    for token in NUMBER_PATTERN.findall(output_text(output)):
        percent = token.endswith("%")
        value = float(token.rstrip("%").replace(",", ".")) / (100.0 if percent else 1.0)
        if not any(math.isclose(value, candidate, rel_tol=0.0, abs_tol=5e-7) for candidate in allowed):
            unexpected.append(token)
    unexpected = sorted(set(unexpected))
    check("no_unexpected_narrative_numbers", not unexpected, [], unexpected)
    return {"passed": all(item["passed"] for item in checks), "checks": checks, "unexpected_text_numbers": unexpected}
