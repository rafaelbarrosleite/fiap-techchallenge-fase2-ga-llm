"""Validacao factual independente da explicacao individual."""

from __future__ import annotations

import math
from typing import Any

from .schemas import validate_output


def _check(checks: list[dict[str, Any]], name: str, expected: Any, actual: Any) -> None:
    if isinstance(expected, float):
        passed = isinstance(actual, (int, float)) and math.isclose(float(actual), expected, abs_tol=1e-12)
    else:
        passed = actual == expected
    checks.append({"check": name, "passed": passed, "expected": expected, "actual": actual})


def validate_factuality(output: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    validate_output(output)
    checks: list[dict[str, Any]] = []
    case, classification = source["case_context"], output["classificacao_do_modelo"]
    _check(checks, "case_reference", case["case_reference"], output["case_reference"])
    _check(checks, "candidate_id", source["model_context"]["candidate_id"], classification["candidate_id"])
    for key in ("predicted_pattern", "probability_malignant", "classification_threshold"):
        _check(checks, f"classification.{key}", case[key], classification[key])
    expected_signals = source["explanation_signals"]
    actual_signals = output["fatores_explicativos"]
    _check(checks, "signal_count", len(expected_signals), len(actual_signals))
    for expected, actual in zip(expected_signals, actual_signals, strict=True):
        for key in (
            "rank", "feature", "display_name", "observed_band", "influence_direction",
            "relative_importance_percent",
        ):
            _check(checks, f"signal.{expected['rank']}.{key}", expected[key], actual[key])
    _check(checks, "prediction_is_not_diagnosis", True, output["predicao_nao_e_diagnostico"])
    _check(checks, "clinical_use_authorized", False, output["uso_clinico_autorizado"])
    _check(checks, "future_text_ready", True, output["preparacao_modulo3"]["ready_for_future_text"])
    _check(checks, "current_text_used", False, output["preparacao_modulo3"]["current_text_data_used"])
    return {
        "passed": all(item["passed"] for item in checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "checks": checks,
    }
