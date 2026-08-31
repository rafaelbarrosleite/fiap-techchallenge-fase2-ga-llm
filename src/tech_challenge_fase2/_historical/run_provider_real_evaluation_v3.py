"""CLI explicita da Missao 7.3; a suite de testes nunca chama a API."""

from __future__ import annotations

import json

from .provider_real_evaluation_v3 import (
    finalize_v3_manifest,
    prepare_v3,
    run_adversarial_v3,
    run_scientific_v3,
    validate_v3,
)


def prepare_main() -> None:
    result = prepare_v3()
    print(json.dumps({
        "passed": result["passed"], "model": result["configuration"]["model"],
        "temperature_sent": result["configuration"]["temperature_sent"],
        "store": result["configuration"]["store"],
        "privacy_valid": result["privacy_valid"],
        "technical_probe_repeated": result["technical_probe_repeated"],
        "provider_calls_performed": result["provider_calls_performed"],
    }, ensure_ascii=False, indent=2))


def scientific_main() -> None:
    result = run_scientific_v3()
    print(json.dumps({
        "approved": result["approved"], "provider": result["provider"],
        "requested_model": result["requested_model"], "response_model": result["response_model"],
        "response_id": result["response_id"], "response_status": result["response_status"],
    }, ensure_ascii=False, indent=2))
    if result["approved"] is not True:
        raise SystemExit(1)


def adversarial_main() -> None:
    result = run_adversarial_v3()
    print(json.dumps({
        "status": result["status"], "provider_calls": result["provider_calls"],
        "scenarios": [
            {"scenario": item["scenario"], "passed": item["scenario_passed"]}
            for item in result["scenarios"]
        ],
    }, ensure_ascii=False, indent=2))
    if result["status"] != "approved":
        raise SystemExit(1)


def validate_main() -> None:
    result = validate_v3()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def finalize_main() -> None:
    result = finalize_v3_manifest()
    print(json.dumps({
        "status": result["status"], "approved": result["approved"],
        "call_budget": result["call_budget"],
    }, ensure_ascii=False, indent=2))
