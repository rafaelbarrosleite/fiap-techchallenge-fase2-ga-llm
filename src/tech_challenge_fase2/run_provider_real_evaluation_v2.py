"""CLI explícita da Missão 7.2; pytest nunca chama estes comandos reais."""

from __future__ import annotations

import json

from .provider_real_evaluation_v2 import (
    finalize_v2_manifest,
    prepare_v2,
    run_adversarial_v2,
    run_scientific_v2,
    run_technical_probe,
    validate_v2,
)


def prepare_main() -> None:
    result = prepare_v2()
    print(json.dumps({
        "passed": result["passed"], "model": result["configuration"]["model"],
        "temperature_sent": result["configuration"]["temperature_sent"],
        "privacy_valid": result["privacy_valid"], "api_calls_performed": 0,
    }, ensure_ascii=False, indent=2))


def probe_main() -> None:
    result = run_technical_probe()
    finalize_v2_manifest()
    print(json.dumps({
        "status": result["status"], "http_status": result.get("http_status"),
        "request_id": result.get("request_id"), "duration_seconds": result["duration_seconds"],
        "usage": result["usage"], "temperature_sent": result["temperature_sent"],
        "retries": result["retries"],
    }, ensure_ascii=False, indent=2))
    if result["status"] != "approved":
        raise SystemExit(1)


def scientific_main() -> None:
    result = run_scientific_v2()
    finalize_v2_manifest()
    print(json.dumps({
        "approved": result["approved"], "schema_valid": result["schema_valid"],
        "response_id": result.get("response_id"),
    }, ensure_ascii=False, indent=2))
    if not result["approved"]:
        raise SystemExit(1)


def adversarial_main() -> None:
    result = run_adversarial_v2()
    manifest = finalize_v2_manifest()
    print(json.dumps({
        "status": result["status"], "provider_calls": result["provider_calls"],
        "total_calls": manifest["call_budget"]["total"],
        "scenarios": [item["scenario"] for item in result["scenarios"]],
    }, ensure_ascii=False, indent=2))
    if result["status"] != "approved":
        raise SystemExit(1)


def validate_main() -> None:
    result = validate_v2()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
