"""CLIs opt-in da Missao 7.5."""

from __future__ import annotations

import json

from .provider_real_evaluation_v4 import (
    prepare_v4,
    run_adversarial_v4,
    run_scientific_v4,
    validate_v4,
)


def prepare_main() -> None:
    result = prepare_v4()
    print(json.dumps({
        "passed": result["passed"], "mission": result["mission"],
        "contract_version": result["configuration"]["contract_version"],
        "provider_calls_performed": result["provider_calls_performed"],
    }, ensure_ascii=False))


def scientific_main() -> None:
    result = run_scientific_v4()
    print(json.dumps({
        "scientific_evaluation_approved": result["scientific_evaluation_approved"],
        "provider": result.get("provider"), "response_status": result.get("response_status"),
    }, ensure_ascii=False))


def adversarial_main() -> None:
    result = run_adversarial_v4()
    print(json.dumps({
        "status": result["status"], "provider_calls": result["provider_calls"],
        "retries": result["retries"],
    }, ensure_ascii=False))


def validate_main() -> None:
    result = validate_v4()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)
