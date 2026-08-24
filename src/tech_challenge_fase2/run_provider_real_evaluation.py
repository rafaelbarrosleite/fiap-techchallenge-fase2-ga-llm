"""CLI explicita da Missao 7; nunca e chamada pela suite de testes."""

from __future__ import annotations

import json

from .provider_real_evaluation import (
    finalize_failed_openai_evaluation, prepare_openai_evaluation,
    run_openai_adversarial, run_openai_main, validate_openai_evaluation,
)


def prepare_main() -> None:
    snapshot = prepare_openai_evaluation()
    print(json.dumps({
        "status": "prepared", "run_identity": snapshot["run_identity"],
        "model": snapshot["provider_configuration"]["model"],
        "privacy_passed": snapshot["privacy_validation"]["passed"],
    }, ensure_ascii=False))


def run_main() -> None:
    result = run_openai_main()
    print(json.dumps({
        "approved": result["approved"], "provider": result["provider"],
        "model": result["requested_model"], "response_id": result["response_id"],
    }, ensure_ascii=False))


def adversarial_main() -> None:
    result = run_openai_adversarial()
    print(json.dumps({
        "status": result["status"], "provider_calls": result["provider_calls"],
        "scenarios": [item["scenario"] for item in result["scenarios"]],
    }, ensure_ascii=False))


def validate_main() -> None:
    result = validate_openai_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def finalize_failure_main() -> None:
    result = finalize_failed_openai_evaluation()
    print(json.dumps({
        "status": "completed_invalid", "stage": result["stage"],
        "automatic_retry_performed": result["automatic_retry_performed"],
    }, ensure_ascii=False))
