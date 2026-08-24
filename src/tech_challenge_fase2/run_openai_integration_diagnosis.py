"""Comandos explícitos do diagnóstico isolado da integração OpenAI."""

from __future__ import annotations

import argparse
import json

from .openai_integration_diagnosis import prepare_diagnosis, run_minimal_diagnostic, validate_diagnosis


def diagnose_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Constrói e valida sem chamar a API.")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("Use --dry-run; a chamada real possui comando separado e explícito.")
    result = prepare_diagnosis()
    print(json.dumps({
        "model": result["model"], "provider": result["provider"],
        "schema_valid": True, "privacy_valid": result["privacy_valid"],
        "store": result["store"], "request_hash": result["request_hash"],
        "api_call_performed": result["api_call_performed"],
    }, ensure_ascii=False, indent=2))


def test_structured_output_main() -> None:
    result = run_minimal_diagnostic()
    print(json.dumps({
        "status": result["status"], "request_success": result["request_success"],
        "requested_model": result["requested_model"],
        "response_id": result.get("response_id"), "duration_seconds": result["duration_seconds"],
        "automatic_retry_performed": result["automatic_retry_performed"],
    }, ensure_ascii=False, indent=2))
    if not result["request_success"]:
        raise SystemExit(1)


def validate_main() -> None:
    result = validate_diagnosis()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
