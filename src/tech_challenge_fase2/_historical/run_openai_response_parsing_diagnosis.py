"""CLI explícita da Missão 7.2.1."""

from __future__ import annotations

import json

from .openai_response_parsing_diagnosis import (
    prepare_parsing_diagnosis,
    run_parsing_probe,
    validate_parsing_diagnosis,
)


def prepare_main() -> None:
    result = prepare_parsing_diagnosis()
    print(json.dumps({
        "model": result["model"], "temperature_sent": result["temperature_sent"],
        "store": result["store"], "schema_valid": result["schema_valid"],
        "privacy_valid": result["privacy_valid"],
        "parser_supports_output_array": result["parser_supports_output_array"],
        "api_call_performed": result["api_call_performed"],
    }, ensure_ascii=False, indent=2))


def probe_main() -> None:
    result = run_parsing_probe()
    print(json.dumps({
        "status": result["status"], "http_status": result["http_status"],
        "response_status": result["response_status"], "request_id": result["request_id"],
        "response_id": result["response_id"], "output_text_found": result["output_text_found"],
        "schema_valid": result["schema_valid"], "usage": result["usage"],
        "ready_for_scientific_evaluation": result["ready_for_scientific_evaluation"],
    }, ensure_ascii=False, indent=2))
    if result["status"] != "approved":
        raise SystemExit(1)


def validate_main() -> None:
    result = validate_parsing_diagnosis()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
