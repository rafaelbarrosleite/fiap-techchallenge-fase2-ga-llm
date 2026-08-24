"""CLI offline da Missao 7.4."""

from __future__ import annotations

import json

from .llm_contract_v2 import build_contract_v2_artifacts, validate_contract_v2


def build_main() -> None:
    result = build_contract_v2_artifacts()
    print(json.dumps({
        "status": result["status"], "checks_audited": result["checks_audited"],
        "ambiguous_checks": result["ambiguous_checks"],
        "fake_v2_approved": result["fake_v2_approved"],
        "ready_for_real_v2_evaluation": result["ready_for_real_v2_evaluation"],
        "external_provider_calls": result["execution"]["external_provider_calls"],
    }, ensure_ascii=False, indent=2))
    if result["status"] != "approved":
        raise SystemExit(1)


def validate_main() -> None:
    result = validate_contract_v2()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
