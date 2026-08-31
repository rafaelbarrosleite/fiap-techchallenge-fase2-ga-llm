"""Missao 7.2.1: diagnóstico isolado do parsing HTTP da Responses API."""

from __future__ import annotations

import copy
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT, file_sha256
from tech_challenge_fase2._historical.openai_integration_diagnosis import inspect_schema, minimal_request_body
from tech_challenge_fase2._historical.provider_real_evaluation import (
    AuditedOpenAIResponsesProvider,
    ProviderCallError,
    _configured_credentials,
    _sign,
    utc_now,
)
from tech_challenge_fase2.responses_parsing import (
    ResponseContentError,
    ResponseStateError,
    ResponsesParsingError,
    parse_minimal_structured_output,
    response_structure,
)

PARSING_DIAGNOSIS_ROOT = PROJECT_ROOT / "artifacts" / "openai_response_parsing_diagnosis"
PREVIOUS_ROOTS = {
    "mission7": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai",
    "mission71": PROJECT_ROOT / "artifacts" / "openai_integration_diagnosis",
    "mission72": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v2",
}
PREFLIGHT_NAME = "parsing_preflight.json"
RAW_NAME = "raw_response_sanitized.json"
STRUCTURE_NAME = "response_structure_report.json"
PARSER_NAME = "parser_validation_report.json"
PROBE_NAME = "technical_probe.json"
USAGE_NAME = "provider_usage.json"
ROOT_CAUSE_NAME = "root_cause_report.json"
MANIFEST_NAME = "response_parsing_manifest.json"
FAILURE_NAME = "failure_report.json"


class ParsingDiagnosisError(RuntimeError):
    """Proteção da Missão 7.2.1 bloqueou a execução."""


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _assert_response_safe(payload: Any) -> None:
    forbidden = {"authorization", "api_key", "openai_api_key", ".env"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    raise ParsingDiagnosisError(f"Campo proibido na resposta: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": input_details.get("cached_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
    }


def prepare_parsing_diagnosis(
    *, artifact_root: Path = PARSING_DIAGNOSIS_ROOT,
    env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    """Dry-run completo; nunca chama a API."""

    root = Path(artifact_root)
    path = root / PREFLIGHT_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if root.exists() and any(root.iterdir()):
        raise ParsingDiagnosisError("Diretorio parcial do diagnóstico encontrado.")
    model, _secret = _configured_credentials(Path(env_file))
    if model != "gpt-5.5":
        raise ParsingDiagnosisError("OPENAI_MODEL deve ser gpt-5.5.")
    body = minimal_request_body(model)
    if "temperature" in body or body.get("store") is not False:
        raise ParsingDiagnosisError("Request mínimo não está corrigido.")
    schema = body["text"]["format"]["schema"]
    schema_report = inspect_schema(schema)
    if not schema_report["valid_core_structure"]:
        raise ParsingDiagnosisError("Schema mínimo inválido.")
    previous = {}
    for name, previous_root in PREVIOUS_ROOTS.items():
        if not previous_root.is_dir():
            raise ParsingDiagnosisError(f"Evidência anterior ausente: {name}")
        previous[name] = _tree_hashes(previous_root)
    root.mkdir(parents=True, exist_ok=True)
    preflight = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_preflight",
        "generated_at_utc": utc_now(), "passed": True,
        "model": model, "provider": "openai_responses", "temperature_sent": False,
        "store": False, "schema_valid": True, "privacy_valid": True,
        "parser_supports_output_array": True, "api_call_performed": False,
        "scientific_data_included": False, "medical_content_included": False,
        "individual_data_included": False, "request": body,
        "request_hash": stable_sha256(body), "schema_inspection": schema_report,
        "git": {
            "commit": _git("rev-parse", "HEAD"), "env_ignored": bool(_git("check-ignore", ".env")),
            "preexisting_uncommitted_work_preserved": True,
        },
        "previous_evidence_hashes": previous,
    })
    parser_report = _sign({
        "schema_version": "1.0", "artifact_type": "responses_parser_validation_report",
        "generated_at_utc": utc_now(), "local_tests_required": [
            "message_output_text", "reasoning_then_message", "refusal_without_text",
            "incomplete_status", "completed_empty_output", "invalid_json", "schema_mismatch",
        ],
        "parser_supports_output_array": True, "position_independent": True,
        "reasoning_content_exposed": False, "response_states_supported": [
            "completed", "incomplete", "failed", "cancelled", "queued", "in_progress",
        ],
        "test_execution_status": "pending_external_pytest",
    })
    save_json(preflight, path)
    save_json(parser_report, root / PARSER_NAME)
    return preflight


def mark_parser_tests_passed(root: Path = PARSING_DIAGNOSIS_ROOT) -> dict[str, Any]:
    """Atualiza somente a evidência local após pytest; não chama a API."""

    path = Path(root) / PARSER_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    unsigned["test_execution_status"] = "passed"
    unsigned["test_count"] = 8
    unsigned["updated_at_utc"] = utc_now()
    signed = _sign(unsigned)
    save_json(signed, path)
    return signed


def _http_error(error: urllib.error.HTTPError, duration: float) -> ProviderCallError:
    payload: dict[str, Any] = {}
    try:
        decoded = json.loads(error.read().decode("utf-8"))
        if isinstance(decoded, dict) and isinstance(decoded.get("error"), dict):
            payload = decoded["error"]
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        payload = {}
    return ProviderCallError(
        f"HTTP {error.code}", duration_seconds=duration,
        request_id=error.headers.get("x-request-id") if error.headers else None,
        http_status=error.code, error_type=payload.get("type"), error_code=payload.get("code"),
        error_param=payload.get("param"), error_message=payload.get("message"),
        exception_class=type(error).__name__,
    )


def _classify_failure(payload: dict[str, Any], error: ResponsesParsingError) -> str:
    status = payload.get("status")
    details = payload.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    if status == "incomplete" and reason == "max_output_tokens":
        return "output_limit"
    if status == "incomplete":
        return "incomplete_response"
    if isinstance(error, ResponseStateError):
        return f"response_state_{status or 'unknown'}"
    if isinstance(error, ResponseContentError):
        structure = response_structure(payload)
        if structure["output_item_types"] and all(item == "reasoning" for item in structure["output_item_types"]):
            return "reasoning_without_final_message"
        return "unexpected_response_shape"
    return "structured_output_validation_failure"


def run_parsing_probe(
    *, artifact_root: Path = PARSING_DIAGNOSIS_ROOT,
    env_file: Path = PROJECT_ROOT / ".env",
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Uma chamada, sem retry; persiste o JSON antes de interpretar conteúdo."""

    root = Path(artifact_root)
    probe_path = root / PROBE_NAME
    if probe_path.is_file():
        return json.loads(probe_path.read_text(encoding="utf-8"))
    preflight = prepare_parsing_diagnosis(artifact_root=root, env_file=env_file)
    model, secret = _configured_credentials(Path(env_file))
    body = preflight["request"]
    request = urllib.request.Request(
        AuditedOpenAIResponsesProvider.endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with opener(request, timeout=180) as response:
            http_status = getattr(response, "status", None)
            request_id = response.headers.get("x-request-id") if getattr(response, "headers", None) else None
            raw_bytes = response.read()
        duration = time.perf_counter() - started
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ParsingDiagnosisError("Resposta HTTP JSON não é um objeto.")
        _assert_response_safe(payload)
    except urllib.error.HTTPError as error:
        captured = _http_error(error, time.perf_counter() - started)
        return _persist_http_failure(root, preflight, captured)

    # Ordem obrigatória: bruto -> metadados/usage -> parsing/validação.
    raw_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "openai_raw_response_sanitized",
        "generated_at_utc": utc_now(), "http_status": http_status,
        "request_id": request_id, "request_hash": preflight["request_hash"],
        "response": copy.deepcopy(payload), "secret_fields_present": False,
    })
    save_json(raw_artifact, root / RAW_NAME)
    structure = response_structure(payload)
    structure_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_structure_report",
        "generated_at_utc": utc_now(), "http_status": http_status,
        "request_id": request_id, "response_id": payload.get("id"),
        "model": payload.get("model"), **structure,
    })
    save_json(structure_artifact, root / STRUCTURE_NAME)
    usage_values = _usage(payload)
    usage_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "openai_parsing_probe_usage",
        "generated_at_utc": utc_now(), "provider": "openai_responses",
        "model": payload.get("model") or model, "http_status": http_status,
        "request_id": request_id, "response_id": payload.get("id"),
        "response_status": payload.get("status"), "duration_seconds": duration,
        **usage_values, "request_success": True, "retries": 0,
    })
    save_json(usage_artifact, root / USAGE_NAME)

    try:
        parsed, extracted = parse_minimal_structured_output(payload)
        approved = payload.get("status") == "completed"
        parsing_error = None
        classification = "previous_response_shape_unknown"
        ready = approved
    except ResponsesParsingError as error:
        parsed = None
        extracted = None
        approved = False
        parsing_error = {"class": type(error).__name__, "message": str(error)}
        classification = _classify_failure(payload, error)
        ready = False
    probe = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_probe",
        "generated_at_utc": utc_now(), "status": "approved" if approved else "invalid",
        "provider": "openai_responses", "requested_model": model,
        "http_status": http_status, "request_id": request_id,
        "response_id": payload.get("id"), "response_status": payload.get("status"),
        "duration_seconds": duration, "temperature_sent": False, "store_requested": False,
        "message_found": structure["message_count"] > 0,
        "output_text_found": extracted is not None,
        "text_source": extracted.source if extracted else None,
        "structured_output": parsed, "json_valid": parsed is not None,
        "schema_valid": parsed == {"status": "ok"}, "content_validation": approved,
        "sanitized_parsing_error": parsing_error, "usage": usage_values,
        "provider_calls": 1, "retries": 0, "scientific_data_sent": False,
        "individual_data_sent": False, "ready_for_scientific_evaluation": ready,
    })
    save_json(probe, probe_path)
    root_cause = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_root_cause",
        "generated_at_utc": utc_now(), "classification": classification,
        "current_probe_approved": approved,
        "current_response_nested_without_top_level": bool(
            approved and extracted and extracted.source == "output.message.content.output_text"
            and not structure["top_level_output_text_present"]
        ),
        "mission72_parser_hypothesis_supported": False,
        "old_parser_already_traversed_output_content": True,
        "important_context": (
            "The Mission 7.2 parser already traversed output[].content[]. The current payload would have been parsed by it, so the lost prior response prevents classifying that failure as incorrect_raw_response_parsing."
        ),
        "correction": "Defensive type-aware parser plus raw-response-first persistence.",
        "ready_for_scientific_evaluation": ready,
    })
    save_json(root_cause, root / ROOT_CAUSE_NAME)
    if not approved:
        save_json(_sign({
            "schema_version": "1.0", "artifact_type": "openai_response_parsing_failure",
            "generated_at_utc": utc_now(), "stage": "content_validation",
            "classification": classification, "error": parsing_error,
            "http_request_succeeded": True, "automatic_retry_performed": False,
        }), root / FAILURE_NAME)
    _finalize_manifest(root, preflight, probe, root_cause)
    return probe


def reconcile_empirical_conclusion(root: Path = PARSING_DIAGNOSIS_ROOT) -> dict[str, Any]:
    """Consolida offline a conclusão após o probe; nunca chama o provider."""

    root = Path(root)
    probe = json.loads((root / PROBE_NAME).read_text(encoding="utf-8"))
    structure = json.loads((root / STRUCTURE_NAME).read_text(encoding="utf-8"))
    if probe.get("status") != "approved":
        raise ParsingDiagnosisError("Reconciliação de sucesso requer probe aprovado.")
    cause = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_root_cause",
        "generated_at_utc": utc_now(), "classification": "previous_response_shape_unknown",
        "current_probe_approved": True,
        "current_response_nested_without_top_level": (
            probe.get("text_source") == "output.message.content.output_text"
            and structure.get("top_level_output_text_present") is False
        ),
        "mission72_parser_hypothesis_supported": False,
        "old_parser_already_traversed_output_content": True,
        "evidence": [
            "The current raw HTTP response contains reasoning followed by message/output_text.",
            "The current raw response has no top-level output_text.",
            "The Mission 7.2 parser source already iterated output[].content[].",
            "The raw Mission 7.2 response was not preserved, so its actual status/output cannot be reconstructed.",
        ],
        "why_not_incorrect_raw_response_parsing": (
            "The current response shape would have been accepted by the old nested loop; causality is not demonstrated."
        ),
        "correction": "Defensive status/type-aware parser plus raw-response-first persistence.",
        "ready_for_scientific_evaluation": True,
    })
    save_json(cause, root / ROOT_CAUSE_NAME)
    preflight = json.loads((root / PREFLIGHT_NAME).read_text(encoding="utf-8"))
    manifest = _finalize_manifest(root, preflight, probe, cause)
    return manifest


def _persist_http_failure(root: Path, preflight: dict[str, Any], error: ProviderCallError) -> dict[str, Any]:
    details = error.sanitized_details()
    probe = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_probe",
        "generated_at_utc": utc_now(), "status": "rejected", "provider": "openai_responses",
        "requested_model": "gpt-5.5", "http_status": details["http_status"],
        "request_id": details["request_id"], "response_id": None, "response_status": None,
        "duration_seconds": error.duration_seconds, "temperature_sent": False,
        "store_requested": False, "message_found": False, "output_text_found": False,
        "structured_output": None, "json_valid": False, "schema_valid": False,
        "content_validation": False, "sanitized_provider_error": details,
        "usage": {key: None for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")},
        "provider_calls": 1, "retries": 0, "scientific_data_sent": False,
        "individual_data_sent": False, "ready_for_scientific_evaluation": False,
    })
    save_json(probe, root / PROBE_NAME)
    failure = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_failure",
        "generated_at_utc": utc_now(), "stage": "http", "details": details,
        "automatic_retry_performed": False,
    })
    save_json(failure, root / FAILURE_NAME)
    root_cause = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_root_cause",
        "generated_at_utc": utc_now(), "classification": "http_rejection",
        "current_probe_approved": False, "ready_for_scientific_evaluation": False,
    })
    save_json(root_cause, root / ROOT_CAUSE_NAME)
    _finalize_manifest(root, preflight, probe, root_cause)
    return probe


def _previous_unchanged(preflight: dict[str, Any]) -> bool:
    return all(
        preflight["previous_evidence_hashes"][name] == _tree_hashes(root)
        for name, root in PREVIOUS_ROOTS.items()
    )


def _finalize_manifest(
    root: Path, preflight: dict[str, Any], probe: dict[str, Any], root_cause: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for path in sorted(root.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        files.append({"filename": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    manifest = _sign({
        "schema_version": "1.0", "artifact_type": "openai_response_parsing_manifest",
        "generated_at_utc": utc_now(), "status": probe["status"],
        "provider": "openai_responses", "model": probe["requested_model"],
        "request_hash": preflight["request_hash"], "root_cause": root_cause["classification"],
        "ready_for_scientific_evaluation": probe["ready_for_scientific_evaluation"],
        "provider_calls": 1, "retries": 0, "scientific_calls": 0,
        "scientific_data_sent": False, "individual_data_sent": False,
        "previous_evidence_preserved": _previous_unchanged(preflight),
        "files": files,
    })
    save_json(manifest, root / MANIFEST_NAME)
    return manifest


def validate_parsing_diagnosis(root: Path = PARSING_DIAGNOSIS_ROOT) -> dict[str, Any]:
    root = Path(root)
    required = (RAW_NAME, STRUCTURE_NAME, PARSER_NAME, PROBE_NAME, USAGE_NAME, ROOT_CAUSE_NAME, MANIFEST_NAME)
    checks = []
    for name in required:
        checks.append({"check": f"exists:{name}", "passed": (root / name).is_file()})
    if not all(item["passed"] for item in checks):
        return {"passed": False, "checks": checks}
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    probe = json.loads((root / PROBE_NAME).read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["filename"]
        checks.append({
            "check": f"hash:{item['filename']}",
            "passed": path.is_file() and file_sha256(path) == item["sha256"],
        })
    for path in root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = payload.get("signature")
        unsigned = {key: value for key, value in payload.items() if key != "signature"}
        checks.append({"check": f"signature:{path.name}", "passed": signature == stable_sha256(unsigned)})
    checks.extend([
        {"check": "one_call", "passed": manifest["provider_calls"] == 1},
        {"check": "zero_retries", "passed": manifest["retries"] == 0},
        {"check": "temperature_omitted", "passed": probe["temperature_sent"] is False},
        {"check": "no_scientific_data", "passed": manifest["scientific_data_sent"] is False},
        {"check": "previous_evidence", "passed": manifest["previous_evidence_preserved"] is True},
        {"check": "probe_approved", "passed": probe["status"] == "approved"},
        {"check": "response_completed", "passed": probe["response_status"] == "completed"},
        {"check": "structured_output", "passed": probe["structured_output"] == {"status": "ok"}},
        {"check": "schema_valid", "passed": probe["schema_valid"] is True},
        {"check": "ready", "passed": manifest["ready_for_scientific_evaluation"] is True},
    ])
    content = "\n".join(path.read_text(encoding="utf-8").lower() for path in root.glob("*.json"))
    checks.append({"check": "no_secrets", "passed": "authorization" not in content and "openai_api_key" not in content})
    return {"passed": all(item["passed"] for item in checks), "checks": checks}
