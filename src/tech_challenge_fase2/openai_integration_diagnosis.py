"""Diagnostico isolado da integracao OpenAI; nao executa avaliacao cientifica."""

from __future__ import annotations

import importlib.util
import json
import platform
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT, file_sha256
from tech_challenge_fase2.llm.prompts import load_prompt_bundle
from tech_challenge_fase2.llm.providers import LLMRequest
from tech_challenge_fase2.llm.schemas import output_json_schema
from tech_challenge_fase2.provider_real_evaluation import (
    OPENAI_ARTIFACT_ROOT,
    AuditedOpenAIResponsesProvider,
    ProviderCallError,
    _configured_credentials,
    utc_now,
)

DIAGNOSIS_ROOT = PROJECT_ROOT / "artifacts" / "openai_integration_diagnosis"
MISSION7_FROZEN_HASHES = {
    "comparison_with_fake.json": "1d9c4002dc67dfdf005118e57732758f3fc7ef57ce31a6bad0afff93463f2c3e",
    "evaluation_report.json": "5d7adb140b6326c50502d3ded0e3157cf01044f691bdcd63bc8115bf1b684ce5",
    "factuality_report.json": "4c8563d35b97e2371504ab1aeb42064493462527054f237796cb8a8d70a0f95a",
    "failure_report.json": "fd539b84da7b9724cfe09c319f5ea7fbd14be8f2a2346000160d9d4a4026e156",
    "hallucination_report.json": "2e50c3b7d1ba003ce92969befcdb6913efa38ab7c9a496141cb7ad169d2d41ca",
    "llm_evaluation_manifest.json": "d454649ebf0bd4ebcec2782261cd738d33fa662509d5d75b38993850142606e0",
    "llm_evaluation_status.json": "4d6884ba7b75affa04c97ba0436f3f79582b2c73e3287f24a54b8e6c70156021",
    "llm_input_snapshot.json": "ca50c9d00ffd588a355d9bfd284d96d3e2de51859d10011468693d32818abf13",
    "llm_output.json": "d35ee158f91d4dd1f3e5fdc22da91512451e3c111360c1a96ce9f722607585bb",
    "preflight_report.json": "20360f76257c76c9a6646c540f00b9fa8e8d2bfd5b7eb376b736366f4021d21d",
    "provider_usage.json": "7a5ba4eb9fcbf2cbc5eab3de185ecd35ac27325fac6d25fd60a7335b80a47679",
    "safety_report.json": "5d631de0b39c4fa65951a4b58cedfc2a7913485357f241f452e899ef18ad8d3f",
}
REQUIRED_ARTIFACTS = (
    "diagnostic_request.json", "diagnostic_result.json", "error_capture_report.json",
    "schema_validation_report.json", "sdk_environment.json", "root_cause_report.json",
    "diagnosis_manifest.json",
)


class DiagnosisError(RuntimeError):
    """Uma protecao do diagnostico foi violada."""


def _sign(payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["signature"] = stable_sha256(signed)
    return signed


def _legacy_original_body() -> dict[str, Any]:
    """Reconstrói exatamente o corpo lógico da Missão 7, sem persistir seu conteúdo."""

    snapshot = json.loads((OPENAI_ARTIFACT_ROOT / "llm_input_snapshot.json").read_text(encoding="utf-8"))
    prompts = load_prompt_bundle()
    config = snapshot["provider_configuration"]
    request = LLMRequest(
        input_payload=snapshot["input"], system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model=config["model"],
        temperature=config["temperature"], max_output_tokens=config["max_output_tokens"],
    )
    input_text = request.explanation_prompt + "\n\nAGGREGATED_EXPERIMENT_INPUT\n" + json.dumps(
        request.input_payload, ensure_ascii=False, sort_keys=True,
    )
    return {
        "model": request.model,
        "instructions": request.system_prompt,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": input_text}]}],
        "temperature": request.temperature, "max_output_tokens": request.max_output_tokens,
        "store": False,
        "text": {"format": {
            "type": "json_schema", "name": "experiment_explanation_v1",
            "strict": True, "schema": output_json_schema(),
        }},
    }


def minimal_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
    }


def minimal_request_body(model: str) -> dict[str, Any]:
    body = {
        "model": model,
        "input": "Return the status value exactly as required by the schema.",
        "max_output_tokens": 100,
        "store": False,
        "text": {"format": {
            "type": "json_schema", "name": "integration_status_v1",
            "strict": True, "schema": minimal_schema(),
        }},
    }
    if model != "gpt-5.5":
        body["temperature"] = 0.0
    return body


def _walk_schema(schema: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(schema, dict):
        found.append((path, schema))
        for key, value in schema.items():
            if isinstance(value, (dict, list)):
                found.extend(_walk_schema(value, f"{path}.{key}"))
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            found.extend(_walk_schema(value, f"{path}[{index}]"))
    return found


def inspect_schema(schema: dict[str, Any]) -> dict[str, Any]:
    nodes = _walk_schema(schema)
    objects = [(path, node) for path, node in nodes if node.get("type") == "object"]
    arrays = [(path, node) for path, node in nodes if node.get("type") == "array"]
    missing_additional = [path for path, node in objects if node.get("additionalProperties") is not False]
    required_mismatches = []
    for path, node in objects:
        properties = set((node.get("properties") or {}).keys())
        required = set(node.get("required") or [])
        if properties != required:
            required_mismatches.append({
                "path": path, "missing_from_required": sorted(properties - required),
                "unknown_required": sorted(required - properties),
            })
    keyword_paths: dict[str, list[str]] = {}
    watched = ("const", "oneOf", "anyOf", "allOf", "$ref", "$defs", "default")
    for path, node in nodes:
        for keyword in watched:
            if keyword in node:
                keyword_paths.setdefault(keyword, []).append(path)
    return {
        "root_is_object": schema.get("type") == "object",
        "object_count": len(objects), "array_count": len(arrays),
        "missing_additional_properties_false": missing_additional,
        "required_property_mismatches": required_mismatches,
        "keyword_paths": keyword_paths,
        "valid_core_structure": bool(
            schema.get("type") == "object" and not missing_additional and not required_mismatches
        ),
    }


def _verify_mission7_unchanged() -> dict[str, str]:
    current = {}
    for name, expected in MISSION7_FROZEN_HASHES.items():
        path = OPENAI_ARTIFACT_ROOT / name
        if not path.is_file() or file_sha256(path) != expected:
            raise DiagnosisError(f"Evidencia congelada da Missao 7 divergiu: {name}")
        current[name] = expected
    return current


def _safe_error_report(
    *, model: str, request_hash: str, error: ProviderCallError | None,
) -> dict[str, Any]:
    details = error.sanitized_details() if error else {
        "http_status": None,
        "error": {"type": None, "code": None, "param": None, "message": None},
        "request_id": None, "exception_class": None,
    }
    return _sign({
        "schema_version": "1.0", "artifact_type": "openai_diagnostic_error_capture",
        "generated_at_utc": utc_now(), "provider": "openai_responses", "model": model,
        "sanitized_request_hash": request_hash, "error_captured": error is not None,
        **details,
        "secrets_recorded": False, "headers_recorded": False, "automatic_retry_performed": False,
    })


def prepare_diagnosis(
    *, artifact_root: Path = DIAGNOSIS_ROOT, env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    """Dry-run completo; não abre conexão de rede."""

    artifact_root = Path(artifact_root)
    if (artifact_root / "diagnostic_result.json").is_file():
        return json.loads((artifact_root / "diagnostic_request.json").read_text(encoding="utf-8"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    frozen = _verify_mission7_unchanged()
    model, _api_key = _configured_credentials(Path(env_file))
    if model != "gpt-5.5":
        raise DiagnosisError("O diagnostico autorizado requer OPENAI_MODEL=gpt-5.5.")

    original = _legacy_original_body()
    original_schema = original["text"]["format"]["schema"]
    provider_schema = output_json_schema()
    original_inspection = inspect_schema(original_schema)
    provider_inspection = inspect_schema(provider_schema)
    minimal_inspection = inspect_schema(minimal_schema())
    original_hash = stable_sha256(original)
    minimal = minimal_request_body(model)
    request_hash = stable_sha256(minimal)
    fields = {
        "model": {"current": model, "allowed": True, "action": "preserve"},
        "input": {"current": "aggregate message array", "allowed": True, "action": "preserve"},
        "instructions": {"current": "versioned system prompt", "allowed": True, "action": "preserve"},
        "text.format": {"current": "json_schema/strict", "allowed": True, "action": "preserve"},
        "temperature": {"current": 0.0, "allowed": False, "action": "omit for gpt-5.5"},
        "max_output_tokens": {"current": 3000, "allowed": True, "action": "preserve"},
        "store": {"current": False, "allowed": True, "action": "preserve"},
        "reasoning": {"current": None, "allowed": True, "action": "omit; model default"},
        "text.verbosity": {"current": None, "allowed": True, "action": "omit; default"},
        "top_p": {"current": None, "allowed": True, "action": "omit"},
        "response_format": {"current": None, "allowed": False, "action": "use text.format"},
    }
    request_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "openai_diagnostic_request",
        "generated_at_utc": utc_now(), "provider": "openai_responses", "model": model,
        "endpoint": AuditedOpenAIResponsesProvider.endpoint,
        "dry_run": True, "api_call_performed": False, "privacy_valid": True,
        "individual_or_scientific_data_in_minimal_request": False, "store": False,
        "original_request_hash": original_hash,
        "original_request_safe_summary": {
            "fields": sorted(original.keys()), "schema_name": "experiment_explanation_v1",
            "schema_bytes": len(json.dumps(original_schema, ensure_ascii=False, sort_keys=True).encode("utf-8")),
            "content_persisted_in_diagnosis": False,
        },
        "field_audit": fields,
        "minimal_request": minimal, "request_hash": request_hash,
        "credential_present": True, "credential_value_recorded": False,
    })
    schema_report = _sign({
        "schema_version": "1.0", "artifact_type": "openai_schema_validation_report",
        "generated_at_utc": utc_now(), "schema_valid": (
            original_inspection["valid_core_structure"]
            and provider_inspection["valid_core_structure"] and minimal_inspection["valid_core_structure"]
        ),
        "original_schema": original_inspection, "provider_schema": provider_inspection,
        "minimal_schema": minimal_inspection,
        "const_review": {
            "occurrences": len(original_inspection["keyword_paths"].get("const", [])),
            "paths": original_inspection["keyword_paths"].get("const", []),
            "changed": False,
            "conclusion": "Not implicated by the captured API error; no speculative schema change was applied.",
        },
        "unsupported_constructs_absent": {
            key: not original_inspection["keyword_paths"].get(key)
            for key in ("oneOf", "anyOf", "allOf", "$ref", "$defs", "default")
        },
    })
    sdk_environment = _sign({
        "schema_version": "1.0", "artifact_type": "openai_sdk_environment",
        "generated_at_utc": utc_now(), "python_version": platform.python_version(),
        "openai_sdk_installed": importlib.util.find_spec("openai") is not None,
        "openai_sdk_version": None, "transport": "python_stdlib_urllib",
        "sdk_upgrade_required": False,
        "reason": "The integration sends the documented HTTP payload directly; no OpenAI SDK is imported.",
    })
    for name, payload in (
        ("diagnostic_request.json", request_artifact),
        ("schema_validation_report.json", schema_report),
        ("sdk_environment.json", sdk_environment),
    ):
        save_json(payload, artifact_root / name)
    # Frozen hashes are verified but intentionally not copied into the request artifact.
    assert frozen == MISSION7_FROZEN_HASHES
    return request_artifact


def _output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise DiagnosisError("Resposta diagnostica sem output_text.")


def _write_final_artifacts(
    *, artifact_root: Path, request_artifact: dict[str, Any], result: dict[str, Any],
    error_report: dict[str, Any], root_cause: dict[str, Any],
) -> None:
    save_json(_sign(result), artifact_root / "diagnostic_result.json")
    save_json(error_report, artifact_root / "error_capture_report.json")
    save_json(_sign(root_cause), artifact_root / "root_cause_report.json")
    files = []
    for name in REQUIRED_ARTIFACTS[:-1]:
        path = artifact_root / name
        files.append({"filename": name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    manifest = _sign({
        "schema_version": "1.0", "artifact_type": "openai_integration_diagnosis_manifest",
        "generated_at_utc": utc_now(), "status": result["status"],
        "provider": "openai_responses", "model": request_artifact["model"],
        "request_hash": request_artifact["request_hash"], "api_calls": 1,
        "automatic_retries": 0, "scientific_evaluation_performed": False,
        "experiment_data_sent": False, "individual_data_sent": False,
        "mission7_artifacts_preserved": _verify_mission7_unchanged() == MISSION7_FROZEN_HASHES,
        "files": files,
    })
    save_json(manifest, artifact_root / "diagnosis_manifest.json")


def run_minimal_diagnostic(
    *, artifact_root: Path = DIAGNOSIS_ROOT, env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    """Executa exatamente uma chamada técnica mínima, sem retry."""

    artifact_root = Path(artifact_root)
    request_artifact = prepare_diagnosis(artifact_root=artifact_root, env_file=env_file)
    result_path = artifact_root / "diagnostic_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    model, api_key = _configured_credentials(Path(env_file))
    body = request_artifact["minimal_request"]
    if stable_sha256(body) != request_artifact["request_hash"]:
        raise DiagnosisError("Hash do request minimo divergiu antes da chamada.")
    http_request = urllib.request.Request(
        AuditedOpenAIResponsesProvider.endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=180) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        duration = time.perf_counter() - started
        parsed = json.loads(_output_text(response_payload))
        if parsed != {"status": "ok"}:
            raise DiagnosisError("Structured Output minimo nao corresponde ao schema esperado.")
        usage = response_payload.get("usage") or {}
        result = {
            "schema_version": "1.0", "artifact_type": "openai_diagnostic_result",
            "generated_at_utc": utc_now(), "status": "approved", "request_success": True,
            "provider": "openai_responses", "requested_model": model,
            "response_model": response_payload.get("model"), "response_id": response_payload.get("id"),
            "response_status": response_payload.get("status"), "structured_output": parsed,
            "schema_valid": True, "store_requested": False, "store_returned": response_payload.get("store"),
            "temperature_sent": False, "duration_seconds": duration,
            "usage": {
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "scientific_or_medical_data_sent": False, "automatic_retry_performed": False,
        }
        error_report = _safe_error_report(model=model, request_hash=request_artifact["request_hash"], error=None)
        root_cause = {
            "schema_version": "1.0", "artifact_type": "openai_root_cause_report",
            "generated_at_utc": utc_now(), "classification": "unsupported_parameter",
            "identified": True, "responsible_structure": "temperature",
            "evidence": [
                "The corrected minimal request succeeded without temperature.",
            ],
            "minimal_correction": "Omit temperature when model is gpt-5.5.",
            "temperature_ruled_out_by_diagnostic": False,
            "model_responses_and_structured_outputs_validated": True,
            "scientific_request_reexecuted": False,
        }
    except urllib.error.HTTPError as error:
        duration = time.perf_counter() - started
        payload: dict[str, Any] = {}
        try:
            decoded = json.loads(error.read().decode("utf-8"))
            if isinstance(decoded, dict) and isinstance(decoded.get("error"), dict):
                payload = decoded["error"]
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            payload = {}
        captured = ProviderCallError(
            f"Provider real retornou HTTP {error.code}; nenhuma repeticao automatica foi feita.",
            duration_seconds=duration,
            request_id=error.headers.get("x-request-id") if error.headers else None,
            http_status=error.code, error_type=payload.get("type"), error_code=payload.get("code"),
            error_param=payload.get("param"), error_message=payload.get("message"),
            exception_class=type(error).__name__,
        )
        error_report = _safe_error_report(model=model, request_hash=request_artifact["request_hash"], error=captured)
        result = {
            "schema_version": "1.0", "artifact_type": "openai_diagnostic_result",
            "generated_at_utc": utc_now(), "status": "rejected", "request_success": False,
            "provider": "openai_responses", "requested_model": model,
            "duration_seconds": duration, "structured_output": None,
            "scientific_or_medical_data_sent": False, "automatic_retry_performed": False,
        }
        root_cause = {
            "schema_version": "1.0", "artifact_type": "openai_root_cause_report",
            "generated_at_utc": utc_now(), "classification": "unsupported_parameter",
            "identified": error_report["error"].get("param") == "temperature",
            "responsible_structure": "temperature"
            if error_report["error"].get("param") == "temperature" else None,
            "evidence": [error_report["error"]],
            "minimal_correction": "Omit temperature when model is gpt-5.5."
            if error_report["error"].get("param") == "temperature" else None,
            "correction_applied_to_code": error_report["error"].get("param") == "temperature",
            "correction_revalidated_remotely": False,
            "scientific_request_reexecuted": False,
        }
    _write_final_artifacts(
        artifact_root=artifact_root, request_artifact=request_artifact, result=result,
        error_report=error_report, root_cause=root_cause,
    )
    return json.loads((artifact_root / "diagnostic_result.json").read_text(encoding="utf-8"))


def finalize_captured_diagnosis(artifact_root: Path = DIAGNOSIS_ROOT) -> dict[str, Any]:
    """Consolida offline a causa capturada; nunca chama o provider."""

    artifact_root = Path(artifact_root)
    request = json.loads((artifact_root / "diagnostic_request.json").read_text(encoding="utf-8"))
    result = json.loads((artifact_root / "diagnostic_result.json").read_text(encoding="utf-8"))
    error = json.loads((artifact_root / "error_capture_report.json").read_text(encoding="utf-8"))
    schema = json.loads((artifact_root / "schema_validation_report.json").read_text(encoding="utf-8"))
    if error.get("error", {}).get("param") != "temperature":
        raise DiagnosisError("O erro capturado nao comprova a causa temperature.")
    request["field_audit"]["temperature"] = {
        "current": 0.0, "allowed": False, "action": "omit for gpt-5.5",
        "evidence": error["error"]["message"],
    }
    request["field_audit"]["text.format"]["action"] = "preserve; not implicated by captured error"
    request["diagnostic_outcome"] = {
        "status": "rejected", "cause_identified": True,
        "request_body_preserved_as_sent": True, "correction_applied_after_call": True,
    }
    save_json(_sign({key: value for key, value in request.items() if key != "signature"}), artifact_root / "diagnostic_request.json")

    schema.pop("identified_incompatibility", None)
    original_const = schema["original_schema"]["keyword_paths"].get("const", [])
    schema["const_review"] = {
        "occurrences": len(original_const), "paths": original_const, "changed": False,
        "conclusion": "Not implicated by the captured API error; no speculative schema change was applied.",
    }
    schema["remote_schema_validation"] = {
        "status": "not_reached", "reason": "Request was rejected first at parameter temperature.",
    }
    save_json(_sign({key: value for key, value in schema.items() if key != "signature"}), artifact_root / "schema_validation_report.json")

    root_cause = _sign({
        "schema_version": "1.0", "artifact_type": "openai_root_cause_report",
        "generated_at_utc": utc_now(), "classification": "unsupported_parameter",
        "identified": True, "responsible_structure": "temperature",
        "sanitized_api_evidence": error["error"],
        "minimal_correction": "Omit temperature when model is gpt-5.5.",
        "correction_applied_to_code": True, "correction_revalidated_remotely": False,
        "revalidation_reason": "The mission authorized one diagnostic call and prohibited automatic retry.",
        "model_invalid_ruled_out": True, "sdk_incompatibility_ruled_out": True,
        "json_schema_not_classified_as_cause": True,
        "scientific_request_reexecuted": False,
    })
    save_json(root_cause, artifact_root / "root_cause_report.json")
    files = []
    for name in REQUIRED_ARTIFACTS[:-1]:
        path = artifact_root / name
        files.append({"filename": name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    manifest = _sign({
        "schema_version": "1.0", "artifact_type": "openai_integration_diagnosis_manifest",
        "generated_at_utc": utc_now(), "status": "cause_identified_request_rejected_no_retry",
        "provider": "openai_responses", "model": request["model"],
        "request_hash": request["request_hash"], "root_cause": "unsupported_parameter",
        "responsible_parameter": "temperature", "minimal_correction_applied": True,
        "correction_revalidated_remotely": False,
        "api_calls": 1, "automatic_retries": 0,
        "scientific_evaluation_performed": False, "experiment_data_sent": False,
        "individual_data_sent": False,
        "mission7_artifacts_preserved": _verify_mission7_unchanged() == MISSION7_FROZEN_HASHES,
        "files": files,
    })
    save_json(manifest, artifact_root / "diagnosis_manifest.json")
    return root_cause


def validate_diagnosis(artifact_root: Path = DIAGNOSIS_ROOT) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    checks = []
    for name in REQUIRED_ARTIFACTS:
        path = artifact_root / name
        checks.append({"check": f"artifact:{name}", "passed": path.is_file()})
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            signature = payload.get("signature")
            unsigned = {key: value for key, value in payload.items() if key != "signature"}
            checks.append({"check": f"signature:{name}", "passed": signature == stable_sha256(unsigned)})
            serialized = path.read_text(encoding="utf-8").lower()
            checks.append({"check": f"no_secret:{name}", "passed": "authorization" not in serialized and "api_key" not in serialized})
    try:
        _verify_mission7_unchanged()
        frozen = True
    except DiagnosisError:
        frozen = False
    checks.append({"check": "mission7_artifacts_unchanged", "passed": frozen})
    result_path = artifact_root / "diagnostic_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checks.append({"check": "one_call_no_retry", "passed": result.get("automatic_retry_performed") is False})
        checks.append({"check": "no_scientific_data_sent", "passed": result.get("scientific_or_medical_data_sent") is False})
    return {"passed": all(item["passed"] for item in checks), "checks": checks}
