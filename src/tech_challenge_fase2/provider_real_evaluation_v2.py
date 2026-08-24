"""Missao 7.2: validacao tecnica corrigida e avaliacao real com gates."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT, file_sha256
from tech_challenge_fase2.llm.privacy import validate_sanitized_input
from tech_challenge_fase2.llm.prompts import load_prompt_bundle
from tech_challenge_fase2.llm.providers import LLMRequest
from tech_challenge_fase2.openai_integration_diagnosis import minimal_request_body
from tech_challenge_fase2.provider_real_evaluation import (
    ADVERSARIAL_NAME,
    EVALUATION_NAME,
    FACTUALITY_NAME,
    FAILURE_NAME,
    HALLUCINATION_NAME,
    INPUT_NAME,
    MANIFEST_NAME,
    OUTPUT_NAME,
    SAFETY_NAME,
    USAGE_NAME,
    AuditedOpenAIResponsesProvider,
    ProviderCallError,
    ProviderRealEvaluationError,
    _configured_credentials,
    _ensure_failure_placeholders,
    _load_signed,
    _sign,
    _write_manifest,
    run_openai_adversarial,
    run_openai_main,
    prepare_openai_evaluation,
    utc_now,
    validate_openai_evaluation,
)

V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v2"
MISSION7_ROOT = PROJECT_ROOT / "artifacts" / "llm_evaluation_openai"
DIAGNOSIS_ROOT = PROJECT_ROOT / "artifacts" / "openai_integration_diagnosis"
MISSION72_START_COMMIT = "1ed89fa4f3034a0f6f480d10c88c72f46aaa8525"
TECHNICAL_PROBE_NAME = "technical_probe.json"
PREFLIGHT_V2_NAME = "mission72_preflight.json"


class Mission72Error(RuntimeError):
    """Um gate da Missao 7.2 impediu a proxima etapa."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _safe_error(error: ProviderCallError) -> dict[str, Any]:
    return error.sanitized_details()


def _parse_http_error(error: urllib.error.HTTPError, duration: float) -> ProviderCallError:
    payload: dict[str, Any] = {}
    try:
        decoded = json.loads(error.read().decode("utf-8"))
        if isinstance(decoded, dict) and isinstance(decoded.get("error"), dict):
            payload = decoded["error"]
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        payload = {}
    return ProviderCallError(
        f"Provider real retornou HTTP {error.code}; nenhuma repeticao automatica foi feita.",
        duration_seconds=duration,
        request_id=error.headers.get("x-request-id") if error.headers else None,
        http_status=error.code, error_type=payload.get("type"), error_code=payload.get("code"),
        error_param=payload.get("param"), error_message=payload.get("message"),
        exception_class=type(error).__name__,
    )


def _extract_output_text(payload: dict[str, Any]) -> str:
    return AuditedOpenAIResponsesProvider._output_text(payload)


def prepare_v2(
    *, artifact_root: Path = V2_ARTIFACT_ROOT, env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    """Preflight local completo, sem chamada externa."""

    artifact_root = Path(artifact_root)
    preflight_path = artifact_root / PREFLIGHT_V2_NAME
    if preflight_path.is_file():
        return _load_signed(preflight_path, "openai_v2_preflight")
    model, _secret = _configured_credentials(Path(env_file))
    if model != "gpt-5.5":
        raise Mission72Error("OPENAI_MODEL deve ser gpt-5.5 nesta missao.")
    head = _git("rev-parse", "HEAD")
    if head != MISSION72_START_COMMIT:
        raise Mission72Error("HEAD divergiu do commit registrado no inicio da Missao 7.2.")
    ignored = bool(_git("check-ignore", ".env"))
    if not ignored:
        raise Mission72Error(".env nao esta ignorado pelo Git.")
    if not MISSION7_ROOT.is_dir() or not DIAGNOSIS_ROOT.is_dir():
        raise Mission72Error("Evidencias anteriores obrigatorias ausentes.")
    previous_hashes = {
        "mission7": _tree_hashes(MISSION7_ROOT),
        "mission71": _tree_hashes(DIAGNOSIS_ROOT),
    }
    snapshot = prepare_openai_evaluation(artifact_root=artifact_root, env_file=env_file)
    validate_sanitized_input(snapshot["input"])
    prompts = load_prompt_bundle()
    scientific_request = LLMRequest(
        input_payload=snapshot["input"], system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model=model,
        temperature=snapshot["provider_configuration"]["temperature"],
        max_output_tokens=snapshot["provider_configuration"]["max_output_tokens"],
    )
    scientific_body = AuditedOpenAIResponsesProvider.request_body(scientific_request)
    probe_body = minimal_request_body(model)
    if "temperature" in scientific_body or "temperature" in probe_body:
        raise Mission72Error("Preflight detectou temperature no request de gpt-5.5.")
    preflight = _sign({
        "schema_version": "1.0", "artifact_type": "openai_v2_preflight",
        "generated_at_utc": utc_now(), "passed": True,
        "repository": {
            "mission_start_clean": True, "mission_start_commit": MISSION72_START_COMMIT,
            "head_before_calls": head, "head_unchanged": head == MISSION72_START_COMMIT,
            "env_ignored": ignored, "automatic_commit_performed": False,
        },
        "configuration": {
            "provider": "openai_responses", "model": model, "store": False,
            "temperature_sent": False, "retry_count": 0,
            "credential_present": True, "credential_recorded": False,
        },
        "schema_valid": True, "privacy_valid": True,
        "individual_data_included": False, "final_predictions_included": False,
        "input_sha256": snapshot["input_sha256"],
        "prompt_versions": snapshot["prompt_versions"],
        "probe_request_hash": stable_sha256(probe_body),
        "scientific_request_hash": stable_sha256(scientific_body),
        "previous_evidence_hashes": previous_hashes,
        "api_calls_performed": 0,
    })
    save_json(preflight, preflight_path)
    return preflight


def run_technical_probe(
    *, artifact_root: Path = V2_ARTIFACT_ROOT, env_file: Path = PROJECT_ROOT / ".env",
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Executa uma unica chamada minima corrigida e bloqueia retries."""

    artifact_root = Path(artifact_root)
    path = artifact_root / TECHNICAL_PROBE_NAME
    if path.is_file():
        return _load_signed(path, "openai_v2_technical_probe")
    preflight = prepare_v2(artifact_root=artifact_root, env_file=env_file)
    model, secret = _configured_credentials(Path(env_file))
    body = minimal_request_body(model)
    if "temperature" in body:
        raise Mission72Error("Probe corrigido nao pode enviar temperature.")
    request = urllib.request.Request(
        AuditedOpenAIResponsesProvider.endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with opener(request, timeout=180) as response:
            http_status = getattr(response, "status", 200)
            response_payload = json.loads(response.read().decode("utf-8"))
        duration = time.perf_counter() - started
        usage = response_payload.get("usage") or {}
        try:
            parsed = json.loads(_extract_output_text(response_payload))
            schema_valid = parsed == {"status": "ok"}
            local_error = None
        except (ProviderRealEvaluationError, json.JSONDecodeError) as error:
            parsed = None
            schema_valid = False
            local_error = {"class": type(error).__name__, "message": str(error)}
        artifact = _sign({
            "schema_version": "1.0", "artifact_type": "openai_v2_technical_probe",
            "generated_at_utc": utc_now(), "status": "approved" if schema_valid else "invalid",
            "provider": "openai_responses", "requested_model": model,
            "response_model": response_payload.get("model"), "http_status": http_status,
            "request_id": response_payload.get("id"), "response_status": response_payload.get("status"),
            "duration_seconds": duration, "store_requested": False,
            "store_returned": response_payload.get("store"), "temperature_sent": False,
            "request_hash": preflight["probe_request_hash"], "schema_valid": schema_valid,
            "structured_output": parsed, "sanitized_local_error": local_error,
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "provider_calls": 1, "retries": 0, "scientific_data_sent": False,
            "medical_content_sent": False, "individual_data_sent": False,
            "secret_recorded": False,
        })
    except urllib.error.HTTPError as error:
        captured = _parse_http_error(error, time.perf_counter() - started)
        artifact = _sign({
            "schema_version": "1.0", "artifact_type": "openai_v2_technical_probe",
            "generated_at_utc": utc_now(), "status": "rejected", "provider": "openai_responses",
            "requested_model": model, "duration_seconds": captured.duration_seconds,
            "store_requested": False, "temperature_sent": False,
            "request_hash": preflight["probe_request_hash"], "schema_valid": False,
            "structured_output": None, "usage": {
                "input_tokens": None, "output_tokens": None, "total_tokens": None,
            },
            "provider_calls": 1, "retries": 0, "scientific_data_sent": False,
            "medical_content_sent": False, "individual_data_sent": False,
            "sanitized_error": _safe_error(captured), "secret_recorded": False,
        })
    save_json(artifact, path)
    if artifact["status"] != "approved":
        _write_blocked_artifacts(artifact_root, stage="technical_probe", reason="Probe tecnico corrigido falhou.")
    return artifact


def _write_blocked_artifacts(artifact_root: Path, *, stage: str, reason: str) -> None:
    failure_path = artifact_root / FAILURE_NAME
    if not failure_path.exists():
        snapshot = _load_signed(artifact_root / INPUT_NAME, "openai_llm_input_snapshot")
        save_json(_sign({
            "schema_version": "1.0", "artifact_type": "openai_failure_report",
            "generated_at_utc": utc_now(), "run_identity": snapshot["run_identity"],
            "stage": stage, "reason": reason, "severity": "mission72_blocked",
            "cause_classification": stage, "field_level_api_error_available": False,
            "sanitized_api_error": None, "automatic_retry_performed": False,
            "prompt_changed": False, "scientific_call_performed": stage != "technical_probe",
            "original_evidence_preserved": True,
        }), failure_path)
    adversarial_path = artifact_root / ADVERSARIAL_NAME
    if not adversarial_path.exists():
        save_json(_sign({
            "schema_version": "1.0", "artifact_type": "openai_adversarial_results",
            "generated_at_utc": utc_now(), "status": "not_run_blocked",
            "provider_calls": 0, "maximum_authorized_calls": 3,
            "all_inputs_aggregate": True, "individual_data_sent": False,
            "scenarios": [], "blocked_by": stage,
        }), adversarial_path)


def record_observed_probe_extraction_failure(
    artifact_root: Path = V2_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Preserva offline a falha observada nesta execução; não chama a API."""

    root = Path(artifact_root)
    probe_path = root / TECHNICAL_PROBE_NAME
    if probe_path.is_file():
        return _load_signed(probe_path, "openai_v2_technical_probe")
    preflight = _load_signed(root / PREFLIGHT_V2_NAME, "openai_v2_preflight")
    snapshot = _load_signed(root / INPUT_NAME, "openai_llm_input_snapshot")
    probe = _sign({
        "schema_version": "1.0", "artifact_type": "openai_v2_technical_probe",
        "generated_at_utc": utc_now(), "status": "invalid",
        "provider": "openai_responses", "requested_model": "gpt-5.5",
        "response_model": None, "http_status": None, "request_id": None,
        "response_status": None, "duration_seconds": None,
        "store_requested": False, "store_returned": None, "temperature_sent": False,
        "request_hash": preflight["probe_request_hash"], "schema_valid": False,
        "structured_output": None,
        "sanitized_local_error": {
            "class": "ProviderRealEvaluationError",
            "message": "Resposta real nao contem output_text.",
        },
        "transport_observation": {
            "http_error_raised": False, "json_response_body_received": True,
            "response_body_persisted": False,
            "note": "The original process ended before response metadata could be persisted; unavailable values remain null.",
        },
        "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        "provider_calls": 1, "retries": 0, "scientific_data_sent": False,
        "medical_content_sent": False, "individual_data_sent": False,
        "secret_recorded": False,
    })
    save_json(probe, probe_path)
    failure = _sign({
        "schema_version": "1.0", "artifact_type": "openai_failure_report",
        "generated_at_utc": utc_now(), "run_identity": snapshot["run_identity"],
        "stage": "technical_probe_output_extraction",
        "reason": "Resposta recebida sem output_text; probe nao aprovado.",
        "severity": "mission72_scientific_call_blocked",
        "cause_classification": "response_without_output_text",
        "field_level_api_error_available": False, "sanitized_api_error": None,
        "automatic_retry_performed": False, "prompt_changed": False,
        "original_evidence_preserved": True, "scientific_call_performed": False,
    })
    save_json(failure, root / FAILURE_NAME)
    usage = _sign({
        "schema_version": "1.0", "artifact_type": "openai_provider_usage",
        "generated_at_utc": utc_now(), "provider": "openai_responses", "model": "gpt-5.5",
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "duration_seconds": None, "request_success": None, "response_id": None,
        "store_requested": False, "cost_estimate": None,
        "scientific_call_performed": False,
        "reason": "Probe process ended before response metadata was persisted; null is not zero.",
    })
    save_json(usage, root / USAGE_NAME)
    _ensure_failure_placeholders(root, snapshot, failure=failure, usage=usage)
    _write_blocked_artifacts(root, stage="technical_probe", reason=failure["reason"])
    _write_manifest(
        root, snapshot, status="invalid", approved=False, usage=usage,
        adversarial={"status": "not_run_technical_probe_invalid", "provider_calls": 0, "scenarios": []},
    )
    save_json(_sign({
        "schema_version": "1.0", "artifact_type": "openai_llm_evaluation_status",
        "status": "technical_probe_invalid", "updated_at_utc": utc_now(),
        "run_identity": snapshot["run_identity"], "technical_provider_calls": 1,
        "main_provider_calls": 0, "adversarial_provider_calls": 0,
    }), root / "llm_evaluation_status.json")
    finalize_v2_manifest(root)
    return probe


def run_scientific_v2(
    *, artifact_root: Path = V2_ARTIFACT_ROOT, env_file: Path = PROJECT_ROOT / ".env",
    provider: Any | None = None,
) -> dict[str, Any]:
    probe = run_technical_probe(artifact_root=artifact_root, env_file=env_file) if not (Path(artifact_root) / TECHNICAL_PROBE_NAME).is_file() else _load_signed(Path(artifact_root) / TECHNICAL_PROBE_NAME, "openai_v2_technical_probe")
    if probe["status"] != "approved":
        raise Mission72Error("Avaliacao cientifica bloqueada: probe tecnico nao aprovado.")
    result = run_openai_main(artifact_root=artifact_root, env_file=env_file, provider=provider)
    if not result.get("approved"):
        _write_blocked_artifacts(Path(artifact_root), stage="scientific_main", reason="Validacao deterministica principal reprovou a resposta.")
    return result


def run_adversarial_v2(
    *, artifact_root: Path = V2_ARTIFACT_ROOT, env_file: Path = PROJECT_ROOT / ".env",
    provider: Any | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    output = _load_signed(root / OUTPUT_NAME, "openai_llm_output")
    if output.get("approved") is not True:
        raise Mission72Error("Cenarios adversariais bloqueados: chamada principal nao aprovada.")
    return run_openai_adversarial(artifact_root=root, env_file=env_file, provider=provider)


def finalize_v2_manifest(artifact_root: Path = V2_ARTIFACT_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    probe = _load_signed(root / TECHNICAL_PROBE_NAME, "openai_v2_technical_probe")
    if (root / MANIFEST_NAME).is_file():
        manifest = _load_signed(root / MANIFEST_NAME, "openai_llm_evaluation_manifest")
        unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    else:
        unsigned = {
            "schema_version": "1.0", "artifact_type": "openai_llm_evaluation_manifest",
            "generated_at_utc": utc_now(), "status": "technical_failure", "approved": False,
            "provider": "openai_responses", "model": probe["requested_model"],
            "files": [],
        }
    adversarial = json.loads((root / ADVERSARIAL_NAME).read_text(encoding="utf-8")) if (root / ADVERSARIAL_NAME).is_file() else None
    usage_record = _load_signed(root / USAGE_NAME, "openai_provider_usage") if (root / USAGE_NAME).is_file() else None
    main_calls = 1 if usage_record and usage_record.get("scientific_call_performed", True) is not False else 0
    adversarial_calls = adversarial.get("provider_calls", 0) if adversarial else 0
    total_calls = probe["provider_calls"] + main_calls + adversarial_calls
    if total_calls > 5:
        raise Mission72Error("Orcamento absoluto de cinco chamadas excedido.")
    unsigned.update({
        "mission_version": "7.2", "technical_probe": {
            "status": probe["status"], "request_id": probe.get("request_id"),
            "duration_seconds": probe["duration_seconds"], "usage": probe["usage"],
            "temperature_sent": probe["temperature_sent"], "retries": probe["retries"],
        },
        "call_budget": {
            "technical": probe["provider_calls"], "scientific_main": main_calls,
            "adversarial": adversarial_calls, "total": total_calls, "maximum": 5,
            "automatic_retries": 0,
        },
        "repository": {
            "mission_start_commit": MISSION72_START_COMMIT,
            "head_at_finalization": _git("rev-parse", "HEAD"),
            "automatic_commit_performed": False, "env_ignored": bool(_git("check-ignore", ".env")),
        },
        "previous_evidence_preserved": _previous_evidence_unchanged(root),
    })
    files = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {MANIFEST_NAME, "llm_evaluation_status.json"}:
            continue
        files.append({
            "filename": str(path.relative_to(root)), "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        })
    unsigned["files"] = files
    signed = _sign(unsigned)
    save_json(signed, root / MANIFEST_NAME)
    return signed


def _previous_evidence_unchanged(artifact_root: Path = V2_ARTIFACT_ROOT) -> bool:
    preflight = _load_signed(Path(artifact_root) / PREFLIGHT_V2_NAME, "openai_v2_preflight")
    return (
        preflight["previous_evidence_hashes"]["mission7"] == _tree_hashes(MISSION7_ROOT)
        and preflight["previous_evidence_hashes"]["mission71"] == _tree_hashes(DIAGNOSIS_ROOT)
    )


def validate_v2(artifact_root: Path = V2_ARTIFACT_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    required = (
        TECHNICAL_PROBE_NAME, INPUT_NAME, OUTPUT_NAME, USAGE_NAME, FACTUALITY_NAME,
        SAFETY_NAME, EVALUATION_NAME, HALLUCINATION_NAME, "comparison_with_fake.json",
        ADVERSARIAL_NAME, MANIFEST_NAME,
    )
    for name in required:
        add(f"exists:{name}", (root / name).is_file(), name)
    if not all((root / name).is_file() for name in required):
        return {"passed": False, "check_count": len(checks), "checks": checks}
    manifest = finalize_v2_manifest(root)
    probe = _load_signed(root / TECHNICAL_PROBE_NAME, "openai_v2_technical_probe")
    add("technical_probe", probe["status"] == "approved" and probe["schema_valid"] is True, probe["status"])
    add("temperature_omitted", probe["temperature_sent"] is False, probe["temperature_sent"])
    add("call_budget", manifest["call_budget"]["total"] <= 5, manifest["call_budget"])
    add("zero_retries", manifest["call_budget"]["automatic_retries"] == 0, manifest["call_budget"])
    add("previous_evidence", manifest["previous_evidence_preserved"] is True, manifest["previous_evidence_preserved"])
    generic = validate_openai_evaluation(root)
    add("generic_evaluation_integrity", generic["passed"], generic["status"])
    content = "\n".join(path.read_text(encoding="utf-8").lower() for path in root.rglob("*.json"))
    add("no_secret", "authorization" not in content and "openai_api_key" not in content, "safe")
    add("scope", all(not manifest["scope_confirmations"][key] for key in manifest["scope_confirmations"]), manifest["scope_confirmations"])
    return {"passed": all(item["passed"] for item in checks), "check_count": len(checks), "checks": checks}
