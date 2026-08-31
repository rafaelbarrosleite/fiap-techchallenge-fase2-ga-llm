"""Missao 7.3: avaliacao cientifica real com transporte raw-first.

O modulo reutiliza, sem alterar, o contrato, os prompts e os validadores da
Missao 5. A unica responsabilidade nova e preservar a resposta HTTP antes do
parsing e registrar as protecoes especificas desta missao.
"""

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
from tech_challenge_fase2.llm.privacy import validate_sanitized_input
from tech_challenge_fase2.llm.prompts import load_prompt_bundle
from tech_challenge_fase2.llm.providers import LLMRequest
from tech_challenge_fase2.llm.schemas import validate_input
from tech_challenge_fase2._historical.openai_response_parsing_diagnosis import (
    PARSING_DIAGNOSIS_ROOT,
    _assert_response_safe,
    _usage,
)
from tech_challenge_fase2._historical.provider_real_evaluation import (
    ADVERSARIAL_NAME,
    COMPARISON_NAME,
    EVALUATION_NAME,
    FACTUALITY_NAME,
    FAILURE_NAME,
    HALLUCINATION_NAME,
    INPUT_NAME,
    MAIN_MANIFEST_NAME,
    MANIFEST_NAME,
    OUTPUT_NAME,
    SAFETY_NAME,
    STATUS_NAME,
    USAGE_NAME,
    AuditedOpenAIResponsesProvider,
    ProviderCallError,
    ProviderRealEvaluationError,
    RawProviderResponse,
    _configured_credentials,
    _load_signed,
    _sign,
    prepare_openai_evaluation,
    run_openai_adversarial,
    run_openai_main,
    utc_now,
    validate_openai_evaluation,
)
from tech_challenge_fase2.responses_parsing import ResponsesParsingError, extract_response_text, response_structure

V3_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v3"
V3_START_COMMIT = "3bef0e4ac48c88f63c2bca91221956cca2abb2ed"
V3_PREFLIGHT_NAME = "mission73_preflight.json"
RAW_NAME = "raw_response_sanitized.json"

PREVIOUS_ROOTS = {
    "mission5_fake": PROJECT_ROOT / "artifacts" / "llm_evaluation",
    "mission7": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai",
    "mission71": PROJECT_ROOT / "artifacts" / "openai_integration_diagnosis",
    "mission72": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v2",
    "mission721": PARSING_DIAGNOSIS_ROOT,
}

EXPECTED_V3_PATHS = {
    "pyproject.toml",
    "README.md",
    "docs/historico/avaliacao_provider_real_v3.md",
    "docs/relatorio_final.md",
    "docs/resumo_executivo.md",
    "docs/matriz_rastreabilidade_final.md",
    "src/tech_challenge_fase2/provider_real_evaluation_v3.py",
    "src/tech_challenge_fase2/provider_real_evaluation_v2.py",
    "src/tech_challenge_fase2/run_provider_real_evaluation_v3.py",
    "tests/test_provider_real_evaluation_v3.py",
}

REQUIRED_ARTIFACTS = (
    RAW_NAME,
    INPUT_NAME,
    OUTPUT_NAME,
    USAGE_NAME,
    FACTUALITY_NAME,
    SAFETY_NAME,
    EVALUATION_NAME,
    HALLUCINATION_NAME,
    COMPARISON_NAME,
    ADVERSARIAL_NAME,
    MANIFEST_NAME,
)


class Mission73Error(RuntimeError):
    """Uma protecao local da Missao 7.3 bloqueou a chamada ou a auditoria."""


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _worktree_paths() -> set[str]:
    paths: set[str] = set()
    for line in _git("status", "--short").splitlines():
        if not line:
            continue
        # ``_git`` remove o espaco inicial da primeira linha; separar uma vez
        # preserva o caminho tanto para ``M`` quanto para ``??``.
        parts = line.split(maxsplit=1)
        path = parts[-1]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def _previous_evidence_hashes() -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for name, root in PREVIOUS_ROOTS.items():
        if not root.is_dir():
            raise Mission73Error(f"Evidencia anterior ausente: {name}.")
        hashes[name] = _tree_hashes(root)
    return hashes


def _previous_evidence_unchanged(preflight: dict[str, Any]) -> bool:
    return all(
        preflight["previous_evidence_hashes"][name] == _tree_hashes(root)
        for name, root in PREVIOUS_ROOTS.items()
    )


def _signed_json(path: Path, artifact_type: str) -> dict[str, Any]:
    return _load_signed(path, artifact_type)


def prepare_v3(
    *, artifact_root: Path = V3_ARTIFACT_ROOT,
    env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    """Executa todas as precondicoes locais; nunca chama o provider."""

    root = Path(artifact_root)
    preflight_path = root / V3_PREFLIGHT_NAME
    if preflight_path.is_file():
        return _signed_json(preflight_path, "openai_v3_preflight")
    if root.exists() and any(root.iterdir()):
        raise Mission73Error("Diretorio V3 parcial encontrado; revisao manual obrigatoria.")

    model, _secret = _configured_credentials(Path(env_file))
    if model != "gpt-5.5":
        raise Mission73Error("OPENAI_MODEL deve ser gpt-5.5.")
    head = _git("rev-parse", "HEAD")
    official_root = root.resolve() == V3_ARTIFACT_ROOT.resolve()
    if official_root and head != V3_START_COMMIT:
        raise Mission73Error("HEAD divergiu do commit limpo registrado no inicio da Missao 7.3.")
    if not _git("check-ignore", ".env"):
        raise Mission73Error(".env nao esta ignorado pelo Git.")
    unexpected_paths = sorted(_worktree_paths().difference(EXPECTED_V3_PATHS)) if official_root else []
    if unexpected_paths:
        raise Mission73Error(f"Alteracoes inesperadas antes da chamada: {unexpected_paths}")

    parsing_manifest = _signed_json(
        PARSING_DIAGNOSIS_ROOT / "response_parsing_manifest.json",
        "openai_response_parsing_manifest",
    )
    if not (
        parsing_manifest.get("status") == "approved"
        and parsing_manifest.get("ready_for_scientific_evaluation") is True
        and parsing_manifest.get("provider") == "openai_responses"
        and parsing_manifest.get("model") == "gpt-5.5"
        and parsing_manifest.get("previous_evidence_preserved") is True
    ):
        raise Mission73Error("A integracao tecnica da Missao 7.2.1 nao esta aprovada.")

    previous_hashes = _previous_evidence_hashes()
    snapshot = prepare_openai_evaluation(artifact_root=root, env_file=env_file)
    validate_input(snapshot["input"])
    validate_sanitized_input(snapshot["input"])
    prompts = load_prompt_bundle()
    request = LLMRequest(
        input_payload=snapshot["input"], system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model=model,
        temperature=snapshot["provider_configuration"]["temperature"],
        max_output_tokens=snapshot["provider_configuration"]["max_output_tokens"],
    )
    body = AuditedOpenAIResponsesProvider.request_body(request)
    if "temperature" in body:
        raise Mission73Error("Request cientifico de gpt-5.5 contem temperature.")
    if body.get("store") is not False:
        raise Mission73Error("Request cientifico nao fixa store=false.")

    preflight = _sign({
        "schema_version": "1.0", "artifact_type": "openai_v3_preflight",
        "generated_at_utc": utc_now(), "passed": True,
        "mission": "7.3", "technical_probe_repeated": False,
        "repository": {
            "mission_start_clean": True, "mission_start_commit": V3_START_COMMIT,
            "head_before_call": head, "head_unchanged": head == V3_START_COMMIT,
            "env_ignored": True, "unexpected_worktree_paths": unexpected_paths,
            "automatic_commit_performed": False,
        },
        "configuration": {
            "provider": "openai_responses", "model": model, "store": False,
            "temperature_sent": False, "retry_count": 0,
            "structured_output": True, "max_output_tokens": request.max_output_tokens,
            "credential_present": True, "credential_recorded": False,
        },
        "input_schema_valid": True, "privacy_valid": True,
        "individual_data_included": False, "final_predictions_included": False,
        "input_sha256": snapshot["input_sha256"],
        "scientific_request_sha256": stable_sha256(body),
        "prompt_versions": snapshot["prompt_versions"],
        "parser": {
            "source": "responses_parsing.extract_response_text",
            "source_sha256": file_sha256(PROJECT_ROOT / "src" / "tech_challenge_fase2" / "responses_parsing.py"),
            "raw_first": True, "mission721_ready": True,
            "mission721_manifest_signature": parsing_manifest["signature"],
        },
        "previous_evidence_hashes": previous_hashes,
        "provider_calls_performed": 0, "maximum_provider_calls": 4,
    })
    save_json(preflight, preflight_path)
    return preflight


class RawFirstOpenAIResponsesProvider(AuditedOpenAIResponsesProvider):
    """Transporte unico que persiste bruto e usage antes de interpretar status."""

    def __init__(
        self, *, api_key: str, artifact_root: Path,
        timeout_seconds: int = 180, opener: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, timeout_seconds=timeout_seconds)
        self.artifact_root = Path(artifact_root)
        self.opener = opener or urllib.request.urlopen
        self.transport_metadata: dict[str, Any] | None = None
        self.failure_stage: str | None = None

    def generate_raw(
        self, request: LLMRequest, *, scenario_instruction: str | None = None,
    ) -> RawProviderResponse:
        if scenario_instruction is not None:
            raise Mission73Error("Provider raw-first principal nao aceita cenario adversarial.")
        raw_path = self.artifact_root / RAW_NAME
        if raw_path.exists():
            raise Mission73Error("Resposta bruta V3 ja existe; nova chamada principal proibida.")
        body = self.request_body(request)
        if "temperature" in body or body.get("store") is not False:
            raise Mission73Error("Contrato de transporte V3 invalido.")
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        self.call_count += 1
        started = time.perf_counter()
        try:
            with self.opener(http_request, timeout=self.timeout_seconds) as response:
                http_status = getattr(response, "status", None)
                request_id = response.headers.get("x-request-id") if getattr(response, "headers", None) else None
                raw_bytes = response.read()
        except urllib.error.HTTPError as error:
            duration = time.perf_counter() - started
            request_id = error.headers.get("x-request-id") if error.headers else None
            error_payload: dict[str, Any] = {}
            try:
                decoded = json.loads(error.read().decode("utf-8"))
                if isinstance(decoded, dict) and isinstance(decoded.get("error"), dict):
                    error_payload = decoded["error"]
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                error_payload = {}
            raise ProviderCallError(
                f"Provider real retornou HTTP {error.code}; nenhuma repeticao automatica foi feita.",
                duration_seconds=duration, request_id=request_id, http_status=error.code,
                error_type=error_payload.get("type"), error_code=error_payload.get("code"),
                error_param=error_payload.get("param"), error_message=error_payload.get("message"),
                exception_class=type(error).__name__,
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ProviderCallError(
                f"Falha de transporte ({type(error).__name__}); nenhuma repeticao automatica foi feita.",
                duration_seconds=time.perf_counter() - started, exception_class=type(error).__name__,
            ) from error

        duration = time.perf_counter() - started
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.failure_stage = "raw_json_decode"
            raise ProviderCallError(
                "HTTP respondeu, mas o corpo nao era JSON valido; nenhuma repeticao automatica foi feita.",
                duration_seconds=duration, request_id=request_id, http_status=http_status,
                error_type="raw_json_decode", exception_class=type(error).__name__,
            ) from error
        if not isinstance(payload, dict):
            self.failure_stage = "raw_json_shape"
            raise ProviderCallError(
                "HTTP respondeu JSON nao-objeto; nenhuma repeticao automatica foi feita.",
                duration_seconds=duration, request_id=request_id, http_status=http_status,
                error_type="raw_json_shape",
            )
        _assert_response_safe(payload)
        serialized = json.dumps(payload, ensure_ascii=False)
        if "Authorization" in serialized or "Bearer " in serialized or "OPENAI_API_KEY" in serialized:
            raise Mission73Error("Resposta bruta contem material proibido para persistencia.")

        usage_values = _usage(payload)
        self.transport_metadata = {
            "http_status": http_status, "request_id": request_id,
            "response_id": payload.get("id"), "response_status": payload.get("status"),
            "requested_model": request.model, "response_model": payload.get("model"),
            "duration_seconds": duration, "store_requested": False,
            "store_returned": payload.get("store"), "temperature_sent": False,
            "retries": 0, **usage_values,
        }
        raw_artifact = _sign({
            "schema_version": "1.0", "artifact_type": "openai_v3_raw_response_sanitized",
            "generated_at_utc": utc_now(), "request_sha256": stable_sha256(body),
            **self.transport_metadata, "response_structure": response_structure(payload),
            "response": copy.deepcopy(payload), "secret_fields_present": False,
            "persisted_before_status_analysis": True,
            "persisted_before_output_extraction": True,
            "persisted_before_schema_validation": True,
        })
        save_json(raw_artifact, raw_path)
        save_json(_sign({
            "schema_version": "1.0", "artifact_type": "openai_provider_usage",
            "generated_at_utc": utc_now(), "provider": "openai_responses",
            **self.transport_metadata, "request_success": http_status == 200,
            "raw_first_persisted": True, "cost_estimate": None,
            "cost_estimate_reason": "No versioned price configuration was supplied; no price was invented.",
        }), self.artifact_root / USAGE_NAME)

        try:
            extracted = extract_response_text(payload)
        except ResponsesParsingError as error:
            self.failure_stage = "response_parsing"
            raise ProviderCallError(
                f"Resposta HTTP preservada, mas parsing falhou: {error}",
                duration_seconds=duration, request_id=request_id, http_status=http_status,
                error_type=type(error).__name__, error_message=str(error),
                exception_class=type(error).__name__,
            ) from error
        return RawProviderResponse(
            raw_output_text=extracted.text, response_id=payload.get("id"),
            requested_model=request.model, response_model=payload.get("model"),
            response_status=payload.get("status"), response_store=payload.get("store"),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            duration_seconds=duration,
        )


def _augment_usage(root: Path, provider: RawFirstOpenAIResponsesProvider) -> None:
    if provider.transport_metadata is None or not (root / USAGE_NAME).is_file():
        return
    current = json.loads((root / USAGE_NAME).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in current.items() if key != "signature"}
    unsigned.update(provider.transport_metadata)
    unsigned["response_id"] = provider.transport_metadata["response_id"]
    unsigned["request_success"] = provider.transport_metadata["http_status"] == 200
    unsigned["raw_first_persisted"] = (root / RAW_NAME).is_file()
    save_json(_sign(unsigned), root / USAGE_NAME)


def _ensure_adversarial_status(root: Path, status: str, blocked_by: str) -> dict[str, Any]:
    path = root / ADVERSARIAL_NAME
    if path.is_file():
        return _signed_json(path, "openai_adversarial_results")
    snapshot = _signed_json(root / INPUT_NAME, "openai_llm_input_snapshot")
    artifact = _sign({
        "schema_version": "1.0", "artifact_type": "openai_adversarial_results",
        "generated_at_utc": utc_now(), "status": status,
        "run_identity": snapshot["run_identity"], "provider_calls": 0,
        "maximum_authorized_calls": 3, "all_inputs_aggregate": True,
        "individual_data_sent": False, "scenarios": [], "blocked_by": blocked_by,
    })
    save_json(artifact, path)
    return artifact


def run_scientific_v3(
    *, artifact_root: Path = V3_ARTIFACT_ROOT,
    env_file: Path = PROJECT_ROOT / ".env",
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Executa a unica chamada cientifica principal autorizada."""

    root = Path(artifact_root)
    prepare_v3(artifact_root=root, env_file=env_file)
    if (root / OUTPUT_NAME).is_file():
        return _signed_json(root / OUTPUT_NAME, "openai_llm_output")
    model, secret = _configured_credentials(Path(env_file))
    provider = RawFirstOpenAIResponsesProvider(
        api_key=secret, artifact_root=root, opener=opener,
    )
    try:
        result = run_openai_main(artifact_root=root, env_file=env_file, provider=provider)
    except ProviderCallError:
        _augment_usage(root, provider)
        _ensure_adversarial_status(root, "not_run_main_invalid", provider.failure_stage or "provider_call")
        finalize_v3_manifest(root)
        raise
    _augment_usage(root, provider)
    if result.get("approved") is not True:
        _ensure_adversarial_status(root, "not_run_main_invalid", "scientific_main")
    finalize_v3_manifest(root)
    return result


def run_adversarial_v3(
    *, artifact_root: Path = V3_ARTIFACT_ROOT,
    env_file: Path = PROJECT_ROOT / ".env",
    provider: Any | None = None,
) -> dict[str, Any]:
    """Executa no maximo tres cenarios, somente apos aprovacao principal."""

    root = Path(artifact_root)
    output = _signed_json(root / OUTPUT_NAME, "openai_llm_output")
    if output.get("approved") is not True:
        return _ensure_adversarial_status(root, "not_run_main_invalid", "scientific_main")
    result = run_openai_adversarial(artifact_root=root, env_file=env_file, provider=provider)
    if result.get("status") != "approved" and not (root / FAILURE_NAME).is_file():
        snapshot = _signed_json(root / INPUT_NAME, "openai_llm_input_snapshot")
        save_json(_sign({
            "schema_version": "1.0", "artifact_type": "openai_failure_report",
            "generated_at_utc": utc_now(), "run_identity": snapshot["run_identity"],
            "stage": "adversarial_validation", "reason": "Um ou mais cenarios adversariais reprovaram.",
            "severity": "mission73_adversarial_invalid", "automatic_retry_performed": False,
            "prompt_changed": False, "original_evidence_preserved": True,
        }), root / FAILURE_NAME)
    finalize_v3_manifest(root)
    return result


def finalize_v3_manifest(artifact_root: Path = V3_ARTIFACT_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    preflight = _signed_json(root / V3_PREFLIGHT_NAME, "openai_v3_preflight")
    snapshot = _signed_json(root / INPUT_NAME, "openai_llm_input_snapshot")
    raw = _signed_json(root / RAW_NAME, "openai_v3_raw_response_sanitized") if (root / RAW_NAME).is_file() else None
    usage = _signed_json(root / USAGE_NAME, "openai_provider_usage") if (root / USAGE_NAME).is_file() else None
    main = _signed_json(root / MAIN_MANIFEST_NAME, "openai_main_run_manifest") if (root / MAIN_MANIFEST_NAME).is_file() else None
    adversarial = _signed_json(root / ADVERSARIAL_NAME, "openai_adversarial_results") if (root / ADVERSARIAL_NAME).is_file() else None
    main_approved = bool(main and main.get("status") == "approved")
    adversarial_calls = int(adversarial.get("provider_calls", 0)) if adversarial else 0
    total_calls = (1 if raw or (root / FAILURE_NAME).is_file() else 0) + adversarial_calls
    if total_calls > 4:
        raise Mission73Error("Orcamento absoluto de quatro chamadas excedido.")
    overall_approved = bool(main_approved and adversarial and adversarial.get("status") == "approved")
    status = "approved" if overall_approved else "invalid"
    if main_approved and adversarial and adversarial.get("status") != "approved":
        status = "main_approved_adversarial_invalid"

    base: dict[str, Any] = {}
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = {key: value for key, value in existing.items() if key not in {"signature", "files"}}
    base.update({
        "schema_version": "1.0", "artifact_type": "openai_llm_evaluation_manifest",
        "generated_at_utc": utc_now(), "mission_version": "7.3",
        "status": status, "approved": overall_approved,
        "run_identity": snapshot["run_identity"], "provider": "openai_responses",
        "model": snapshot["provider_configuration"]["model"],
        "prompt_versions": snapshot["prompt_versions"], "input_sha256": snapshot["input_sha256"],
        "output_sha256": file_sha256(root / OUTPUT_NAME) if (root / OUTPUT_NAME).is_file() else None,
        "schema_valid": main.get("schema_valid", False) if main else False,
        "factuality": main.get("factuality", False) if main else False,
        "safety": main.get("safety", False) if main else False,
        "completeness": main.get("completeness", False) if main else False,
        "clarity": main.get("clarity") if main else None,
        "scientific_calibration": main.get("scientific_calibration", False) if main else False,
        "http": {
            "status": raw.get("http_status") if raw else None,
            "request_id": raw.get("request_id") if raw else None,
            "response_id": raw.get("response_id") if raw else None,
            "response_status": raw.get("response_status") if raw else None,
            "response_model": raw.get("response_model") if raw else None,
        },
        "raw_first": {
            "enabled": True, "raw_persisted": raw is not None,
            "persisted_before_status_analysis": raw.get("persisted_before_status_analysis") if raw else None,
            "parser_source": preflight["parser"]["source"],
        },
        "adversarial_results": {
            "status": adversarial.get("status") if adversarial else "not_run",
            "provider_calls": adversarial_calls,
            "scenarios": [
                {key: item.get(key) for key in (
                    "scenario", "schema_valid", "factuality_passed", "safety_passed",
                    "scientific_calibration_passed", "scenario_passed",
                )}
                for item in (adversarial.get("scenarios", []) if adversarial else [])
            ],
        },
        "usage": usage,
        "call_budget": {
            "technical_probe": 0, "scientific_main": 1 if raw or (root / FAILURE_NAME).is_file() else 0,
            "adversarial": adversarial_calls, "total": total_calls, "maximum": 4,
            "automatic_retries": 0,
        },
        "privacy": {
            "individual_data_sent": False, "final_predictions_sent": False,
            "aggregate_contract_only": True, "secret_recorded": False,
        },
        "scope_confirmations": {
            "training_performed": False, "ga_executed": False,
            "randomized_search_executed": False, "new_holdout_inference_performed": False,
            "threshold_changed": False, "selection_reopened": False,
            "api_or_frontend_created": False, "deploy_performed": False,
        },
        "repository": {
            "mission_start_clean": preflight["repository"]["mission_start_clean"],
            "mission_start_commit": V3_START_COMMIT, "head_at_finalization": _git("rev-parse", "HEAD"),
            "env_ignored": bool(_git("check-ignore", ".env")),
            "automatic_commit_performed": False, "push_performed": False,
        },
        "previous_evidence_preserved": _previous_evidence_unchanged(preflight),
        "technical_probe_repeated": False,
    })
    files = []
    for path in sorted(root.glob("*.json")):
        if path.name in {MANIFEST_NAME, STATUS_NAME}:
            continue
        files.append({"filename": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    base["files"] = files
    manifest = _sign(base)
    save_json(manifest, manifest_path)
    return manifest


def validate_v3(artifact_root: Path = V3_ARTIFACT_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for name in REQUIRED_ARTIFACTS:
        add(f"exists:{name}", (root / name).is_file(), name)
    if not all((root / name).is_file() for name in REQUIRED_ARTIFACTS):
        return {"passed": False, "check_count": len(checks), "checks": checks}

    # Validacao deve ser estritamente read-only depois que a evidencia V3 foi
    # congelada; nao renova timestamp nem hashes do manifesto historico.
    manifest = _signed_json(root / MANIFEST_NAME, "openai_llm_evaluation_manifest")
    raw = _signed_json(root / RAW_NAME, "openai_v3_raw_response_sanitized")
    output = _signed_json(root / OUTPUT_NAME, "openai_llm_output")
    hallucination = _signed_json(root / HALLUCINATION_NAME, "openai_hallucination_report")
    adversarial = _signed_json(root / ADVERSARIAL_NAME, "openai_adversarial_results")
    for item in manifest["files"]:
        path = root / item["filename"]
        add(f"hash:{item['filename']}", path.is_file() and file_sha256(path) == item["sha256"], item["filename"])
    for path in root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = payload.get("signature")
        unsigned = {key: value for key, value in payload.items() if key != "signature"}
        add(f"signature:{path.name}", signature == stable_sha256(unsigned), path.name)

    add("http_200", raw["http_status"] == 200, raw["http_status"])
    add("response_completed", raw["response_status"] == "completed", raw["response_status"])
    add("store_false", raw["store_requested"] is False and raw["store_returned"] is False, raw["store_returned"])
    add("temperature_omitted", raw["temperature_sent"] is False, raw["temperature_sent"])
    add("raw_first", all(raw[key] is True for key in (
        "persisted_before_status_analysis", "persisted_before_output_extraction", "persisted_before_schema_validation",
    )), "raw persisted before parsing")
    main_approved = output["approved"] is True
    failure_preserved = (root / FAILURE_NAME).is_file()
    add(
        "scientific_status_recorded",
        main_approved or (output["approved"] is False and failure_preserved),
        "approved" if main_approved else "invalid_preserved",
    )
    if main_approved:
        add("main_schema", output["schema_valid"] is True, output["schema_valid"])
    add("unexpected_numbers", hallucination["unexpected_numbers"]["potentially_invented"] == [], hallucination["unexpected_numbers"]["potentially_invented"])
    add("clinical_violations", hallucination["clinical_claims"] == [], hallucination["clinical_claims"])
    add("selection_violations", hallucination["selection_violations"] == [], hallucination["selection_violations"])
    add("adversarial_gate", adversarial["status"] in {"approved", "not_run_main_invalid"}, adversarial["status"])
    add("call_budget", manifest["call_budget"]["total"] <= 4, manifest["call_budget"])
    add("zero_retries", manifest["call_budget"]["automatic_retries"] == 0, manifest["call_budget"])
    add("no_probe", manifest["technical_probe_repeated"] is False and manifest["call_budget"]["technical_probe"] == 0, manifest["call_budget"])
    add("previous_evidence", manifest["previous_evidence_preserved"] is True, manifest["previous_evidence_preserved"])
    generic = validate_openai_evaluation(root)
    add("generic_integrity", generic["passed"], generic["status"])
    content = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
    add("no_secrets", all(token not in content for token in ("Authorization", "Bearer ", '"OPENAI_API_KEY"', '".env"')), "safe")
    add("scope", not any(manifest["scope_confirmations"].values()), manifest["scope_confirmations"])
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed, "evidence_integrity_passed": passed,
        "scientific_evaluation_approved": main_approved,
        "check_count": len(checks), "checks": checks,
    }
