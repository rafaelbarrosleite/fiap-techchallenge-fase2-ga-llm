"""Orquestracao protegida, idempotente e auditavel da explicacao agregada."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256

from .evaluation import evaluate_output
from .factuality import validate_factuality
from .input_builder import FINAL_ROOT, PROJECT_ROOT, build_llm_input, file_sha256
from .privacy import validate_sanitized_input, validate_user_instruction
from .prompts import PromptBundle, load_prompt_bundle
from .providers import FakeLLMProvider, LLMProvider, LLMRequest, OpenAIResponsesProvider, load_env_value
from .safety import validate_safety
from .schemas import validate_output

LLM_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "llm_evaluation"
INPUT_NAME = "llm_input_snapshot.json"
OUTPUT_NAME = "llm_output.json"
FACTUALITY_NAME = "factuality_report.json"
SAFETY_NAME = "safety_report.json"
EVALUATION_NAME = "evaluation_report.json"
MANIFEST_NAME = "llm_evaluation_manifest.json"
STATUS_NAME = "llm_evaluation_status.json"
REQUIRED_ARTIFACTS = (INPUT_NAME, OUTPUT_NAME, FACTUALITY_NAME, SAFETY_NAME, EVALUATION_NAME)


class LLMEvaluationError(RuntimeError):
    """A execucao nao pode prosseguir sem quebrar protecoes da missao."""


class ManualInterventionRequired(LLMEvaluationError):
    """Ha uma execucao parcial ou uma identidade diferente no mesmo destino."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sign(payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["signature"] = stable_sha256(signed)
    return signed


def _load_signed(path: Path, artifact_type: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if signature != stable_sha256(unsigned):
        raise LLMEvaluationError(f"Assinatura invalida: {path.name}.")
    if artifact_type is not None and payload.get("artifact_type") != artifact_type:
        raise LLMEvaluationError(f"Tipo de artefato invalido: {path.name}.")
    return payload


def llm_code_signature() -> dict[str, Any]:
    root = Path(__file__).parent
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".txt"})
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in files}
    return {"sha256": stable_sha256(hashes), "files": hashes}


def _configuration(provider: str, model: str, temperature: float, max_output_tokens: int) -> dict[str, Any]:
    return {
        "provider": provider, "model": model, "temperature": temperature,
        "max_output_tokens": max_output_tokens, "network_required": provider != "fake",
    }


def _identity(input_payload: dict[str, Any], prompts: PromptBundle, configuration: dict[str, Any]) -> dict[str, Any]:
    code = llm_code_signature()
    components = {
        "input_sha256": stable_sha256(input_payload),
        "system_prompt_sha256": prompts.system_sha256,
        "explanation_prompt_sha256": prompts.explanation_sha256,
        "provider_configuration": configuration,
        "llm_code_sha256": code["sha256"],
    }
    return {"sha256": stable_sha256(components), "components": components, "code": code}


def _completed_manifest_valid(artifact_root: Path, expected_identity: str | None = None) -> bool:
    manifest_path = artifact_root / MANIFEST_NAME
    status_path = artifact_root / STATUS_NAME
    if not manifest_path.is_file() or not status_path.is_file():
        return False
    try:
        manifest = _load_signed(manifest_path, "llm_evaluation_manifest")
        status = _load_signed(status_path, "llm_evaluation_status")
        if status.get("status") != "completed" or status.get("manifest_signature") != manifest["signature"]:
            return False
        if expected_identity is not None and manifest.get("run_identity") != expected_identity:
            return False
        for item in manifest["files"]:
            path = artifact_root / item["filename"]
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def prepare_evaluation(
    *,
    artifact_root: Path = LLM_ARTIFACT_ROOT,
    final_root: Path = FINAL_ROOT,
    provider_name: str = "fake",
    model: str = "deterministic-explainer-v1",
    temperature: float = 0.0,
    max_output_tokens: int = 3000,
) -> dict[str, Any]:
    prompts = load_prompt_bundle()
    payload = build_llm_input(final_root=final_root)
    validate_sanitized_input(payload)
    configuration = _configuration(provider_name, model, temperature, max_output_tokens)
    identity = _identity(payload, prompts, configuration)
    artifact_root = Path(artifact_root)
    snapshot_path = artifact_root / INPUT_NAME
    if _completed_manifest_valid(artifact_root, identity["sha256"]):
        return _load_signed(snapshot_path, "llm_input_snapshot")
    if (artifact_root / MANIFEST_NAME).exists():
        raise ManualInterventionRequired(
            "Ja existe avaliacao no destino com identidade diferente ou integridade invalida; nao sera sobrescrita."
        )
    if snapshot_path.exists():
        existing = _load_signed(snapshot_path, "llm_input_snapshot")
        if existing.get("run_identity") != identity["sha256"]:
            raise ManualInterventionRequired("Snapshot preparado possui identidade diferente.")
        return existing
    snapshot = _sign({
        "schema_version": "1.0", "artifact_type": "llm_input_snapshot",
        "generated_at_utc": utc_now(), "run_identity": identity["sha256"],
        "input_sha256": stable_sha256(payload), "input": payload,
        "privacy_validation": {
            "passed": True, "individual_data_included": False,
            "forbidden_fields_found": [], "validated_before_provider": True,
        },
        "prompt_versions": {
            "system": prompts.system_version, "explanation": prompts.explanation_version,
            "system_sha256": prompts.system_sha256, "explanation_sha256": prompts.explanation_sha256,
        },
        "provider_configuration": configuration,
        "identity_components": identity["components"],
    })
    save_json(snapshot, snapshot_path)
    save_json(_sign({
        "schema_version": "1.0", "artifact_type": "llm_evaluation_status",
        "status": "prepared", "updated_at_utc": utc_now(), "run_identity": identity["sha256"],
    }), artifact_root / STATUS_NAME)
    return snapshot


def _provider_for(name: str, *, env_file: Path) -> LLMProvider:
    if name == "fake":
        return FakeLLMProvider()
    if name == "openai_responses":
        return OpenAIResponsesProvider(env_file=env_file)
    raise LLMEvaluationError(f"Provider desconhecido: {name}.")


def run_evaluation(
    *,
    artifact_root: Path = LLM_ARTIFACT_ROOT,
    final_root: Path = FINAL_ROOT,
    provider_name: str = "fake",
    model: str = "deterministic-explainer-v1",
    temperature: float = 0.0,
    max_output_tokens: int = 3000,
    provider: LLMProvider | None = None,
    user_instruction: str | None = None,
    env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    validate_user_instruction(user_instruction)
    snapshot = prepare_evaluation(
        artifact_root=artifact_root, final_root=final_root, provider_name=provider_name,
        model=model, temperature=temperature, max_output_tokens=max_output_tokens,
    )
    artifact_root = Path(artifact_root)
    identity = snapshot["run_identity"]
    if _completed_manifest_valid(artifact_root, identity):
        return _load_signed(artifact_root / OUTPUT_NAME, "llm_output")
    status_path = artifact_root / STATUS_NAME
    status = _load_signed(status_path, "llm_evaluation_status")
    if status.get("status") == "started":
        raise ManualInterventionRequired("Execucao iniciada sem manifesto concluido; revisao manual obrigatoria.")
    for name in (OUTPUT_NAME, FACTUALITY_NAME, SAFETY_NAME, EVALUATION_NAME):
        if (artifact_root / name).exists():
            raise ManualInterventionRequired("Artefatos parciais encontrados; nada sera sobrescrito.")
    save_json(_sign({
        "schema_version": "1.0", "artifact_type": "llm_evaluation_status",
        "status": "started", "updated_at_utc": utc_now(), "run_identity": identity,
    }), status_path)
    prompts = load_prompt_bundle()
    validate_sanitized_input(snapshot["input"])
    selected_provider = provider or _provider_for(provider_name, env_file=Path(env_file))
    request = LLMRequest(
        input_payload=snapshot["input"], system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model=model,
        temperature=temperature, max_output_tokens=max_output_tokens,
    )
    response = selected_provider.generate(request)
    validate_output(response.output)
    factuality = validate_factuality(response.output, snapshot["input"])
    safety = validate_safety(response.output)
    evaluation = evaluate_output(response.output, snapshot["input"])
    approved = factuality["passed"] and safety["passed"] and evaluation["approved"]

    output_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_output", "generated_at_utc": utc_now(),
        "run_identity": identity, "approved": approved, "provider": response.provider,
        "model": response.model, "response_id": response.response_id, "usage": response.usage,
        "structured_output": response.output,
    })
    factuality_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_factuality_report",
        "generated_at_utc": utc_now(), "run_identity": identity, **factuality,
    })
    safety_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_safety_report",
        "generated_at_utc": utc_now(), "run_identity": identity, **safety,
    })
    evaluation_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_evaluation_report",
        "generated_at_utc": utc_now(), "run_identity": identity, **evaluation,
    })
    for name, value in (
        (OUTPUT_NAME, output_artifact), (FACTUALITY_NAME, factuality_artifact),
        (SAFETY_NAME, safety_artifact), (EVALUATION_NAME, evaluation_artifact),
    ):
        save_json(value, artifact_root / name)

    files = [
        {"filename": name, "sha256": file_sha256(artifact_root / name), "bytes": (artifact_root / name).stat().st_size}
        for name in REQUIRED_ARTIFACTS
    ]
    manifest = _sign({
        "schema_version": "1.0", "artifact_type": "llm_evaluation_manifest",
        "generated_at_utc": utc_now(), "status": "approved" if approved else "invalid",
        "run_identity": identity,
        "prompt_versions": snapshot["prompt_versions"],
        "provider": response.provider, "model": response.model,
        "generation_configuration": snapshot["provider_configuration"],
        "hashes": {
            "input_sha256": snapshot["input_sha256"],
            "llm_code_sha256": snapshot["identity_components"]["llm_code_sha256"],
        },
        "factuality_passed": factuality["passed"], "safety_passed": safety["passed"],
        "evaluation_approved": evaluation["approved"], "individual_data_sent": False,
        "provider_received_aggregate_contract_only": True, "secrets_recorded": False,
        "new_training_performed": False, "ga_executed": False,
        "randomized_search_executed": False, "threshold_changed": False,
        "selection_reopened": False, "files": files,
        "software": {"python": platform.python_version()},
    })
    save_json(manifest, artifact_root / MANIFEST_NAME)
    save_json(_sign({
        "schema_version": "1.0", "artifact_type": "llm_evaluation_status", "status": "completed",
        "updated_at_utc": utc_now(), "run_identity": identity,
        "manifest_signature": manifest["signature"], "approved": approved,
    }), status_path)
    return output_artifact


def evaluate_existing_output(artifact_root: Path = LLM_ARTIFACT_ROOT) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    if not _completed_manifest_valid(artifact_root):
        raise LLMEvaluationError("Avaliacao concluida e integra nao encontrada.")
    snapshot = _load_signed(artifact_root / INPUT_NAME, "llm_input_snapshot")
    output = _load_signed(artifact_root / OUTPUT_NAME, "llm_output")
    recomputed = evaluate_output(output["structured_output"], snapshot["input"])
    stored = _load_signed(artifact_root / EVALUATION_NAME, "llm_evaluation_report")
    comparable_stored = {key: stored[key] for key in ("approved", "overall_score", "dimensions", "factuality", "safety", "evaluation_is_deterministic", "llm_judge_used")}
    if recomputed != comparable_stored:
        raise LLMEvaluationError("Relatorio armazenado diverge da reavaliacao deterministica.")
    return recomputed


def configured_real_model(env_file: Path = PROJECT_ROOT / ".env") -> str:
    model = load_env_value("OPENAI_MODEL", env_file)
    if not model or model.startswith("replace_with_"):
        raise LLMEvaluationError("OPENAI_MODEL deve ser preenchido explicitamente no .env.")
    return model

