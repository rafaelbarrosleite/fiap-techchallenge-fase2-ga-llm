"""Orquestracao auditavel da explicacao individual, offline por padrao."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT
from tech_challenge_fase2.llm.providers import LLMRequest, LLMResponse, load_env_value
from tech_challenge_fase2.llm.safety import validate_safety

from .case_builder import build_individual_input
from .evaluation import evaluate
from .factuality import validate_factuality
from .privacy import validate_privacy
from .prompts import load_prompts
from .providers import FakeIndividualProvider, OpenAIIndividualProvider
from .schemas import validate_output

FAKE_ROOT = PROJECT_ROOT / "artifacts" / "llm_individual_explanation"
OPENAI_ROOT = PROJECT_ROOT / "artifacts" / "llm_individual_explanation_openai"
INPUT_NAME = "individual_input_snapshot.json"
OUTPUT_NAME = "individual_output.json"
PRIVACY_NAME = "privacy_report.json"
FACTUALITY_NAME = "factuality_report.json"
SAFETY_NAME = "safety_report.json"
EVALUATION_NAME = "evaluation_report.json"
USAGE_NAME = "provider_usage.json"
MANIFEST_NAME = "individual_explanation_manifest.json"
FAILURE_NAME = "failure_report.json"
PREVIOUS_EVALUATION_NAME = "evaluation_report_before_semantic_fix.json"
MAX_OUTPUT_TOKENS = 5000


class IndividualExplanationError(RuntimeError):
    """A execucao individual nao pode prosseguir com seguranca."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sign(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["signature"] = stable_sha256(result)
    return result


def _load_signed(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("artifact_type") != artifact_type or signature != stable_sha256(unsigned):
        raise IndividualExplanationError(f"Artefato invalido: {path}.")
    return payload


def _root(provider_name: str, artifact_root: Path | None) -> Path:
    if artifact_root is not None:
        return Path(artifact_root)
    return OPENAI_ROOT if provider_name == "openai_responses" else FAKE_ROOT


def _model(provider_name: str, model: str | None, env_file: Path) -> str:
    if model:
        return model
    if provider_name == "fake":
        return "deterministic-individual-explainer-v1"
    configured = load_env_value("OPENAI_MODEL", env_file)
    if not configured:
        raise IndividualExplanationError("OPENAI_MODEL deve estar explicitamente configurado.")
    return configured


def prepare(
    *, provider_name: str = "fake", model: str | None = None,
    artifact_root: Path | None = None, env_file: Path = PROJECT_ROOT / ".env",
) -> dict[str, Any]:
    root = _root(provider_name, artifact_root)
    selected_model = _model(provider_name, model, Path(env_file))
    payload = build_individual_input()
    validate_privacy(payload)
    prompts = load_prompts()
    identity = stable_sha256({
        "contract": "individual_v1", "input": payload, "provider": provider_name,
        "model": selected_model, "system_sha256": prompts.system_sha256,
        "explanation_sha256": prompts.explanation_sha256,
        "store": False, "temperature_sent": False, "max_output_tokens": MAX_OUTPUT_TOKENS,
    })
    input_path = root / INPUT_NAME
    if input_path.is_file():
        existing = _load_signed(input_path, "individual_explanation_input")
        if existing["run_identity"] != identity:
            raise IndividualExplanationError("Diretorio contem outra identidade; nada sera sobrescrito.")
        return existing
    if root.exists() and any(root.iterdir()):
        raise IndividualExplanationError("Diretorio parcial encontrado; revisao manual obrigatoria.")
    snapshot = _sign({
        "schema_version": "3.0", "artifact_type": "individual_explanation_input",
        "generated_at_utc": _now(), "run_identity": identity,
        "provider_configuration": {
            "provider": provider_name, "model": selected_model, "store": False,
            "temperature_sent": False, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "automatic_retries": 0, "structured_output": True,
        },
        "prompt_versions": {
            "system": prompts.system_version, "explanation": prompts.explanation_version,
            "system_sha256": prompts.system_sha256,
            "explanation_sha256": prompts.explanation_sha256,
        },
        "privacy": {
            "passed": True, "development_only": True, "deidentified": True,
            "patient_identifiers_sent": False, "raw_feature_values_sent": False,
            "ground_truth_sent": False, "original_index_sent": False,
            "holdout_case_sent": False,
        },
        "input_sha256": stable_sha256(payload), "input": payload,
    })
    save_json(snapshot, input_path)
    save_json(_sign({
        "schema_version": "1.0", "artifact_type": "individual_privacy_report",
        "generated_at_utc": _now(), "run_identity": identity, "passed": True,
        "checks": snapshot["privacy"],
    }), root / PRIVACY_NAME)
    return snapshot


def _provider(name: str, *, env_file: Path) -> Any:
    if name == "fake":
        return FakeIndividualProvider()
    if name == "openai_responses":
        return OpenAIIndividualProvider(env_file=env_file)
    raise IndividualExplanationError(f"Provider desconhecido: {name}.")


def _usage(response: LLMResponse, provider: Any, duration: float) -> dict[str, Any]:
    raw = response.usage or {}
    return _sign({
        "schema_version": "1.0", "artifact_type": "individual_provider_usage",
        "generated_at_utc": _now(), "provider": response.provider, "model": response.model,
        "request_id": response.response_id,
        "input_tokens": raw.get("input_tokens"), "output_tokens": raw.get("output_tokens"),
        "total_tokens": raw.get("total_tokens"),
        "duration_seconds": getattr(provider, "last_duration_seconds", None) or duration,
        "http_status": getattr(provider, "last_http_status", None),
        "request_success": True, "store": False, "temperature_sent": False,
        "automatic_retries": 0,
    })


def run(
    *, provider_name: str = "fake", model: str | None = None,
    artifact_root: Path | None = None, env_file: Path = PROJECT_ROOT / ".env",
    provider: Any | None = None,
) -> dict[str, Any]:
    root = _root(provider_name, artifact_root)
    snapshot = prepare(
        provider_name=provider_name, model=model, artifact_root=root, env_file=Path(env_file),
    )
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = _load_signed(manifest_path, "individual_explanation_manifest")
        if manifest["run_identity"] != snapshot["run_identity"]:
            raise IndividualExplanationError("Manifesto pertence a outra identidade.")
        return _load_signed(root / OUTPUT_NAME, "individual_explanation_output")
    for name in (OUTPUT_NAME, FACTUALITY_NAME, SAFETY_NAME, EVALUATION_NAME, USAGE_NAME):
        if (root / name).exists():
            raise IndividualExplanationError("Execucao parcial preservada; nada sera sobrescrito.")
    prompts = load_prompts()
    request = LLMRequest(
        input_payload=snapshot["input"], system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text,
        model=snapshot["provider_configuration"]["model"], temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    selected_provider = provider or _provider(provider_name, env_file=Path(env_file))
    started = time.perf_counter()
    try:
        response = selected_provider.generate(request)
        duration = time.perf_counter() - started
        validate_output(response.output)
        factuality = validate_factuality(response.output, snapshot["input"])
        safety = validate_safety(response.output)
        evaluation = evaluate(response.output, snapshot["input"])
    except Exception as error:
        save_json(_sign({
            "schema_version": "1.0", "artifact_type": "individual_explanation_failure",
            "generated_at_utc": _now(), "run_identity": snapshot["run_identity"],
            "error_type": type(error).__name__, "error_message": str(error),
            "provider_called": provider_name == "openai_responses", "automatic_retries": 0,
            "secret_recorded": False,
        }), root / FAILURE_NAME)
        raise
    approved = factuality["passed"] and safety["passed"] and evaluation["approved"]
    output_artifact = _sign({
        "schema_version": "3.0", "artifact_type": "individual_explanation_output",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"],
        "approved": approved, "provider": response.provider, "model": response.model,
        "response_id": response.response_id, "structured_output": response.output,
    })
    factuality_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "individual_factuality_report",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"], **factuality,
    })
    safety_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "individual_safety_report",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"], **safety,
    })
    evaluation_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "individual_evaluation_report",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"], **evaluation,
    })
    usage_artifact = _usage(response, selected_provider, duration)
    for name, value in (
        (OUTPUT_NAME, output_artifact), (FACTUALITY_NAME, factuality_artifact),
        (SAFETY_NAME, safety_artifact), (EVALUATION_NAME, evaluation_artifact),
        (USAGE_NAME, usage_artifact),
    ):
        save_json(value, root / name)
    manifest = _sign({
        "schema_version": "1.0", "artifact_type": "individual_explanation_manifest",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"],
        "status": "approved" if approved else "invalid", "approved": approved,
        "provider": response.provider, "model": response.model,
        "contract_version": "individual_v1", "prompt_versions": snapshot["prompt_versions"],
        "input_sha256": snapshot["input_sha256"],
        "output_sha256": stable_sha256(response.output),
        "quality": {
            "factuality": factuality["passed"], "safety": safety["passed"],
            "completeness": evaluation["dimensions"]["completeness"]["passed"],
            "clarity": evaluation["dimensions"]["clarity"]["passed"],
            "medical_context_relevance": evaluation["dimensions"]["medical_context_relevance"]["passed"],
            "scientific_calibration": evaluation["dimensions"]["scientific_calibration"]["passed"],
        },
        "privacy": snapshot["privacy"],
        "execution_scope": {
            "new_training": False, "ga_executed": False, "randomized_search_executed": False,
            "threshold_changed": False, "holdout_inference": False, "selection_reopened": False,
            "external_calls": (
                1 + int((root / FAILURE_NAME).is_file())
                if provider_name == "openai_responses" else 0
            ),
            "prior_failure_preserved": (root / FAILURE_NAME).is_file(),
            "manual_reexecution_after_fix": (
                (root / FAILURE_NAME).is_file() if provider_name == "openai_responses" else False
            ),
            "automatic_retries": 0,
        },
    })
    save_json(manifest, manifest_path)
    return output_artifact


def evaluate_existing(*, artifact_root: Path = FAKE_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    snapshot = _load_signed(root / INPUT_NAME, "individual_explanation_input")
    output = _load_signed(root / OUTPUT_NAME, "individual_explanation_output")
    report = evaluate(output["structured_output"], snapshot["input"])
    manifest = _load_signed(root / MANIFEST_NAME, "individual_explanation_manifest")
    return {
        **report,
        "manifest_valid": manifest["run_identity"] == snapshot["run_identity"],
        "privacy_valid": _load_signed(root / PRIVACY_NAME, "individual_privacy_report")["passed"],
    }


def revalidate_existing(*, artifact_root: Path = OPENAI_ROOT) -> dict[str, Any]:
    """Recalcula os gates offline sem alterar a resposta nem chamar provider."""

    root = Path(artifact_root)
    snapshot = _load_signed(root / INPUT_NAME, "individual_explanation_input")
    output_artifact = _load_signed(root / OUTPUT_NAME, "individual_explanation_output")
    manifest = _load_signed(root / MANIFEST_NAME, "individual_explanation_manifest")
    previous_path = root / PREVIOUS_EVALUATION_NAME
    if not previous_path.exists():
        previous = _load_signed(root / EVALUATION_NAME, "individual_evaluation_report")
        save_json(previous, previous_path)
    output = output_artifact["structured_output"]
    factuality = validate_factuality(output, snapshot["input"])
    safety = validate_safety(output)
    report = evaluate(output, snapshot["input"])
    approved = factuality["passed"] and safety["passed"] and report["approved"]
    updated_output = _sign({
        **{key: value for key, value in output_artifact.items() if key != "signature"},
        "approved": approved,
    })
    factuality_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "individual_factuality_report",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"], **factuality,
    })
    safety_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "individual_safety_report",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"], **safety,
    })
    evaluation_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "individual_evaluation_report",
        "generated_at_utc": _now(), "run_identity": snapshot["run_identity"], **report,
        "offline_revalidation": {
            "provider_called": False,
            "output_changed": False,
            "reason": "semantic_calibration_rule_accepts_explicit_non_causality_paraphrases",
            "previous_report_preserved_as": PREVIOUS_EVALUATION_NAME,
        },
    })
    updated_manifest = _sign({
        **{key: value for key, value in manifest.items() if key != "signature"},
        "generated_at_utc": _now(), "status": "approved" if approved else "invalid",
        "approved": approved,
        "quality": {
            "factuality": factuality["passed"], "safety": safety["passed"],
            "completeness": report["dimensions"]["completeness"]["passed"],
            "clarity": report["dimensions"]["clarity"]["passed"],
            "medical_context_relevance": report["dimensions"]["medical_context_relevance"]["passed"],
            "scientific_calibration": report["dimensions"]["scientific_calibration"]["passed"],
        },
        "offline_revalidation": {
            "performed": True, "provider_called": False, "output_changed": False,
            "previous_status": manifest["status"],
            "previous_evaluation_preserved": True,
        },
    })
    for name, value in (
        (OUTPUT_NAME, updated_output), (FACTUALITY_NAME, factuality_artifact),
        (SAFETY_NAME, safety_artifact), (EVALUATION_NAME, evaluation_artifact),
        (MANIFEST_NAME, updated_manifest),
    ):
        save_json(value, root / name)
    return {**report, "manifest_valid": True, "privacy_valid": True}
