import json
from pathlib import Path

import pytest

from tech_challenge_fase2.llm.input_builder import build_llm_input
from tech_challenge_fase2.llm.prompts import load_prompt_bundle
from tech_challenge_fase2.llm.providers import LLMRequest, build_deterministic_output
from tech_challenge_fase2.llm.schemas import output_json_schema
from tech_challenge_fase2.provider_real_evaluation import (
    ADVERSARIAL_NAME, COMPARISON_NAME, FAILURE_NAME, HALLUCINATION_NAME,
    MANIFEST_NAME, OUTPUT_NAME, PREFLIGHT_NAME, USAGE_NAME,
    AuditedOpenAIResponsesProvider, ProviderCallError, RawProviderResponse,
    finalize_failed_openai_evaluation, hallucination_report, prepare_openai_evaluation, run_openai_adversarial,
    run_openai_main, validate_openai_evaluation,
)


def _env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=sk-test-not-real-mission7\nOPENAI_MODEL=gpt-test\n", encoding="utf-8")
    return path


class OfflineAuditedProvider:
    name = "openai_responses"

    def __init__(self) -> None:
        self.call_count = 0

    def generate_raw(self, request: LLMRequest, *, scenario_instruction: str | None = None) -> RawProviderResponse:
        self.call_count += 1
        output = build_deterministic_output(request.input_payload)
        return RawProviderResponse(
            raw_output_text=json.dumps(output, ensure_ascii=False),
            response_id=f"resp-offline-{self.call_count}", requested_model=request.model,
            response_model=request.model, response_status="completed", response_store=False,
            usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            duration_seconds=0.01,
        )


class FailingProvider:
    name = "openai_responses"
    call_count = 0

    def generate_raw(self, request: LLMRequest, *, scenario_instruction: str | None = None) -> RawProviderResponse:
        self.call_count += 1
        raise ProviderCallError("falha sintetica sem retry", duration_seconds=0.02, request_id="req-safe")


def test_preflight_reuses_exact_fake_payload_without_persisting_secret(tmp_path: Path) -> None:
    root = tmp_path / "openai"
    snapshot = prepare_openai_evaluation(artifact_root=root, env_file=_env(tmp_path))
    assert snapshot["input"] == build_llm_input()
    assert snapshot["privacy_validation"]["individual_data_included"] is False
    assert snapshot["credential_validation"]["secret_value_recorded"] is False
    assert (root / PREFLIGHT_NAME).is_file()
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
    assert "sk-test-not-real-mission7" not in persisted
    assert "final_predictions.json" not in persisted


def test_request_contract_uses_structured_output_store_false_and_aggregate_only() -> None:
    prompts = load_prompt_bundle()
    request = LLMRequest(
        input_payload=build_llm_input(), system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model="gpt-5.5",
    )
    body = AuditedOpenAIResponsesProvider.request_body(request)
    serialized = json.dumps(body)
    assert body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert "temperature" not in body
    assert body["text"]["format"]["schema"] == output_json_schema()
    assert "patient_id" not in serialized
    assert "final_predictions.json" not in serialized


def test_offline_stub_generates_main_comparison_usage_and_hallucination_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used by pytest"))
    root = tmp_path / "openai"
    provider = OfflineAuditedProvider()
    result = run_openai_main(artifact_root=root, env_file=_env(tmp_path), provider=provider)
    assert result["approved"] is True
    assert provider.call_count == 1
    assert {OUTPUT_NAME, USAGE_NAME, COMPARISON_NAME, HALLUCINATION_NAME}.issubset({path.name for path in root.iterdir()})
    comparison = json.loads((root / COMPARISON_NAME).read_text(encoding="utf-8"))
    assert comparison["rows"]["factuality"] == {"fake": True, "openai": True}
    hallucination = json.loads((root / HALLUCINATION_NAME).read_text(encoding="utf-8"))
    assert hallucination["unexpected_numbers"]["potentially_invented"] == []
    assert hallucination["clinical_claims"] == []


def test_three_adversarial_calls_run_only_after_approved_main(tmp_path: Path) -> None:
    root = tmp_path / "openai"
    env = _env(tmp_path)
    main_provider = OfflineAuditedProvider()
    assert run_openai_main(artifact_root=root, env_file=env, provider=main_provider)["approved"] is True
    adversarial_provider = OfflineAuditedProvider()
    result = run_openai_adversarial(artifact_root=root, env_file=env, provider=adversarial_provider)
    assert result["status"] == "approved"
    assert result["provider_calls"] == 3
    assert adversarial_provider.call_count == 3
    assert len(result["scenarios"]) == 3
    assert all(item["scenario_passed"] for item in result["scenarios"])
    assert (root / ADVERSARIAL_NAME).is_file()
    assert (root / MANIFEST_NAME).is_file()
    assert validate_openai_evaluation(root)["passed"] is True


def test_provider_failure_is_preserved_and_never_retried(tmp_path: Path) -> None:
    root = tmp_path / "openai"
    provider = FailingProvider()
    with pytest.raises(ProviderCallError, match="sem retry"):
        run_openai_main(artifact_root=root, env_file=_env(tmp_path), provider=provider)
    assert provider.call_count == 1
    assert (root / FAILURE_NAME).is_file()
    assert (root / OUTPUT_NAME).is_file()
    assert (root / COMPARISON_NAME).is_file()
    failure = json.loads((root / FAILURE_NAME).read_text(encoding="utf-8"))
    assert failure["automatic_retry_performed"] is False
    repeated = run_openai_main(artifact_root=root, env_file=_env(tmp_path), provider=provider)
    assert repeated["approved"] is False
    assert provider.call_count == 1
    finalize_failed_openai_evaluation(root)
    audit = validate_openai_evaluation(root)
    assert audit["evidence_integrity_passed"] is True
    assert audit["evaluation_approved"] is False


def test_hallucination_report_distinguishes_data_structural_and_invented_numbers() -> None:
    source = build_llm_input()
    output = build_deterministic_output(source)
    output["conclusao"] += " Existem 3 modelos; valor inventado 0,123456."
    report = hallucination_report(output, source)
    assert "3" in report["unexpected_numbers"]["from_data"] or "3" in report["unexpected_numbers"]["structural_legitimate"]
    assert "0,123456" in report["unexpected_numbers"]["potentially_invented"]
    assert report["passed"] is False
