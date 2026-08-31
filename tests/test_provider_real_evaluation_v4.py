"""Testes offline da Missao 7.5; nenhuma chamada externa e permitida."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_challenge_fase2.llm_v2.input_builder import build_llm_input_v2
from tech_challenge_fase2.llm_v2.privacy import validate_sanitized_input_v2
from tech_challenge_fase2.llm_v2.providers import build_deterministic_output_v2
from tech_challenge_fase2.llm_v2.schemas import SchemaV2Error
from tech_challenge_fase2.provider_real_evaluation_v4 import (
    ADVERSARIAL_NAME,
    RAW_NAME,
    RawFirstOpenAIResponsesProviderV2,
    _request,
    hallucination_report_v2,
    prepare_v4,
    run_adversarial_v4,
    run_scientific_v4,
    validate_v4,
)


class _Response:
    status = 200
    headers = {"x-request-id": "req_offline_v4"}

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class _DynamicOfflineOpener:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request, timeout):
        assert timeout == 240
        body = json.loads(request.data.decode("utf-8"))
        assert body["store"] is False
        assert "temperature" not in body
        text = body["input"][0]["content"][0]["text"]
        aggregate = text.split("AGGREGATED_EXPERIMENT_INPUT\n", 1)[1]
        source = json.loads(aggregate)
        output = build_deterministic_output_v2(source)
        self.calls += 1
        payload = {
            "id": f"resp_offline_v4_{self.calls}", "status": "completed",
            "model": "gpt-5.5-offline-fixture", "store": False,
            "output": [{
                "type": "message", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(output, ensure_ascii=False)}],
            }],
            "usage": {
                "input_tokens": 100, "output_tokens": 200, "total_tokens": 300,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 10},
            },
        }
        return _Response(payload)


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "mission75.env"
    path.write_text(
        "OPENAI_API_KEY=offline-fixture-not-a-real-secret\nOPENAI_MODEL=gpt-5.5\n",
        encoding="utf-8",
    )
    return path


def test_v2_privacy_rejects_individual_field() -> None:
    payload = build_llm_input_v2()
    payload["patient_id"] = "forbidden"
    with pytest.raises(SchemaV2Error):
        validate_sanitized_input_v2(payload)


def test_v4_request_uses_v2_without_temperature() -> None:
    payload = build_llm_input_v2()
    request = _request(payload, "gpt-5.5")
    body = RawFirstOpenAIResponsesProviderV2.request_body(request)
    assert "temperature" not in body
    assert body["store"] is False
    assert body["text"]["format"]["name"] == "experiment_explanation_v2"
    assert body["text"]["format"]["schema"]["properties"]["contract_version"]["const"] == "v2"


def test_prepare_v4_is_offline_and_idempotent(tmp_path: Path, env_file: Path) -> None:
    artifact_root = tmp_path / "v4"
    first = prepare_v4(artifact_root=artifact_root, env_file=env_file)
    second = prepare_v4(artifact_root=artifact_root, env_file=env_file)
    assert first == second
    assert first["passed"] is True
    assert first["provider_calls_performed"] == 0
    assert first["configuration"]["contract_version"] == "v2"
    assert first["configuration"]["temperature_sent"] is False


def test_full_v4_flow_is_offline_reproducible_and_valid(tmp_path: Path, env_file: Path) -> None:
    artifact_root = tmp_path / "v4"
    opener = _DynamicOfflineOpener()
    main = run_scientific_v4(
        artifact_root=artifact_root, env_file=env_file, opener=opener,
    )
    assert main["scientific_evaluation_approved"] is True
    assert opener.calls == 1
    repeated = run_scientific_v4(
        artifact_root=artifact_root, env_file=env_file, opener=opener,
    )
    assert repeated == main
    assert opener.calls == 1
    adversarial = run_adversarial_v4(
        artifact_root=artifact_root, env_file=env_file, opener=opener,
    )
    assert adversarial["status"] == "approved"
    assert adversarial["provider_calls"] == 3
    assert opener.calls == 4
    assert (artifact_root / RAW_NAME).is_file()
    assert (artifact_root / ADVERSARIAL_NAME).is_file()
    validation = validate_v4(artifact_root)
    assert validation["passed"] is True
    assert validation["scientific_evaluation_approved"] is True


def test_pair_misattribution_is_visible_in_hallucination_report() -> None:
    source = build_llm_input_v2()
    output = build_deterministic_output_v2(source)
    rf_baseline_ga = next(
        item for item in output["comparison_findings"]
        if item["comparison_id"] == "random_forest__baseline_vs_ga"
    )
    rf_baseline_ga["same_confusion_matrix"] = True
    from tech_challenge_fase2.llm_v2.evaluation import evaluate_output_v2

    evaluation = evaluate_output_v2(output, source)
    report = hallucination_report_v2(
        output, source, evaluation["factuality"], evaluation["safety"], evaluation,
    )
    assert report["comparison_pair_violations"]
    assert not report["mcnemar_violations"]
