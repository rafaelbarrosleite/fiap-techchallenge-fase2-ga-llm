import json
from pathlib import Path

import pytest

from tech_challenge_fase2.llm.providers import LLMRequest, build_deterministic_output
from tech_challenge_fase2.provider_real_evaluation import RawProviderResponse
from tech_challenge_fase2.provider_real_evaluation_v3 import (
    RAW_NAME,
    RawFirstOpenAIResponsesProvider,
    finalize_v3_manifest,
    prepare_v3,
    run_adversarial_v3,
    run_scientific_v3,
    validate_v3,
)


def _env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=sk-test-not-real-mission73\nOPENAI_MODEL=gpt-5.5\n",
        encoding="utf-8",
    )
    return path


class _HttpResponse:
    status = 200
    headers = {"x-request-id": "req-v3-offline"}

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class OfflineAdversarialProvider:
    name = "openai_responses"

    def __init__(self) -> None:
        self.call_count = 0

    def generate_raw(
        self, request: LLMRequest, *, scenario_instruction: str | None = None,
    ) -> RawProviderResponse:
        self.call_count += 1
        output = build_deterministic_output(request.input_payload)
        return RawProviderResponse(
            raw_output_text=json.dumps(output, ensure_ascii=False),
            response_id=f"resp-adv-{self.call_count}", requested_model=request.model,
            response_model=request.model, response_status="completed", response_store=False,
            usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            duration_seconds=0.01,
        )


def _scientific_payload() -> dict:
    from tech_challenge_fase2.llm.input_builder import build_llm_input

    source = build_llm_input()
    output = build_deterministic_output(source)
    return {
        "id": "resp-v3-offline", "model": "gpt-5.5-test", "status": "completed",
        "store": False,
        "output": [
            {"type": "reasoning", "id": "rs-v3", "summary": []},
            {
                "type": "message", "role": "assistant", "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": json.dumps(output, ensure_ascii=False),
                }],
            },
        ],
        "usage": {
            "input_tokens": 700, "output_tokens": 900, "total_tokens": 1600,
            "input_tokens_details": {"cached_tokens": 50},
            "output_tokens_details": {"reasoning_tokens": 80},
        },
    }


def test_v3_preflight_is_offline_and_omits_temperature(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used by preflight"),
    )
    result = prepare_v3(artifact_root=tmp_path / "v3", env_file=_env(tmp_path))
    assert result["passed"] is True
    assert result["configuration"]["temperature_sent"] is False
    assert result["configuration"]["store"] is False
    assert result["technical_probe_repeated"] is False
    assert result["privacy_valid"] is True
    assert result["provider_calls_performed"] == 0


def test_raw_first_scientific_main_and_adversarial_are_fully_offline(tmp_path: Path) -> None:
    root = tmp_path / "v3"
    env = _env(tmp_path)
    response = _HttpResponse(_scientific_payload())
    main = run_scientific_v3(
        artifact_root=root, env_file=env, opener=lambda *args, **kwargs: response,
    )
    assert main["approved"] is True
    raw = json.loads((root / RAW_NAME).read_text(encoding="utf-8"))
    assert raw["http_status"] == 200
    assert raw["persisted_before_status_analysis"] is True
    assert raw["response_structure"]["output_item_types"] == ["reasoning", "message"]
    usage = json.loads((root / "provider_usage.json").read_text(encoding="utf-8"))
    assert usage["request_id"] == "req-v3-offline"
    assert usage["reasoning_tokens"] == 80

    adversarial_provider = OfflineAdversarialProvider()
    adversarial = run_adversarial_v3(
        artifact_root=root, env_file=env, provider=adversarial_provider,
    )
    assert adversarial["status"] == "approved"
    assert adversarial_provider.call_count == 3
    assert finalize_v3_manifest(root)["call_budget"] == {
        "technical_probe": 0, "scientific_main": 1, "adversarial": 3,
        "total": 4, "maximum": 4, "automatic_retries": 0,
    }
    assert validate_v3(root)["passed"] is True


def test_raw_and_usage_survive_incomplete_response_without_retry(tmp_path: Path) -> None:
    root = tmp_path / "v3"
    env = _env(tmp_path)
    payload = {
        "id": "resp-v3-incomplete", "model": "gpt-5.5-test", "status": "incomplete",
        "store": False, "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning", "id": "rs-only", "summary": []}],
        "usage": {
            "input_tokens": 700, "output_tokens": 3000, "total_tokens": 3700,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 3000},
        },
    }
    with pytest.raises(Exception, match="parsing falhou"):
        run_scientific_v3(
            artifact_root=root, env_file=env,
            opener=lambda *args, **kwargs: _HttpResponse(payload),
        )
    assert (root / RAW_NAME).is_file()
    raw = json.loads((root / RAW_NAME).read_text(encoding="utf-8"))
    assert raw["response_status"] == "incomplete"
    usage = json.loads((root / "provider_usage.json").read_text(encoding="utf-8"))
    assert usage["total_tokens"] == 3700
    assert usage["raw_first_persisted"] is True
    adversarial = json.loads((root / "adversarial_results.json").read_text(encoding="utf-8"))
    assert adversarial["status"] == "not_run_main_invalid"


def test_raw_provider_rejects_a_second_main_call(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / RAW_NAME).write_text("{}", encoding="utf-8")
    provider = RawFirstOpenAIResponsesProvider(
        api_key="sk-test-not-real-mission73", artifact_root=root,
        opener=lambda *args, **kwargs: pytest.fail("network should not run"),
    )
    request = LLMRequest(
        input_payload={}, system_prompt="system", explanation_prompt="explain", model="gpt-5.5",
    )
    with pytest.raises(Exception, match="nova chamada principal proibida"):
        provider.generate_raw(request)
    assert provider.call_count == 0
