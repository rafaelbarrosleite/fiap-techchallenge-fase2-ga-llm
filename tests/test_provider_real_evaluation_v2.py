import io
import json
import urllib.error
from pathlib import Path

import pytest

from tech_challenge_fase2.llm.providers import LLMRequest, build_deterministic_output
from tech_challenge_fase2.provider_real_evaluation import RawProviderResponse
from tech_challenge_fase2.provider_real_evaluation_v2 import (
    Mission72Error,
    finalize_v2_manifest,
    prepare_v2,
    run_adversarial_v2,
    run_scientific_v2,
    run_technical_probe,
    validate_v2,
)


def _env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=sk-test-not-real-v2\nOPENAI_MODEL=gpt-5.5\n", encoding="utf-8")
    return path


class FakeHTTPResponse:
    status = 200

    def __init__(self) -> None:
        self.payload = {
            "id": "resp-probe-offline", "model": "gpt-5.5", "status": "completed",
            "store": False,
            "output": [{
                "type": "message", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": '{"status":"ok"}'}],
            }],
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class OfflineProvider:
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


def test_v2_preflight_is_offline_private_and_omits_temperature(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    root = tmp_path / "v2"
    result = prepare_v2(artifact_root=root, env_file=_env(tmp_path))
    assert result["passed"] is True
    assert result["configuration"]["temperature_sent"] is False
    assert result["privacy_valid"] is True
    assert result["api_calls_performed"] == 0
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
    assert "sk-test-not-real-v2" not in persisted


def test_corrected_probe_calls_once_without_temperature(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    requests = []

    def opener(request, timeout):
        requests.append(json.loads(request.data.decode()))
        return FakeHTTPResponse()

    result = run_technical_probe(artifact_root=root, env_file=_env(tmp_path), opener=opener)
    assert result["status"] == "approved"
    assert result["provider_calls"] == 1
    assert result["retries"] == 0
    assert result["structured_output"] == {"status": "ok"}
    assert len(requests) == 1
    assert "temperature" not in requests[0]


def test_failed_probe_blocks_scientific_call_without_retry(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    error_body = io.BytesIO(json.dumps({
        "error": {"type": "invalid_request_error", "code": "bad", "param": "text.format", "message": "bad"}
    }).encode())
    error = urllib.error.HTTPError(
        "https://api.openai.com/v1/responses", 400, "Bad Request", {"x-request-id": "req-fail"}, error_body,
    )
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        raise error

    probe = run_technical_probe(artifact_root=root, env_file=_env(tmp_path), opener=opener)
    assert probe["status"] == "rejected"
    provider = OfflineProvider()
    with pytest.raises(Mission72Error, match="bloqueada"):
        run_scientific_v2(artifact_root=root, env_file=_env(tmp_path), provider=provider)
    assert len(calls) == 1
    assert provider.call_count == 0


def test_probe_without_output_text_is_persisted_as_invalid(tmp_path: Path) -> None:
    root = tmp_path / "v2"

    class MissingOutputResponse(FakeHTTPResponse):
        def __init__(self) -> None:
            super().__init__()
            self.payload["output"] = [{"type": "reasoning"}]

    probe = run_technical_probe(
        artifact_root=root, env_file=_env(tmp_path),
        opener=lambda *args, **kwargs: MissingOutputResponse(),
    )
    assert probe["status"] == "invalid"
    assert probe["schema_valid"] is False
    assert probe["sanitized_local_error"]["class"] == "ProviderRealEvaluationError"
    assert probe["provider_calls"] == 1
    assert probe["retries"] == 0


def test_complete_v2_flow_offline_respects_five_call_budget(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    run_technical_probe(artifact_root=root, env_file=_env(tmp_path), opener=lambda *args, **kwargs: FakeHTTPResponse())
    main_provider = OfflineProvider()
    main = run_scientific_v2(artifact_root=root, env_file=_env(tmp_path), provider=main_provider)
    assert main["approved"] is True
    assert main_provider.call_count == 1
    adversarial_provider = OfflineProvider()
    adversarial = run_adversarial_v2(
        artifact_root=root, env_file=_env(tmp_path), provider=adversarial_provider,
    )
    assert adversarial["status"] == "approved"
    assert adversarial_provider.call_count == 3
    manifest = finalize_v2_manifest(root)
    assert manifest["call_budget"] == {
        "technical": 1, "scientific_main": 1, "adversarial": 3,
        "total": 5, "maximum": 5, "automatic_retries": 0,
    }
    assert validate_v2(root)["passed"] is True
