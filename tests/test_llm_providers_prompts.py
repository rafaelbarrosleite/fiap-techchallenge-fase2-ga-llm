import json
from pathlib import Path

import pytest

from tech_challenge_fase2.llm.input_builder import build_llm_input
from tech_challenge_fase2.llm.prompts import load_prompt_bundle
from tech_challenge_fase2.llm.providers import (
    FakeLLMProvider, LLMRequest, OpenAIResponsesProvider, load_env_value,
)


def _request():
    prompts = load_prompt_bundle()
    return LLMRequest(
        input_payload=build_llm_input(), system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model="offline-test",
    )


def test_prompts_are_versioned_complete_and_centralized() -> None:
    prompts = load_prompt_bundle()
    assert prompts.system_version == "system_v1"
    assert prompts.explanation_version == "explanation_v1"
    assert len(prompts.system_sha256) == len(prompts.explanation_sha256) == 64
    assert "Nao diagnostique" in prompts.system_text
    assert "JSON estrito" in prompts.explanation_text


def test_fake_provider_is_deterministic_and_offline(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    provider = FakeLLMProvider()
    first = provider.generate(_request())
    second = provider.generate(_request())
    assert first.output == second.output
    assert first.usage == {"paid_tokens": 0}
    assert provider.call_count == 2


def test_generic_env_value_is_not_accepted_as_real_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=replace_with_your_openai_api_key\n", encoding="utf-8")
    assert load_env_value("OPENAI_API_KEY", env) == "replace_with_your_openai_api_key"
    with pytest.raises(ValueError, match="preencha"):
        OpenAIResponsesProvider(env_file=env)


def test_real_provider_request_uses_structured_output_and_store_false(monkeypatch) -> None:
    request = _request()
    fake_output = FakeLLMProvider().generate(request).output
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                "id": "resp-test", "output": [{"content": [{"type": "output_text", "text": json.dumps(fake_output)}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }).encode()

    def fake_urlopen(http_request, timeout):
        captured["body"] = json.loads(http_request.data)
        captured["authorization"] = http_request.headers["Authorization"]
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAIResponsesProvider(api_key="sk-test-not-real")
    response = provider.generate(request)
    assert response.output == fake_output
    assert captured["body"]["store"] is False
    assert captured["body"]["text"]["format"]["type"] == "json_schema"
    assert captured["body"]["text"]["format"]["strict"] is True
    assert "patient_id" not in json.dumps(captured["body"])

