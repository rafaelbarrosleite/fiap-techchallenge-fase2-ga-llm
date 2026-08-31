import io
import json
import urllib.error
from pathlib import Path

import pytest

from tech_challenge_fase2.genetic.serialization import stable_sha256
from tech_challenge_fase2.llm.schemas import output_json_schema
from tech_challenge_fase2._historical.openai_integration_diagnosis import (
    inspect_schema,
    minimal_request_body,
    minimal_schema,
    prepare_diagnosis,
)
from tech_challenge_fase2._historical.provider_real_evaluation import (
    AuditedOpenAIResponsesProvider,
    ProviderCallError,
)


def _env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=sk-test-not-real-diagnosis\nOPENAI_MODEL=gpt-5.5\n", encoding="utf-8")
    return path


def test_provider_schema_preserves_local_contract_without_speculative_change() -> None:
    local = output_json_schema()
    local_report = inspect_schema(local)
    assert len(local_report["keyword_paths"]["const"]) == 4
    assert local_report["valid_core_structure"] is True


def test_minimal_request_is_strict_private_and_stable() -> None:
    body = minimal_request_body("gpt-5.5")
    assert body["store"] is False
    assert "temperature" not in body
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert inspect_schema(minimal_schema())["valid_core_structure"] is True
    assert stable_sha256(body) == stable_sha256(minimal_request_body("gpt-5.5"))
    serialized = json.dumps(body).lower()
    assert "patient" not in serialized
    assert "diagnos" not in serialized
    assert "recall" not in serialized
    assert "probability" not in serialized


def test_dry_run_never_calls_network_and_never_persists_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used by dry-run"))
    root = tmp_path / "diagnosis"
    result = prepare_diagnosis(artifact_root=root, env_file=_env(tmp_path))
    assert result["api_call_performed"] is False
    assert result["privacy_valid"] is True
    assert result["original_request_safe_summary"]["content_persisted_in_diagnosis"] is False
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
    assert "sk-test-not-real-diagnosis" not in persisted
    assert "AGGREGATED_EXPERIMENT_INPUT" not in persisted


def test_http_error_capture_preserves_sanitized_fields_without_secret(monkeypatch) -> None:
    body = json.dumps({
        "error": {
            "type": "invalid_request_error", "code": "invalid_json_schema",
            "param": "text.format.schema", "message": "Unsupported keyword: const",
        }
    }).encode()
    http_error = urllib.error.HTTPError(
        "https://api.openai.com/v1/responses", 400, "Bad Request",
        {"x-request-id": "req-safe", "authorization": "Bearer secret-must-not-leak"},
        io.BytesIO(body),
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(http_error))
    provider = AuditedOpenAIResponsesProvider(api_key="sk-test-not-real-diagnosis")
    request_body = minimal_request_body("gpt-5.5")
    from tech_challenge_fase2.llm.providers import LLMRequest
    request = LLMRequest(
        input_payload={}, system_prompt="system", explanation_prompt="explain", model="gpt-5.5",
    )
    monkeypatch.setattr(provider, "request_body", lambda *args, **kwargs: request_body)
    with pytest.raises(ProviderCallError) as captured:
        provider.generate_raw(request)
    details = captured.value.sanitized_details()
    assert details == {
        "http_status": 400,
        "error": {
            "type": "invalid_request_error", "code": "invalid_json_schema",
            "param": "text.format.schema", "message": "Unsupported keyword: const",
        },
        "request_id": "req-safe", "exception_class": "HTTPError",
    }
    assert "secret-must-not-leak" not in json.dumps(details)
