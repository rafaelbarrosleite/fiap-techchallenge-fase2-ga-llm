import json
from pathlib import Path

import pytest

from tech_challenge_fase2._historical.openai_response_parsing_diagnosis import run_parsing_probe
from tech_challenge_fase2.responses_parsing import (
    ResponseContentError,
    ResponseStateError,
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    extract_response_text,
    parse_minimal_structured_output,
    response_structure,
)


def _message(content: dict) -> dict:
    return {"type": "message", "status": "completed", "role": "assistant", "content": [content]}


def _completed(output: list) -> dict:
    return {"id": "resp-test", "model": "gpt-5.5", "status": "completed", "output": output}


def test_case_a_extracts_message_output_text() -> None:
    payload = _completed([_message({"type": "output_text", "text": '{"status":"ok"}'})])
    parsed, extracted = parse_minimal_structured_output(payload)
    assert parsed == {"status": "ok"}
    assert extracted.source == "output.message.content.output_text"


def test_case_b_ignores_reasoning_before_message() -> None:
    payload = _completed([
        {"type": "reasoning", "id": "rs-test", "summary": []},
        _message({"type": "output_text", "text": '{"status":"ok"}'}),
    ])
    assert extract_response_text(payload).text == '{"status":"ok"}'
    structure = response_structure(payload)
    assert structure["output_item_types"] == ["reasoning", "message"]


def test_case_c_refusal_does_not_invent_output_text() -> None:
    payload = _completed([_message({"type": "refusal", "refusal": "no"})])
    with pytest.raises(ResponseContentError, match="refusal"):
        extract_response_text(payload)
    assert response_structure(payload)["refusal_count"] == 1


def test_case_d_incomplete_preserves_details() -> None:
    payload = {
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning"}],
    }
    with pytest.raises(ResponseStateError) as captured:
        extract_response_text(payload)
    assert captured.value.status == "incomplete"
    assert captured.value.incomplete_details == {"reason": "max_output_tokens"}


def test_case_e_completed_empty_output_fails_explicitly() -> None:
    with pytest.raises(ResponseContentError, match="empty output"):
        extract_response_text(_completed([]))


def test_case_f_invalid_json_fails_structured_parse() -> None:
    payload = _completed([_message({"type": "output_text", "text": "not json"})])
    with pytest.raises(StructuredOutputParseError):
        parse_minimal_structured_output(payload)


def test_case_g_valid_json_with_wrong_schema_fails() -> None:
    payload = _completed([_message({"type": "output_text", "text": '{"status":"wrong"}'})])
    with pytest.raises(StructuredOutputSchemaError):
        parse_minimal_structured_output(payload)


def test_raw_response_and_usage_are_persisted_before_incomplete_parsing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-not-real-parsing\nOPENAI_MODEL=gpt-5.5\n", encoding="utf-8")
    payload = {
        "id": "resp-incomplete", "model": "gpt-5.5", "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning", "id": "rs-test", "summary": []}],
        "usage": {
            "input_tokens": 20, "output_tokens": 100, "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 100},
        },
        "store": False,
    }

    class Response:
        status = 200
        headers = {"x-request-id": "req-incomplete"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    root = tmp_path / "parsing"
    probe = run_parsing_probe(artifact_root=root, env_file=env, opener=lambda *args, **kwargs: Response())
    assert probe["status"] == "invalid"
    assert probe["sanitized_parsing_error"]["class"] == "ResponseStateError"
    assert (root / "raw_response_sanitized.json").is_file()
    usage = json.loads((root / "provider_usage.json").read_text(encoding="utf-8"))
    assert usage["reasoning_tokens"] == 100
    cause = json.loads((root / "root_cause_report.json").read_text(encoding="utf-8"))
    assert cause["classification"] == "output_limit"
