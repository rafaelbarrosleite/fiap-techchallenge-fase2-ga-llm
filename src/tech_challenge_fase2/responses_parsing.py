"""Parsing defensivo do JSON HTTP bruto retornado pela Responses API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

RESPONSE_STATES = {"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"}


class ResponsesParsingError(ValueError):
    """Base para falhas explícitas de estado, conteúdo ou contrato."""


class ResponseStateError(ResponsesParsingError):
    def __init__(self, status: str | None, incomplete_details: Any = None) -> None:
        super().__init__(f"Response status is {status!r}, not 'completed'.")
        self.status = status
        self.incomplete_details = incomplete_details


class ResponseContentError(ResponsesParsingError):
    """Resposta completed sem texto final extraível."""


class StructuredOutputParseError(ResponsesParsingError):
    """Texto final não é JSON válido."""


class StructuredOutputSchemaError(ResponsesParsingError):
    """JSON válido não cumpre o schema mínimo."""


@dataclass(frozen=True)
class ExtractedText:
    text: str
    source: str
    text_count: int


def response_structure(payload: dict[str, Any]) -> dict[str, Any]:
    """Retorna somente metadados estruturais; nunca expõe reasoning interno."""

    output = payload.get("output")
    items = output if isinstance(output, list) else []
    item_types: list[str | None] = []
    content_types: list[dict[str, Any]] = []
    message_count = 0
    refusal_count = 0
    output_text_count = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            item_types.append(None)
            continue
        item_type = item.get("type")
        item_types.append(item_type)
        if item_type != "message":
            continue
        message_count += 1
        contents = item.get("content") if isinstance(item.get("content"), list) else []
        types = []
        for content in contents:
            content_type = content.get("type") if isinstance(content, dict) else None
            types.append(content_type)
            if content_type == "output_text":
                output_text_count += 1
            elif content_type == "refusal":
                refusal_count += 1
        content_types.append({"output_index": index, "types": types})
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "response_status": payload.get("status"),
        "known_status": payload.get("status") in RESPONSE_STATES,
        "incomplete_details": payload.get("incomplete_details"),
        "output_is_array": isinstance(output, list),
        "output_item_types": item_types,
        "message_count": message_count,
        "content_types_by_message": content_types,
        "output_text_count": output_text_count,
        "refusal_count": refusal_count,
        "top_level_output_text_present": isinstance(payload.get("output_text"), str),
        "usage_present": bool(usage),
    }


def extract_response_text(payload: dict[str, Any]) -> ExtractedText:
    """Extrai texto por tipo sem assumir posição fixa no array ``output``."""

    status = payload.get("status")
    if status != "completed":
        raise ResponseStateError(status, payload.get("incomplete_details"))
    texts: list[str] = []
    output = payload.get("output")
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        contents = item.get("content")
        for content in contents if isinstance(contents, list) else []:
            if not isinstance(content, dict) or content.get("type") != "output_text":
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    if texts:
        return ExtractedText(text="\n".join(texts), source="output.message.content.output_text", text_count=len(texts))
    top_level = payload.get("output_text")
    if isinstance(top_level, str) and top_level:
        return ExtractedText(text=top_level, source="top_level_output_text_fallback", text_count=1)
    structure = response_structure(payload)
    if structure["refusal_count"]:
        raise ResponseContentError("Completed response contains refusal content and no output_text.")
    if not structure["output_item_types"]:
        raise ResponseContentError("Completed response has an empty output array.")
    raise ResponseContentError(
        "Completed response contains no output_text in message content or top-level fallback."
    )


def parse_minimal_structured_output(payload: dict[str, Any]) -> tuple[dict[str, Any], ExtractedText]:
    extracted = extract_response_text(payload)
    try:
        parsed = json.loads(extracted.text)
    except json.JSONDecodeError as error:
        raise StructuredOutputParseError("Extracted output_text is not valid JSON.") from error
    if not isinstance(parsed, dict) or set(parsed) != {"status"} or parsed.get("status") != "ok":
        raise StructuredOutputSchemaError("Structured output must be exactly {'status': 'ok'}.")
    return parsed, extracted
