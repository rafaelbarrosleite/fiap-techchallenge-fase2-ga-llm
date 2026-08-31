"""Barreira de privacidade do contrato V2.

O schema V2 e validado antes da varredura recursiva ja consolidada na Missao 5.
Isso evita adaptar o contrato V1 ou duplicar a lista de campos proibidos.
"""

from __future__ import annotations

from typing import Any

from tech_challenge_fase2.llm.privacy import (
    FORBIDDEN_KEY_PARTS,
    INDIVIDUAL_ARRAY_KEYS,
    PrivacyError,
    _normalize,
)

from .schemas import validate_input_v2


def validate_sanitized_input_v2(payload: Any) -> dict[str, Any]:
    """Rejeita estrutura fora do schema e qualquer indicio de dado individual."""

    validated = validate_input_v2(payload)
    _scan_v2(validated)
    return validated


def _scan_v2(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _normalize(str(raw_key))
            safe_negative_flags = {
                "diagnosis_allowed", "individual_data_included",
                "individual_predictions_used",
            }
            safe_negative = key in safe_negative_flags and child is False
            if not safe_negative and any(part in key for part in FORBIDDEN_KEY_PARTS):
                raise PrivacyError(f"Campo proibido em {path}.{raw_key}.")
            if key in INDIVIDUAL_ARRAY_KEYS and isinstance(child, list):
                raise PrivacyError(f"Colecao de registros individuais proibida em {path}.{raw_key}.")
            _scan_v2(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_v2(child, f"{path}[{index}]")
