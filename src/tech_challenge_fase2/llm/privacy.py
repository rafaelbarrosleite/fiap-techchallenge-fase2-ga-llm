"""Barreira deterministica contra dados individuais e instrucoes clinicas."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .schemas import validate_input


class PrivacyError(ValueError):
    """A entrada contem estrutura ou finalidade proibida."""


FORBIDDEN_KEY_PARTS = {
    "patient", "paciente", "diagnosis", "diagnostico", "feature", "prediction",
    "predicao", "probability", "probabilidade", "individual_index", "dataset_index",
    "record", "registro_individual", "row_data", "y_true", "y_pred", "sample_id",
}
INDIVIDUAL_ARRAY_KEYS = {"rows", "records", "patients", "samples", "observations", "features"}


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def _scan(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _normalize(str(raw_key))
            safe_negative_flag = key in {"diagnosis_allowed", "individual_data_included"} and child is False
            if not safe_negative_flag and any(part in key for part in FORBIDDEN_KEY_PARTS):
                raise PrivacyError(f"Campo proibido em {path}.{raw_key}.")
            if key in INDIVIDUAL_ARRAY_KEYS and isinstance(child, list):
                raise PrivacyError(f"Colecao de registros individuais proibida em {path}.{raw_key}.")
            _scan(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan(child, f"{path}[{index}]")


def validate_sanitized_input(payload: Any) -> dict[str, Any]:
    validated = validate_input(payload)
    _scan(validated)
    return validated


CLINICAL_INSTRUCTION_PATTERNS = (
    r"diagnostic(?:ar|o)", r"qual modelo.*pacient", r"tratamento", r"prescrev",
    r"recomend(?:e|ar).*medic", r"uso clinico", r"substitu(?:ir|a).*medic",
)


def validate_user_instruction(instruction: str | None) -> None:
    if instruction is None:
        return
    normalized = _normalize(instruction).replace("_", " ")
    if any(re.search(pattern, normalized) for pattern in CLINICAL_INSTRUCTION_PATTERNS):
        raise PrivacyError("Instrucao clinica adversarial rejeitada antes do provider.")
    raise PrivacyError("Instrucoes livres nao fazem parte do contrato desta missao.")
