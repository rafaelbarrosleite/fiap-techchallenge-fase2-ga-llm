"""Privacy gate especifico para uma explicacao individual desidentificada."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .schemas import validate_input


class IndividualPrivacyError(ValueError):
    """A entrada permitiria identificar ou reconstruir um registro."""


FORBIDDEN_KEYS = {
    "id", "patient_id", "paciente_id", "dataset_index", "original_index", "row_index",
    "record_id", "sample_id", "name", "nome", "birth_date", "date_of_birth",
    "ground_truth", "diagnosis", "diagnostico", "y_true", "raw_features", "feature_values",
    "source_row", "original_row", "final_predictions",
}


def _normalize(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9_]+", "_", plain.lower()).strip("_")


def _scan(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _normalize(str(raw_key))
            if key in FORBIDDEN_KEYS:
                raise IndividualPrivacyError(f"Campo identificavel ou reconstruivel proibido em {path}.{raw_key}.")
            _scan(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan(child, f"{path}[{index}]")


def validate_privacy(payload: Any) -> dict[str, Any]:
    validated = validate_input(payload)
    _scan(validated)
    case, provenance = validated["case_context"], validated["source_provenance"]
    if any((
        case["source_record_reconstructible"], case["raw_feature_values_included"],
        case["ground_truth_included"], provenance["patient_identifiers_included"],
        provenance["original_row_index_included"], provenance["test_or_holdout_case_used"],
    )):
        raise IndividualPrivacyError("Protecoes de privacidade individual foram violadas.")
    return validated
