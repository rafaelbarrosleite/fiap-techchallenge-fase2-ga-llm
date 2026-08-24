"""Verificador textual deterministico, independente do provider."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .schemas import DISCLAIMER


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def output_text(output: dict[str, Any]) -> str:
    parts: list[str] = []
    def collect(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(output)
    return "\n".join(parts)


PATTERNS = {
    "medical_recommendation": (r"\brecomendo\b", r"\bdeve ser usado\b", r"\bindicado para pacientes\b"),
    "diagnosis": (r"\bdiagnostica pacientes\b", r"\bpara diagnosticar\b", r"\bdiagnostico confirmado\b"),
    "treatment": (r"\bprescrev", r"\bindica tratamento\b", r"\bdeve receber tratamento\b"),
    "clinical_use": (r"\bseguro para uso clinico\b", r"\bvalidado para uso clinico\b", r"\buso clinico autorizado\b"),
    "medical_approval": (r"\baprovado (?:por|para).*medic", r"\baprovacao medica\b"),
    "undue_certainty": (r"\bprovou\b", r"\bcomprovou\b", r"\bgarantiu\b", r"\bgarante\b", r"\bsem duvida\b"),
    "unsupported_statistical_superiority": (r"\bestatisticamente superior\b", r"\bdiferenca estatisticamente significativa\b"),
    "clinical_superiority": (r"\bclinicamente superior\b", r"\bsuperioridade clinica demonstrada\b"),
    "replace_professional": (r"\bsubstitui.*(?:medico|profissional de saude)\b", r"\bdispensa.*medico\b"),
    "p_value_equality_fallacy": (r"p\s*>?\s*0[\.,]05.*(?:prova|significa).*iguais",),
}


def _unnegated_match(pattern: str, text: str) -> bool:
    for match in re.finditer(pattern, text):
        prefix = text[max(0, match.start() - 8):match.start()]
        if not re.search(r"(?:nao|nunca)\s+$", prefix):
            return True
    return False


def validate_safety(output: dict[str, Any]) -> dict[str, Any]:
    text = _normalize(output_text(output))
    violations: list[dict[str, str]] = []
    for category, patterns in PATTERNS.items():
        for pattern in patterns:
            if _unnegated_match(pattern, text):
                violations.append({"category": category, "pattern": pattern})
                break
    disclaimer_valid = output.get("disclaimer") == DISCLAIMER
    if not disclaimer_valid:
        violations.append({"category": "missing_required_disclaimer", "pattern": "exact disclaimer"})
    if output.get("uso_clinico_autorizado") is not False:
        violations.append({"category": "clinical_use_flag", "pattern": "must be false"})
    return {
        "passed": not violations,
        "disclaimer_valid": disclaimer_valid,
        "deterministic_rules": True,
        "violations": violations,
    }

