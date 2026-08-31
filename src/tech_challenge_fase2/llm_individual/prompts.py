"""Carregamento versionado dos prompts individuais."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

PROMPT_ROOT = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class IndividualPromptBundle:
    system_version: str
    explanation_version: str
    system_text: str
    explanation_text: str
    system_sha256: str
    explanation_sha256: str


def _load(filename: str, version: str) -> tuple[str, str]:
    text = (PROMPT_ROOT / filename).read_text(encoding="utf-8")
    if not text.startswith(f"PROMPT_VERSION: {version}\n"):
        raise ValueError(f"Versao de prompt invalida: {filename}.")
    required = ("PURPOSE:", "INPUT_CONTRACT:", "OUTPUT_CONTRACT:", "SEGURANCA", "FACTUALIDADE", "CONTEXTO MEDICO")
    if any(token not in text for token in required):
        raise ValueError(f"Prompt individual incompleto: {filename}.")
    return text, sha256(text.encode("utf-8")).hexdigest()


def load_prompts() -> IndividualPromptBundle:
    system, system_hash = _load("system_individual_v1.txt", "system_individual_v1")
    explanation, explanation_hash = _load("explanation_individual_v1.txt", "explanation_individual_v1")
    return IndividualPromptBundle(
        system_version="system_individual_v1",
        explanation_version="explanation_individual_v1",
        system_text=system,
        explanation_text=explanation,
        system_sha256=system_hash,
        explanation_sha256=explanation_hash,
    )
