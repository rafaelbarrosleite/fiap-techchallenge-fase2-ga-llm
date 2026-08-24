"""Carregamento centralizado e auditavel de prompts versionados."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

PROMPT_ROOT = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class PromptBundle:
    system_version: str
    explanation_version: str
    system_text: str
    explanation_text: str
    system_sha256: str
    explanation_sha256: str


def _load(name: str, expected_version: str) -> tuple[str, str]:
    path = PROMPT_ROOT / name
    text = path.read_text(encoding="utf-8")
    if not text.startswith(f"PROMPT_VERSION: {expected_version}\n"):
        raise ValueError(f"Cabecalho de versao invalido em {path}.")
    required = ("PURPOSE:", "INPUT_CONTRACT:", "OUTPUT_CONTRACT:", "SEGURANCA", "FACTUALIDADE")
    if any(token not in text for token in required):
        raise ValueError(f"Prompt incompleto: {path}.")
    return text, sha256(text.encode("utf-8")).hexdigest()


def load_prompt_bundle() -> PromptBundle:
    system, system_hash = _load("system_v1.txt", "system_v1")
    explanation, explanation_hash = _load("explanation_v1.txt", "explanation_v1")
    return PromptBundle(
        system_version="system_v1", explanation_version="explanation_v1",
        system_text=system, explanation_text=explanation,
        system_sha256=system_hash, explanation_sha256=explanation_hash,
    )
