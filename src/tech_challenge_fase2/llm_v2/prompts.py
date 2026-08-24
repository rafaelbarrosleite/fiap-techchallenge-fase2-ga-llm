"""Carregamento exclusivo dos prompts V2; V1 permanece intocado."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "llm" / "prompts"


@dataclass(frozen=True)
class PromptBundleV2:
    system_version: str
    explanation_version: str
    system_text: str
    explanation_text: str
    system_sha256: str
    explanation_sha256: str


def _load(name: str, version: str) -> tuple[str, str]:
    path = PROMPT_ROOT / name
    text = path.read_text(encoding="utf-8")
    if not text.startswith(f"PROMPT_VERSION: {version}\n"):
        raise ValueError(f"Cabecalho V2 invalido em {path}.")
    required = (
        "PURPOSE:", "INPUT_CONTRACT:", "OUTPUT_CONTRACT:", "REGRAS DE SEGURANCA",
        "REGRAS DE FACTUALIDADE", "comparison_id", "left_method", "right_method",
        "nao infira estatistica ausente", "holdout nao reabre selecao",
    )
    normalized = text.lower()
    if any(token.lower() not in normalized for token in required):
        raise ValueError(f"Prompt V2 incompleto: {path}.")
    return text, sha256(text.encode("utf-8")).hexdigest()


def load_prompt_bundle_v2() -> PromptBundleV2:
    system, system_hash = _load("system_v2.txt", "system_v2")
    explanation, explanation_hash = _load("explanation_v2.txt", "explanation_v2")
    return PromptBundleV2(
        system_version="system_v2", explanation_version="explanation_v2",
        system_text=system, explanation_text=explanation,
        system_sha256=system_hash, explanation_sha256=explanation_hash,
    )
