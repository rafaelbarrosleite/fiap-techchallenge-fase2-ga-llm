"""Selecao explicita de contrato; nenhuma migracao silenciosa entre V1 e V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ContractComponents:
    contract_version: str
    build_input: Callable[[], dict[str, Any]]
    validate_input: Callable[[Any], dict[str, Any]]
    validate_output: Callable[[Any], dict[str, Any]]
    output_json_schema: Callable[[], dict[str, Any]]
    load_prompts: Callable[[], Any]
    fake_provider_factory: Callable[[], Any]


def load_contract(contract_version: str) -> ContractComponents:
    if contract_version == "v1":
        from tech_challenge_fase2.llm.input_builder import build_llm_input
        from tech_challenge_fase2.llm.prompts import load_prompt_bundle
        from tech_challenge_fase2.llm.providers import FakeLLMProvider
        from tech_challenge_fase2.llm.schemas import output_json_schema, validate_input, validate_output
        return ContractComponents("v1", build_llm_input, validate_input, validate_output, output_json_schema, load_prompt_bundle, FakeLLMProvider)
    if contract_version == "v2":
        from tech_challenge_fase2.llm_v2.input_builder import build_llm_input_v2
        from tech_challenge_fase2.llm_v2.prompts import load_prompt_bundle_v2
        from tech_challenge_fase2.llm_v2.providers import FakeLLMProviderV2
        from tech_challenge_fase2.llm_v2.schemas import output_json_schema_v2, validate_input_v2, validate_output_v2
        return ContractComponents("v2", build_llm_input_v2, validate_input_v2, validate_output_v2, output_json_schema_v2, load_prompt_bundle_v2, FakeLLMProviderV2)
    raise ValueError("contract_version deve ser selecionado explicitamente como 'v1' ou 'v2'.")
