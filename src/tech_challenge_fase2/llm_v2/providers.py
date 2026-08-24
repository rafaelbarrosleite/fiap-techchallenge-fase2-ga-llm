"""Provider fake V2 deterministico e completamente offline."""

from __future__ import annotations

from typing import Any

from tech_challenge_fase2.llm.providers import LLMRequest, LLMResponse
from tech_challenge_fase2.llm.schemas import DISCLAIMER

from .schemas import CONTRACT_VERSION_V2, SCHEMA_VERSION_V2, validate_input_v2, validate_output_v2


def _tradeoff(pair: dict[str, Any]) -> bool:
    directions = {
        pair["recall_relation"], pair["f1_relation"], pair["roc_auc_relation"],
    }.difference({"equal"})
    return "left_higher" in directions and "right_higher" in directions


def build_deterministic_output_v2(payload: dict[str, Any]) -> dict[str, Any]:
    validate_input_v2(payload)
    findings = []
    for pair in payload["comparison_pairs"]:
        relation_text = (
            f"No par {pair['comparison_id']}, a comparacao usa exclusivamente "
            f"{pair['left_method']} como left_method e {pair['right_method']} como right_method. "
            "As relacoes descrevem o holdout agregado e nao estabelecem causalidade ou relevancia clinica."
        )
        findings.append({**pair, "tradeoff_present": _tradeoff(pair), "interpretation": relation_text})
    uncertainty = []
    for item in payload["uncertainty_comparisons"]:
        count = item["mcnemar"]["discordant_total"]
        interpretation = (
            f"No par {item['comparison_id']}, McNemar possui {count} pares discordantes agregados. "
            "A baixa contagem limita a inferencia; o valor-p nao prova igualdade nem relevancia clinica."
            if item["mcnemar"]["low_count_warning"]
            else f"No par {item['comparison_id']}, a interpretacao usa somente a evidencia agregada fornecida."
        )
        uncertainty.append({
            **item,
            "limited_by_few_discordances": item["mcnemar"]["low_count_warning"],
            "interpretation": interpretation,
        })
    selected = payload["selected_model"]
    output = {
        "schema_version": SCHEMA_VERSION_V2, "contract_version": CONTRACT_VERSION_V2,
        "resumo_executivo": (
            "Os resultados agregados sugerem ganhos observados em alguns pares e trade-offs em outros. "
            "Cada conclusao esta vinculada a um comparison_id explicito; nenhuma relacao foi transferida entre pares. "
            "A leitura e academica, descritiva e nao representa validacao clinica."
        ),
        "modelo_selecionado": {
            **selected,
            "explanation": (
                "A escolha foi congelada por validacao cruzada antes do holdout; "
                "o teste confirmatorio nao reabriu a selecao."
            ),
        },
        "model_results": payload["model_results"], "comparison_findings": findings,
        "interpretacao_comparacoes": (
            "Baseline, algoritmo genetico e busca aleatoria foram comparados somente pelos nove pares nomeados. "
            "Mesma matriz de confusao e ROC-AUC diferente sao propriedades do par identificado, nao da familia isoladamente."
        ),
        "incerteza_estatistica": (
            "Os intervalos dos deltas de recall incluem zero, portanto nao ha evidencia suficiente para afirmar superioridade estatistica. "
            "As contagens agregadas de discordancias de McNemar sao baixas e limitam a inferencia; valor-p alto nao prova igualdade."
        ),
        "uncertainty_findings": uncertainty, "limitacoes": list(payload["limitations"]),
        "conclusao": (
            "Foi observado desempenho confirmatorio que deve ser interpretado com os pares e intervalos fornecidos. "
            "O estudo tem carater academico e experimental, nao representa validacao clinica e nao autoriza uso medico."
        ),
        "disclaimer": DISCLAIMER, "holdout_nao_reabriu_selecao": True,
        "uso_clinico_autorizado": False,
    }
    validate_output_v2(output)
    return output


class FakeLLMProviderV2:
    name = "fake_v2"
    contract_version = CONTRACT_VERSION_V2

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            output=build_deterministic_output_v2(request.input_payload),
            provider=self.name, model=request.model,
            response_id="offline-deterministic-v2", usage={"paid_tokens": 0, "external_calls": 0},
        )
