"""Providers offline e OpenAI para explicacoes individuais estruturadas."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tech_challenge_fase2.llm.providers import LLMRequest, LLMResponse, load_env_value
from tech_challenge_fase2.llm.schemas import DISCLAIMER
from tech_challenge_fase2.responses_parsing import extract_response_text

from .schemas import output_json_schema, validate_output


def build_deterministic_output(payload: dict[str, Any]) -> dict[str, Any]:
    case = payload["case_context"]
    factors = []
    band_labels = {"low": "baixa", "typical": "intermediária", "high": "elevada"}
    direction_labels = {
        "toward_benign": "no sentido da classe benigna",
        "toward_malignant": "no sentido da classe maligna",
    }
    for signal in payload["explanation_signals"]:
        factors.append({
            **signal,
            "explanation": (
                f"{signal['display_name'].capitalize()} ficou na faixa {band_labels[signal['observed_band']]} "
                f"e contribuiu {direction_labels[signal['influence_direction']]} na decisão matemática. "
                "Essa associação não demonstra causa biológica."
            ),
        })
    pattern = "maligno" if case["predicted_pattern"] == "malignant_pattern" else "benigno"
    probability_percent = round(case["probability_malignant"] * 100.0, 4)
    return {
        "schema_version": "3.0",
        "contract_version": "individual_v1",
        "case_reference": case["case_reference"],
        "resumo_executivo": (
            f"O modelo classificou o caso demonstrativo desidentificado como padrão {pattern}, "
            f"com probabilidade estimada de {probability_percent}% para a classe maligna. "
            "Essa probabilidade descreve a saída matemática do modelo e não constitui diagnóstico."
        ),
        "classificacao_do_modelo": {
            "candidate_id": payload["model_context"]["candidate_id"],
            "predicted_pattern": case["predicted_pattern"],
            "probability_malignant": case["probability_malignant"],
            "classification_threshold": case["classification_threshold"],
            "interpretation": (
                "A classe foi definida pela comparação da probabilidade estimada com o limiar fixo de 0,5. "
                "O resultado requer revisão humana e não informa o diagnóstico real do caso."
            ),
        },
        "fatores_explicativos": factors,
        "insights_acionaveis_para_medicos": [
            {
                "action": "Revisar a coerência dos cinco sinais que mais influenciaram a classificação.",
                "rationale": "Os sinais mostram como o modelo chegou à saída e ajudam a identificar dependência excessiva de uma variável.",
                "scope": "human_review_only", "patient_care_decision": False,
            },
            {
                "action": "Confrontar a saída com avaliação profissional e evidências clínicas independentes.",
                "rationale": "A classificação isolada não foi validada para orientar diagnóstico, tratamento ou decisão médica.",
                "scope": "human_review_only", "patient_care_decision": False,
            },
        ],
        "limitacoes": [
            "O provider recebeu apenas uma representação parcial e desidentificada, sem valores brutos nem diagnóstico real.",
            "As contribuições explicam a decisão matemática do modelo, não relações causais ou mecanismos biológicos.",
            "O modelo foi avaliado academicamente e não possui validação clínica, prospectiva ou regulatória.",
        ],
        "preparacao_modulo3": {
            "ready_for_future_text": True,
            "current_text_data_used": False,
            "explanation": (
                "O contrato reserva campos para futuros resumos textuais desidentificados de notas e laudos, "
                "sempre sujeitos a autorização, proveniência, validação de schema e revisão de segurança."
            ),
        },
        "conclusao": (
            "A explicação torna auditável uma classificação individual do modelo e oferece ações de revisão humana. "
            "Ela não confirma doença, não prescreve conduta e não autoriza utilização clínica."
        ),
        "disclaimer": DISCLAIMER,
        "predicao_nao_e_diagnostico": True,
        "uso_clinico_autorizado": False,
    }


class FakeIndividualProvider:
    name = "fake_individual"

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        output = build_deterministic_output(request.input_payload)
        validate_output(output)
        return LLMResponse(
            output=output, provider=self.name, model=request.model,
            response_id="offline-individual-v1",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "external_calls": 0},
        )


class OpenAIIndividualProvider:
    """Responses API sem temperature, sem retry e com store=false."""

    name = "openai_responses"
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, *, api_key: str | None = None, env_file: Path | None = None, timeout_seconds: int = 120) -> None:
        self.api_key = api_key or load_env_value("OPENAI_API_KEY", env_file)
        self.timeout_seconds = timeout_seconds
        self.last_duration_seconds: float | None = None
        self.last_http_status: int | None = None
        if not self.api_key or self.api_key.startswith("replace_with_"):
            raise ValueError("OPENAI_API_KEY ausente ou placeholder.")

    @staticmethod
    def request_body(request: LLMRequest) -> dict[str, Any]:
        input_text = request.explanation_prompt + "\n\nINDIVIDUAL_MODEL_EXPLANATION_INPUT\n" + json.dumps(
            request.input_payload, ensure_ascii=False, sort_keys=True,
        )
        return {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": input_text}]}],
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema", "name": "individual_model_explanation_v1",
                    "strict": True, "schema": output_json_schema(),
                }
            },
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        body = self.request_body(request)
        if "temperature" in body or body["store"] is not False:
            raise RuntimeError("Configuracao insegura da chamada individual.")
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                self.last_http_status = getattr(response, "status", 200)
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            self.last_http_status = error.code
            raise RuntimeError(f"Provider individual falhou com HTTP {error.code}.") from error
        finally:
            self.last_duration_seconds = time.perf_counter() - started
        extracted = extract_response_text(response_payload)
        output = json.loads(extracted.text)
        validate_output(output)
        return LLMResponse(
            output=output,
            provider=self.name,
            model=str(response_payload.get("model") or request.model),
            response_id=response_payload.get("id"),
            usage=response_payload.get("usage"),
        )
