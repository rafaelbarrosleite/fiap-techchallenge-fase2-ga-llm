"""Providers desacoplados: mock deterministico e OpenAI Responses configuravel."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .schemas import DISCLAIMER, output_json_schema, validate_output


@dataclass(frozen=True)
class LLMRequest:
    input_payload: dict[str, Any]
    system_prompt: str
    explanation_prompt: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 3000


@dataclass(frozen=True)
class LLMResponse:
    output: dict[str, Any]
    provider: str
    model: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class LLMProvider(Protocol):
    name: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...


def _change(after: float, before: float, tolerance: float = 1e-12) -> str:
    delta = after - before
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "worsened"
    return "unchanged"


def build_deterministic_output(payload: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    for item in payload["model_comparison"]:
        baseline = item["baseline"]["metrics"]
        ga = item["ga"]["metrics"]
        random_search = item["random_search"]["metrics"]
        changes = {
            "ga_recall_change": _change(ga["recall_malignant"], baseline["recall_malignant"]),
            "ga_f1_change": _change(ga["f1_malignant"], baseline["f1_malignant"]),
            "ga_auc_change": _change(ga["roc_auc"], baseline["roc_auc"]),
        }
        directions = set(changes.values()).difference({"unchanged"})
        threshold_keys = ("true_positives", "true_negatives", "false_positives", "false_negatives")
        same_threshold = all(ga[key] == baseline[key] for key in threshold_keys)
        different_auc = abs(ga["roc_auc"] - baseline["roc_auc"]) > 1e-12
        cv_improved = item["cv_recall_ga"] > item["cv_recall_baseline"] + 1e-12
        holdout_improved = ga["recall_malignant"] > baseline["recall_malignant"] + 1e-12
        model_label = {"logistic_regression": "Regressão Logística", "random_forest": "Random Forest", "knn": "KNN"}[item["model"]]
        if changes["ga_recall_change"] == "improved":
            interpretation = f"No {model_label}, foi observado aumento de recall no holdout, sem atribuir causalidade ou significado clínico."
        elif changes["ga_recall_change"] == "worsened":
            interpretation = f"No {model_label}, foi observada piora de recall no holdout; o resultado negativo foi preservado."
        else:
            interpretation = f"No {model_label}, o recall no holdout permaneceu igual ao baseline."
        if same_threshold and different_auc:
            interpretation += " As decisões no threshold foram iguais, mas a ordenação das probabilidades, resumida pelo ROC-AUC, diferiu."
        comparisons.append({
            "model": item["model"],
            "baseline": {"method": "baseline", **baseline},
            "ga": {"method": "ga", **ga},
            "random_search": {"method": "random_search", **random_search},
            **changes,
            "tradeoff_present": len(directions) > 1,
            "cv_gain_confirmed_on_holdout": bool(cv_improved and holdout_improved),
            "same_threshold_outcomes_different_auc": bool(same_threshold and different_auc),
            "interpretation": interpretation,
        })
    selected = payload["selected_model"]
    result = {
        "schema_version": "1.0",
        "resumo_executivo": (
            "Os resultados agregados sugerem ganhos observados em algumas comparações e ausência de confirmação em outras. "
            "A leitura é descritiva, com incerteza explícita, e não representa validação clínica."
        ),
        "modelo_selecionado": {
            "candidate_id": selected["candidate_id"], "model": selected["model"], "method": selected["method"],
            "explanation": "A escolha foi congelada por validação cruzada antes do holdout; o teste confirmatório não reabriu a seleção.",
        },
        "comparacao_modelos": comparisons,
        "interpretacao_ga": (
            "O algoritmo genético foi comparado com o baseline e com a busca aleatória. "
            "Ganhos observados não demonstram causalidade, superioridade estatística ou relevância clínica."
        ),
        "incerteza_estatistica": (
            "Os intervalos dos deltas de recall incluem zero. Portanto, não há evidência suficiente para afirmar superioridade estatística; "
            "um valor p acima de 0,05 também não prova igualdade entre métodos."
        ),
        "incerteza_por_modelo": list(payload["uncertainty_summary"]),
        "limitacoes": list(payload["limitations"]),
        "conclusao": (
            "Foi observado desempenho confirmatório compatível com a análise agregada, mas a amostra é limitada e os intervalos permanecem amplos. "
            "O estudo tem caráter acadêmico e experimental."
        ),
        "disclaimer": DISCLAIMER,
        "holdout_nao_reabriu_selecao": True,
        "uso_clinico_autorizado": False,
    }
    validate_output(result)
    return result


class FakeLLMProvider:
    name = "fake"

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            output=build_deterministic_output(request.input_payload), provider=self.name,
            model=request.model, response_id="offline-deterministic-response", usage={"paid_tokens": 0},
        )


def load_env_value(name: str, env_file: Path | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if env_file is None or not Path(env_file).is_file():
        return None
    for raw_line in Path(env_file).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            return candidate.strip().strip("'\"") or None
    return None


class OpenAIResponsesProvider:
    """Provider real opt-in; usa Structured Outputs e store=false."""

    name = "openai_responses"
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, *, api_key: str | None = None, env_file: Path | None = None, timeout_seconds: int = 90) -> None:
        self.api_key = api_key or load_env_value("OPENAI_API_KEY", env_file)
        self.timeout_seconds = timeout_seconds
        if not self.api_key or self.api_key.startswith("replace_with_"):
            raise ValueError("OPENAI_API_KEY ausente; preencha .env explicitamente.")

    @staticmethod
    def _output_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        raise RuntimeError("Resposta do provider nao contem output_text.")

    def generate(self, request: LLMRequest) -> LLMResponse:
        input_text = request.explanation_prompt + "\n\nAGGREGATED_EXPERIMENT_INPUT\n" + json.dumps(
            request.input_payload, ensure_ascii=False, sort_keys=True,
        )
        body = {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": input_text}]}],
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "experiment_explanation_v1", "strict": True, "schema": output_json_schema()}},
        }
        http_request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Provider real falhou com HTTP {error.code}.") from error
        output = json.loads(self._output_text(response_payload))
        validate_output(output)
        return LLMResponse(
            output=output, provider=self.name, model=request.model,
            response_id=response_payload.get("id"), usage=response_payload.get("usage"),
        )
