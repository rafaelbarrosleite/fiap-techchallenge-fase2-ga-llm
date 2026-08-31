# Guia de demonstração offline

## Princípio de segurança

A demonstração oficial valida e carrega estados `completed`. Ela não chama comandos de GA, busca aleatória, baseline ou provider real. Antes de apresentar, confirme que `artifacts/final_evaluation/final_evaluation_status.json` e `artifacts/llm_evaluation/llm_evaluation_status.json` estão `completed`.

## 1. Preparar e validar o ambiente

```bash
uv sync
uv run validate-deliverable
```

Mostre a versão `0.7.0`, o status `passed` e a ausência de downloads/chamadas externas durante a validação. `validate-deliverable` é somente leitura.

## 2. Executar a suíte

```bash
uv run pytest
```

Explique que os testes de consolidação comparam a tabela mestre aos JSONs congelados e inspecionam o código para proibir treino, inferência e rede.

Resultado esperado desta entrega: `182 passed`. Avisos de depreciação de `pyparsing`/Matplotlib podem aparecer, sem afetar o status.

## 3. Abrir as evidências principais

```bash
python -m json.tool artifacts/final_evaluation/final_test_results.json | less
python -m json.tool artifacts/final_evaluation/uncertainty_results.json | less
python -m json.tool artifacts/llm_evaluation/llm_evaluation_manifest.json | less
python -m json.tool artifacts/final_summary/final_delivery_manifest.json | less
```

Mostre `new_optimization_performed=false`, `selection_reopened=false`, `provider=fake`, `individual_data_sent=false` e as confirmações do manifesto final.

## 4. Demonstrar a avaliação final sem nova inferência

```bash
uv run run-final-evaluation
```

Com manifesto íntegro e status `completed`, o comando somente valida hashes e carrega o resultado existente. Não há `fit`, predição ou abertura nova do holdout.

## 5. Demonstrar a LLM offline

```bash
uv run run-llm-evaluation
uv run evaluate-llm-output
```

O primeiro comando reutiliza a execução mock concluída para a mesma identidade. O segundo recalcula factualidade, segurança e as cinco dimensões sem chamar provider.

Como evidência complementar, abra `docs/avaliacao_provider_real_v4.md` e mostre que a única resposta real V2 obteve 327/327 fatos e zero violações clínicas, mas permaneceu não aprovada pelo gate lexical de calibração. Não execute novamente o provider real durante a demonstração.

Abra:

```bash
python -m json.tool artifacts/llm_evaluation/factuality_report.json | less
python -m json.tool artifacts/llm_evaluation/safety_report.json | less
python -m json.tool artifacts/llm_evaluation/evaluation_report.json | less
```

Destaque 139 checks factuais, zero violações, disclaimer válido e score geral `1.0`.

### Explicação individual

```bash
uv run run-individual-explanation
uv run evaluate-individual-explanation
python -m json.tool artifacts/llm_individual_explanation/individual_output.json | less
```

Mostre a classe e a probabilidade, os cinco fatores, os insights com `scope=human_review_only`, o bloco do Módulo 3 e o disclaimer. Em seguida, abra `docs/examples/individual_explanation_v1.json` para mostrar que há uma saída completa versionada. A execução OpenAI real já está preservada em `artifacts/llm_individual_explanation_openai/`; não a repita durante a apresentação.

## 6. Demonstrar idempotência

```bash
shasum -a 256 artifacts/llm_evaluation/llm_evaluation_manifest.json
uv run run-llm-evaluation
shasum -a 256 artifacts/llm_evaluation/llm_evaluation_manifest.json
```

Os hashes devem ser idênticos. O mesmo raciocínio vale para `run-final-evaluation`: estado concluído íntegro é carregado, não recalculado.

## 7. Provar ausência de nova otimização ou inferência

```bash
python - <<'PY'
import json
from pathlib import Path

final = json.loads(Path("artifacts/final_evaluation/final_test_results.json").read_text())
delivery = json.loads(Path("artifacts/final_summary/final_delivery_manifest.json").read_text())
print("new_optimization_performed:", final["new_optimization_performed"])
print("selection_reopened:", final["selection_reopened"])
print(delivery["scope_confirmations"])
PY
```

Não execute `run-ga-battery`, `run-ga-experiment`, `run-ga-analysis`, `run-baseline` ou provider `openai_responses` durante a apresentação.

## Demonstração em 5 minutos

1. `uv run validate-deliverable` — mostrar status e hashes.
2. Abrir `model_results.csv` — localizar as nove linhas e o vencedor global.
3. Abrir `02_falsos_negativos_baseline_vs_ga.png` — explicar `3→1`, `4→3`, `4→4`.
4. `uv run run-final-evaluation` — mostrar carregamento do estado concluído.
5. `uv run run-individual-explanation` e `uv run evaluate-individual-explanation` — mostrar explicação individual e score `1.0`.
6. Abrir `final_delivery_manifest.json` — encerrar com as confirmações de escopo.

## Plano de contingência

Se `uv sync` exigir rede no ambiente da banca, use o `.venv` já preparado e execute `.venv/bin/pytest`, `.venv/bin/run-final-evaluation`, `.venv/bin/run-llm-evaluation` e `.venv/bin/validate-deliverable`. Não instale dependências durante a fala. Se qualquer manifesto falhar, interrompa a demo e mostre o relatório; não regenere resultados no palco.
