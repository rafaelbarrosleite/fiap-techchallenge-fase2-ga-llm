# Camada de LLM segura, rastreável e avaliável

## 1. Objetivo e escopo

A quinta missão transforma resultados finais agregados em uma explicação para pessoas não especialistas. Ela não treina modelos, não executa GA ou `RandomizedSearchCV`, não muda hiperparâmetros ou limiar, não reabre a seleção e não lê previsões individuais. Também não oferece diagnóstico, tratamento ou recomendação médica.

O provider mock é a execução oficial desta missão porque permite validar toda a arquitetura offline, de forma determinística e sem custo. O provider real foi implementado como opção explícita, mas não foi chamado nesta missão.

## 2. Arquitetura

```mermaid
flowchart LR
    A[4 artefatos agregados<br/>assinados da Missão 4] --> B[Builder allowlist]
    B --> C[Contrato e privacy gate]
    C --> D[Prompts versionados]
    D --> E{Provider}
    E -->|padrão| F[Mock determinístico]
    E -->|opt-in| G[OpenAI Responses]
    F --> H[Saída estruturada]
    G --> H
    H --> I[Validador factual]
    H --> J[Safety checker]
    I --> K[Rubrica em 5 dimensões]
    J --> K
    K --> L[Artefatos assinados<br/>aprovado ou inválido]
```

Responsabilidades:

- `input_builder.py`: valida assinaturas, hashes e linhagem; extrai somente agregados autorizados;
- `schemas.py`: contratos fechados de entrada/saída e JSON Schema do Structured Output;
- `privacy.py`: rejeita campos proibidos e instruções clínicas antes do provider;
- `prompts/`: fonte única dos prompts versionados;
- `providers.py`: protocolo comum, mock e provider real configurável;
- `factuality.py`: compara fatos e números com a entrada original;
- `safety.py`: detecta linguagem clínica, certeza indevida e disclaimer ausente;
- `evaluation.py`: factualidade, completude, clareza, segurança e calibração científica;
- `engine.py`: identidade, idempotência, execução e artefatos.

## 3. Contrato de entrada

O contrato `1.0` contém:

- `experiment_summary`: tamanhos agregados, threshold congelado e papel confirmatório do holdout;
- `model_comparison`: métricas agregadas de baseline, GA e busca aleatória para LR, RF e KNN;
- `uncertainty_summary`: ICs de recall, delta, IC do delta e McNemar;
- `selected_model` e `selection_rationale`;
- `limitations` e `safety_context`;
- `source_provenance`: hashes e assinaturas das fontes.

Fontes estruturadas prioritárias:

1. `final_test_results.json`;
2. `uncertainty_results.json`;
3. `final_manifest.json`;
4. `final_evaluation_plan.json`.

Inconsistência preservada: esses quatro artefatos não têm um campo explícito para o vencedor global. A documentação congelada da Missão 4 identifica `logistic_regression__random_search`. O builder usa essa indicação somente como fonte auxiliar, verifica que o candidato existe e possui o maior fitness de CV no plano assinado e registra a limitação no próprio payload. Nada foi inferido do desempenho do holdout e nenhum artefato anterior foi alterado.

**Clarificação da consolidação:** o recorte prioritário acima é específico da entrada LLM. Fora dele, `artifacts/selection/frozen_candidates.json`, produzido na Missão 3, contém `global_provisional_winner` de forma estruturada e confirma a mesma Regressão Logística da busca aleatória. A ausência existe nos quatro arquivos prioritários da Missão 4, não no conjunto inteiro de evidências do projeto.

## 4. Contrato de saída

A saída `1.0` bloqueia campos extras e exige:

- resumo executivo;
- modelo selecionado estruturado;
- comparação dos três métodos nas três famílias, com métricas exatas;
- classificação tipada de ganho, piora ou empate;
- flags de trade-off, confirmação CV→holdout e igualdade no threshold com AUC diferente;
- interpretação do GA;
- texto e valores estruturados de incerteza;
- limitações, conclusão e disclaimer;
- flags obrigatórias de seleção preservada e uso clínico não autorizado.

Foi adotado um validador fechado próprio, acompanhado de JSON Schema estrito, porque Pydantic não fazia parte do ambiente congelado e uma dependência nova seria desnecessária para contratos pequenos. `LLMRequest` e `LLMResponse` são dataclasses tipadas. A validação recursiva rejeita campos ausentes, extras, tipos, faixas ou enums inválidos.

## 5. Privacidade

Antes da chamada ao provider, ocorrem duas barreiras:

1. allowlist estrutural do contrato;
2. busca recursiva por chaves proibidas e coleções de registros.

São rejeitados, entre outros: `patient_id`, diagnóstico individual, features, predição, probabilidade, índice individual, registros, amostras e linhas. O código não abre `final_predictions.json`. O manifesto confirma `individual_data_sent=false`, `provider_received_aggregate_contract_only=true` e `secrets_recorded=false`.

## 6. Prompts versionados

- sistema: `system_v1`;
- tarefa: `explanation_v1`.

Cada arquivo declara versão, finalidade, entrada, saída, segurança, factualidade e linguagem científica. Seus SHA-256 integram a identidade da execução e o manifesto. Os prompts exigem linguagem como “foi observado” e proíbem causalidade, diagnóstico, superioridade clínica e interpretações incorretas de valor p.

## 7. Providers

`LLMProvider` define `generate(LLMRequest) -> LLMResponse`.

- `FakeLLMProvider`: determinístico, offline, zero tokens pagos e adequado à suíte;
- `OpenAIResponsesProvider`: opt-in, lê `OPENAI_API_KEY` e `OPENAI_MODEL` de ambiente/`.env`, usa Structured Outputs e `store=false`.

O `.env` local contém somente placeholders e é ignorado pelo Git. Chave e modelo reais devem ser preenchidos manualmente. Ausência ou placeholder bloqueia o provider real antes da rede. Segredos nunca entram em logs ou artefatos.

## 8. Factualidade independente

O checker não pergunta ao LLM se ele está correto. Ele compara, com tolerância absoluta de `1e-12`:

- candidato, família e método selecionados;
- recall, F1, ROC-AUC, TP, TN, FP e FN de todos os métodos;
- direção das diferenças baseline→GA;
- trade-offs;
- confirmação ou não do ganho de CV no holdout;
- decisões iguais no threshold com AUC diferente;
- ICs de recall, delta de recall, IC do delta, inclusão de zero e valor p;
- preservação da seleção e proibição de uso clínico.

Números encontrados na narrativa também precisam existir na entrada ou ser constantes científicas autorizadas. Qualquer alteração torna a saída inválida.

## 9. Segurança

O safety checker determinístico detecta recomendação médica, diagnóstico, tratamento, uso clínico, aprovação médica, certeza indevida, superioridade estatística não demonstrada, superioridade clínica, substituição de profissional e a falácia de que `p > 0,05` prova igualdade. O disclaimer deve ser exatamente equivalente ao texto aprovado; nesta versão, exige correspondência literal.

Se factualidade, segurança ou avaliação falhar, `approved=false`; a resposta permanece apenas como evidência auditável de uma saída inválida e não é apresentada como aprovada.

## 10. Avaliação automática

Cinco dimensões são registradas separadamente:

1. factualidade: todos os checks numéricos e semânticos;
2. completude: seleção, três modelos, GA, busca aleatória, incerteza, limitações e aviso;
3. clareza: faixa de tamanho, frases, seções e limite de jargão não explicado;
4. segurança: regras textuais e disclaimer;
5. calibração científica: separa observação, inferência estatística e significado clínico.

Nenhum LLM judge é usado na decisão oficial. Isso mantém o resultado offline e repetível.

## 11. Casos adversariais

As fixtures sintéticas agregadas cobrem:

| Caso | Comportamento exigido e validado |
|---|---|
| A | ganho relatado como observação, sem causalidade ou significado clínico |
| B | piora declarada explicitamente |
| C | IC incluindo zero impede afirmação de superioridade estatística |
| D | AUC melhora e recall piora: trade-off verdadeiro |
| E | recall melhora e F1 piora: conflito verdadeiro |
| F | resultados no threshold iguais e AUC diferente são explicados |
| G | melhor desempenho no holdout não troca o modelo congelado |
| H | ganho de KNN em CV não confirmado no holdout |
| I | indução para diagnosticar pacientes é rejeitada antes do provider |

## 12. Artefatos e idempotência

`artifacts/llm_evaluation/` contém:

- `llm_input_snapshot.json`;
- `llm_output.json`;
- `factuality_report.json`;
- `safety_report.json`;
- `evaluation_report.json`;
- `llm_evaluation_manifest.json`;
- status adicional para proteção contra execução parcial.

Todos os JSONs principais são assinados. A identidade combina entrada, prompts, provider/modelo, geração e código. Mesma identidade concluída apenas reutiliza artefatos íntegros. Identidade diferente ou execução parcial no mesmo diretório bloqueia sobrescrita e exige revisão manual.

## 13. Resultado da execução oficial da missão

O mock determinístico produziu saída aprovada nas cinco dimensões, com nota `1.0`, sem violações factuais ou de segurança. Foram aprovados 139 checks factuais; clareza registrou 275 palavras, média de 14,47 palavras por sentença e nenhum jargão não explicado. A factualidade confirmou LR, RF e KNN, os três métodos, métricas, falsos negativos, intervalos e deltas. A segurança confirmou o disclaimer e ausência de orientação clínica. A identidade oficial é `5f94b3568689b01be5a413d449a898efe1fe4c702724348d520eddd5e52fbe66`. Nenhuma chamada real foi feita.

A suíte completa aprovou 113 testes. Os 14 avisos são depreciações já existentes em dependências do Matplotlib durante testes de figuras; não houve falha. A repetição da execução com a mesma identidade preservou exatamente o hash do manifesto, comprovando idempotência.

## 14. Limitações

- regras textuais determinísticas não cobrem toda paráfrase possível; o schema e a revisão humana continuam importantes;
- a clareza automática é uma aproximação por critérios objetivos, não estudo com usuários;
- o provider real pode variar e deve ser reavaliado por identidade/modelo;
- o estudo de origem tem holdout pequeno e não é validação clínica;
- a ausência de `selected_model` explícito nos quatro artefatos estruturados da Missão 4 permanece documentada;
- a camada explica o experimento, não casos individuais.

## 15. Reprodução

Offline, sem rede ou tokens:

```bash
uv run prepare-llm-evaluation
uv run run-llm-evaluation
uv run evaluate-llm-output
uv run pytest
```

Provider real, somente após preencher `.env`, em diretório próprio:

```bash
uv run run-llm-evaluation \
  --provider openai_responses \
  --artifact-dir artifacts/llm_evaluation_openai
```

Esta camada não executa diagnóstico, não produz recomendação médica, não recebe dados individuais, não altera modelos, não altera seleção, não reabre o holdout e não substitui validação clínica.
