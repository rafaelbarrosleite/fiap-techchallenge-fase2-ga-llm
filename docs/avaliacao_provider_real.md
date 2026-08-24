# Avaliação complementar do provider real — Missão 7

## 1. Objetivo

Esta missão avaliou se a camada LLM segura da Missão 5 continuaria factual, completa, clara, cientificamente calibrada e segura com um provider real. A modelagem permaneceu fora do processo: não houve treino, GA, busca aleatória, inferência no holdout, alteração de threshold ou reabertura de seleção.

## 2. Configuração

| Item | Valor |
|---|---|
| Provider | `openai_responses` |
| Modelo solicitado | `gpt-5.5` |
| Endpoint | Responses API |
| Structured Output | JSON Schema estrito |
| Store | `false` |
| Temperatura solicitada | `0,0` |
| Máximo de saída | 3.000 tokens |
| Retry automático | zero |
| Prompt de sistema | `system_v1` |
| Prompt de explicação | `explanation_v1` |

Os prompts não foram alterados depois da resposta da API.

## 3. Contrato enviado

O payload possui SHA-256 lógico `44ca9f33e8a8553ed837441507c97c4cebfd9e6bcc57170e81e9e97e4936ee13`, exatamente igual ao snapshot oficial do `FakeLLMProvider`. Ele contém somente resumo experimental, métricas agregadas dos nove candidatos, incerteza agregada, seleção congelada, limitações, contexto de segurança e proveniência.

Não foram enviados `final_predictions.json`, linhas, features, diagnósticos, probabilidades, previsões ou índices individuais.

## 4. Privacidade e preflight

O preflight ocorreu antes da chamada e aprovou schema, privacy checker, hashes dos artefatos da Missão 4, igualdade com o fake, versões dos prompts, configuração explícita do modelo, presença não revelada da credencial e `store=false`.

A chave não foi impressa, copiada ou persistida. Os artefatos registram somente `credential_present=true` e `secret_value_recorded=false`.

## 5. Resultado da chamada principal

A única chamada principal foi rejeitada pela API com HTTP 400 antes de qualquer saída estruturada. A duração foi `2,048108 s`, e o identificador seguro disponibilizado foi `req_c06505f920e84cb3adc296f7c6768097`.

Não houve retry. O provider não retornou texto estruturado, modelo efetivo ou status de resposta concluída. O cliente original registrou o código HTTP e o request ID, mas não reteve o corpo detalhado do erro. Assim, a causa comprovável é **requisição rejeitada pela API antes da geração**; não há evidência suficiente para atribuir a rejeição especificamente ao modelo, à temperatura ou ao JSON Schema.

## 6. Consumo de tokens

| Campo | Resultado |
|---|---:|
| Input tokens | `null` |
| Output tokens | `null` |
| Total tokens | `null` |
| Request success | `false` |
| Estimativa de custo | `null` |

A API não forneceu usage na resposta de erro. Nenhum preço foi inventado.

## 7. Factualidade

Status: `not_evaluated`. Sem saída estruturada, os 139 checks não puderam ser aplicados ao provider real. Isso não equivale a reprovação factual do texto; significa ausência de texto para avaliar.

## 8. Segurança

Status: `not_evaluated`. Não houve texto real para verificar disclaimer, recomendação clínica ou certeza indevida. A privacidade da entrada, porém, foi aprovada antes da chamada, e nenhum dado individual foi enviado.

## 9. Completude, clareza e calibração

As três dimensões ficaram `not_evaluated`. Não é metodologicamente válido atribuir nota zero ou um a uma resposta inexistente. O manifesto mantém `approved=false` porque o critério da missão exige saída válida e aprovação em todas as barreiras.

## 10. Hallucination report

`unexpected_numbers`, `unexpected_model_names`, `unsupported_claims`, `clinical_claims`, `statistical_overclaims` e `selection_violations` ficaram `null/not_evaluated`. Listas vazias poderiam sugerir incorretamente que uma resposta foi examinada; por isso o estado ausente foi preservado explicitamente.

## 11. Comparação Fake versus OpenAI

| Dimensão | Fake | OpenAI |
|---|---|---|
| Schema válido | aprovado | não houve saída |
| Factualidade | aprovado | não avaliada |
| Completude | aprovada | não avaliada |
| Segurança | aprovada | não avaliada |
| Calibração científica | aprovada | não avaliada |
| Clareza | 1,0 | não avaliada |
| Números inesperados | zero | não avaliados |
| Violações | zero | não avaliadas |
| Disclaimer correto | sim | não avaliado |
| Seleção congelada respeitada | sim | não avaliada |

Diferença de estilo não entrou na comparação.

## 12. Avaliação adversarial

Os cenários A, B e C não foram enviados. A regra da missão permitia essas chamadas somente depois de aprovação integral da execução principal. O manifesto registra `not_run_main_invalid` e zero chamadas adversariais.

## 13. Limitações e melhoria proposta

A limitação central é operacional: o cliente não preservou o corpo seguro do HTTP 400, impedindo diagnóstico do campo rejeitado. Uma execução futura, versionada como experimento separado e somente após nova autorização, deveria capturar de forma sanitizada `error.type`, `error.code`, `error.param` e `error.message`, sem headers ou credenciais. Ela também deveria ser preparada em outro diretório e nunca sobrescrever esta primeira evidência.

Essa melhoria não foi aplicada à execução nem houve nova tentativa.

## 14. Conclusão

A Missão 7 produziu uma evidência científica válida de limitação, mas a execução principal **não foi aprovada**. Não é possível concluir que `gpt-5.5` preservou factualidade, completude, segurança ou calibração, porque a API rejeitou a requisição antes de gerar resposta.

O resultado positivo é de engenharia de segurança: o preflight funcionou, o payload foi o mesmo do fake, a privacidade foi preservada, a falha ficou visível, não houve retry, os cenários condicionais foram bloqueados e os artefatos das Missões 3–6 permaneceram intactos.

Artefatos prioritários:

- [`failure_report.json`](../artifacts/llm_evaluation_openai/failure_report.json)
- [`provider_usage.json`](../artifacts/llm_evaluation_openai/provider_usage.json)
- [`comparison_with_fake.json`](../artifacts/llm_evaluation_openai/comparison_with_fake.json)
- [`llm_evaluation_manifest.json`](../artifacts/llm_evaluation_openai/llm_evaluation_manifest.json)

