# Diagnóstico do parsing da OpenAI Responses API

## 1. Problema observado na Missão 7.2

A chamada mínima corrigida não lançou HTTPError e retornou JSON, mas o processo terminou com `ProviderRealEvaluationError: Resposta real nao contem output_text`. Como o JSON bruto não foi persistido antes do parsing, status, output e usage daquela execução não puderam ser reconstruídos.

## 2. JSON HTTP bruto e conveniences de SDK

A [referência oficial da Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) define `output` como um array heterogêneo e recomenda não assumir que o primeiro item é uma mensagem. O texto final canônico aparece em itens `type=message`, dentro de `content` com `type=output_text`. Algumas superfícies e SDKs também expõem um convenience `output_text`; o parser HTTP não deve depender dele.

O projeto não possui SDK OpenAI instalado e consome o JSON bruto com `urllib`.

## 3. Estrutura real observada

A única chamada da Missão 7.2.1 retornou:

```text
response.output[0].type = reasoning
response.output[1].type = message
response.output[1].content[0].type = output_text
top-level output_text = ausente
```

O conteúdo de reasoning não foi exposto; apenas seu tipo foi registrado. A resposta bruta sanitizada integral foi salva porque o probe usou somente `{status: "ok"}` e não continha dados sensíveis.

## 4. Status e identificadores

- HTTP status: `200`
- Response status: `completed`
- Request ID: `req_345377926f7f4a1eb087bf176ab76228`
- Response ID: `resp_0859b4798d7fc887016a8ca58dace887d2b507fc8568b6bde0`
- Modelo retornado: `gpt-5.5-2026-04-23`
- `incomplete_details`: `null`
- `store` solicitado: `false`
- `temperature` enviado: `false`

## 5. Parsing implementado

O parser agora:

1. valida explicitamente os estados `completed`, `incomplete`, `failed`, `cancelled`, `queued` e `in_progress`;
2. percorre todos os itens de `output` sem assumir posição;
3. ignora metadados de `reasoning` e outros tipos não textuais;
4. inspeciona somente itens `message`;
5. coleta conteúdos `output_text` com texto válido;
6. reconhece refusal sem inventar texto;
7. mantém fallback top-level apenas para compatibilidade;
8. produz falhas diferentes para estado, ausência de conteúdo, JSON inválido e schema incompatível.

## 6. Ordem de persistência

A execução respeitou:

```text
HTTP response
→ raw_response_sanitized.json
→ response_structure_report.json e provider_usage.json
→ validação de status
→ inspeção de output[]
→ extração textual
→ parse JSON
→ validação do schema
```

Uma exceção posterior não elimina mais a evidência HTTP.

## 7. Structured Output

O texto extraído foi:

```json
{"status":"ok"}
```

O JSON foi parseado e validado contra o schema mínimo estrito. `json_valid=true`, `schema_valid=true` e `content_validation=true`.

## 8. Usage

| Métrica | Valor |
|---|---:|
| Input tokens | 46 |
| Output tokens | 39 |
| Reasoning tokens | 22 |
| Cached tokens | 0 |
| Total tokens | 85 |
| Duração | 1,893936 s |

## 9. Resultado do probe

O probe foi aprovado com uma chamada, zero retries, zero dados científicos, zero conteúdo médico e zero dados individuais. Nenhuma avaliação científica ou adversarial foi iniciada.

## 10. Causa raiz

A resposta atual confirmou a estrutura nested sem `output_text` top-level. Porém, a hipótese de que o parser da Missão 7.2 procurava apenas no top-level foi refutada pela inspeção do código: ele já percorria `output[].content[]`. O payload atual teria sido aceito pelo loop antigo.

Por isso, não é metodologicamente correto classificar retroativamente a falha anterior como `incorrect_raw_response_parsing`. A classificação preservada é `previous_response_shape_unknown`: sem o JSON bruto da Missão 7.2, não é possível saber se aquela resposta estava incompleta, continha somente reasoning ou possuía outra estrutura.

Essa divergência em relação à hipótese inicial foi registrada explicitamente, sem alterar silenciosamente a evidência.

## 11. Correção

A melhoria aplicada combina:

- parser defensivo orientado a `status` e `type`;
- tolerância a reasoning e outros itens;
- validação estruturada isolada;
- persistência raw-first;
- captura de status, IDs, usage, `incomplete_details` e tipos antes do parsing.

Mesmo sem atribuir causa incorreta à Missão 7.2, o transporte atual ficou observável e testável.

## 12. Testes e prontidão

Os sete casos obrigatórios foram cobertos: mensagem, reasoning antes da mensagem, refusal, incomplete, output vazio, JSON inválido e schema incompatível. Um oitavo teste comprova que raw response e usage sobrevivem a uma resposta incomplete.

A suíte completa passou antes da chamada. O resultado técnico atual permite:

```text
ready_for_scientific_evaluation = true
```

Isso não equivale a aprovação científica do provider. Uma futura missão científica deve usar nova pasta versionada e manter o FakeLLMProvider como caminho oficial offline.
