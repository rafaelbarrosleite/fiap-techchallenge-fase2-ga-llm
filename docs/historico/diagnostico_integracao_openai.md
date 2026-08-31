# Diagnóstico da integração OpenAI

## 1. Problema

A chamada única da Missão 7 foi rejeitada pela Responses API com HTTP 400 antes da geração. O corpo detalhado do erro não havia sido preservado. Esta missão isolada investigou somente a validade técnica da integração; não repetiu a avaliação científica.

## 2. Evidência preservada da Missão 7

Os 12 arquivos de `artifacts/llm_evaluation_openai/` foram verificados por SHA-256 antes e depois do diagnóstico. Nenhum foi alterado. A evidência congelada continua registrando uma chamada, ausência de retry, nenhuma saída e o request id original.

## 3. Documentação oficial consultada

- [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5): confirma o identificador `gpt-5.5`, a Responses API e Structured Outputs.
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): confirma `text.format`, `type=json_schema`, `name`, `schema`, `strict`, objetos fechados com `additionalProperties=false` e campos obrigatórios.
- [Responses API — create](https://developers.openai.com/api/reference/resources/responses/methods/create): confirma `model`, `input`, `instructions`, `max_output_tokens`, `store` e `text` no contrato geral.
- [Orientação para GPT-5.5](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.5): confirma o uso da Responses API, `reasoning.effort` e `text.verbosity`.

O contrato geral da API lista `temperature`, mas a validação efetiva do modelo é mais restritiva: a resposta sanitizada do servidor declarou explicitamente que `temperature` não é suportado com `gpt-5.5`.

## 4. SDK e transporte

O pacote Python `openai` não está instalado no ambiente (`openai_sdk_version = null`). A integração não usa `client.responses.create`; ela envia o JSON documentado diretamente com `urllib` da biblioteca padrão do Python 3.12.13. Portanto, não houve incompatibilidade de versão do SDK e nenhuma dependência foi atualizada.

## 5. Requisição original analisada

O corpo lógico original foi reconstruído a partir do snapshot congelado e dos prompts versionados. Seu conteúdo científico não foi copiado para a nova pasta; apenas estrutura, tamanho e hash foram registrados.

| Campo | Estado na Missão 7 | Avaliação final | Ação |
|---|---|---|---|
| `model` | `gpt-5.5` | válido | preservar |
| `input` | mensagem agregada | estrutura válida | preservar |
| `instructions` | prompt versionado | válido | preservar |
| `text.format` | JSON Schema estrito | sintaxe correta; não alcançada na chamada diagnóstica | preservar |
| `temperature` | `0.0` | rejeitado pelo modelo | omitir para `gpt-5.5` |
| `max_output_tokens` | `3000` | permitido | preservar |
| `store` | `false` | permitido | preservar |
| `reasoning` | ausente | permitido; usa padrão do modelo | preservar ausente |
| `text.verbosity` | ausente | permitido; usa padrão | preservar ausente |
| `top_p` | ausente | sem ação | preservar ausente |
| `response_format` | ausente | correto para Responses; usa-se `text.format` | preservar ausente |

## 6. Structured Outputs e JSON Schema

A inspeção local confirmou:

- raiz do tipo `object`;
- todos os objetos com `additionalProperties=false`;
- todas as propriedades presentes em `required`;
- ausência de `oneOf`, `anyOf`, `allOf`, `$ref`, `$defs` e `default`;
- arrays, objetos aninhados e enums estruturalmente coerentes.

O schema contém quatro usos de `const`. Como o servidor rejeitou primeiro o parâmetro `temperature`, a compatibilidade remota completa do schema não chegou a ser exercitada. Nenhuma adaptação especulativa de schema foi mantida; o contrato científico local permanece intacto.

## 7. Captura sanitizada de erros

O provider agora preserva, quando disponíveis: `http_status`, `error.type`, `error.code`, `error.param`, `error.message`, `request_id`, classe da exceção, provider, modelo, timestamp e hash do request sanitizado. Authorization, API key, `.env`, headers sensíveis e secrets não são persistidos.

## 8. Dry-run

O comando `uv run diagnose-openai-request --dry-run` concluiu com:

- modelo `gpt-5.5`;
- provider `openai_responses`;
- schema local válido;
- privacidade válida;
- `store=false`;
- hash `4d925f6a52f6b75d8009305a97152224e4c128529ca013a8a0039b7d279c6976` para o request diagnóstico efetivamente enviado;
- `api_call_performed=false`.

## 9. Chamada técnica mínima

Foi feita exatamente uma chamada, sem retry, com prompt trivial, schema `{status: "ok"}`, `gpt-5.5`, Responses API, Structured Outputs e `store=false`. Nenhuma métrica, informação médica, agregado científico ou dado individual foi enviado.

Resultado: HTTP 400 em 1,4145 s, antes da geração. Mensagem sanitizada:

```text
type: invalid_request_error
code: null
param: temperature
message: Unsupported parameter: 'temperature' is not supported with this model.
```

O request id foi preservado somente no artefato de auditoria. Não houve segunda chamada.

## 10. Causa raiz

Classificação: `unsupported_parameter`.

O parâmetro responsável foi `temperature`. O próprio servidor o indicou em `error.param` e na mensagem. Isso afasta, para esta rejeição, as hipóteses de modelo inválido, incompatibilidade de SDK e erro de autenticação. Como a validação parou nesse campo, a chamada não demonstra ainda a aceitação remota do schema completo.

## 11. Correção mínima

O provider passou a omitir `temperature` somente quando o modelo solicitado é exatamente `gpt-5.5`. Para outros modelos, o comportamento anterior foi preservado até existir evidência específica. Prompts, dados agregados, factuality checker, safety checker, schema local e resultados congelados não foram modificados.

A correção não foi revalidada remotamente nesta missão: uma nova chamada seria um retry, expressamente proibido após a única tentativa diagnóstica.

## 12. Testes

Antes da chamada, 130 testes passaram. A cobertura adicionada verifica dry-run sem rede, estrutura do schema, request mínimo, hash estável, omissão de `temperature` para `gpt-5.5`, captura detalhada de HTTP 400 e ausência de secrets. A suíte comum continua totalmente offline.

Os comandos finais de validação são:

```bash
uv run pytest
uv run validate-deliverable
uv run validate-openai-diagnosis
```

## 13. Próximos passos

Antes de qualquer Missão 7.2 científica, deve ser autorizada uma nova execução técnica versionada do request mínimo corrigido, agora sem `temperature`. Se ela chegar à validação do schema, será possível confirmar ou delimitar qualquer incompatibilidade posterior sem misturar o diagnóstico com dados do experimento. A avaliação científica original não deve ser repetida nesta missão.
