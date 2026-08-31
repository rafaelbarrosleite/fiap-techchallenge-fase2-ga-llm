# Avaliação do provider real — Missão 7.2

## 1. Contexto da Missão 7

A primeira avaliação real foi rejeitada antes da geração com HTTP 400. Seus artefatos permanecem congelados em `artifacts/llm_evaluation_openai/`.

## 2. Causa raiz da Missão 7.1

O diagnóstico isolado identificou `temperature` como parâmetro incompatível com `gpt-5.5`. A evidência sanitizada foi preservada em `artifacts/openai_integration_diagnosis/`.

## 3. Correção

O provider omite `temperature` quando o modelo é exatamente `gpt-5.5`. O preflight da Missão 7.2 confirmou que o campo estava ausente tanto do request mínimo quanto do request científico preparado. Nenhum prompt, dado agregado, schema científico ou checker foi alterado.

## 4. Estado inicial e preflight

- worktree limpo no início da missão;
- commit: `1ed89fa4f3034a0f6f480d10c88c72f46aaa8525`;
- `.env` ignorado pelo Git;
- credencial presente sem leitura ou persistência de seu valor;
- `OPENAI_MODEL=gpt-5.5`;
- schema e privacidade aprovados localmente;
- payload científico lógico com SHA-256 `44ca9f33e8a8553ed837441507c97c4cebfd9e6bcc57170e81e9e97e4936ee13`;
- hashes das Missões 7 e 7.1 registrados antes da chamada;
- 134 testes locais aprovados antes da chamada.

## 5. Chamada mínima corrigida

Foi executada uma única chamada técnica com Responses API, `gpt-5.5`, Structured Outputs, `store=false`, sem `temperature`, prompt trivial e schema `{status: "ok"}`. Nenhum dado científico, médico ou individual foi enviado.

O transporte recebeu um corpo JSON sem lançar HTTPError, mas a resposta não continha `output_text`. A extração local encerrou com:

```text
ProviderRealEvaluationError: Resposta real nao contem output_text.
```

Como a versão inicial do probe ainda não persistia o payload quando `output_text` estava ausente, HTTP status, request id, duração, status da resposta e usage ficaram indisponíveis e foram registrados como `null`, nunca como zero. O código foi reforçado depois da ocorrência para preservar esses metadados em futuras execuções, mas nenhuma nova chamada foi feita.

Resultado do probe: `invalid`.

## 6. Avaliação científica

Não executada. O gate obrigatório bloqueou a chamada científica principal após a reprovação do probe técnico. Assim, nenhum agregado do experimento foi enviado ao provider nesta missão.

## 7. Factualidade

Não avaliada. O relatório JSON é um placeholder explícito e não deve ser interpretado como reprovação factual de uma resposta inexistente.

## 8. Segurança

Não avaliada sobre saída real, pois não houve saída aprovada. As proteções de entrada passaram: zero dados individuais, zero `final_predictions.json`, zero conteúdo médico no probe e segredo ausente dos artefatos.

## 9. Completude e clareza

Não avaliadas. Nenhuma resposta científica foi gerada.

## 10. Calibração científica

Não avaliada. Não houve texto científico do provider a ser confrontado com as regras de linguagem calibrada.

## 11. Comparação com o fake

O FakeLLMProvider permanece aprovado como caminho oficial offline. A coluna OpenAI está marcada como `not_evaluated`, pois diferença estilística ou de qualidade não pode ser inferida sem uma saída científica.

## 12. Testes adversariais

Não executados. O orçamento consumido foi uma chamada técnica; chamada principal e três cenários permaneceram em zero.

## 13. Usage e custo

| Etapa | Chamadas | Input tokens | Output tokens | Total tokens | Duração |
|---|---:|---:|---:|---:|---:|
| Probe técnico | 1 | `null` | `null` | `null` | `null` |
| Principal científica | 0 | — | — | — | — |
| Adversariais | 0 | — | — | — | — |

Não foi calculado custo: usage não foi preservado e `null` não equivale a zero.

## 14. Limitações

A ausência do payload bruto impede determinar se a resposta estava `incomplete`, continha apenas itens de reasoning, foi recusada ou usou todo o limite de saída antes de emitir texto. `max_output_tokens=100` e o reasoning padrão do modelo são hipóteses técnicas para uma investigação futura, não uma causa demonstrada. Nenhum parâmetro foi alterado com base nessa hipótese.

## 15. Conclusão

A correção de `temperature` permitiu avançar além do HTTP 400 anterior, mas o critério técnico completo ainda não foi satisfeito porque não houve Structured Output `{status: "ok"}`. A Missão 7.2 termina como `invalid / scientific_call_blocked`, com uma chamada, zero retries e toda evidência anterior preservada.

Uma próxima missão deve repetir somente o probe em um novo diretório versionado, com captura completa já corrigida. A avaliação científica só poderá ser retomada se esse novo probe for aprovado. O relatório final, o resumo executivo, o README e a matriz de rastreabilidade não foram atualizados como sucesso.
