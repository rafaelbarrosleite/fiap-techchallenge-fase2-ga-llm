# Histórico da integração com o provider real

Índice narrativo das cinco etapas que antecederam a avaliação real aprovada. Os documentos originais estão preservados em [`historico/`](historico/), sem reescrita: cada um registra o estado de conhecimento no momento em que foi escrito, incluindo conclusões que etapas seguintes corrigiram.

Este material existe porque o enunciado pede desafios enfrentados e soluções implementadas. Apagar as tentativas apagaria a resposta.

O resultado final e autoritativo está em [`avaliacao_provider_real_v4.md`](avaliacao_provider_real_v4.md); o caminho oficial de reprodução continua sendo o provider mock offline.

## A sequência

| # | Etapa | O que se descobriu | Documento |
|---|---|---|---|
| 1 | Primeira avaliação real V1 | A resposta obteve 138/139 checks. A única falha revelou um problema de contrato, não de modelo: um booleano sobre matriz de confusão e AUC não identificava explicitamente qual par de métodos estava sendo comparado. | [`avaliacao_provider_real.md`](historico/avaliacao_provider_real.md) |
| 2 | Diagnóstico da integração | Parâmetros incompatíveis com a Responses API impediam a chamada. O diagnóstico isolou o corpo mínimo de requisição válido. | [`diagnostico_integracao_openai.md`](historico/diagnostico_integracao_openai.md) |
| 3 | Correção da chamada V2 | A requisição passou a omitir `temperature` e usar saída estruturada com `store=false`. | [`avaliacao_provider_real_v2.md`](historico/avaliacao_provider_real_v2.md) |
| 4 | Diagnóstico do parsing | A resposta chegava íntegra, mas o texto não era extraído do formato tipado da Responses API. Passou-se a preservar o payload bruto antes de qualquer parsing, para que uma falha de leitura não destruísse a evidência. | [`diagnostico_parsing_responses_api.md`](historico/diagnostico_parsing_responses_api.md) |
| 5 | Transporte raw-first V3 | Consolidou a preservação do bruto e do uso de tokens mesmo em resposta incompleta, sem retry. | [`avaliacao_provider_real_v3.md`](historico/avaliacao_provider_real_v3.md) |

## O que a sequência ensinou

**Falha de contrato não é falha de modelo.** A etapa 1 reprovou por uma ambiguidade do próprio esquema de entrada, não por alucinação. A correção foi criar o contrato V2 com `comparison_id` explícito, e não ajustar o julgamento da resposta. O resultado histórico não foi reclassificado.

**Preservar o bruto antes de interpretar.** As etapas 2 e 4 falharam em pontos diferentes do mesmo caminho: montar a requisição e ler a resposta. Depois delas, o payload bruto e o uso de tokens passaram a ser gravados antes de qualquer parsing. Uma falha de leitura deixou de poder destruir a evidência de uma chamada que não se repete.

**Chamada única, sem retry.** Nenhuma etapa reexecutou uma chamada reprovada para obter um resultado melhor. As reprovações foram preservadas com o mesmo peso das aprovações.

## Código correspondente

Os módulos vivem em `src/tech_challenge_fase2/_historical/` e continuam executáveis por módulo, fora de `[project.scripts]`:

```bash
uv run python -m tech_challenge_fase2._historical.run_provider_real_evaluation_v4 --help
```

Não os execute durante a demonstração: vários chamam o provider real quando recebem credencial, e a evidência preservada não deve ser regerada.
