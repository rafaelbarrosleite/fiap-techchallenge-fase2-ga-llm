# Contrato LLM V2 — semântica comparativa explícita

## 1. Motivação

A Missão 7.4 cria uma versão paralela do contrato de entrada, saída e prompts da camada LLM. O objetivo é tornar cada relação comparativa inequívoca e garantir que toda conclusão estatística exigida possua evidência agregada autoritativa no payload.

A missão foi executada totalmente offline. Não houve chamada ao provider OpenAI, uso de tokens, acesso a previsões individuais, novo treino, otimização ou inferência.

## 2. Falha observada na Missão 7.3

Na primeira avaliação científica real, o `gpt-5.5` passou o schema, segurança, completude, clareza, calibração e 138 dos 139 checks factuais. A única divergência ocorreu no campo V1:

`random_forest.same_threshold_different_auc`

O provider atribuiu `true` porque Random Forest GA e RandomizedSearchCV possuem a mesma matriz de confusão e ROC-AUC diferente. Essa interpretação é numericamente correta. O factuality checker V1, porém, interpretava o campo como baseline versus GA, para o qual a matriz é diferente e o valor esperado é `false`.

A Missão 7.3 continua congelada como `scientific_evaluation_approved=false`. A V2 não reclassifica retroativamente essa execução.

## 3. Por que a falha é ambiguidade semântica

O booleano V1 informa uma propriedade, mas não identifica os dois métodos comparados. Como três métodos estão presentes no mesmo objeto, mais de um par é plausível. O valor isolado não permite distinguir:

- baseline versus GA;
- GA versus RandomizedSearchCV;
- baseline versus RandomizedSearchCV.

A divergência não envolveu número inventado ou método inexistente. Ela decorreu da ausência do sujeito comparativo no contrato.

## 4. Limitações auditadas no V1

Os 139 checks factuais foram auditados individualmente:

| Classificação | Quantidade |
|---|---:|
| `unambiguous` | 99 |
| `needs_explicit_pair` | 39 |
| `redundant` | 1 |
| `needs_more_evidence` | 0 dentro dos 139 |
| `unsafe_to_require` | 0 dentro dos 139 |

Os 39 checks com par implícito abrangiam mudanças GA versus baseline, trade-offs, matriz de confusão versus ROC-AUC, confirmação CV/holdout e incerteza baseline versus GA. Eles foram substituídos por checks indexados por `comparison_id`.

O check redundante de números narrativos foi mantido como defesa em profundidade. Nenhum check V1 foi removido ou alterado in-place.

Fora dos 139 checks, a exigência textual de afirmar baixo poder do McNemar por poucas discordâncias foi classificada como `needs_more_evidence` no V1, pois o payload fornecia valores-p, mas não as contagens agregadas.

## 5. Decisões do V2

O contrato exige seleção explícita:

- `contract_version="v1"` preserva reprodução histórica;
- `contract_version="v2"` ativa schema, prompts, fake e evaluator V2;
- versões vazias ou desconhecidas são rejeitadas;
- não existe migração silenciosa.

O V2 declara:

- `schema_version="2.0"`;
- `contract_version="v2"`;
- `system_v2` e `explanation_v2`;
- nove pares comparativos fechados;
- direção de todos os deltas como `right_minus_left`;
- dados de McNemar vinculados ao par baseline versus GA.

Os novos campos centrais são:

- `comparison_id`;
- `left_method` e `right_method`;
- `left_candidate_id` e `right_candidate_id`;
- `evaluation_scope`;
- `same_confusion_matrix`;
- `different_roc_auc`;
- `recall_relation`, `f1_relation` e `roc_auc_relation`;
- `metric_delta.direction`;
- `left_wrong_right_correct`;
- `left_correct_right_wrong`;
- `discordant_total`;
- `low_count_warning`;
- `evidence_source`;
- `limited_by_few_discordances`.

## 6. Pares explícitos

Cada família possui exatamente três pares:

1. `baseline_vs_ga`;
2. `ga_vs_random_search`;
3. `baseline_vs_random_search`.

O identificador inclui a família, por exemplo:

```json
{
  "comparison_id": "random_forest__ga_vs_random_search",
  "model": "random_forest",
  "left_method": "ga",
  "right_method": "random_search",
  "same_confusion_matrix": true,
  "different_roc_auc": true,
  "roc_auc_relation": "right_higher",
  "metric_delta": {
    "direction": "right_minus_left"
  }
}
```

Para a ambiguidade da Missão 7.3, o V2 registra simultaneamente e sem conflito:

- `random_forest__baseline_vs_ga`: matriz diferente, ROC-AUC diferente;
- `random_forest__ga_vs_random_search`: mesma matriz, ROC-AUC diferente;
- `random_forest__baseline_vs_random_search`: matriz diferente, ROC-AUC diferente.

O schema rejeita um `comparison_id` combinado com métodos ou candidatos de outro par. Se valores corretos forem atribuídos ao par errado sem violar o tipo, o factuality checker V2 identifica os campos trocados pelo ID.

## 7. Tratamento de McNemar

Foi adotada a Estratégia A. O artefato assinado `uncertainty_results.json` já continha as discordâncias agregadas:

| Par baseline versus GA | Baseline errado/GA correto | Baseline correto/GA errado | Total | p | Baixa contagem |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 2 | 0 | 2 | 0,5 | sim |
| Random Forest | 1 | 0 | 1 | 1,0 | sim |
| KNN | 2 | 1 | 3 | 1,0 | sim |

Esses valores foram apenas reutilizados. O dataset, `final_predictions.json` e previsões individuais não foram lidos. Nenhuma estatística foi recalculada.

O LLM V2 só pode declarar limitação por poucas discordâncias quando `low_count_warning` e `discordant_total` estiverem presentes no mesmo `comparison_id`. Valor-p alto continua não sendo evidência de igualdade.

## 8. Compatibilidade histórica

Os arquivos de schema, factuality, providers e prompts V1 tiveram seus hashes registrados antes e depois da construção V2. Todos permaneceram idênticos.

Os artefatos das Missões 5, 7, 7.1, 7.2, 7.2.1 e 7.3 também foram protegidos por hashes. O validador V3 foi tornado read-only para não renovar timestamp ou manifesto durante validações futuras.

A resposta real da Missão 7.3 foi analisada apenas retrospectivamente:

- seus 138 checks válidos continuam válidos como evidência V1;
- o booleano histórico é correto para `random_forest__ga_vs_random_search`;
- ele é incorreto para `random_forest__baseline_vs_ga`;
- a narrativa RF pode ser reutilizada somente quando vinculada ao par explícito adequado;
- o objeto V1 completo não satisfaz o schema V2;
- o status histórico não foi alterado.

## 9. Avaliação offline

O `FakeLLMProviderV2` é determinístico, não usa rede e registra zero tokens pagos. A avaliação oficial offline apresentou:

- schema: passou;
- factualidade: 327 checks aprovados;
- segurança: passou;
- completude: passou;
- clareza: passou;
- calibração científica: passou;
- score global: 1,0;
- nove pares explícitos: presentes;
- três comparações de incerteza: presentes;
- seleção congelada: preservada;
- disclaimer: preservado.

Os testes adversariais reproduzem a classe de erro da 7.3 e comprovam que o evaluator detecta valores atribuídos ao par errado.

## 10. Critérios para futura chamada real

O manifesto V2 registra `ready_for_real_v2_evaluation=true`. Isso autoriza apenas o planejamento de uma missão futura; nenhuma avaliação real V2 ocorreu nesta etapa.

Uma futura chamada deve:

1. selecionar `contract_version="v2"` explicitamente;
2. usar `system_v2` e `explanation_v2` sem alteração silenciosa;
3. enviar somente o payload agregado V2 validado;
4. usar Structured Outputs com o JSON Schema V2;
5. persistir a resposta raw-first;
6. executar os 327 checks factuais V2 e as demais barreiras independentes;
7. preservar a resposta original diante de qualquer falha;
8. não reclassificar a Missão 7.3.

Comandos offline:

```bash
uv run build-llm-contract-v2
uv run validate-llm-contract-v2
uv run pytest
```
