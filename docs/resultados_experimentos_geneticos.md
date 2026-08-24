# Resultados dos experimentos genéticos oficiais

## Escopo e protocolo

Os nove experimentos foram executados em série em 19 de agosto de 2026, com seed 42, somente nas 455 linhas de desenvolvimento. Cada indivíduo foi avaliado nos mesmos cinco folds de `StratifiedKFold(shuffle=True, random_state=42)`. O teste final de 114 linhas não participou de fitness, gráficos, seleção ou desempate.

O fitness foi:

```text
0,60 * recall_maligno_medio
+ 0,25 * F1_maligno_medio
+ 0,15 * ROC_AUC_medio
- 0,10 * desvio_padrao_do_recall
```

## Resultado dos nove experimentos

| Configuração | Modelo | Fitness | Recall | Desvio recall | F1 | ROC-AUC | Únicos | Cache hits | Tempo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Regressão Logística | 0,973150 | 0,970588 | 0,018602 | 0,973479 | 0,995253 | 119 | 101 | 17,87 s |
| A | Random Forest | 0,955954 | 0,952941 | 0,023529 | 0,953383 | 0,987977 | 159 | 61 | 232,38 s |
| A | KNN | 0,953990 | 0,947059 | 0,034300 | 0,966708 | 0,983385 | 78 | 142 | 1,34 s |
| B | Regressão Logística | 0,973166 | 0,970588 | 0,018602 | 0,973479 | 0,995356 | 450 | 390 | 24,15 s |
| B | Random Forest | 0,959515 | 0,958824 | 0,029994 | 0,955910 | 0,988287 | 732 | 108 | 789,92 s |
| B | KNN | 0,953990 | 0,947059 | 0,034300 | 0,966708 | 0,983385 | 119 | 721 | 2,12 s |
| C | Regressão Logística | 0,973166 | 0,970588 | 0,018602 | 0,973479 | 0,995356 | 1.080 | 780 | 35,13 s |
| C | Random Forest | 0,961648 | 0,958824 | 0,029994 | 0,964503 | 0,988184 | 1.638 | 222 | 1.962,29 s |
| C | KNN | 0,953990 | 0,947059 | 0,034300 | 0,966708 | 0,983385 | 120 | 1.740 | 2,29 s |

Todos terminaram por `max_generations`. Não houve falhas de ajuste. Houve três `ConvergenceWarning` do Liblinear em um candidato não vencedor, uma ocorrência em cada configuração logística devido à população inicial reproduzida pela seed comum; os melhores candidatos não tiveram avisos. Os artefatos individuais contêm as cinco métricas por dobra, genoma completo, histórico e assinatura. Os operadores detectaram e repararam 502 combinações KNN incompatíveis após crossover/mutação; nenhuma chegou inválida ao fitness.

## Melhores genomas

### Regressão Logística — configuração C

```json
{
  "log10_c": -0.7543542804560831,
  "penalty": "l2",
  "class_weight": "balanced"
}
```

O fitness e todas as médias de C empataram com B na precisão numérica registrada, mas os valores contínuos de `C` diferem. Como tempo não pode desempatar, a chave canônica fixa reteve C. Para equilíbrio entre qualidade e custo, B é mais eficiente: mesma métrica observada com 450 em vez de 1.080 avaliações únicas.

### Random Forest — configuração C

```json
{
  "n_estimators": 107,
  "max_depth": 7,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": 0.5,
  "class_weight": null
}
```

C superou B em 0,002133 de fitness. A melhora ocorreu na geração 21; houve platôs intermediários, mas não parada antecipada.

### KNN — solução comum a A, B e C

```json
{
  "n_neighbors": 3,
  "weights": "uniform",
  "metric": "minkowski",
  "p": 1
}
```

As três configurações chegaram à mesma solução e às mesmas métricas. C cobriu todas as 120 combinações únicas, mas não melhorou A. A foi retida como primeira configuração na ordem fixa para a solução idêntica.

## Custo e cache

A bateria levou 3.067,47 s, ou 51,12 minutos. Foram 8.760 solicitações, 4.495 avaliações únicas, 4.265 cache hits e 22.475 ajustes de modelos. O teto sem cache era 43.800 ajustes.

| Configuração | Avaliações únicas | Fits | Tempo total |
|---|---:|---:|---:|
| A | 356 | 1.780 | 251,58 s |
| B | 1.301 | 6.505 | 816,18 s |
| C | 2.838 | 14.190 | 1.999,71 s |

A Random Forest respondeu por aproximadamente 97,3% do tempo total. O KNN acumulou muitos cache hits porque seu espaço finito foi quase ou totalmente coberto.

## Evolução e diversidade

- Regressão Logística A encontrou 0,973150 até a geração 9; B e C terminaram em um platô praticamente idêntico.
- Random Forest A e B melhoraram em etapas. C permaneceu em 0,956705 entre as gerações 5 e 13, avançou novamente e encontrou 0,961648 na geração 21.
- KNN convergiu cedo: A chegou à solução final na geração 7; B na geração 3; C na geração 1.
- LR e Random Forest mantiveram diversidade populacional próxima de 1,0 porque o mecanismo força filhos distintos dentro de cada geração.
- No KNN, a diversidade oscilou quando o espaço de 120 soluções ficou saturado; ainda assim, a população permaneceu válida.

## Respostas às perguntas de análise

- **O GA melhorou o fitness?** Sim nos três modelos em relação ao baseline CV: +0,015013 em LR, +0,014510 em RF e +0,021006 em KNN.
- **Melhorou recall maligno?** Sim: +0,017647 em LR, +0,017647 em RF e +0,029412 em KNN.
- **Reduziu instabilidade?** LR reduziu o desvio em 0,021294; RF reduziu em 0,002225; KNN permaneceu igual.
- **Qual configuração equilibrou qualidade e custo?** A foi suficiente para KNN; B foi o melhor compromisso para LR; C trouxe ganho mensurável apenas em RF.
- **Populações maiores sempre ajudaram?** Não. Aumentar orçamento não mudou KNN e quase não mudou LR. Em RF, C encontrou resultado melhor, mas com custo alto.
- **Houve convergência?** Sim, com platôs claros. O salto tardio de RF C mostra que um platô curto não bastaria para concluir estagnação definitiva.
- **Os modelos responderam igualmente?** Não. KNN esgotou um espaço pequeno; LR formou um platô amplo; RF se beneficiou mais da exploração longa.
- **A diferença é clinicamente relevante?** Não é possível afirmar. São diferenças de uma única validação cruzada e não uma validação clínica.

## Figuras

As sete figuras em `reports/figures/` foram geradas exclusivamente com CV: melhor fitness, fitness médio, diversidade, A/B/C, métodos, recall com variabilidade e custo.

## Confirmação posterior no teste final

A quarta missão não reexecutou nenhum experimento desta página. Ela reconstruiu e ajustou apenas os candidatos previamente congelados e confirmou:

| Modelo | Δ recall GA−baseline no teste | Δ F1 | Δ ROC-AUC | Δ FN |
|---|---:|---:|---:|---:|
| Regressão Logística | +0,047619 | +0,024971 | +0,001653 | −2 |
| Random Forest | +0,023810 | +0,012963 | −0,004630 | −1 |
| KNN | 0,000000 | +0,011728 | −0,008763 | 0 |

Assim, o ganho de recall observado em CV generalizou para LR e RF, mas não para KNN. A Regressão Logística avaliada como melhor GA foi `GA_B`, conforme o manifesto assinado regenerado após a correção canônica; referências anteriores a C descrevem o empate antes dessa autoridade final.

Detalhes, intervalos e matrizes estão em `avaliacao_final.md`. Os resultados de teste não alteraram esta busca nem seus vencedores históricos.
