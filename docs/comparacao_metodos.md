# Comparação entre baseline, GA e busca aleatória

> Nota de proveniência: a tabela histórica de CV denomina o melhor GA logístico como C. Após a correção canônica da seleção, o plano assinado da Missão 4 avaliou GA B, empatado nas métricas agregadas registradas. As duas origens são preservadas; os resultados confirmatórios usam a autoridade do plano final.

## Comparabilidade

Os três métodos usam as mesmas 455 linhas de desenvolvimento, cinco folds estratificados fixos, seed 42, pipelines com normalização interna quando necessária, limiar 0,5 e a mesma fórmula de fitness. Métricas históricas do teste final em `baseline_results.json` foram deliberadamente excluídas.

O `RandomizedSearchCV` recebeu múltiplos scorers para recall maligno, F1 maligno e ROC-AUC. Foi executado com `refit=False`; depois, o código calculou fitness base, penalidade do desvio do recall e desempates sobre `cv_results_`. Assim, nenhum estimador vencedor foi reajustado em todo o desenvolvimento nesta missão.

## Resultado por modelo

| Modelo | Método | Fitness | Recall | Desvio recall | F1 | ROC-AUC | Candidatos | Tempo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | Baseline CV | 0,958152 | 0,952941 | 0,039896 | 0,964048 | 0,995769 | 1 | 0,02 s |
| Regressão Logística | Melhor GA C | 0,973166 | 0,970588 | 0,018602 | 0,973479 | 0,995356 | 1.080 | 35,13 s |
| Regressão Logística | RandomizedSearchCV | 0,973166 | 0,970588 | 0,018602 | 0,973479 | 0,995356 | 1.080 | 529,06 s |
| Random Forest | Baseline CV | 0,947138 | 0,941176 | 0,032219 | 0,949490 | 0,988545 | 1 | 0,59 s |
| Random Forest | Melhor GA C | 0,961648 | 0,958824 | 0,029994 | 0,964503 | 0,988184 | 1.638 | 1.962,29 s |
| Random Forest | RandomizedSearchCV | 0,958248 | 0,958824 | 0,029994 | 0,950438 | 0,988958 | 1.638 | 2.260,37 s |
| KNN | Baseline CV | 0,932984 | 0,917647 | 0,034300 | 0,951042 | 0,987100 | 1 | 0,04 s |
| KNN | Melhor GA A | 0,953990 | 0,947059 | 0,034300 | 0,966708 | 0,983385 | 78 | 1,34 s |
| KNN | RandomizedSearchCV | 0,953990 | 0,947059 | 0,034300 | 0,966708 | 0,983385 | 78 | 1,82 s |

## Diferença do melhor GA para o baseline

| Modelo | Fitness absoluto | Fitness relativo | Recall | Desvio recall |
|---|---:|---:|---:|---:|
| Regressão Logística | +0,015013 | +1,567% | +0,017647 | -0,021294 |
| Random Forest | +0,014510 | +1,532% | +0,017647 | -0,002225 |
| KNN | +0,021006 | +2,252% | +0,029412 | 0,000000 |

O ROC-AUC não aumentou em todos os casos: o fitness melhorou porque recall e F1 têm maior peso. Isso precisa ser lido como trade-off do objetivo aprovado, não como melhoria universal.

## GA versus RandomizedSearchCV

- Em Regressão Logística, os métodos empataram em todas as métricas agregadas exibidas. O desempate canônico reteve a busca aleatória com `C=0,1767893108`, `penalty=l2` e `class_weight=balanced`. Esse desempate não indica qualidade superior.
- Em Random Forest, o GA superou a busca aleatória em 0,003400 de fitness, com mesmo recall e desvio, F1 maior em 0,014065 e AUC menor em 0,000774.
- Em KNN, ambos encontraram exatamente `k=3`, peso uniforme e Manhattan via Minkowski `p=1`, com métricas idênticas. O GA A foi retido como primeira origem na ordem determinística.

A busca aleatória levou 2.791,26 s (46,52 min). O tempo não entrou no ranking. O `RandomizedSearchCV` capturou 29 avisos de convergência do Liblinear entre os 5.400 ajustes logísticos; não houve avisos em RF ou KNN. A API agrega esses avisos durante `fit` e não os associa de forma segura a um índice de candidato, por isso eles são preservados como lista da execução e não usados para ocultar ou descartar resultados.

## Vencedores provisórios

| Modelo | Origem congelada | Motivo |
|---|---|---|
| Regressão Logística | RandomizedSearchCV | Empate métrico; chave canônica final |
| Random Forest | GA C | Maior fitness final |
| KNN | GA A | Solução idêntica à busca aleatória; primeira origem fixa |

O vencedor global provisório é a Regressão Logística da busca aleatória, com fitness 0,973166. Ela merece seguir para a avaliação final única porque lidera o ranking de CV; isso ainda não permite afirmar desempenho no teste nem valor clínico.

## Limitações

- Uma única seed oficial foi usada para manter comparabilidade, não para estimar a distribuição do GA.
- Os folds são os mesmos para todos, mas diferenças pequenas podem refletir particularidades dessa divisão.
- A comparação não é um teste estatístico de superioridade.
- O orçamento mede candidatos únicos, mas operadores e padrões de amostragem diferem entre GA e busca aleatória.
- O KNN tem espaço finito de 120 soluções; o orçamento comparado foi 78 porque A foi a origem GA retida.
- Nenhuma conclusão sobre generalização final pode ser feita antes da abertura única do holdout.

## Avaliação confirmatória concluída

O holdout foi avaliado somente após o congelamento. Resultado complementar:

| Modelo | Método | Recall teste | F1 teste | ROC-AUC teste | FN |
|---|---|---:|---:|---:|---:|
| Regressão Logística | Baseline | 0,928571 | 0,951220 | 0,996032 | 3 |
|  | GA B | 0,976190 | 0,976190 | 0,997685 | 1 |
|  | RandomizedSearchCV | 0,976190 | 0,976190 | 0,997685 | 1 |
| Random Forest | Baseline | 0,904762 | 0,950000 | 0,996032 | 4 |
|  | GA C | 0,928571 | 0,962963 | 0,991402 | 3 |
|  | RandomizedSearchCV | 0,928571 | 0,962963 | 0,997024 | 3 |
| KNN | Baseline | 0,904762 | 0,938272 | 0,982308 | 4 |
|  | GA A | 0,904762 | 0,950000 | 0,973545 | 4 |
|  | RandomizedSearchCV | 0,904762 | 0,950000 | 0,973545 | 4 |

No KNN, GA e busca aleatória são a mesma solução e não contam como evidências independentes. Em LR, configurações distintas produziram saídas idênticas neste teste. Em RF, as matrizes foram iguais entre GA e busca aleatória, com diferença apenas no ranking probabilístico refletido pela AUC.

O vencedor global congelado não foi alterado pelo teste. A Regressão Logística da busca aleatória permanece prevista para a demonstração futura por causa da regra pré-holdout, não porque o teste iniciou novo desempate.
