# Protocolo de seleção e congelamento final

## Objetivo

Este protocolo impede que o conjunto de teste influencie hiperparâmetros, limiar, modelo ou narrativa. Os candidatos abaixo são “provisórios”: venceram somente por validação cruzada no desenvolvimento.

## Dados e seeds bloqueados

- dataset SHA-256: `1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33`;
- split 80/20 estratificado: seed 42;
- cinco folds estratificados com embaralhamento: seed 42;
- engine GA: seed 42;
- estimadores compatíveis: seed 42;
- limiar: 0,5;
- `n_jobs=1`.

## Regra determinística

1. maior fitness final, comparado com tolerância de `1e-12`;
2. maior recall maligno médio;
3. menor desvio-padrão do recall;
4. maior F1 maligno médio;
5. maior ROC-AUC médio;
6. menor complexidade objetiva para Random Forest: menos árvores e menor profundidade finita;
7. chave canônica normalizada dos hiperparâmetros.

Tempo nunca desempata. Se dois métodos encontrarem exatamente a mesma chave e as mesmas métricas, representam a mesma solução preditiva; a primeira origem na ordem fixa é retida apenas para serialização.

## Correção auditada da chave canônica

Depois da primeira seleção, foi detectado que o GA representava regularização logística como `log10_c`, enquanto a busca aleatória usava `C`. O KNN também incluía o nome do modelo em uma origem e não em outra. Isso podia atribuir uma origem diferente a hiperparâmetros equivalentes.

A correção converte todas as origens para a mesma representação preditiva antes do último desempate. Não houve mudança de fitness, folds, espaços, operadores ou avaliações. Foram regenerados somente `methods_summary.json`, `frozen_candidates.json`, `selection_manifest.json` e as figuras derivadas. Os nove experimentos e o `RandomizedSearchCV` permaneceram válidos.

## Candidatos congelados

### Regressão Logística

- origem: `RandomizedSearchCV`;
- `C=0,17678931080693083`;
- `penalty=l2`;
- `class_weight=balanced`;
- fitness CV: `0,9731659423904974`.

### Random Forest

- origem: `GA_C`;
- `n_estimators=107`;
- `max_depth=7`;
- `min_samples_split=2`;
- `min_samples_leaf=1`;
- `max_features=0.5`;
- `class_weight=None`;
- fitness CV: `0,961648019213057`.

### KNN

- origem serializada: `GA_A`;
- `n_neighbors=3`;
- `weights=uniform`;
- `metric=minkowski`;
- `p=1`;
- fitness CV: `0,9539899923880164`;
- observação: a busca aleatória encontrou exatamente a mesma solução.

O vencedor global provisório é a Regressão Logística da busca aleatória.

## Artefatos de autoridade

- `artifacts/selection/frozen_candidates.json`: candidatos e métricas congelados;
- `artifacts/selection/selection_manifest.json`: candidatos considerados, critérios e assinaturas de origem;
- `artifacts/official/execution_manifest.json`: protocolo e assinatura do código GA;
- `artifacts/comparison/baseline_cv.json`: baseline metodologicamente comparável;
- `artifacts/comparison/randomized_search_cv.json`: busca aleatória e orçamentos.

Esses arquivos são gerados localmente e ignorados pelo Git. A assinatura atual de `frozen_candidates.json` deve ser verificada diretamente no arquivo antes da quarta missão.

## Procedimento para a quarta missão

1. validar hashes, schemas e assinaturas dos artefatos congelados;
2. recusar qualquer alteração de hiperparâmetros, espaço, limiar ou regra após essa validação;
3. construir pipelines novos a partir dos candidatos congelados;
4. reajustar cada candidato uma única vez nas 455 linhas de desenvolvimento;
5. abrir o teste final uma única vez para métricas previamente definidas;
6. registrar matriz de confusão, falsos negativos, precision, recall, F1, ROC-AUC e tempos;
7. relatar todos os resultados, inclusive se o ranking de CV não se confirmar;
8. não retornar à seleção com base no teste.

Não implementar LLM, API, interface ou cloud antes de encerrar e auditar essa avaliação final, salvo nova decisão explícita de escopo.

## Execução do protocolo

O procedimento foi concluído em 24 de agosto de 2026:

1. 79 testes e todas as assinaturas foram validados sem fit/predição no holdout;
2. o plano foi congelado sob `b94cbc663473ab040e89961144c69600062a03a09803c73027ca4163bc4fe1f7`;
3. nove origens foram reconstruídas em oito Pipelines canônicos;
4. os oito Pipelines foram ajustados somente nas 455 linhas;
5. as 114 linhas foram inferidas na sessão confirmatória;
6. 1.026 registros de previsão foram persistidos;
7. métricas, pareamentos, ICs, modelos e figuras foram assinados;
8. uma nova chamada ao comando apenas carregou o manifesto concluído.

O manifesto registra oito fits, oito grupos de inferência, zero GA, zero `RandomizedSearchCV` e zero alteração de limiar. Nenhum resultado de teste retornou à seleção.

### Ressalva de origem logística

Este documento citava GA C como melhor GA logístico. O `selection_manifest.json` vigente após a correção de chave canônica contém `GA_B` entre os três candidatos considerados. Como a quarta missão exige seguir os artefatos assinados, GA B foi avaliado. Isso não altera as métricas de CV do empate e foi registrado explicitamente em vez de ser corrigido em silêncio.

### Estado após a avaliação

- vencedor global pré-holdout: Regressão Logística da busca aleatória, mantido;
- LR GA: reduziu FN de 3 para 1;
- RF GA: reduziu FN de 4 para 3;
- KNN GA: manteve 4 FN e o mesmo recall;
- conclusão: eficácia confirmada no objetivo prioritário para duas famílias, sem validade clínica.
