# Avaliação final confirmatória

## Resumo em linguagem simples

“Abrir o teste” significa usar, pela primeira vez nesta etapa de otimização, as 114 linhas reservadas para medir como os candidatos escolhidos anteriormente se comportam em dados que não orientaram a busca. Isso ocorreu somente depois de concluir o algoritmo genético, a busca aleatória, a comparação por validação cruzada e o congelamento dos candidatos.

A validação cruzada ajudou a escolher hiperparâmetros dentro das 455 linhas de desenvolvimento. O teste final tem outro papel: estimar generalização. Ele não é uma nova competição nem autoriza trocar de candidato depois de observar o resultado.

## Barreira de pré-validação

Antes de qualquer ajuste ou predição no holdout, `prepare-final-evaluation`:

- aprovou 79 testes automatizados;
- confirmou o SHA-256 `1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33`;
- validou as assinaturas dos candidatos e do manifesto de seleção;
- confirmou 455 registros de desenvolvimento e 114 de teste, sem sobreposição;
- confirmou classes 285/170 no desenvolvimento e 72/42 no teste;
- instanciou os nove Pipelines sem ajustá-los;
- confirmou limiar 0,5 e ausência de componentes de busca;
- registrou zero chamadas de `fit`, `predict`, `predict_proba` e `score`;
- congelou o plano com assinatura `b94cbc663473ab040e89961144c69600062a03a09803c73027ca4163bc4fe1f7`.

O histórico `artifacts/baseline_results.json` já continha métricas do holdout desde a primeira missão. Essa exceção não foi apagada: o preflight comprova que o arquivo não integra a linhagem assinada de seleção. Os artefatos de GA, baseline CV, busca aleatória e congelamento usados para escolher candidatos permaneceram restritos ao desenvolvimento.

## Candidatos executados

Foram preservadas nove origens e realizados oito treinos canônicos:

| Família | Método | Origem | Configuração resumida |
|---|---|---|---|
| Regressão Logística | Baseline | `baseline_cv` | C=1, L2, lbfgs, sem peso de classe |
| Regressão Logística | GA | `GA_B` | C=0,1766293561, L2, liblinear, balanced |
| Regressão Logística | Aleatória | `RandomizedSearchCV` | C=0,1767893108, L2, liblinear, balanced |
| Random Forest | Baseline | `baseline_cv` | 200 árvores, profundidade livre, balanced |
| Random Forest | GA | `GA_C` | 107 árvores, profundidade 7, sem peso de classe |
| Random Forest | Aleatória | `RandomizedSearchCV` | 101 árvores, profundidade 16, balanced |
| KNN | Baseline | `baseline_cv` | k=5, uniforme, Minkowski p=2 |
| KNN | GA | `GA_A` | k=3, uniforme, Minkowski p=1 |
| KNN | Aleatória | `RandomizedSearchCV` | mesma solução canônica do GA |

A documentação anterior citava GA C para a Regressão Logística. O manifesto assinado vigente, regenerado após a correção de chave canônica da terceira missão, contém GA B. A avaliação seguiu o manifesto de autoridade, sem alterar silenciosamente a origem.

## Resultado primário: baseline versus GA

Valores completos permanecem no JSON; a tabela arredonda somente para leitura.

| Modelo | Método | Recall M | F1 M | ROC-AUC | FN | Matriz `[[TN,FP],[FN,TP]]` |
|---|---|---:|---:|---:|---:|---|
| Regressão Logística | Baseline | 0,928571 | 0,951220 | 0,996032 | 3 | `[[71,1],[3,39]]` |
|  | GA | 0,976190 | 0,976190 | 0,997685 | 1 | `[[71,1],[1,41]]` |
|  | Δ GA−base | **+0,047619** | **+0,024971** | +0,001653 | **−2** | — |
| Random Forest | Baseline | 0,904762 | 0,950000 | 0,996032 | 4 | `[[72,0],[4,38]]` |
|  | GA | 0,928571 | 0,962963 | 0,991402 | 3 | `[[72,0],[3,39]]` |
|  | Δ GA−base | **+0,023810** | **+0,012963** | −0,004630 | **−1** | — |
| KNN | Baseline | 0,904762 | 0,938272 | 0,982308 | 4 | `[[71,1],[4,38]]` |
|  | GA | 0,904762 | 0,950000 | 0,973545 | 4 | `[[72,0],[4,38]]` |
|  | Δ GA−base | 0,000000 | +0,011728 | −0,008763 | 0 | — |

## Métricas adicionais

| Modelo | Método | Accuracy | Precision M | Especificidade | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| Regressão Logística | Baseline | 0,964912 | 0,975000 | 0,986111 | 0,957341 |
|  | GA | 0,982456 | 0,976190 | 0,986111 | 0,981151 |
| Random Forest | Baseline | 0,964912 | 1,000000 | 1,000000 | 0,952381 |
|  | GA | 0,973684 | 1,000000 | 1,000000 | 0,964286 |
| KNN | Baseline | 0,956140 | 0,974359 | 0,986111 | 0,945437 |
|  | GA | 0,964912 | 1,000000 | 1,000000 | 0,952381 |

## Comparação complementar com busca aleatória

| Modelo | Método | Recall M | F1 M | ROC-AUC | FN |
|---|---|---:|---:|---:|---:|
| Regressão Logística | Busca aleatória | 0,976190 | 0,976190 | 0,997685 | 1 |
| Random Forest | Busca aleatória | 0,928571 | 0,962963 | 0,997024 | 3 |
| KNN | Busca aleatória | 0,904762 | 0,950000 | 0,973545 | 4 |

LR GA e busca aleatória têm hiperparâmetros contínuos ligeiramente diferentes, mas produziram exatamente as mesmas previsões e probabilidades neste teste. Em RF, GA e busca aleatória tiveram a mesma matriz, porém a busca aleatória obteve ROC-AUC maior. No KNN, GA e busca aleatória são a mesma configuração canônica, compartilharam treino e não contam como evidências independentes.

## Casos alterados

- Regressão Logística GA corrigiu os índices técnicos 190 e 205, sem introduzir erro novo.
- Random Forest GA corrigiu o índice 100, sem introduzir erro novo.
- KNN GA corrigiu os índices 99 e 208 e introduziu erro no índice 39. O saldo foi um acerto adicional, mas o recall e os quatro FN permaneceram iguais porque as mudanças envolveram classes/resultados diferentes.

Os índices são apenas identificadores técnicos do dataset público. Nenhuma variável preditora ou dado pessoal foi persistido no artefato de previsões.

## Incerteza

Os intervalos de recall usam Wilson 95% sobre 42 casos malignos:

| Modelo | Baseline — recall [IC95%] | GA — recall [IC95%] |
|---|---|---|
| Regressão Logística | 0,9286 [0,8099; 0,9754] | 0,9762 [0,8768; 0,9958] |
| Random Forest | 0,9048 [0,7793; 0,9623] | 0,9286 [0,8099; 0,9754] |
| KNN | 0,9048 [0,7793; 0,9623] | 0,9048 [0,7793; 0,9623] |

O bootstrap pareado de 5.000 réplicas, seed 42, estimou os deltas GA−baseline:

| Modelo | Δ recall | IC95% bootstrap | McNemar exato | Discordantes |
|---|---:|---|---:|---:|
| Regressão Logística | +0,047619 | [0,000000; 0,121951] | p=0,5 | 2 |
| Random Forest | +0,023810 | [0,000000; 0,080027] | p=1,0 | 1 |
| KNN | 0,000000 | [−0,069767; 0,068182] | p=1,0 | 3 |

Todas as contagens discordantes são inferiores a dez. Os testes têm baixo poder: p alto não prova igualdade, p baixo não provaria relevância clínica. A largura dos intervalos confirma que uma ou duas observações alteram materialmente o resultado.

## CV versus teste

| Modelo | Método | Recall CV | Recall teste | Teste−CV |
|---|---|---:|---:|---:|
| Regressão Logística | Baseline | 0,952941 | 0,928571 | −0,024370 |
|  | GA | 0,970588 | 0,976190 | +0,005602 |
| Random Forest | Baseline | 0,941176 | 0,904762 | −0,036415 |
|  | GA | 0,958824 | 0,928571 | −0,030252 |
| KNN | Baseline | 0,917647 | 0,904762 | −0,012885 |
|  | GA | 0,947059 | 0,904762 | −0,042297 |

O ganho de recall observado em CV generalizou no holdout para LR e RF. No KNN, não se confirmou: o recall otimizado caiu até o mesmo valor do baseline. O ROC-AUC do GA também caiu em RF e KNN, mostrando que a melhoria não foi universal em todas as métricas.

## Execução única e auditoria

A execução começou em `2026-08-24T15:08:26Z`, terminou em `2026-08-24T15:09:05Z` e durou 39,65 s. O manifesto registra oito `fit` e oito grupos de inferência, nove origens relatadas, zero GA, zero `RandomizedSearchCV` e zero mudança de limiar.

Durante o acompanhamento, o status foi consultado enquanto o processo ainda estava ativo e aparecia como `started`. O procedimento de recuperação autorizado depois disso abortou na primeira verificação ao encontrar `completed`; não carregou dados, modelos nem repetiu inferência. Não foi criado artefato de recuperação. Uma nova chamada de `run-final-evaluation` apenas validou hashes e carregou o resultado existente.

As 1.026 previsões foram usadas para recalcular independentemente todas as métricas. Os oito modelos, sete JSONs assinados, seis figuras e demais arquivos do manifesto tiveram hashes conferidos.

## Conclusão

Neste holdout, a otimização genética generalizou melhor para Regressão Logística e Random Forest no objetivo prioritário de reduzir falsos negativos. Para KNN, não confirmou o ganho de recall observado na CV, embora tenha eliminado o único falso positivo e melhorado F1/accuracy. As diferenças permanecem incertas e não constituem validação clínica.

O candidato previsto para a futura demonstração permanece a Regressão Logística da busca aleatória, porque era o vencedor global congelado antes do teste. A avaliação final não reabriu a seleção, mesmo que LR GA tenha produzido o mesmo resultado de teste.

## Comandos

```bash
uv run prepare-final-evaluation
uv run run-final-evaluation
uv run pytest
```

Com resultado `completed` íntegro, `run-final-evaluation` apenas carrega o JSON existente e não repete `fit` ou predição.
