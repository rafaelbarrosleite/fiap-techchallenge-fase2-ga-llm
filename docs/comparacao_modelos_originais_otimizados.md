# Comparação dos modelos originais e otimizados

## Resultado principal exigido pela FIAP

| Modelo | Recall baseline | Recall GA | Δ recall | F1 baseline | F1 GA | Δ F1 | AUC baseline | AUC GA | FN baseline→GA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,928571 | 0,976190 | **+0,047619** | 0,951220 | 0,976190 | **+0,024971** | 0,996032 | 0,997685 | **3→1** |
| Random Forest | 0,904762 | 0,928571 | **+0,023810** | 0,950000 | 0,962963 | **+0,012963** | 0,996032 | 0,991402 | **4→3** |
| KNN | 0,904762 | 0,904762 | 0,000000 | 0,938272 | 0,950000 | +0,011728 | 0,982308 | 0,973545 | 4→4 |

### Leitura por família

- **Regressão Logística:** o GA corrigiu dois casos malignos sem introduzir erro novo. O ganho de recall de CV foi confirmado e o número de FN caiu de três para um.
- **Random Forest:** o GA corrigiu um caso maligno sem introduzir erro novo. Recall e F1 aumentaram, mas ROC-AUC caiu 0,004630.
- **KNN:** o GA manteve o mesmo recall e os mesmos quatro FN. Eliminou o falso positivo do baseline, melhorando F1 e accuracy, mas reduziu ROC-AUC e trocou parte dos casos errados.

## Matrizes de confusão

| Modelo | Baseline | GA |
|---|---|---|
| Regressão Logística | `[[71,1],[3,39]]` | `[[71,1],[1,41]]` |
| Random Forest | `[[72,0],[4,38]]` | `[[72,0],[3,39]]` |
| KNN | `[[71,1],[4,38]]` | `[[72,0],[4,38]]` |

## Comparação complementar com RandomizedSearchCV

| Modelo | GA: recall/F1/AUC/FN | Aleatória: recall/F1/AUC/FN | Diferença relevante |
|---|---|---|---|
| Regressão Logística | 0,976190 / 0,976190 / 0,997685 / 1 | 0,976190 / 0,976190 / 0,997685 / 1 | Mesmas previsões e probabilidades no teste; parâmetros C não são canonicamente idênticos |
| Random Forest | 0,928571 / 0,962963 / 0,991402 / 3 | 0,928571 / 0,962963 / 0,997024 / 3 | Mesma matriz; busca aleatória teve AUC +0,005622 |
| KNN | 0,904762 / 0,950000 / 0,973545 / 4 | 0,904762 / 0,950000 / 0,973545 / 4 | Mesma solução canônica e mesmo treino; não são evidências independentes |

## Custo observado da otimização

| Família | Melhor GA | Avaliações únicas | Fits CV | Tempo de busca |
|---|---|---:|---:|---:|
| Regressão Logística | B | 450 | 2.250 | 24,15 s |
| Random Forest | C | 1.638 | 8.190 | 1.962,28 s |
| KNN | A | 78 | 390 | 1,33 s |

Esse custo é o esforço histórico para encontrar os candidatos, não o custo da inferência final. Tempo não participou do ranking. A Random Forest consumiu muito mais processamento para obter uma redução de um FN neste holdout; isso é um trade-off acadêmico, não uma decisão clínica.

## Conclusão responsável

O GA foi eficaz no objetivo primário para duas das três famílias neste conjunto final: LR e RF reduziram falsos negativos. No KNN, o ganho de recall da CV não generalizou. Os intervalos são amplos e há somente 42 casos malignos; portanto, o resultado sustenta uma conclusão de engenharia sobre este experimento, não superioridade clínica.
