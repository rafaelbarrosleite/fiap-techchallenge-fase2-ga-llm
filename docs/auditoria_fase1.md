# Auditoria da Fase 1

## Escopo e integridade

A auditoria foi feita em modo somente leitura sobre:

- `README.md`;
- `relatorio_tecnico.md`;
- `requirements.txt`;
- `Dockerfile`;
- as 114 células de `Tech_Challenge_B.ipynb` (97 de código e 17 Markdown);
- a estrutura e o conteúdo agregado de `data.csv`;
- as saídas persistidas no notebook.

Antes da auditoria, foram registrados hashes SHA-256 dos arquivos da Fase 1. Ao final da missão, os hashes foram comparados novamente. Nenhum arquivo da Fase 1 foi alterado.

## 1. Dataset

O notebook carrega `data.csv` com `pandas.read_csv`. O arquivo tem 569 linhas de dados e um cabeçalho com 33 campos. Cada linha possui 32 valores porque há uma vírgula final; o pandas materializa esse campo como `Unnamed: 32`, totalmente vazio.

Constatações verificadas:

| Item | Resultado |
|---|---|
| Registros | 569 |
| Colunas lidas | 33 |
| Preditores usados | 30 numéricos |
| Classes | 357 benignos (62,74%) e 212 malignos (37,26%) |
| Ausências úteis | 0 |
| Ausências em `Unnamed: 32` | 569 |
| IDs duplicados | 0 |
| Linhas completas duplicadas | 0 |
| SHA-256 | `1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33` |

Origem declarada pela Fase 1: Breast Cancer Wisconsin (Diagnostic), distribuído no Kaggle em <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data>.

## 2. Tratamento dos dados

1. `Unnamed: 32` é removida por estar completamente vazia.
2. `id` é removida por ser identificador, sem papel preditivo justificável.
3. `diagnosis` é convertida com `{"B": 0, "M": 1}`.
4. Outliers são mantidos. A justificativa é adequada: extremos podem representar justamente tumores malignos.
5. Não há imputação, pois não existem ausências nos 30 preditores.
6. Não há seleção de atributos; os 30 preditores são usados.

## 3. Divisão e prevenção de vazamento

O notebook usa dois cortes com `random_state=42` e `stratify`:

1. 80% treino+validação e 20% teste;
2. 75% dos 80% para treino e 25% dos 80% para validação.

Resultado: 341 linhas de treino (60%), 114 de validação (20%) e 114 de teste (20%). As classes mantêm aproximadamente a proporção original.

`StandardScaler` está dentro de `Pipeline` para Regressão Logística e KNN. Assim, média e desvio são aprendidos somente no conjunto passado a `fit`. Random Forest não recebe escala, o que é adequado.

### Riscos encontrados

- Não há vazamento direto do teste para o ajuste do scaler ou dos modelos.
- A EDA e as correlações com o alvo foram calculadas sobre toda a base antes do corte. Como nenhum atributo foi removido com base nelas, isso não alterou diretamente o ajuste, mas é um risco de vazamento de processo. Na Fase 2, o teste deve ser reservado antes de qualquer análise supervisionada que possa influenciar decisões.
- O conjunto de validação foi consultado repetidamente para diferentes valores de `k` do KNN. Isso pode superajustar decisões à validação. A Fase 2 usará validação cruzada fixa dentro dos 80% de desenvolvimento.
- O teste foi usado apenas no modelo escolhido, o que é correto para a Fase 1. Porém, depois da escolha, a Regressão Logística não foi reajustada com treino+validação; permaneceu treinada em apenas 341 linhas.
- O notebook contém `!pip install shap`, uma mutação de ambiente no meio da execução. Isso dificulta automação e reprodutibilidade.

## 4. Modelos e hiperparâmetros originais

| Modelo | Pipeline e hiperparâmetros explícitos |
|---|---|
| Regressão Logística | `StandardScaler`; `LogisticRegression(max_iter=1000)`; demais padrões da versão instalada |
| Random Forest | `RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")` |
| KNN | `StandardScaler`; `KNeighborsClassifier(n_neighbors=5)`; demais padrões da versão instalada |

O notebook também testa `k` em 3, 5, 7, 9 e 11 e calcula erro para 1 a 20, mas mantém `k=5` no comparativo principal.

## 5. Reprodução dos resultados da Fase 1

A lógica de modelagem foi reimplementada e executada com Python 3.12.13, NumPy 2.3.2, pandas 2.3.1 e scikit-learn 1.7.1. As métricas da validação coincidiram com as saídas persistidas no notebook.

| Modelo na validação | Accuracy | Precision M | Recall M | F1 M | ROC-AUC | Matriz `[[TN, FP], [FN, TP]]` | FN |
|---|---:|---:|---:|---:|---:|---|---:|
| Regressão Logística | 0,973684 | 0,954545 | **0,976744** | 0,965517 | 0,995742 | `[[69, 2], [1, 42]]` | 1 |
| Random Forest | 0,973684 | 0,976190 | 0,953488 | 0,964706 | 0,990992 | `[[70, 1], [2, 41]]` | 2 |
| KNN | 0,973684 | 1,000000 | 0,930233 | 0,963855 | 0,994759 | `[[71, 0], [3, 40]]` | 3 |

A Regressão Logística foi escolhida pelo maior recall maligno na validação e por sua interpretabilidade. No teste original, ainda treinada somente nos 60%, os resultados reproduzidos foram:

| Accuracy | Precision M | Recall M | F1 M | ROC-AUC | Matriz | FN |
|---:|---:|---:|---:|---:|---|---:|
| 0,973684 | 0,975610 | 0,952381 | 0,963855 | 0,995370 | `[[71, 1], [2, 40]]` | 2 |

As quatro métricas publicadas na Fase 1 foram confirmadas. ROC-AUC foi calculada agora porque não fazia parte do relatório original.

## 6. Baseline corrigido da Fase 2

Para formar uma referência justa para a futura otimização, cada modelo foi reajustado nos mesmos 455 registros de desenvolvimento e avaliado nos 114 registros do teste final já reservado. Nenhuma otimização foi feita.

| Modelo no teste final | Accuracy | Precision M | Recall M | F1 M | ROC-AUC | Matriz | FN |
|---|---:|---:|---:|---:|---:|---|---:|
| Regressão Logística | 0,964912 | 0,975000 | **0,928571** | **0,951220** | **0,996032** | `[[71, 1], [3, 39]]` | 3 |
| Random Forest | 0,964912 | 1,000000 | 0,904762 | 0,950000 | **0,996032** | `[[72, 0], [4, 38]]` | 4 |
| KNN | 0,956140 | 0,974359 | 0,904762 | 0,938272 | 0,982308 | `[[71, 1], [4, 38]]` | 4 |

A verificação final dos dois protocolos levou 0,278 s neste ambiente. Tempos variam por máquina e não devem ser comparados sem controlar hardware e carga; o valor de cada execução fica registrado no JSON gerado.

O desempenho de recall da Regressão Logística caiu de 0,952381 para 0,928571 após o reajuste com 80%. Isso não é erro de implementação: três observações malignas mudaram de lado da fronteira de decisão. O resultado ilustra por que a futura otimização deve ser avaliada empiricamente e por que o teste não pode orientar a busca.

## 7. Reprodutibilidade

### O que já era reproduzível

- dataset local e íntegro;
- cortes estratificados com seed 42;
- seed do Random Forest;
- scaler dentro dos pipelines;
- hiperparâmetros principais explícitos.

### O que impedia reprodução forte

- dependências da Fase 1 sem versões;
- Python sem versão exata no notebook;
- defaults de bibliotecas não congelados;
- instalação de SHAP dentro do notebook;
- mistura de EDA, treino, comparação, interpretação e instalação em um único artefato;
- ausência de testes e de saída estruturada com metadados.

A Fase 2 corrige isso com `pyproject.toml`, `uv.lock`, código em `src/`, testes, hash do dataset, saída JSON e logging de agregados.

## 8. Reuso e reorganização

### Reutilizar

- o dataset e seu mapeamento de classes;
- `random_state=42` e estratificação;
- os três modelos como baseline;
- pipelines com escala apenas onde necessária;
- prioridade ao recall da classe maligna;
- alerta de que o sistema não substitui diagnóstico médico;
- justificativas para manter outliers e remover `id`/coluna vazia.

### Corrigir ou reorganizar

- reservar teste antes de EDA supervisionada;
- usar validação cruzada no desenvolvimento para fitness e comparação;
- reajustar o vencedor nos 80% antes de uma única avaliação final;
- congelar dependências;
- mover lógica reutilizável do notebook para módulos testáveis;
- registrar ROC-AUC, matriz, falsos negativos, versões e tempos;
- separar seleção de hiperparâmetros, avaliação final e explicação pela LLM;
- criar avaliação objetiva das explicações e proibir dados reais/identificáveis na LLM.

## 9. Limitações

- Amostra pequena, de uma única fonte e sem validação externa.
- O teste contém apenas 42 casos malignos; a diferença de uma observação altera o recall em cerca de 2,38 pontos percentuais.
- Métricas pontuais não trazem intervalo de confiança.
- Nenhum resultado autoriza uso clínico.
- O notebook completo não foi reexecutado célula por célula porque contém instalação interativa de SHAP e visualizações fora do baseline. Os passos de carga, corte, pipelines, treino e métricas foram reproduzidos diretamente e conferidos contra as saídas salvas.
