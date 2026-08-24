# Algoritmo genético

## Objetivo e escopo desta etapa

O algoritmo genético busca combinações de hiperparâmetros para Regressão Logística, Random Forest e KNN. O mecanismo e a bateria oficial A/B/C foram executados e validados para os três modelos.

O avaliador recebe somente as 455 linhas de desenvolvimento. As 114 linhas do teste final nunca são passadas ao fitness, aos operadores ou ao engine. Portanto, todos os números deste documento sobre candidatos são métricas de validação cruzada, não métricas do teste final.

> Nota de estado: este documento registra principalmente as Missões 2–3. A avaliação confirmatória foi concluída depois, na Missão 4, sem reexecutar o GA. Referências abaixo ao teste “reservado para missão posterior” devem ser lidas nesse contexto histórico.

## Conceitos em linguagem simples

- **Gene:** um hiperparâmetro, como o número de vizinhos do KNN.
- **Indivíduo ou cromossomo:** uma configuração completa de um modelo.
- **População:** conjunto de indivíduos avaliados na mesma geração.
- **Fitness:** nota calculada a partir das cinco dobras de validação cruzada.
- **Seleção:** escolha dos indivíduos que terão chance de gerar filhos.
- **Crossover:** combinação dos genes de dois pais.
- **Mutação:** alteração controlada de um ou mais genes.
- **Elitismo:** cópia direta dos melhores indivíduos para a próxima geração.
- **Geração:** uma rodada completa de seleção, crossover, mutação, avaliação e substituição.

## Representação dos indivíduos

Os indivíduos são dataclasses imutáveis com campos nomeados. Eles não são vetores sem significado e podem ser serializados diretamente em JSON.

### Regressão Logística

| Gene | Domínio | Motivo |
|---|---|---|
| `log10_c` | real em [-4, 3] | Representa `C=10^log10_c` e cobre várias ordens de grandeza |
| `penalty` | `l1` ou `l2` | Controla a forma da regularização |
| `class_weight` | `None` ou `balanced` | Permite ponderar a classe maligna |

`solver="liblinear"` e `max_iter=2000` são fixos. O solver aceita L1 e L2, evitando combinações incompatíveis. `max_iter` é condição de estabilidade, não gene usado para favorecer fitness.

### Random Forest

| Gene | Domínio |
|---|---|
| `n_estimators` | inteiro [100, 500] |
| `max_depth` | `None` ou inteiro [3, 20] |
| `min_samples_split` | inteiro [2, 20] |
| `min_samples_leaf` | inteiro [1, 10] |
| `max_features` | `sqrt`, `log2`, 0,5 ou 1,0 |
| `class_weight` | `None`, `balanced` ou `balanced_subsample` |

O modelo recebe `random_state=42` e `n_jobs=1` durante a avaliação. Não há normalização porque árvores não dependem da escala dos atributos.

### KNN

| Gene | Domínio | Regra condicional |
|---|---|---|
| `n_neighbors` | ímpares entre 3 e 31 | Evita empates frequentes |
| `weights` | `uniform` ou `distance` | Peso igual ou por distância |
| `metric` | `minkowski`, `euclidean`, `manhattan` | Define a distância |
| `p` | 1 ou 2 | Só existe quando `metric="minkowski"`; caso contrário é `None` |

O `leaf_size`, presente na proposta preliminar, foi removido: ele afeta principalmente eficiência interna e não as previsões usadas pelo fitness. Mantê-lo criaria um gene neutro. O KNN usa `StandardScaler` dentro do pipeline.

## Fitness

Para cada indivíduo, o código calcula as métricas em cinco dobras e depois aplica:

```text
fitness_base = 0,60 * media_recall_maligno
             + 0,25 * media_F1_maligno
             + 0,15 * media_ROC_AUC

fitness_final = fitness_base
              - 0,10 * desvio_padrao_recall_maligno
```

O peso 0,10 foi mantido conforme a decisão aprovada. Como o recall está em [0, 1], seu desvio-padrão máximo teórico é 0,5; logo, a penalidade máxima é 0,05 ponto. Um teste específico comprova que, mantendo as médias iguais, maior instabilidade produz fitness menor.

Cada resultado registra separadamente fitness final, fitness base, médias, desvio do recall, métricas por dobra, tempo, avisos de convergência e falhas. Uma falha de ajuste recebe fitness `-1`, permanece visível no artefato e não encerra toda a busca.

## Por que usar cinco dobras

As 455 linhas são divididas cinco vezes em treino e validação. Cada linha participa exatamente uma vez da validação. A estratificação mantém benignos e malignos em todas as dobras. Como o scaler está dentro do pipeline, ele é reajustado somente no treino de cada dobra.

Isso reduz a dependência de um único corte. O teste final continua isolado porque não participa da criação das dobras, do fitness, da seleção ou da escolha dos hiperparâmetros.

## Operadores implementados

### População inicial

Cada espaço sorteia valores válidos com `numpy.random.Generator`. Chaves JSON canônicas impedem duplicatas na população inicial. Se não for possível atingir a diversidade pedida após tentativas limitadas, a execução falha explicitamente.

### Seleção por torneio

O engine sorteia participantes sem reposição e escolhe o de melhor ranking. O tamanho do torneio é configurável.

### Crossover uniforme

Para cada gene, um sorteio decide de qual pai virá o valor do primeiro filho; o outro valor vai para o segundo. Os filhos mantêm a dataclass do modelo e passam por reparação e validação.

### Mutação tipada

- categórico: troca por outra categoria;
- inteiro: escolhe outro inteiro válido;
- real: perturba `log10_c` na escala logarítmica e limita ao domínio;
- opcional: `max_depth` pode entrar ou sair de `None`;
- condicional: `p` só sofre mutação quando a métrica é Minkowski.

A função oferece `force_change=True` para garantir mudança quando usado em testes e na correção de duplicatas.

### Reparação

Valores numéricos são limitados às faixas; `n_neighbors` é tornado ímpar; categorias inválidas recebem padrão seguro; e `metric`/`p` são reconciliados. Depois, o indivíduo é validado novamente.

### Elitismo e substituição

Os melhores indivíduos são copiados. O restante da nova população é produzido por seleção, crossover e mutação. A geração anterior é então substituída integralmente pela nova. O cache evita reavaliar cromossomos já vistos.

### Critério de parada

Toda execução para ao atingir `max_generations`. Opcionalmente, `stagnation_generations` interrompe quando o melhor fitness deixa de melhorar pelo número configurado de gerações. O motivo fica no JSON.

## Exemplo fictício de uma geração

Considere quatro indivíduos de KNN:

| Indivíduo | Genes resumidos | Fitness fictício |
|---|---|---:|
| A | 5 vizinhos, uniforme, Euclidiana | 0,91 |
| B | 9 vizinhos, distância, Manhattan | 0,94 |
| C | 15 vizinhos, uniforme, Minkowski p=2 | 0,89 |
| D | 7 vizinhos, distância, Minkowski p=1 | 0,93 |

Com um elite, B passa diretamente. Um torneio pode escolher B e D como pais. No crossover, um filho recebe `n_neighbors=9` de B e `metric="minkowski", p=1` de D. Uma mutação troca `weights` para `uniform`. O filho é reparado, validado e avaliado nas cinco dobras. O processo continua até completar quatro indivíduos na nova geração.

Os números acima são apenas didáticos e não são resultados do dataset.

## Configurações cadastradas

| Configuração | População | Gerações após a inicial | Crossover | Mutação por gene | Elites | Torneio | Candidatos máximos/modelo | Fits máximos/modelo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A - pequena | 20 | 10 | 0,70 | 0,10 | 2 | 3 | 220 | 1.100 |
| B - equilibrada | 40 | 20 | 0,80 | 0,20 | 2 | 3 | 840 | 4.200 |
| C - exploratória | 60 | 30 | 0,75 | 0,30 | 4 | 4 | 1.860 | 9.300 |

O teto para os três modelos e as três configurações é 43.800 fits. Esse teto assume que todo candidato é único e que não há parada antecipada.

## Smoke tests validados

Os smoke tests usam população 4 e duas gerações após a inicial. Cada um fez 12 solicitações, encontrou 9 indivíduos únicos e realizou 45 fits. São testes de integração, não experimentos oficiais.

| Modelo | Fitness CV | Recall CV | Desvio recall | F1 CV | ROC-AUC CV | Tempo | Falhas/avisos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,952532 | 0,947059 | 0,022010 | 0,950015 | 0,993292 | 0,239 s | 0/0 |
| Random Forest | 0,947913 | 0,947059 | 0,028818 | 0,938905 | 0,985552 | 14,679 s | 0/0 |
| KNN | 0,932719 | 0,917647 | 0,034300 | 0,950942 | 0,985501 | 0,156 s | 0/0 |

Nenhuma métrica do teste final foi calculada para esses candidatos.

## Reprodutibilidade

São controladas separadamente:

- seed do engine genético;
- seed fixa das cinco dobras;
- seed fixa dos estimadores que aceitam `random_state`;
- versões das bibliotecas;
- hash do dataset;
- configuração completa.

Uma assinatura SHA-256 exclui data/hora e tempos, que variam naturalmente. Duas execuções da Regressão Logística com seed 42 produziram a mesma assinatura `b4e76f0...ca2b69`, o mesmo histórico e o mesmo melhor genoma. Com seed 43, a assinatura foi `0fe46715...be490`, com trajetória e melhor genoma diferentes.

## Artefatos e logging

O JSON de schema `1.0` contém configuração, seed, modelo, data/hora, escopo dos dados, melhor indivíduo, métricas por dobra, histórico, tempo, parada, solicitações, avaliações únicas, fits, falhas, avisos e assinatura. Um validador próprio rejeita campos obrigatórios ausentes ou qualquer artefato que declare uso do holdout.

Logs e resultados são gravados em `logs/` e `artifacts/`, ambos ignorados pelo Git.

## Execução oficial e retomada

Cada experimento possui artefato, status e checkpoint próprios. O checkpoint registra população avaliada, melhor global, histórico, cache, contador de solicitações, estado do gerador NumPy, estagnação e tempo acumulado. A escrita usa arquivo temporário no mesmo diretório, `fsync` e troca atômica.

Ao retomar, um resultado só é pulado quando artefato e status declaram conclusão e a identidade SHA-256 coincide com modelo, configuração, seeds, dataset, índices de desenvolvimento, versão e código relevante. Um checkpoint de identidade diferente é ignorado.

## Resultados oficiais resumidos

| Modelo | A | B | C | Melhor GA |
|---|---:|---:|---:|---|
| Regressão Logística | 0,973150 | 0,973166 | 0,973166 | C pelo desempate canônico |
| Random Forest | 0,955954 | 0,959515 | 0,961648 | C |
| KNN | 0,953990 | 0,953990 | 0,953990 | Mesma solução; A retida na ordem fixa |

O KNN chegou às 120 combinações únicas possíveis. O C da Random Forest encontrou sua melhora principal apenas na geração 21, mostrando que o platô intermediário não era definitivo. Detalhes completos estão em `resultados_experimentos_geneticos.md`.

## Custo observado no Mac mini M4/16 GB

Extrapolando os tempos por indivíduo único dos smoke tests e assumindo o pior caso sem cache:

| Configuração | Fits nos 3 modelos | Tempo serial estimado |
|---|---:|---:|
| A | 3.300 | 6,1 min |
| B | 12.600 | 23,4 min |
| C | 27.900 | 51,9 min |
| Total | 43.800 | 81,5 min |

A bateria real fez 4.495 avaliações únicas, correspondentes a 22.475 ajustes em cinco dobras, e levou 3.067,47 s (51,12 min). A Random Forest representou 97,3% do tempo. O cache evitou 4.265 solicitações repetidas; no KNN, a duplicação cresceu porque o espaço inteiro foi coberto.

A comparação `RandomizedSearchCV` usou 1.080, 1.638 e 78 candidatos em LR, RF e KNN, respectivamente, e levou 2.791,26 s. `refit=False` impediu o ajuste final prematuro; o vencedor foi escolhido sobre os resultados multi-scorer com a mesma fórmula composta.

## Limitações conhecidas

- Uma única seed oficial não mede a variabilidade entre execuções completas do GA.
- A estimativa de tempo depende dos `n_estimators` sorteados e da carga do sistema.
- Uma única seed não mede variabilidade estatística do GA.
- O fitness composto expressa uma decisão acadêmica, não uma regra clínica validada.
- Diferenças pequenas em uma única divisão de CV não demonstram superioridade clínica.
- LR B, LR C e a melhor busca aleatória tiveram métricas indistinguíveis nos folds; o desempate canônico não representa ganho de qualidade.
- O teste final permanece reservado para uma missão posterior.
