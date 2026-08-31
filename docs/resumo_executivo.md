# Resumo executivo

## Problema e objetivo

O projeto evoluiu um classificador acadêmico de tumores da Fase 1 em duas frentes: otimização de hiperparâmetros com algoritmo genético (GA) e explicação controlada dos resultados por uma LLM. A meta foi melhorar o recall da classe maligna sem consultar o teste final durante a busca e tornar a explicação factual, segura e reproduzível.

## O que foi construído

- auditoria metodológica da Fase 1 e baseline reproduzível;
- GA autoral para Regressão Logística, Random Forest e KNN;
- fitness com 60% recall, 25% F1, 15% ROC-AUC e penalidade de instabilidade;
- nove experimentos A/B/C em cinco dobras, com cache, seeds, checkpoints e manifestos;
- comparação justa com busca aleatória (`RandomizedSearchCV`);
- seleção e congelamento por validação cruzada antes do holdout;
- avaliação final confirmatória com IC95%, bootstrap e McNemar;
- camada LLM offline com contratos V1/V2 fechados, privacy gate, prompts versionados, factualidade e safety checker;
- avaliação complementar OpenAI raw-first, isolada do pipeline de ML e sem dados individuais;
- documentação, tabela mestre, figuras e validador final somente leitura.

## Principais resultados

| Família | Recall baseline→GA | FN baseline→GA | Leitura |
|---|---:|---:|---|
| Regressão Logística | 0,928571→0,976190 | 3→1 | ganho observado no holdout |
| Random Forest | 0,904762→0,928571 | 4→3 | ganho observado, com AUC menor |
| KNN | 0,904762→0,904762 | 4→4 | ganho de CV não confirmado |

O GA melhorou o fitness de CV nos três modelos. No holdout, reduziu falsos negativos em LR e RF, mas não em KNN. Em LR e KNN, a busca aleatória alcançou métricas agregadas equivalentes ao GA; em RF, o GA obteve maior fitness de CV, com custo computacional elevado.

O modelo para demonstração continua sendo a Regressão Logística da busca aleatória, escolhida antes do holdout. O teste final não reabriu a seleção.

## Contribuição da LLM

A LLM recebe somente resultados agregados. Nenhuma linha, feature, índice, previsão ou probabilidade individual é enviada. O provider oficial de reprodução é um mock determinístico offline: V1 passou 139 checks e V2 passou 327 checks.

Uma chamada complementar real com `gpt-5.5` e contrato V2 passou schema, 327/327 fatos, segurança, completude, clareza, pares explícitos e McNemar. Ela permaneceu cientificamente não aprovada porque três checks lexicais de calibração não reconheceram paráfrases semanticamente adequadas. A evidência negativa foi preservada, sem retry ou mudança posterior de prompt/schema/checker, e não substitui o caminho offline oficial.

## Limitações e conclusão

Há somente 42 casos malignos no holdout; uma observação muda recall em cerca de 2,38 pontos percentuais. Os intervalos são amplos, o bootstrap inclui zero e McNemar tem poucos discordantes. O dataset é pequeno, de fonte única e sem validação externa, prospectiva ou clínica. O baseline histórico já havia registrado métricas do mesmo holdout, embora não tenha participado da seleção.

Conclusão: o GA foi útil em duas famílias no objetivo prioritário e como demonstração de engenharia reproduzível, mas não houve superioridade universal ou evidência clínica. A camada LLM acrescentou explicação segura e auditável sem criar nova evidência; a avaliação real também tornou visível a limitação dos critérios lexicais. O projeto está tecnicamente pronto para apresentação acadêmica e demonstração offline, não para diagnóstico ou decisão médica.
