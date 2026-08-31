# Auditoria documental final

## Objetivo e hierarquia de autoridade

A auditoria verificou se a narrativa das Missões 1–5 corresponde aos artefatos estruturados e se a entrega pode ser demonstrada sem reabrir modelagem. A ordem de autoridade adotada é:

1. JSONs assinados de execução, seleção, avaliação final e LLM;
2. código e testes que validam esses contratos;
3. documentação produzida em cada missão, entendida como fotografia histórica;
4. tabelas expositivas arredondadas.

Nenhuma evidência congelada foi alterada. Divergências foram classificadas e preservadas abaixo.

## Escopo revisado

Foram lidos README, `pyproject.toml`, plano, decisões técnicas, matriz de requisitos, auditoria da Fase 1, algoritmo genético, resultados oficiais, comparação de métodos, protocolo de seleção, avaliação final, comparação baseline/GA, limitações e camada LLM. Também foram verificados os comandos e os artefatos de `official/`, `comparison/`, `selection/`, `final_evaluation/`, `llm_evaluation/`, `llm_contract_v2/` e `llm_evaluation_openai_v4/`.

## Achados

| ID | Tipo | Ocorrência | Fonte autoritativa | Consequência e tratamento |
|---|---|---|---|---|
| D1 | Divergência histórica | Relatórios da Missão 3 chamam o melhor GA de Regressão Logística de GA C; a avaliação final usou GA B | `final_evaluation_plan.json` e `final_test_results.json` assinados contêm `GA_B` | B e C empataram nas métricas agregadas de CV; a correção canônica posterior determinou a origem serializada. A divergência não muda as métricas do empate e permanece citada como história, não é “corrigida” retroativamente. |
| D2 | Campo ausente em recorte prioritário | Os quatro JSONs prioritários da Missão 4 não possuem `selected_model` global | `frozen_candidates.json` possui `global_provisional_winner`; a documentação de seleção confirma a decisão | A camada LLM registrou a ausência no seu recorte e usou fonte auxiliar. A consolidação final referencia diretamente o artefato estruturado da Missão 3 e explica por que o holdout não pode escolher outro candidato. |
| D3 | Exposição histórica do holdout | O baseline histórico em `baseline_results.json` já registrava métricas do mesmo holdout na Missão 1 | preflight e manifesto da Missão 4 provam que esse arquivo não entrou na linhagem de seleção | O holdout não era literalmente desconhecido para o projeto inteiro. A avaliação final é uma confirmação controlada, não um primeiro contato absoluto. |
| D4 | Contagens de testes | Documentos históricos citam 58, 79, 113 e 120 testes | Cada número pertence ao encerramento de uma missão distinta | Não há divergência matemática. O relatório final identifica os números como snapshots; o manifesto consolidado registra 161 testes. |
| D5 | Referência obsoleta | `algoritmo_genetico.md` ainda diz que o teste está reservado para missão futura | Artefatos da Missão 4 mostram avaliação concluída | O texto é preservado como estado ao final da Missão 3. O README e o relatório final apontam o estado atual. |
| D6 | Referência obsoleta | `protocolo_selecao_final.md` proíbe LLM antes da avaliação final | Manifestos das Missões 4 e 5 mostram a sequência correta | A condição foi cumprida; a frase continua válida como regra temporal histórica. |
| D7 | Provider real | O mock está aprovado; a execução real V2 passou 327/327 fatos, mas falhou em três checks lexicais de calibração | `llm_evaluation_openai_v4/` registra `methodologically_complete_not_approved` | A execução real não foi reclassificada. O mock permanece oficial e a limitação lexical aparece no relatório. |
| D7 | Sobreposição documental | `avaliacao_final.md`, `comparacao_metodos.md` e `comparacao_modelos_originais_otimizados.md` repetem métricas | `final_test_results.json` e `uncertainty_results.json` | Não são apagados: um documenta protocolo, outro compara busca e o terceiro responde ao requisito FIAP. O relatório consolidado vira a porta de entrada. |
| D8 | Fluxo de comandos arriscado para demonstração | O README anterior misturava comandos de GA/busca com comandos seguros | CLIs idempotentes e manifestos completed | O README final separa reprodução histórica de demonstração oficial. A demo não executa comandos de otimização. |
| D9 | Terminologia | “teste final” e “holdout” aparecem como sinônimos; “busca aleatória” e `RandomizedSearchCV` alternam | Convenção definida nesta missão | Material novo usa “holdout (teste final)” na primeira ocorrência e depois “holdout”; usa “busca aleatória (`RandomizedSearchCV`)” na primeira ocorrência. |

## Conferência de métricas principais

Os valores narrativos principais coincidem com `final_test_results.json`:

- Regressão Logística: FN `3 → 1`, recall `0,928571 → 0,976190`;
- Random Forest: FN `4 → 3`, recall `0,904762 → 0,928571`;
- KNN: FN `4 → 4`, recall `0,904762 → 0,904762`;
- limiar fixo: `0,5`;
- desenvolvimento/holdout: `455/114`, com `42` malignos no holdout;
- seleção reaberta: `false`;
- nova otimização na Missão 4: `false`.

Os intervalos e deltas narrados coincidem com `uncertainty_results.json`. O provider oficial da Missão 5 é `fake`, factualidade e segurança estão aprovadas, e `individual_data_sent=false` no manifesto LLM. A evidência complementar V4 registra uma chamada real, 327/327 fatos, zero dados individuais e avaliação científica não aprovada.

## Conclusão da auditoria

Não foi encontrado problema grave que invalide a entrega. Os riscos relevantes são interpretativos: histórico do holdout, baixo número de casos positivos, diferenças de origem GA B/C e ausência do vencedor global no recorte prioritário usado pela LLM. Todos estão explicitados e não alteram a prova de que a seleção foi feita por CV antes da avaliação confirmatória.
