# Matriz de rastreabilidade dos requisitos

> Adendo das Missões 7.1–7.5: as falhas técnicas iniciais foram diagnosticadas e preservadas. A avaliação real final com contrato V2 retornou HTTP 200 e obteve 327/327 fatos, segurança, completude e clareza, mas permaneceu não aprovada por três checks lexicais de calibração. Não houve retry nem dados individuais; consulte [`avaliacao_provider_real_v4.md`](avaliacao_provider_real_v4.md). O provider fake continua sendo a demonstração oficial offline.

Fonte oficial: [`docs/references/IADT - Fase 2 - Tech challenge.pdf`](<references/IADT - Fase 2 - Tech challenge.pdf>), páginas 3 a 7. O projeto escolhido é o Projeto 1: Otimização de Modelos de Diagnóstico.

Esta matriz preserva a evolução por requisito. A consolidação final, com missão, implementação, teste e evidência em uma única visão, está em [`matriz_rastreabilidade_final.md`](matriz_rastreabilidade_final.md).

Status usado: **concluído**, **parcial**, **não iniciado**, **condicional** ou **a confirmar**.

## Obrigatórios - solução técnica

| ID | Requisito oficial | Interpretação prática | Evidência necessária | Responsável | Validação | Status |
|---|---|---|---|---|---|---|
| O1 | Implementar algoritmo genético para otimizar hiperparâmetros dos modelos do Módulo 1 | Implementação própria e testável, sem usar o teste no fitness | Código, testes e logs de execução | `src/.../genetic/` | Testes unitários e experimento reproduzível | Concluído para LR, RF e KNN; nove experimentos válidos |
| O1.1 | Definir codificação adequada de genes | Cromossomos tipados e espaços válidos para cada modelo | Tabela de genes e validadores | Código GA + `decisoes_tecnicas.md` | Todo cromossomo decodifica em parâmetros aceitos | Concluído para LR, RF e KNN |
| O1.2 | Implementar seleção, cruzamento e mutação | Torneio, crossover uniforme e mutação por tipo de gene | Funções, testes e logs | Código GA | Testes determinísticos com seeds | Concluído, incluindo elitismo, reparação e substituição |
| O1.3 | Definir fitness com métricas de desempenho | Composição centrada em recall, com F1, ROC-AUC e estabilidade em CV | Fórmula, código e resultados por dobra | Avaliador GA | Recalcular fitness em folds fixos | Concluído e testado em 5-fold CV |
| O1.4 | Comparar modelos otimizados com originais | Mesma divisão e protocolo CV na seleção; avaliação confirmatória no teste congelado | Baseline + resultados GA | Pipeline de avaliação e relatório | Diferenças absolutas, recall, FN, incerteza e tempos | Concluído: baseline × GA no teste para LR, RF e KNN |
| O1.5 | Pelo menos 3 configurações do GA | Três orçamentos/taxas documentados para cada modelo escolhido | Configurações, seeds, curvas e resultados | Configuração GA | Três execuções identificáveis e reproduzíveis | Concluído: A/B/C executados nos três modelos |
| O2.1 | Monitoramento e logging para tracking de desempenho | Eventos estruturados por execução e geração, sem linhas de pacientes | Logs, identidades, tempos, métricas, erros e checkpoints | `logging_utils.py` + monitor GA | Inspeção de log e teste de campos | Concluído para GA local; observabilidade gerenciada não iniciada |
| O2.2 | Documentar arquitetura e decisões | Diagrama, fluxos, decisões e alternativas | Documentos versionados | `decisoes_tecnicas.md` | Revisão contra o código | Concluído para baseline, GA e seleção por CV |
| O3.1 | Integrar LLM pré-treinada | Adaptador de provedor, entrada estruturada e saída controlada | Código, exemplo e teste com mock | Módulos `llm/` e `llm_v2/` | Mock oficial + provider real opt-in | Concluído; mock aprovado e chamada real V2 preservada como não aprovada pelo gate lexical |
| O3.1a | Gerar explicações em linguagem natural dos diagnósticos dos modelos | Explicar somente resultado agregado, com limitações e sem diagnosticar | Prompt, resposta e checklist factual | LLM + prompts | Rubrica de fidelidade e segurança | Concluído para resultados experimentais agregados; explicação individual proibida |
| O3.1b | Transformar dados numéricos e estatísticos em insights acionáveis para médicos | Resumir métricas agregadas e indicar revisão humana, sem prescrição | Exemplos auditáveis | LLM + relatório | Conferência de cada número contra JSON | Concluído como explicação acadêmica não clínica; “acionável” foi limitado por segurança |
| O3.1c | Preparar base para integração textual no Módulo 3 | Contrato extensível, sem coletar texto clínico real agora | Esquema e arquitetura | LLM + documentação | Teste de contrato | Concluído com schemas V1/V2 fechados e seleção explícita de versão |
| O3.2 | Implementar prompt engineering | Templates versionados, contexto, restrições, formato e exemplos | Arquivos de prompt e histórico | `llm/prompts/` | Testes de renderização e revisão | Concluído: system/explanation V1 e V2 com hashes |
| O3.3 | Avaliar qualidade das interpretações | Rubrica de correção, completude, clareza, segurança e alucinação | Fixtures sintéticas, notas e análise | Avaliação LLM | Avaliação repetível | Concluído deterministicamente; mock aprovado e limitação lexical do provider real documentada |
| O4.1 | Projeto Python estruturado e ambiente virtual | `src`, testes, configuração e lock de dependências | Árvore, `pyproject.toml`, `uv.lock` | Raiz do projeto | Instalação limpa | Concluído |
| O4.2 | Documentação detalhada com diagramas de arquitetura | README, plano, decisões e diagrama atualizado | Markdown e Mermaid | `README.md`, `docs/` | Links e correspondência com código | Concluído para baseline, GA, seleção, avaliação final e LLM |
| O4.3 | Testes automatizados | Dados, operadores GA, avaliação final, prompts e integrações | Suíte de testes | `tests/` | `pytest` sem falhas | Concluído para o escopo implementado, incluindo LLM offline e adversarial |

## Obrigatórios - entregáveis

| ID | Requisito oficial | Interpretação prática | Evidência necessária | Responsável | Validação | Status |
|---|---|---|---|---|---|---|
| E1 | Repositório Git com código-fonte completo | Versionar código e documentação, excluindo segredos e gerados | Árvore e histórico Git | Projeto | Clone limpo executável | Concluído; código e documentação publicados sem segredos ou saídas geradas |
| E2 | Scripts ou notebooks de demonstração | Demonstração curta do baseline, GA, avaliação final e LLM | Script/notebook executável | `notebooks/` e CLI | Execução do zero | CLIs de todas as camadas, incluindo prepare/run/evaluate LLM, concluídas |
| E3 | Relatório: implementação e resultados do GA | Método, genes, operadores, configurações e resultados reais | Relatório final | `reports/`/`docs/` | Rastreabilidade com logs | Concluído para GA: nove resultados, comparação e figuras |
| E4 | Relatório: LLM, prompts e avaliação | Abordagem, templates, exemplos e avaliação | Relatório final + prompts | `docs/`, `prompts/` | Conferência com execuções | Concluído em `docs/camada_llm_segura.md` |
| E5 | Relatório: comparativo original versus otimizado | Mesma metodologia e tabela com FN | Relatório final | Avaliação | Recalcular a partir dos artefatos | Concluído para LR, RF e KNN com incerteza e previsões pareadas |
| E6 | Relatório: desafios e soluções | Limitações, falhas, decisões e correções | Seção crítica | Relatório final | Revisão por evidência | Concluído para otimização, avaliação final e LLM |
| E7 | Vídeo no YouTube/Vimeo, público ou não listado, até 15 min | Roteiro e demonstração dentro do limite | Link e roteiro | Entrega final | Duração e acesso ao link | Não iniciado |
| E7.1 | Demonstrar sistema em execução | Mostrar entrada segura, GA, avaliação e LLM | Captura contínua | Vídeo | Checklist de cenas | Não iniciado |
| E7.2 | Explicar componentes | Arquitetura e responsabilidades | Diagrama no vídeo | Vídeo | Conferir contra arquitetura | Não iniciado |
| E7.3 | Apresentar resultados do GA | Resultados reais dos três experimentos | Tabelas/curvas | Vídeo | Valores iguais ao relatório | Não iniciado |
| E7.4 | Demonstrar integração com LLM | Explicação com entrada sintética/agregada | Demo segura | Vídeo | Sem dados pessoais e com disclaimer | Não iniciado |

## Opcionais ou condicionais

| ID | Requisito oficial | Classificação | Evidência/decisão | Status |
|---|---|---|---|---|
| P1 | Documentação da API, se aplicável | Opcional condicional | Só será necessária se uma API for criada; não é necessária ao baseline/CLI | Condicional |
| P2 | Arquivos de configuração para implantação, se houver nuvem | Opcional condicional | Incluir apenas após decisão explícita por cloud | Condicional |
| P3 | Infraestrutura como código, se houver nuvem | Opcional condicional | IaC acompanha a arquitetura cloud escolhida | Condicional |

## Possível pontuação extra

| ID | Requisito oficial | Interpretação | Recomendação | Status |
|---|---|---|---|---|
| X1 | Implementação em nuvem é opcional e pode valer pontuação extra | Deploy, observabilidade e possivelmente autoscaling reais | Adiar até GA, testes, relatório e LLM estarem sólidos | Não iniciado por decisão de escopo |

## Pontos ambíguos a confirmar com o professor

| ID | Ambiguidade | Leitura conservadora adotada | Pergunta sugerida |
|---|---|---|---|
| A1 | O título diz “configurar recursos de escalabilidade automática”, mas a observação diz que nuvem é opcional | Documentar componentes escaláveis e medir desempenho local; tratar autoscaling real como parte da nuvem opcional | É obrigatório demonstrar autoscaling fora da nuvem ou logging/arquitetura satisfazem esse item no projeto local? |
| A2 | “Modelos” aparece no plural, sem quantidade mínima explícita | Otimizar os três baselines para máxima cobertura; priorizar LR e RF se o custo ficar excessivo | É aceitável otimizar dois modelos e manter o terceiro apenas como baseline? |
| A3 | “Explicações dos diagnósticos” pode significar explicação individual | Usar casos sintéticos e resultados do modelo, nunca dados pessoais reais; deixar claro que não é diagnóstico clínico | A demonstração precisa incluir explicação individual ou uma explicação agregada dos resultados atende? |
| A4 | Não há rubrica objetiva para qualidade da LLM | Criar rubrica própria e avaliação humana documentada, com checagem automática de números | Existe rubrica oficial ou número mínimo de exemplos para a avaliação? |
| A5 | Não é dito se bibliotecas de evolução são permitidas | Implementar operadores principais diretamente para evidenciar domínio | É permitido usar DEAP/pygad ou a implementação deve ser autoral? |
