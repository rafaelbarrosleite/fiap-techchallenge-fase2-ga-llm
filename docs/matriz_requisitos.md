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
| O2.0 | Configurar recursos de escalabilidade automática para variações de demanda | Pool autoescalável sobre o modelo congelado, com política de backlog, histerese e teto | Política, benchmark comparativo, relatório assinado e figura | `serving/autoscaling.py` + `serving/load_benchmark.py` | `run-load-benchmark` e `validate-scalability` | Concluído: p95 1,76x menor e vazão 1,70x maior sob o mesmo perfil; limiar de custo por pedido documentado |
| O2.1 | Monitoramento e logging para tracking de desempenho | Eventos estruturados por execução, geração e ciclo de serviço, sem linhas de pacientes | Logs JSON Lines, identidades, tempos, métricas, erros e checkpoints | `logging_utils.py`, monitor GA e `serving/monitoring.py` | Inspeção de log, teste de campos e barreira aplicada na escrita | Concluído para GA e para a camada de serviço; observabilidade gerenciada permanece fora de escopo |
| O2.2 | Documentar arquitetura e decisões | Diagrama, fluxos, decisões e alternativas | Documentos versionados | `decisoes_tecnicas.md` + `escalabilidade_e_monitoramento.md` | Revisão contra o código | Concluído para baseline, GA, seleção por CV, LLM e camada de serviço |
| O3.1 | Integrar LLM pré-treinada | Adaptador de provedor, entrada estruturada e saída controlada | Código, exemplos e testes com mock/real | `llm/`, `llm_v2/` e `llm_individual/` | Mock oficial + providers reais opt-in | Concluído: agregados preservados e explicação individual OpenAI aprovada |
| O3.1a | Gerar explicações em linguagem natural dos diagnósticos dos modelos | Explicar classificação individual desidentificada e resultados agregados; distinguir predição de diagnóstico clínico | Prompt, resposta, exemplo e 40 checks | `llm_individual/` + LLM V1/V2 | Factualidade, privacidade e segurança | Concluído: caso de desenvolvimento explicado pelo fake e OpenAI real, sem ID, índice, target ou linha bruta |
| O3.1b | Transformar dados numéricos e estatísticos em insights acionáveis para médicos | Converter probabilidade e contribuições locais em ações de revisão humana, sem prescrição | `insights_acionaveis_para_medicos` auditáveis | Contrato individual 3.0 | Escopo obrigatório `human_review_only` | Concluído: revisão dos sinais, auditoria da entrada e confronto independente; decisão de cuidado sempre falsa |
| O3.1c | Preparar base para integração textual no Módulo 3 | Reservar campos textuais fechados e salvaguardas sem enviar texto clínico agora | `future_text_integration` e `preparacao_modulo3` | Schema individual 3.0 | Teste de contrato e estado desabilitado | Concluído: campos textuais futuros preparados com autorização, desidentificação e proveniência obrigatórias |
| O3.2 | Implementar prompt engineering | Templates versionados para agregados e classificação individual, contexto médico, restrições e formato | Arquivos de prompt, hashes e testes | `llm/prompts/` + `llm_individual/prompts/` | Testes de versão, schema e requisição | Concluído: V1/V2 agregados e prompts individuais V1 |
| O3.3 | Avaliar qualidade das interpretações | Rubrica de factualidade, completude, clareza, segurança, relevância médica e calibração | Adulterações, fake e avaliação real | Avaliadores determinísticos | 40 fatos individuais + seis dimensões | Concluído: fake e OpenAI individual aprovados; falhas intermediárias preservadas e corrigidas sem alterar a resposta |
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
| E7 | Vídeo no YouTube/Vimeo, público ou não listado, até 15 min | Roteiro e demonstração dentro do limite | Link e roteiro | Entrega final | Duração e acesso ao link | Concluído: https://youtu.be/9obBNc1XHmg |
| E7.1 | Demonstrar sistema em execução | Mostrar entrada segura, GA, avaliação e LLM | Captura contínua | Vídeo | Checklist de cenas | Concluído: suíte, validadores, explicação individual, barreiras e benchmark executados em tela |
| E7.2 | Explicar componentes | Arquitetura e responsabilidades | Código em tela | Vídeo | Conferir contra arquitetura | Concluído: genomas, fitness, operadores, prompt, barreira de privacidade e política de escalabilidade |
| E7.3 | Apresentar resultados do GA | Resultados reais dos três experimentos | Tabelas/curvas | Vídeo | Valores iguais ao relatório | Concluído: tabela mestre e figuras do painel, com os resultados negativos declarados |
| E7.4 | Demonstrar integração com LLM | Explicação agregada e individual desidentificada | Demo segura | Vídeo | Sem dados pessoais e com disclaimer | Concluído: 139 verificações factuais em tela e barreiras recusando dado individual ao vivo |

## Opcionais ou condicionais

| ID | Requisito oficial | Classificação | Evidência/decisão | Status |
|---|---|---|---|---|
| P1 | Documentação da API, se aplicável | Opcional condicional | Não há API HTTP: o serviço processa lotes, não requisições de terceiros. A interface de serviço está documentada em `escalabilidade_e_monitoramento.md` | Não aplicável |
| P2 | Arquivos de configuração para implantação, se houver nuvem | Opcional condicional | `Dockerfile` e `docker-compose.yml` versionados; imagem construída no CI | Concluído e verificado |
| P3 | Infraestrutura como código, se houver nuvem | Opcional condicional | Módulo em `deploy/terraform/` com ECS Fargate e target tracking; `fmt` e `validate` no CI | Concluído como código e validado; não aplicado |

## Possível pontuação extra

| ID | Requisito oficial | Interpretação | Recomendação | Status |
|---|---|---|---|---|
| X1 | Implementação em nuvem é opcional e pode valer pontuação extra | Deploy, observabilidade e possivelmente autoscaling reais | Container, orquestração local, IaC e arquitetura da solução entregues e verificados no CI; provisionamento real não executado | Parcial e verificado: imagem construída e Terraform validado a cada push; arquitetura documentada em `escalabilidade_e_monitoramento.md`; nenhum recurso pago foi criado |

## Pontos ambíguos a confirmar com o professor

| ID | Ambiguidade | Leitura conservadora adotada | Pergunta sugerida |
|---|---|---|---|
| A1 | O título diz “configurar recursos de escalabilidade automática”, mas a observação diz que nuvem é opcional | Resolvido pela leitura mais exigente: autoscaling implementado, medido e documentado localmente, com container e IaC cobrindo a parte de nuvem sem provisionar recursos | Resolvido tecnicamente; a evidência local não depende de a nuvem ser exigida ou não |
| A2 | “Modelos” aparece no plural, sem quantidade mínima explícita | Otimizar os três baselines para máxima cobertura; priorizar LR e RF se o custo ficar excessivo | É aceitável otimizar dois modelos e manter o terceiro apenas como baseline? |
| A3 | “Explicações dos diagnósticos” pode significar explicação individual | Implementado caso individual de desenvolvimento desidentificado e não reconstruível; sem ID, índice, ground truth ou valores brutos | Resolvido tecnicamente com contrato 3.0 e exemplo auditável; continua não sendo diagnóstico clínico |
| A4 | Não há rubrica objetiva para qualidade da LLM | Criar rubrica própria e avaliação humana documentada, com checagem automática de números | Existe rubrica oficial ou número mínimo de exemplos para a avaliação? |
| A5 | Não é dito se bibliotecas de evolução são permitidas | Implementar operadores principais diretamente para evidenciar domínio | É permitido usar DEAP/pygad ou a implementação deve ser autoral? |
