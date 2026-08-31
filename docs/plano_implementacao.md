# Plano de implementação

O plano mantém o teste final separado e produz evidência verificável em cada etapa. Nenhuma etapa inclui commit, push, deploy ou repositório remoto sem autorização.

## Estado atual

### Etapa 1 - Auditoria e fundação: concluída

Objetivo: entender a Fase 1, provar sua reprodução e criar uma base correta para comparação.

Entregas:

- leitura integral dos arquivos relevantes e do notebook;
- inspeção do CSV e registro de hash;
- reprodução das métricas históricas;
- baseline corrigido 80/20;
- dependências travadas com `pyproject.toml` e `uv.lock`;
- código modular em `src/`;
- oito testes iniciais;
- logging, métricas, matrizes, falsos negativos, versões e tempos;
- matriz oficial e proposta técnica do GA.

Critério cumprido: `pytest` passa e `run-baseline` gera resultados estruturados sem alterar a Fase 1.

### Etapa 2 - Núcleo do algoritmo genético: concluída

Objetivo simples: criar o mecanismo de busca, ainda sem executar os experimentos finais.

Implementado:

- genes tipados e cromossomos por modelo;
- população inicial válida;
- torneio, crossover uniforme, mutação tipada e elitismo;
- cache de indivíduos já avaliados;
- fitness por validação cruzada estratificada;
- seeds e identificador de execução;
- testes unitários dos operadores e da reprodutibilidade.

Validação concluída:

- 45 testes automatizados aprovados na conclusão da segunda missão;
- smoke test com Regressão Logística, Random Forest e KNN;
- cinco dobras estratificadas exclusivamente nos 80% de desenvolvimento;
- artefatos JSON com schema validado;
- duas execuções com seed 42 produziram a mesma assinatura;
- seed 43 produziu trajetória diferente;
- zero falhas e zero avisos problemáticos nos três smoke tests;
- bateria oficial A/B/C ainda não executada naquele marco.

Decisão resolvida: os três modelos serão otimizados.

### Etapa 3 - Experimentos e seleção por CV: concluída

Objetivo simples: comparar configurações de busca com orçamento e folds controlados.

Executado:

- configurações A pequena, B equilibrada e C exploratória;
- mesmas dobras de CV e mesma seed-base;
- curvas de melhor/média/diversidade por geração;
- tabela de hiperparâmetros, fitness, recall, F1, ROC-AUC, desvio e tempo;
- baseline comparável nos mesmos folds;
- busca aleatória com orçamento equivalente;
- congelamento de um vencedor provisório por modelo e um global;
- sete figuras sem uso do teste final;
- 58 testes automatizados.

Validação: nove artefatos oficiais, três status por configuração/modelo, checkpoints, manifesto, comparação e seleção assinados.

Resultado: a bateria levou 51,12 minutos. Foram 4.495 avaliações únicas e 22.475 fits, contra teto de 43.800 fits. A busca aleatória comparável levou 46,52 minutos.

## Próximas etapas propostas

### Etapa 4 - Avaliação final única: concluída

Objetivo simples: escolher apenas com CV e usar o teste uma única vez para a comparação final.

Implementado/executado:

- carregar somente os candidatos já congelados;
- reajustar em todos os 455 registros de desenvolvimento;
- avaliar baseline e otimizado no mesmo teste;
- relatar falsos negativos, métricas, tempos e incerteza;
- não reabrir seleção, espaços, limiar ou hiperparâmetros após o teste.

Validação concluída:

- preflight com 79 testes e zero chamadas ao holdout;
- plano assinado antes da execução;
- nove origens e oito Pipelines canônicos;
- ajuste somente nas 455 linhas de desenvolvimento;
- execução confirmatória concluída em 39,65 s;
- 1.026 previsões persistidas e métricas recalculadas;
- intervalos de Wilson, bootstrap pareado e McNemar exato;
- oito modelos joblib e seis figuras inspecionadas;
- hashes e schemas validados;
- zero GA, zero busca aleatória, zero ajuste de limiar e zero seleção pós-teste.

Resultado: LR GA reduziu FN de 3 para 1; RF GA, de 4 para 3; KNN GA manteve 4 FN. O ganho de recall da CV não se confirmou para KNN.

### Etapa 5 - LLM segura e avaliável: concluída

Objetivo simples: transformar somente resultados fornecidos em explicações fiéis.

Implementado:

- contrato estruturado de entrada;
- templates de prompt versionados;
- adaptador de provedor configurado por ambiente;
- modo offline/mock para testes;
- proibição explícita de diagnóstico e de dados pessoais reais;
- avaliação factual, clareza, completude, segurança e alucinação;
- exemplos sintéticos e agregados;
- checagem factual independente de todos os números estruturados;
- safety checker determinístico e disclaimer obrigatório;
- identidade, hashes, manifesto e idempotência;
- provider mock offline e provider OpenAI real somente opt-in.

Validação concluída: os nove cenários adversariais A–I, privacidade, schema fechado, prompts, providers, hashes e artefatos são testados sem rede. A execução mock foi aprovada nas cinco dimensões. Nenhum treino, GA, busca aleatória, mudança de threshold ou seleção foi realizado.

### Etapa 6 - Relatório e demonstração

Objetivo simples: reunir evidências, limitações e execução em material acadêmico claro.

Estado: concluída.

Entregas:

- relatório técnico final;
- notebook ou roteiro de demonstração;
- diagrama final;
- roteiro de vídeo de até 15 minutos;
- checklist da matriz de requisitos;
- instruções de reprodução em ambiente limpo.

Validação concluída naquele marco: relatório e resumo autocontidos, roteiro de 10–15 minutos, demo de 5 minutos, mapa de evidências, tabela mestre, seis figuras revisadas, matriz final, validador somente leitura e manifesto da entrega. A suíte da Missão 6 aprovou 120 testes; após as missões complementares de LLM, o total consolidado passou a 161, sem novo treino, busca ou inferência.

### Etapa 7 - Avaliação complementar do provider real: concluída metodologicamente, não aprovada cientificamente

As Missões 7.1–7.3 diagnosticaram o parâmetro `temperature`, validaram o parsing raw-first e preservaram a resposta V1 com 138/139 checks. A Missão 7.4 criou o contrato V2 com pares explícitos, McNemar agregado e 327 checks, aprovado integralmente com o provider fake.

Na Missão 7.5, uma única chamada científica real em `artifacts/llm_evaluation_openai_v4/` retornou HTTP 200 e `completed`. O payload continha somente agregados, usou `store=false`, omitiu `temperature` e não teve retry. A resposta passou schema, 327/327 fatos, segurança, completude, clareza, pares comparativos e McNemar. O status final permaneceu `methodologically_complete_not_approved` porque três checks lexicais de calibração não reconheceram paráfrases semanticamente adequadas. Não houve ajuste posterior nem chamadas adversariais.

O provider fake V2 continua sendo o caminho oficial de reprodução offline. A execução real é evidência complementar e não pode ser apresentada como aprovação científica integral.

### Etapa 8 - Consolidação e publicação: concluída, exceto vídeo

- documentação consolidada com o resultado negativo da avaliação real;
- manifesto final atualizado com hashes e escopo correto;
- suíte e validadores executados offline;
- código e documentação versionados e publicados sem segredos;
- vídeo de até 15 minutos permanece como único entregável externo pendente.

### Etapa 9 - Explicação individual do requisito LLM: concluída

- pipeline congelado reutilizado sem treino;
- caso de desenvolvimento convertido em representação desidentificada e não reconstruível;
- contrato fechado 3.0, prompts médicos versionados e Structured Output;
- explicação da classe, probabilidade e cinco contribuições locais;
- insights acionáveis restritos a revisão humana;
- base contratual para notas e laudos textuais futuros;
- fake offline e OpenAI real avaliados em seis dimensões;
- 40/40 checks factuais e zero violação clínica;
- falha intermediária de parsing e revalidação semântica preservadas;
- nenhum caso do holdout, novo treino, GA, busca ou alteração de threshold.

### Extras opcionais

- API, se trouxer valor à demonstração;
- container;
- nuvem, autoscaling real, observabilidade gerenciada e IaC;
- publicação, commit, push e repositório remoto.

Esses itens não devem competir com os requisitos acadêmicos centrais.

## Critérios gerais de qualidade

- teste final isolado da busca;
- nenhuma métrica inventada;
- seeds e versões registradas;
- logs sem linhas individuais do dataset;
- código principal fora do notebook;
- saídas geradas e segredos ignorados pelo Git;
- linguagem explícita de apoio, nunca substituição diagnóstica;
- divergências e resultados negativos preservados no relatório.
