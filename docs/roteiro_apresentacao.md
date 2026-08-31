# Roteiro de apresentação — 10 a 15 minutos

Convenção: Regressão Logística (LR), Random Forest (RF), KNN, baseline, GA, busca aleatória (`RandomizedSearchCV`), desenvolvimento, validação cruzada (CV), holdout (teste final) e modelo para demonstração.

## Orçamento de tempo

| Versão | Cenas | Duração |
|---|---:|---:|
| Completa | 13 | **14 min 20 s** |
| Enxuta, sem o Slide 2 | 12 | **13 min 35 s** |

O Slide 2 (auditoria da Fase 1) é o corte planejado: é a cena que menos participa das quatro exigências do vídeo. Se ainda faltar tempo, o corte seguinte é o Slide 8 (trade-offs). Grave a versão enxuta se estiver inseguro com o relógio — estourar 15 minutos custa mais que omitir a auditoria.

## Cobertura das exigências do vídeo

O enunciado pede quatro coisas explicitamente. Nenhuma pode faltar.

| Exigência do enunciado | Onde é atendida |
|---|---|
| Demonstração do sistema em execução | Slide 13 — comandos e painel em tela |
| Explicação dos diferentes componentes | Slides 3, 4, 10 e 11 |
| Resultados da otimização via algoritmos genéticos | Slides 5, 6, 7, 8 e 9 |
| Demonstração da integração com LLMs | Slide 10, com saída real em tela |

## Antes de gravar

1. Rode `uv run pytest`, `uv run validate-deliverable` e `uv run validate-scalability` e confirme que passam. A demonstração depende de os estados congelados estarem íntegros.
2. **Deixe os comandos do Slide 13 já executados em um terminal aberto**, com a saída visível, e role a tela durante a fala. Executar ao vivo coloca 40 segundos de espera dentro de uma cena de 70 — e uma falha de ambiente no meio da gravação custa a tomada inteira.
3. Abra `reports/dashboard/index.html` em outra aba do navegador, já na seção de verificação, para não navegar em tela.

## Slide 1 — Problema e pergunta acadêmica (45 s)

- **Mensagem principal:** otimizar três modelos sem contaminar o teste e explicar resultados sem criar diagnóstico.
- **Dados:** 569 registros, 30 preditores, 212 malignos.
- **Figura:** diagrama de arquitetura do relatório final.
- **Fala sugerida:** “O desafio não foi apenas melhorar uma métrica; foi construir uma cadeia auditável de seleção, confirmação e explicação.”
- **Risco:** parecer proposta clínica. Dizer explicitamente “acadêmico e experimental”.

## Slide 2 — Auditoria inicial (45 s, primeira cena a cortar)

- **Mensagem principal:** a Fase 1 era reproduzível em partes, mas misturava etapas e usava validação repetidamente.
- **Dados:** 60/20/20 histórico; dependências não congeladas; EDA supervisionada antes do corte.
- **Figura:** fluxo antes/depois simples.
- **Fala sugerida:** “Não encontramos vazamento direto de scaler, mas havia risco de processo e baixa reprodutibilidade.”
- **Risco:** afirmar que a Fase 1 era inválida. Ela foi reproduzida; os riscos foram corrigidos.

## Slide 3 — Arquitetura experimental protegida (60 s)

- **Mensagem principal:** seleção nos 80% de desenvolvimento, holdout bloqueado até o congelamento.
- **Dados:** 455 desenvolvimento, 114 holdout, seed 42, cinco dobras, threshold 0,5.
- **Figura:** Mermaid da arquitetura consolidada.
- **Fala sugerida:** “Toda escolha acontece à esquerda da barreira; o teste só confirma.”
- **Risco:** dizer que o holdout era totalmente desconhecido. O baseline histórico já o havia registrado.

## Slide 4 — Algoritmo Genético (75 s)

- **Mensagem principal:** operadores implementados diretamente e genomas específicos por modelo.
- **Dados:** fitness `0,60 recall + 0,25 F1 + 0,15 AUC − 0,10 std(recall)`.
- **Figura:** ciclo população→torneio→crossover→mutação→reparo→elitismo.
- **Fala sugerida:** “O indivíduo é uma configuração legível, não um vetor opaco.”
- **Risco:** chamar fitness de utilidade clínica; é decisão acadêmica.

## Slide 5 — Experimentos A/B/C e custo (65 s)

- **Mensagem principal:** famílias responderam de maneira diferente ao orçamento.
- **Dados:** 9 experimentos, 4.495 avaliações únicas, 22.475 fits, 51,12 min; RF 97,3% do tempo.
- **Figura:** a tabela A/B/C da seção 5 do `relatorio_final.md`. Não existe figura de avaliações e duração no repositório: as versionadas vão de `01_` a `06_`, e `07_` é a de escalabilidade.
- **Fala sugerida:** “KNN saturou cedo; RF precisou da busca exploratória; LR entrou em platô.”
- **Risco:** sugerir que maior orçamento sempre melhora.

## Slide 6 — GA versus busca aleatória (60 s)

- **Mensagem principal:** GA não venceu universalmente o benchmark.
- **Dados:** LR e KNN empatam nas métricas agregadas; RF fitness GA `0,961648` versus aleatória `0,958248`.
- **Figura:** `06_fitness_ga_vs_busca_aleatoria.png`.
- **Fala sugerida:** “O valor acadêmico está também em relatar empates e custo, não esconder resultado negativo.”
- **Risco:** tratar soluções idênticas como evidências independentes.

## Slide 7 — Avaliação final e resultados (75 s)

- **Mensagem principal:** LR e RF reduziram falsos negativos; KNN não melhorou recall.
- **Dados:** LR `3→1`, RF `4→3`, KNN `4→4`.
- **Figura:** `02_falsos_negativos_baseline_vs_ga.png`.
- **Fala sugerida:** “Este é o resultado confirmatório, não uma nova seleção.”
- **Risco:** chamar LR GA de novo vencedor; o modelo congelado continua LR da busca aleatória.

## Slide 8 — Métricas e trade-offs (50 s)

- **Mensagem principal:** melhoria depende da métrica.
- **Dados:** AUC GA caiu em RF e KNN; RF GA/aleatória têm mesma matriz e AUC distinta.
- **Figura:** `03_roc_auc_por_metodo.png`.
- **Fala sugerida:** “O threshold resume decisões binárias; AUC resume ordenação probabilística.”
- **Risco:** dizer que AUC diferente contradiz a matriz igual.

## Slide 9 — Incerteza (75 s)

- **Mensagem principal:** a evidência não sustenta superioridade estatística ou clínica.
- **Dados:** 42 malignos; ICs amplos; bootstrap inclui zero; McNemar 1–3 discordantes.
- **Figura:** `05_intervalos_recall.png`.
- **Fala sugerida:** “p alto não prova igualdade; um ganho observado não é validação clínica.”
- **Risco:** usar “significativo” como sinônimo de importante.

## Slide 10 — Camada LLM agregada e individual (90 s)

- **Mensagem principal:** a LLM explica resultados agregados e uma classificação individual desidentificada sob contratos separados.
- **Dados:** mock V1 com 139 checks; mock V2 com 327; contrato individual com 40/40 fatos no fake e na OpenAI real.
- **Figura:** fluxo LLM de `camada_llm_segura.md`.
- **Fala sugerida:** “O provider gera; código independente decide se a saída pode ser aprovada. No caso individual, a LLM recebe classe, probabilidade e cinco sinais, mas não recebe ID, índice, diagnóstico real ou valores brutos.”
- **Risco:** sugerir que outro LLM valida a resposta; as barreiras oficiais são determinísticas.
- **Onde cada número aparece:** o painel mostra os **139** checks do contrato agregado V1 e os **40/40** do individual. Os **327** do contrato V2 não estão no painel — vivem em `docs/contrato_llm_v2.md` e em `artifacts/llm_contract_v2/`. Cite o 327 pela documentação, não procurando em tela.

## Slide 11 — Escalabilidade automática e monitoramento (75 s)

- **Mensagem principal:** o requisito de escalabilidade foi implementado, medido e comparado — não apenas descrito.
- **Dados:** mesmo perfil de vale, rajada e drenagem, 146 pedidos em 4 CPUs. Pool fixo mínimo: p95 `131,6 ms`, `177,7 req/s`. Pool autoescalável: p95 `74,6 ms`, `301,9 req/s`. Redução de p95 de `1,76x`, ganho de vazão de `1,70x`.
- **Figura:** `07_escalabilidade_automatica.png` — a linha de workers acompanhando a demanda em cima, o limiar de custo por pedido embaixo.
- **Fala sugerida:** “A primeira medição mostrou o autoscaling *mais lento* que o pool fixo. A causa não era a política: o BLAS já paralelizava internamente e um worker sozinho saturava as CPUs. Fixar uma thread de BLAS por worker inverteu a relação. E mesmo depois disso, escalar réplicas só compensa acima de cerca de 2 ms por pedido — abaixo disso o despacho custa mais que o trabalho. Mantivemos a varredura inteira na evidência em vez de escolher o tamanho de lote que favorecia a conclusão.”
- **Risco:** apresentar os números como característica do modelo. São dependentes do hardware, e o relatório declara isso. Também não dizer que há autoscaling em nuvem rodando: o IaC existe e é validado no CI, mas nada foi provisionado.
- **Se sobrar tempo:** mostrar que o monitoramento recusa dado individual na escrita — a barreira reprovou o próprio código do benchmark, que usava `label` para nomear cenário, e o campo foi renomeado em vez de a regra ser afrouxada.

## Slide 12 — Segurança, divergências e limitações (75 s)

- **Mensagem principal:** transparência sobre GA B/C, holdout histórico e ausência de validação clínica.
- **Dados:** uma observação = 2,38 p.p. de recall; fonte única; sem validação externa.
- **Figura:** quadro de limitações, sem novo gráfico.
- **Fala sugerida:** “Não limpamos divergências históricas; registramos fonte e impacto.”
- **Risco:** esconder que o baseline já conhecia o holdout.

## Slide 13 — Demonstração e conclusão (70 s)

- **Mensagem principal:** pronto para defesa acadêmica e reprodução offline, não para uso clínico.
- **Em tela**, rolando a saída já produzida (ver *Antes de gravar*), nesta ordem:

```bash
uv run pytest                  # 230 testes a partir de um clone limpo
uv run validate-deliverable    # hashes de documentos, artefatos e figuras
uv run evaluate-llm-output     # factualidade, segurança e cinco dimensões
uv run validate-scalability    # escopo e ausência de dado individual no log
```

- **Em tela, o painel:** abra `reports/dashboard/index.html` e vá direto à aba *LLM agregada*, seção *Verificação independente*. Mostrar a resposta da LLM ao lado das 139 checagens que recalculam cada número é a forma mais rápida de tornar visível a tese do projeto.
- **Alternativa:** `notebooks/demonstracao.ipynb` percorre o mesmo caminho em um artefato só, útil se preferir não alternar entre terminal e arquivos.
- **Figura:** badge verde do CI mais o checklist da matriz de rastreabilidade.
- **Fala sugerida:** “A contribuição é otimização reproduzível mais explicação segura, com limites explícitos. O CI roda essa mesma suíte em Python 3.11 e 3.13 a cada push, então a reprodutibilidade não depende da minha máquina.”
- **Risco:** dizer que há serviço em nuvem no ar. O container e o IaC existem, são construídos e validados no CI, mas nenhum recurso foi provisionado. Também não executar `run-llm-evaluation` ao vivo: ele recusa reaproveitar a execução congelada por projeto, e a recusa parece falha para quem não conhece o motivo.
