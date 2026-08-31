# Roteiro de apresentação — 10 a 15 minutos

Convenção: Regressão Logística (LR), Random Forest (RF), KNN, baseline, GA, busca aleatória (`RandomizedSearchCV`), desenvolvimento, validação cruzada (CV), holdout (teste final) e modelo para demonstração.

## Slide 1 — Problema e pergunta acadêmica (45 s)

- **Mensagem principal:** otimizar três modelos sem contaminar o teste e explicar resultados sem criar diagnóstico.
- **Dados:** 569 registros, 30 preditores, 212 malignos.
- **Figura:** diagrama de arquitetura do relatório final.
- **Fala sugerida:** “O desafio não foi apenas melhorar uma métrica; foi construir uma cadeia auditável de seleção, confirmação e explicação.”
- **Risco:** parecer proposta clínica. Dizer explicitamente “acadêmico e experimental”.

## Slide 2 — Auditoria inicial (55 s)

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

## Slide 5 — Experimentos A/B/C e custo (75 s)

- **Mensagem principal:** famílias responderam de maneira diferente ao orçamento.
- **Dados:** 9 experimentos, 4.495 avaliações únicas, 22.475 fits, 51,12 min; RF 97,3% do tempo.
- **Figura:** figura histórica `07_avaliacoes_e_duracao.png` ou tabela A/B/C.
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

## Slide 8 — Métricas e trade-offs (60 s)

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

## Slide 10 — Camada LLM (70 s)

- **Mensagem principal:** a LLM explica apenas agregados sob contrato.
- **Dados:** mock V1 com 139 checks; mock V2 com 327; execução real V2 com 327/327 fatos, mas gate lexical de calibração não aprovado.
- **Figura:** fluxo LLM de `camada_llm_segura.md`.
- **Fala sugerida:** “O provider gera; código independente decide se a saída pode ser aprovada. A execução real acertou todos os fatos, mas foi preservada como não aprovada porque o gate lexical é conservador.”
- **Risco:** sugerir que outro LLM valida a resposta; as barreiras oficiais são determinísticas.

## Slide 11 — Segurança, divergências e limitações (75 s)

- **Mensagem principal:** transparência sobre GA B/C, holdout histórico e ausência de validação clínica.
- **Dados:** uma observação = 2,38 p.p. de recall; fonte única; sem validação externa.
- **Figura:** quadro de limitações, sem novo gráfico.
- **Fala sugerida:** “Não limpamos divergências históricas; registramos fonte e impacto.”
- **Risco:** esconder que o baseline já conhecia o holdout.

## Slide 12 — Conclusão e demonstração (45 s)

- **Mensagem principal:** pronto para defesa acadêmica e reprodução offline, não para uso clínico.
- **Dados:** testes finais e manifesto da entrega.
- **Figura:** checklist da matriz de rastreabilidade.
- **Fala sugerida:** “A contribuição é otimização reproduzível mais explicação segura, com limites explícitos.”
- **Risco:** prometer API, cloud ou deploy; não fazem parte desta missão.
