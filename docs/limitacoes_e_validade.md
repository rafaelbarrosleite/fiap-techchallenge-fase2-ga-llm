# Limitações e validade

## Validade interna

Pontos fortes:

- dataset e split identificados por SHA-256;
- 455 linhas usadas para ajuste e 114 reservadas para confirmação;
- candidatos escolhidos exclusivamente por cinco dobras de CV;
- limiar 0,5 e preprocessing congelados;
- nove origens avaliadas na mesma sessão e oito configurações canônicas treinadas;
- previsões por registro, métricas recalculáveis, hashes e modelos locais;
- comparação pareada entre baseline e GA;
- nenhuma nova busca ou seleção depois de observar o teste.

Ressalva histórica: o baseline da primeira missão já havia registrado métricas no mesmo holdout. Esse arquivo não foi usado na linhagem de seleção da terceira missão, mas significa que o conjunto não era literalmente desconhecido para o projeto como um todo. A avaliação atual deve ser entendida como confirmação controlada dos candidatos congelados, não como primeiro contato absoluto do repositório com aquelas 114 linhas.

## Incerteza estatística

O holdout tem apenas 42 casos malignos. Uma observação muda o recall em aproximadamente 2,38 pontos percentuais. Por isso:

- os IC95% de Wilson do recall são largos e se sobrepõem;
- o bootstrap pareado inclui zero no limite ou no interior dos deltas principais;
- McNemar tem apenas 1 a 3 pares discordantes, contagem insuficiente para conclusão forte;
- p alto não demonstra equivalência;
- nenhum p-valor mede relevância clínica.

## Validade externa

O Breast Cancer Wisconsin (Diagnostic) é uma amostra pequena, de uma única fonte e com variáveis derivadas de imagens de núcleos celulares. Não há validação temporal, multicêntrica, prospectiva ou externa. Distribuições de outros hospitais, equipamentos e populações podem ser diferentes.

## Métricas e objetivo

O fitness atribui 60% ao recall maligno, 25% ao F1, 15% ao ROC-AUC e penaliza instabilidade de recall. Essa função representa uma prioridade acadêmica aprovada. Ela não é uma função de utilidade clínica validada e não inclui calibração, prevalência local, custo de exames, impacto de falsos positivos ou desfechos de pacientes.

O teste mostrou o trade-off esperado: Random Forest e KNN GA melhoraram alguns resultados no limiar, mas reduziram ROC-AUC. Não existe “melhoria universal” de um modelo.

## Multiplicidade e interpretação

Nove origens foram relatadas, mas não devem ser tratadas como nove testes confirmatórios independentes:

- KNN GA e busca aleatória são a mesma solução e compartilham treino;
- LR GA e busca aleatória são configurações próximas que produziram saídas iguais neste holdout;
- os modelos compartilham os mesmos 114 casos;
- o projeto não ajustou p-valores por comparações múltiplas porque não usa os testes para nova seleção.

## Reprodutibilidade e serialização

Os Pipelines foram serializados com joblib e scikit-learn 1.7.1. Arquivos pickle/joblib podem executar código ao carregar; somente os modelos locais com hashes presentes no manifesto devem ser considerados confiáveis. Mudanças de versão podem impedir carregamento ou alterar comportamento.

Tempos de parede refletem o Mac mini M4 e a carga do momento. Não são benchmarks universais.

## Validade da camada LLM

O provider fake V1/V2 é a referência oficial offline porque é determinístico e não depende de rede. A avaliação complementar real representa uma única resposta do modelo retornado `gpt-5.5-2026-04-23`; ela não generaliza para outras versões, providers ou execuções.

Essa resposta obteve 327/327 fatos e passou segurança, completude e clareza, mas permaneceu não aprovada porque três checks de calibração exigiam formulações lexicais específicas. O caso mostra duas limitações simultâneas: respostas reais variam e regras determinísticas estreitas podem reprovar paráfrases semanticamente adequadas. O resultado histórico foi preservado sem retry ou ajuste retrospectivo.

Nenhuma avaliação LLM — fake ou real — acrescenta validade clínica, nova evidência estatística ou autorização de uso em pacientes.

## Escopo de uso

Este projeto é acadêmico. Não substitui profissional de saúde, não emite diagnóstico, não foi validado para decisão clínica e não estabelece segurança, eficácia médica ou conformidade regulatória.

## Próxima evidência necessária

Antes de qualquer pretensão de uso real seriam necessários, no mínimo:

- validação externa e prospectiva;
- avaliação de calibração;
- análise por subgrupos com base ética e estatística;
- protocolo clínico e revisão especializada;
- governança, privacidade, segurança e monitoramento;
- estudo de impacto que não reutilize este holdout para novas escolhas.
