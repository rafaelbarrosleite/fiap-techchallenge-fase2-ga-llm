# Roteiro de gravação do vídeo — Tech Challenge Fase 2

Roteiro de **screencast**: você grava a tela com o sistema rodando de verdade e lê a fala. Não há slides.

O que o enunciado exige, e onde cada item é atendido:

| Exigência | Bloco |
|---|---|
| Upload no YouTube ou Vimeo, até 15 minutos | ver *Depois de gravar*, no fim |
| Demonstração do sistema em execução | 2, 4, 5, 6, 7 |
| Explicação dos diferentes componentes da solução | 1, 3, 5, 6 |
| Apresentação dos resultados da otimização via algoritmos genéticos | 3, 4 |
| Demonstração da integração com LLMs | 5, 6 |

**Duração planejada: 13 min 30 s.** Sobra margem para pausas.

| Bloco | Assunto | Tempo |
|---|---|---:|
| 0 | Preparação (não grava) | — |
| 1 | Abertura e o que é o projeto | 1 min 00 s |
| 2 | O sistema rodando: suíte e validação | 1 min 30 s |
| 3 | Componentes: o algoritmo genético no código | 2 min 00 s |
| 4 | Resultados da otimização genética | 2 min 15 s |
| 5 | Integração com LLM: contrato, prompt e execução | 2 min 45 s |
| 6 | Explicação individual e as barreiras recusando | 2 min 00 s |
| 7 | Escalabilidade automática rodando ao vivo | 1 min 30 s |
| 8 | Fechamento e limitações | 30 s |

---

## Bloco 0 — Preparação (antes de apertar REC)

1. Terminal na raiz do projeto, fonte grande (14–16 pt), janela ocupando a tela toda.
2. Rode uma vez, **sem gravar**, para aquecer cache e confirmar que tudo passa:

```bash
uv sync --reinstall
uv run pytest -q
uv run validate-deliverable
```

O `--reinstall` reinstala o pacote do projeto no ambiente.

**Regra:** rode `uv sync --reinstall` depois de qualquer `git pull`. Alterações em `pyproject.toml` obrigam o uv a reinstalar o projeto, e um install pela metade derruba tanto o `pytest` quanto os comandos como `validate-deliverable`, com `ModuleNotFoundError`. O `uv run` **não** conserta isso sozinho: ele considera o projeto instalado e não refaz a instalação. Só o `--reinstall` resolve.

3. Abra o editor com a pasta do projeto e deixe estes arquivos em abas, nesta ordem:
   - `src/tech_challenge_fase2/genetic/genomes.py`
   - `src/tech_challenge_fase2/genetic/fitness.py`
   - `src/tech_challenge_fase2/genetic/operators.py`
   - `src/tech_challenge_fase2/llm/prompts/system_v1.txt`
   - `src/tech_challenge_fase2/llm/privacy.py`
   - `src/tech_challenge_fase2/serving/autoscaling.py`
4. Abra `reports/dashboard/index.html` no navegador, em outra aba.
5. Abra a aba Actions do repositório no GitHub, mostrando o CI verde.
6. Limpe o terminal (`clear`) e comece.

> **Atenção:** o Bloco 7 roda o benchmark, que **sobrescreve** `artifacts/scalability/scalability_report.json` com números do seu hardware. Depois de gravar, rode `git checkout -- artifacts/scalability/ && uv run build-dashboard` para devolver o repositório ao estado publicado.

---

## Bloco 1 — Abertura (1 min 00 s)

**Mostre:** o editor com a árvore do projeto aberta à esquerda.

> Olá. Este é o Tech Challenge da Fase 2, Projeto 1: otimização de modelos de diagnóstico com algoritmos genéticos e interpretação de resultados com LLM.
>
> O ponto de partida é o classificador de tumores da Fase 1, treinado sobre o Breast Cancer Wisconsin: 569 registros, 30 preditores, 212 casos malignos. Três famílias de modelo — Regressão Logística, Random Forest e KNN.
>
> O trabalho tem duas metades. Na primeira, um algoritmo genético autoral busca hiperparâmetros para essas três famílias. Na segunda, uma camada de LLM transforma os resultados em explicação em linguagem natural, sob barreiras que impedem tanto invenção de número quanto vazamento de dado de paciente.
>
> Antes de tudo, um aviso que vale para o vídeo inteiro: este é um resultado acadêmico e experimental. Os modelos não foram validados para uso clínico e não devem ser usados para diagnóstico, tratamento ou decisão médica.

**Ação na tela:** passe o mouse pela árvore mostrando `src/`, `docs/`, `artifacts/`, `tests/`.

> A lógica fica em `src`, para ser testável. As evidências ficam em `artifacts`, versionadas: quem clonar este repositório reproduz tudo sem depender da minha máquina.

---

## Bloco 2 — O sistema em execução (1 min 30 s)

**Comando 1** — a suíte completa:

```bash
uv run pytest -q
```

*(demora cerca de 30 segundos; fale durante)*

> Estou rodando a suíte inteira a partir do repositório como ele está publicado. São 230 testes: operadores do algoritmo genético, reprodutibilidade com semente fixa, contratos da LLM, barreiras de privacidade, a camada de escalabilidade e a integridade dos artefatos.

*(quando aparecer `230 passed`)*

> Duzentos e trinta testes passando. Esses mesmos testes rodam no GitHub Actions a cada push, em Python 3.11 e 3.13.

**Ação:** troque para a aba do GitHub Actions e mostre os jobs verdes por dois segundos.

> Aqui está o CI: suíte nas duas versões de Python, benchmark de escalabilidade, construção do painel, e a imagem do container mais a validação do Terraform.

**Comando 2** — a validação da entrega:

```bash
uv run validate-deliverable | head -12
```

> Este comando é somente leitura. Ele confere hash por hash os documentos, os artefatos e as figuras contra um manifesto assinado. São 57 verificações. Se qualquer número do relatório divergisse do artefato que o originou, isto falharia.

---

## Bloco 3 — Componentes: o algoritmo genético (2 min 00 s)

**Ação:** abra `src/tech_challenge_fase2/genetic/genomes.py`, linhas 13 a 45.

> O primeiro requisito é a codificação dos genes. Cada indivíduo é uma configuração legível, não um vetor opaco.
>
> A Regressão Logística tem `log10_c` — o parâmetro C representado em log de base 10, para cobrir várias ordens de grandeza —, o tipo de regularização e o peso de classe. O Random Forest tem número de árvores, profundidade, mínimos de amostras, `max_features` e peso de classe. O KNN tem vizinhos, pesos, métrica e o parâmetro `p`, que só existe quando a métrica é Minkowski.

**Ação:** abra `src/tech_challenge_fase2/genetic/fitness.py`, função `calculate_fitness`, linha 59.

> A função de fitness é esta. Sessenta por cento de recall da classe maligna, vinte e cinco por cento de F1, quinze por cento de ROC-AUC, menos dez por cento do desvio-padrão do recall entre as dobras.
>
> Recall pesa mais porque, neste problema, o erro caro é o falso negativo: deixar de sinalizar um caso maligno. O termo de desvio-padrão penaliza soluções instáveis entre dobras — uma configuração que vai muito bem em uma dobra e mal em outra não interessa.
>
> E um detalhe metodológico central: isto é calculado em cinco dobras **somente** sobre os dados de desenvolvimento. O conjunto de teste não participa da busca.

**Ação:** abra `src/tech_challenge_fase2/genetic/operators.py` e role pelas funções `tournament_select` (58), `uniform_crossover` (71), `mutate_genome` (231) e `select_elites` (289).

> Os operadores foram implementados diretamente, sem biblioteca de evolução. Seleção por torneio, crossover uniforme, mutação por tipo de gene — inteiro, real ou categórico, cada um com sua regra —, reparação de indivíduos inválidos e elitismo.
>
> Há também cache de avaliações, histórico por geração e checkpoints. Um ajuste que falha recebe fitness menos um e fica registrado, em vez de ser escondido.

---

## Bloco 4 — Resultados da otimização genética (2 min 15 s)

**Ação:** vá para o navegador, no painel, aba **Visão geral**.

> Este é o painel de resultados, gerado a partir dos artefatos assinados.
>
> Foram nove experimentos: três configurações do algoritmo genético — que variam tamanho de população, número de gerações e taxas de crossover e mutação — aplicadas às três famílias. A bateria fez 4.495 avaliações únicas e 22.475 ajustes de modelo, em 51 minutos.

**Ação:** aponte para a tabela mestre.

> Aqui estão as nove origens avaliadas no mesmo conjunto de teste. A linha destacada é o vencedor global, congelado **antes** de o teste ser aberto.
>
> Na Regressão Logística, o recall subiu de 0,9286 para 0,9762, e os falsos negativos caíram de três para um. No Random Forest, de quatro para três. No KNN, o recall não mudou: permaneceu em 0,9048, com quatro falsos negativos.

**Ação:** aba **Algoritmo genético**, role até as figuras.

> Vale dizer isso com clareza, porque é um resultado negativo que eu escolhi mostrar: o algoritmo genético **não** venceu universalmente. No KNN, o ganho que apareceu na validação cruzada não se confirmou no teste. E o ROC-AUC caiu no Random Forest e no KNN, mesmo com o recall melhorando.
>
> Comparado a uma busca aleatória com o mesmo orçamento, o genético empatou em duas famílias e ganhou em uma. O valor acadêmico está em relatar isso, não em esconder.

**Ação:** role até a figura dos intervalos de recall.

> E a incerteza. São apenas 42 casos malignos no teste, então uma observação muda o recall em 2,4 pontos percentuais. Os intervalos de confiança são amplos, o intervalo do delta inclui zero e o teste de McNemar tem entre um e três pares discordantes.
>
> Ou seja: houve ganho observado, mas não há evidência suficiente para afirmar superioridade estatística. Muito menos clínica.

---

## Bloco 5 — Integração com LLM (2 min 45 s)

**Ação:** abra `src/tech_challenge_fase2/llm/prompts/system_v1.txt`.

> Agora a segunda metade. O prompt é versionado e faz parte do contrato — ele tem hash, e mudá-lo muda a identidade da execução.
>
> Ele declara o propósito, o contrato de entrada, o contrato de saída e as regras de segurança: não diagnosticar, não recomendar tratamento, não indicar modelo para uso em pacientes e não afirmar aprovação médica.

**Ação:** abra `src/tech_challenge_fase2/llm/privacy.py`, linha 16, `FORBIDDEN_KEY_PARTS`.

> E esta é a barreira de entrada. A LLM recebe **apenas** resultados agregados. Identificador, índice, diagnóstico real, valores brutos de atributo — nada disso atravessa.

**Comando 3** — a avaliação da saída da LLM:

```bash
uv run evaluate-llm-output
```

> Aprovado, nota 1,0. Mas o número sozinho não diz nada. Vou mostrar o que está por trás.

**Ação:** navegador, aba **LLM agregada** do painel. Role até *Verificação independente* e abra a lista de checagens.

> Aqui está a explicação que a LLM gerou: resumo executivo, interpretação do algoritmo genético, leitura da incerteza, comparação por família e as limitações que ela própria declara.
>
> E aqui está o que eu considero o ponto central do projeto. Cada afirmação numérica dessa explicação é conferida por código independente, que recalcula o valor a partir do artefato congelado e compara. São 139 verificações, uma para cada fato: métrica, contagem, intervalo, conclusão.
>
> Repare na coluna da direita: `esperado=0.9285714285714286`. Esse valor não veio da LLM — veio do artefato. Se a LLM tivesse escrito outro número, a linha ficaria vermelha e a saída seria reprovada.
>
> Quem julga a resposta é código determinístico, não outro modelo. Não existe LLM avaliando LLM aqui.

**Ação:** role até o cartão de segurança.

> Além da factualidade, um verificador de segurança procura linguagem de diagnóstico, de tratamento, de recomendação clínica e de certeza indevida. Zero violações, e o disclaimer obrigatório presente.
>
> A execução oficial usa um provider mock determinístico, para que a demonstração seja reproduzível offline. Mas houve também uma execução real com a OpenAI, preservada no repositório: ela passou 327 de 327 verificações factuais e não foi aprovada por três checagens lexicais de calibração. Eu mantive essa reprovação registrada em vez de ajustar o verificador depois da resposta.

---

## Bloco 6 — Explicação individual e as barreiras (2 min 00 s)

**Comando 4** — a explicação de uma classificação individual:

```bash
uv run run-individual-explanation
uv run evaluate-individual-explanation
```

> O enunciado pede explicações dos diagnósticos produzidos pelos modelos, não apenas dos agregados. Então existe um contrato separado para o caso individual.

**Ação:** navegador, aba **LLM individual** do painel.

> Este é um caso do conjunto de desenvolvimento, classificado pelo modelo congelado. A probabilidade estimada foi 0,74553, acima do limiar de 0,5, então a classe saiu como padrão maligno.
>
> A explicação lista os cinco sinais que mais influenciaram, com a faixa em que cada um caiu, a direção da influência e a importância relativa. O primeiro responde por quase 56 por cento.
>
> Agora o ponto delicado. A LLM recebe o nome do sinal, a faixa e a importância — sem isso não haveria explicação nenhuma. Mas ela **não** recebe o identificador, o índice, o diagnóstico real, nem os trinta valores medidos. O caso chega como uma referência opaca: `demo_case_001`.

**Ação:** role até os insights.

> E cada ação sugerida é estruturalmente limitada a revisão humana. O campo `patient_care_decision` é falso por contrato: a saída não pode representar decisão de cuidado.

**Comando 5** — as barreiras recusando ao vivo:

```bash
uv run demo-barreiras
```

> Em vez de só afirmar que as barreiras existem, vou mostrá-las funcionando.
>
> Primeiro, o contrato agregado oficial passa. Depois, eu pego exatamente o mesmo contrato e acrescento um `patient_id` — a barreira recusa e nomeia o campo. E terceiro, tento gravar um evento de monitoramento com probabilidade por registro; recusado também.
>
> Essa terceira barreira já reprovou o meu próprio código durante o desenvolvimento: o benchmark usava a palavra `label` para nomear um cenário, e colidia com rótulo de classe. Eu renomeei o campo em vez de afrouxar a regra.

---

## Bloco 7 — Escalabilidade automática (1 min 30 s)

**Ação:** abra `src/tech_challenge_fase2/serving/autoscaling.py`, método `decide`, linha 67.

> O enunciado pede recursos de escalabilidade automática para lidar com variações de demanda. Esta é a política: uma função pura do backlog observado. Ela decide quantos workers manter, com histerese entre duas e seis requisições por worker, para não trocar de tamanho a cada oscilação.

**Comando 6** — o benchmark ao vivo:

```bash
uv run run-load-benchmark
```

*(demora cerca de 3 segundos)*

> Isto está medindo agora, nesta máquina. O mesmo perfil de demanda — vale, rajada e drenagem — servido de duas formas: com um pool fixo mínimo e com o pool autoescalável.

*(leia os números que aparecerem na sua tela, não os do roteiro)*

> Com o pool fixo, a latência p95 fica no valor de cima; com autoscaling, cai para o de baixo, e a vazão sobe. Os números variam conforme o hardware — o relatório declara isso explicitamente e não os apresenta como característica do modelo.

**Ação:** navegador, aba **Escalabilidade**, mostre a figura.

> Em cima, a linha laranja é o número de workers acompanhando a demanda: ocioso em um, sobe até quatro na rajada, drena de volta.
>
> Embaixo há um achado que eu mantive porque mudou o desenho. Escalar réplicas só compensa acima de cerca de dois milissegundos por pedido. Abaixo disso, o custo de despachar é maior que o trabalho, e adicionar workers **piora** o desempenho — as duas barras cinzas.
>
> E antes disso houve um achado ainda mais desconfortável: a primeira medição mostrou o autoscaling mais lento que o pool fixo. A causa não era a política. Era que a biblioteca de álgebra linear já paralelizava internamente, e um worker sozinho saturava as CPUs. Fixar uma thread por worker inverteu a relação. Preferi mostrar a varredura inteira a escolher o tamanho de lote que favorecia a conclusão.

---

## Bloco 8 — Fechamento (30 s)

**Ação:** volte ao painel, aba **Visão geral**, seção *Garantias de escopo*.

> Para encerrar. Estas são as garantias de escopo do manifesto assinado: nenhum treino novo, nenhuma otimização nova, seleção não reaberta, limiar não alterado, nenhum identificador de paciente enviado à LLM e nenhum recurso de nuvem provisionado.
>
> O container e a infraestrutura como código existem e são construídos e validados no CI a cada push, mas nada foi provisionado. Não há serviço no ar.
>
> Resumindo: o algoritmo genético reduziu falsos negativos em duas das três famílias, sem superioridade universal e sem significância estatística demonstrada. A camada de LLM explica os resultados sob verificação determinística de cada número e sob barreiras de privacidade que recusam dado individual.
>
> E encerro repetindo o que abri: isto é acadêmico e experimental. Não é validado para uso clínico e não deve orientar diagnóstico, tratamento ou decisão médica. Obrigado.

---

## Depois de gravar

1. Devolva o repositório ao estado publicado, já que o Bloco 7 sobrescreveu a medição:

```bash
git checkout -- artifacts/scalability/
uv run build-dashboard
git status --short          # precisa sair vazio
```

2. Confira o vídeo: duração abaixo de 15 minutos, áudio audível, texto do terminal legível.
3. Suba no YouTube ou Vimeo como **público ou não listado** — não deixe privado, a banca precisa abrir.
4. Inclua o link na entrega e no `README.md`.

## Se algo falhar durante a gravação

| Sintoma | O que fazer |
|---|---|
| `ModuleNotFoundError: No module named 'tech_challenge_fase2'` | O ambiente perdeu o install do pacote. Rode `uv sync --reinstall` e siga |
| Um comando falha | Pare, rode `uv sync --reinstall`, confirme `uv run pytest -q` e regrave o bloco |
| `validate-deliverable` reprova | Não regrave por cima: algum artefato divergiu do manifesto. Investigue antes |
| O painel abre desatualizado | `uv run build-dashboard` e recarregue a página |
| O tempo estourou | Corte o Bloco 3 pela metade: mostre só `fitness.py`, sem `operators.py` |

**Não execute `uv run run-llm-evaluation` durante a gravação.** Ele recusa reaproveitar a execução congelada, por projeto, e a recusa parece falha para quem não conhece o motivo. A explicação está em `docs/limitacoes_e_validade.md`.
