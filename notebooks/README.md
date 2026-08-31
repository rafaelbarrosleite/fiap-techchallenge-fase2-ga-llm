# Notebooks

[`demonstracao.ipynb`](demonstracao.ipynb) percorre a entrega inteira em modo somente leitura: valida assinaturas, exibe a tabela mestre, o comparativo baseline versus GA, a avaliação da camada LLM, a explicação individual desidentificada e a medição de escalabilidade.

O notebook não treina, não reabre seleção, não altera o limiar, não consulta o holdout e não faz chamada de rede. A lógica permanece em `src/` para ser testável; o notebook apenas a exercita.

```bash
uv run --with jupyter jupyter lab notebooks/demonstracao.ipynb
```

Os comandos de reprodução metodológica — bateria genética, busca aleatória, baseline e providers reais — ficam fora dele de propósito: são caros e sobrescreveriam evidência congelada.
