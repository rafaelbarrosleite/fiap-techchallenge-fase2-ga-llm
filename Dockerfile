# Imagem do servico de inferencia sobre o pipeline congelado da avaliacao final.
#
# A imagem nao treina e nao reabre selecao: ela carrega um modelo ja assinado e
# confere o hash contra o manifesto no arranque. Se o hash divergir, o container
# falha em vez de servir um modelo desconhecido.
FROM python:3.12-slim AS base

# Uma thread de BLAS por replica. Sem isso a algebra linear paraleliza dentro do
# processo, satura as CPUs com uma replica e anula o efeito do autoscaling.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

# As dependencias mudam menos que o codigo, entao resolvem-se antes para
# aproveitar o cache de camadas entre builds.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY data ./data
COPY artifacts ./artifacts
RUN uv sync --frozen

# Falha cedo se o modelo congelado nao conferir com o manifesto assinado.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD uv run python -c "from tech_challenge_fase2.serving import resolve_frozen_model; resolve_frozen_model()"

RUN useradd --create-home --uid 10001 servico && chown -R servico:servico /app
USER servico

ENTRYPOINT ["uv", "run"]
CMD ["run-load-benchmark"]
