"""Barreira de privacidade aplicada ao HTML antes de ele ser escrito.

O painel le artefatos que ja passaram por barreiras proprias, mas a renderizacao
e uma superficie nova: basta um campo a mais em um laco para publicar dado que
nenhuma camada anterior autorizou. A verificacao roda sobre o texto final, nao
sobre a intencao do gerador, e falha a construcao em vez de emitir o arquivo.
"""

from __future__ import annotations

import re

# Rotulos que so apareceriam se o painel expusesse um registro do dataset.
# O risco nao e nomear qual sinal influenciou a decisao: o contrato individual
# expoe justamente isso, com faixa categorica, direcao e importancia relativa,
# e uma explicacao que omitisse o nome do sinal nao explicaria nada. O risco e o
# valor medido do registro, que reconstruiria o caso. Estes marcadores cobrem
# identificacao, alvo e valor bruto -- nao nomes de atributo.
FORBIDDEN_MARKERS = (
    "patient_id",
    "record_id",
    "raw_features",
    "feature_values",
    "observed_value",
    "raw_value",
    "ground_truth",
    "original_index",
    "predict_proba",
)

# Uma sequencia longa de numeros separados por virgula tem a forma de uma linha
# do dataset exportada por engano.
NUMERIC_ROW_PATTERN = re.compile(r"(?:-?\d+\.\d{3,}\s*,\s*){8,}-?\d+\.\d{3,}")


class DashboardPrivacyError(RuntimeError):
    """O HTML renderizado carregaria dado individual."""


def assert_html_has_no_individual_data(html: str) -> None:
    """Recusa o documento se ele contiver marca de registro individual."""

    # Casar o token inteiro. Um substring ingenuo confundiria `patient_id` com
    # `patient_identifier_sent_to_llm`, que e uma confirmacao de ausencia -- o
    # painel seria bloqueado justamente por declarar que nada foi enviado.
    found = sorted(
        marker for marker in FORBIDDEN_MARKERS
        if re.search(rf"\b{re.escape(marker)}\b", html)
    )
    if found:
        raise DashboardPrivacyError(
            f"O painel renderizado contem marcadores individuais proibidos: {found}."
        )

    row = NUMERIC_ROW_PATTERN.search(html)
    if row is not None:
        raise DashboardPrivacyError(
            "O painel renderizado contem uma sequencia numerica longa com forma de "
            f"linha do dataset: {row.group(0)[:60]!r}..."
        )
