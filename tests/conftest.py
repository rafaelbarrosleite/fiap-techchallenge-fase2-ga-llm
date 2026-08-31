"""Garante que a suite encontre o pacote mesmo sem o install editavel.

O projeto usa layout `src/`, entao `tech_challenge_fase2` so e importavel
quando o pacote esta instalado no ambiente. Se o arquivo `.pth` do install
editavel some ou fica obsoleto -- ambiente recriado, `uv sync` interrompido,
troca de versao de Python -- toda a suite falha na coleta com
`ModuleNotFoundError`, sem que exista defeito algum no codigo.

Acrescentar `src` ao caminho de importacao remove essa dependencia. A suite
passa a testar o codigo-fonte da arvore, que e o que se quer verificar, em vez
de depender de um efeito colateral da instalacao.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
