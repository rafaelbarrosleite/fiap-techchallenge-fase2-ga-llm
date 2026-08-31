"""Demonstracao ao vivo das barreiras de privacidade.

Existe para a gravacao do video: mostrar a barreira recusando e mais convincente
que afirmar que ela existe. Nao escreve nada e nao chama rede.
"""

from __future__ import annotations

import json
from pathlib import Path

from .llm.privacy import validate_sanitized_input
from .serving.monitoring import PerformanceMonitor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = PROJECT_ROOT / "artifacts" / "llm_evaluation" / "llm_input_snapshot.json"


def barriers_main() -> None:
    entrada = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["input"]

    validate_sanitized_input(entrada)
    print("1. Contrato agregado oficial: ACEITO pela barreira")

    print("\n2. O mesmo contrato, acrescentando um identificador de paciente:")
    try:
        validate_sanitized_input({**entrada, "patient_id": "8510426"})
        print("   *** PASSOU -- nao deveria")
    except Exception as erro:  # noqa: BLE001 - a mensagem e o resultado exibido
        print(f"   RECUSADO -> {str(erro)[:120]}")

    print("\n3. Evento de desempenho com probabilidade por registro:")
    try:
        PerformanceMonitor().record("ciclo", workers=4, probability=0.97)
        print("   *** PASSOU -- nao deveria")
    except Exception as erro:  # noqa: BLE001
        print(f"   RECUSADO -> {str(erro)[:120]}")
