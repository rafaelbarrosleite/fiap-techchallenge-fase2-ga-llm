"""Monitoramento de desempenho em eventos estruturados, sem dado de paciente.

O projeto ja impede que a camada LLM receba registros individuais. A camada de
servico repete a barreira no plano da observabilidade: um log de desempenho
guarda contagens, tempos e tamanhos, nunca features, probabilidades por
registro ou identificadores. O guard e aplicado na escrita, nao em revisao
posterior, porque log emitido nao pode ser retirado de onde ja foi coletado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

# Chaves proibidas em qualquer evento. Cobrem identificacao direta, o vetor de
# features do dataset e saidas por registro que permitiriam reconstruir um caso.
FORBIDDEN_EVENT_KEYS = frozenset(
    {
        "id", "patient_id", "record_id", "index", "row", "rows",
        "diagnosis", "target", "label", "y", "y_true", "y_pred",
        "features", "feature_values", "x", "probability", "probabilities",
        "prediction", "predictions", "proba", "predict_proba",
    }
)


class MonitoringError(ValueError):
    """Um evento tentou registrar dado individual ou campo proibido."""


def assert_event_is_aggregate(event: dict[str, Any]) -> None:
    """Recusa eventos que carreguem identificacao ou saida por registro."""

    found = sorted(FORBIDDEN_EVENT_KEYS.intersection(event))
    if found:
        raise MonitoringError(
            f"Evento de monitoramento contem campos individuais proibidos: {found}."
        )
    for key, value in event.items():
        if isinstance(value, dict):
            assert_event_is_aggregate(value)
        elif isinstance(value, (list, tuple)) and any(
            isinstance(item, dict) for item in value
        ):
            for item in value:
                if isinstance(item, dict):
                    assert_event_is_aggregate(item)
        elif isinstance(value, (list, tuple)) and len(value) > 64:
            raise MonitoringError(
                f"Campo {key!r} tem {len(value)} itens; series longas podem "
                "reconstruir saidas por registro."
            )


def percentile(values: Iterable[float], fraction: float) -> float:
    """Percentil por interpolacao linear, estavel para amostras pequenas."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise MonitoringError("Percentil exige ao menos uma observacao.")
    if not 0.0 <= fraction <= 1.0:
        raise MonitoringError("A fracao do percentil precisa estar em [0, 1].")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float]:
    """Resumo de latencia usado em relatorio e figura."""

    values = [float(value) for value in latencies_ms]
    if not values:
        raise MonitoringError("Resumo de latencia exige ao menos uma observacao.")
    return {
        "count": len(values),
        "min_ms": round(min(values), 6),
        "mean_ms": round(fmean(values), 6),
        "p50_ms": round(percentile(values, 0.50), 6),
        "p95_ms": round(percentile(values, 0.95), 6),
        "p99_ms": round(percentile(values, 0.99), 6),
        "max_ms": round(max(values), 6),
    }


@dataclass
class PerformanceMonitor:
    """Coletor de eventos de desempenho em memoria, opcionalmente em JSONL.

    Os eventos ficam disponiveis para agregacao mesmo sem arquivo, o que
    mantem os testes offline e sem efeito colateral em disco.
    """

    log_path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Valida e registra um evento agregado."""

        event = {"event": event_type, **fields}
        assert_event_is_aggregate(event)
        self.events.append(event)
        if self.log_path is not None:
            line = json.dumps(event, sort_keys=True, ensure_ascii=False)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        return event

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["event"] == event_type]
