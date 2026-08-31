"""Politica de escalabilidade automatica dirigida pelo backlog de pedidos.

A politica e uma funcao pura do estado observado. Manter a decisao separada
da execucao permite testa-la de forma deterministica, sem relogio, threads ou
modelo carregado, e permite auditar por que cada troca de tamanho ocorreu.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import ceil
from typing import Any


def default_max_workers() -> int:
    """Teto padrao: uma replica por CPU disponivel ao processo."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # plataformas sem sched_getaffinity
        return max(1, os.cpu_count() or 1)

SCALE_UP = "scale_up"
SCALE_DOWN = "scale_down"
HOLD = "hold"
COOLDOWN = "cooldown_block"


class AutoscalingError(ValueError):
    """A politica foi configurada com limites incoerentes."""


@dataclass(frozen=True)
class AutoscalingPolicy:
    """Limites e histerese do dimensionamento automatico.

    `target_backlog_per_worker` define a carga que cada worker deve absorver.
    As margens de subida e descida criam uma faixa morta: sem ela, um backlog
    oscilando ao redor do alvo provocaria troca de tamanho a cada ciclo.
    """

    min_workers: int = 1
    max_workers: int = field(default_factory=default_max_workers)
    target_backlog_per_worker: int = 4
    scale_up_backlog_per_worker: int = 6
    scale_down_backlog_per_worker: int = 2
    cooldown_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.min_workers < 1:
            raise AutoscalingError("min_workers precisa ser pelo menos 1.")
        if self.max_workers < self.min_workers:
            raise AutoscalingError("max_workers nao pode ser menor que min_workers.")
        if self.target_backlog_per_worker < 1:
            raise AutoscalingError("target_backlog_per_worker precisa ser positivo.")
        if self.scale_down_backlog_per_worker >= self.scale_up_backlog_per_worker:
            raise AutoscalingError(
                "A margem de descida precisa ser menor que a de subida para haver histerese."
            )
        if self.cooldown_seconds < 0:
            raise AutoscalingError("cooldown_seconds nao pode ser negativo.")

    def clamp(self, workers: int) -> int:
        return max(self.min_workers, min(self.max_workers, workers))

    def decide(
        self,
        *,
        backlog: int,
        current_workers: int,
        seconds_since_last_change: float,
    ) -> "AutoscalingDecision":
        """Escolhe o numero de workers para o proximo ciclo."""

        if backlog < 0:
            raise AutoscalingError("backlog nao pode ser negativo.")
        current = self.clamp(current_workers)
        per_worker = backlog / current

        if per_worker >= self.scale_up_backlog_per_worker:
            action, desired = SCALE_UP, ceil(backlog / self.target_backlog_per_worker)
        elif per_worker <= self.scale_down_backlog_per_worker:
            action, desired = SCALE_DOWN, ceil(backlog / self.target_backlog_per_worker)
        else:
            return AutoscalingDecision(HOLD, current, current, backlog, per_worker)

        desired = self.clamp(desired)
        if desired == current:
            return AutoscalingDecision(HOLD, current, current, backlog, per_worker)
        if seconds_since_last_change < self.cooldown_seconds:
            return AutoscalingDecision(COOLDOWN, current, desired, backlog, per_worker)
        return AutoscalingDecision(action, desired, desired, backlog, per_worker)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "target_backlog_per_worker": self.target_backlog_per_worker,
            "scale_up_backlog_per_worker": self.scale_up_backlog_per_worker,
            "scale_down_backlog_per_worker": self.scale_down_backlog_per_worker,
            "cooldown_seconds": self.cooldown_seconds,
        }


@dataclass(frozen=True)
class AutoscalingDecision:
    """Resultado auditavel de um ciclo de decisao."""

    action: str
    workers: int
    desired_workers: int
    backlog: int
    backlog_per_worker: float

    @property
    def changed(self) -> bool:
        return self.action in (SCALE_UP, SCALE_DOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "workers": self.workers,
            "desired_workers": self.desired_workers,
            "backlog": self.backlog,
            "backlog_per_worker": round(self.backlog_per_worker, 6),
        }


@dataclass
class AutoscalerState:
    """Aplica a politica ao longo do tempo, guardando a trilha de decisoes."""

    policy: AutoscalingPolicy
    workers: int = field(default=0)
    history: list[dict[str, Any]] = field(default_factory=list)
    _last_change_at: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.workers = self.policy.clamp(self.workers or self.policy.min_workers)

    def observe(self, *, backlog: int, now: float) -> AutoscalingDecision:
        elapsed = (
            float("inf") if self._last_change_at is None else now - self._last_change_at
        )
        decision = self.policy.decide(
            backlog=backlog,
            current_workers=self.workers,
            seconds_since_last_change=elapsed,
        )
        if decision.changed:
            self.workers = decision.workers
            self._last_change_at = now
        record = decision.to_dict()
        record["observed_at_offset_seconds"] = round(now, 6)
        self.history.append(record)
        return decision
