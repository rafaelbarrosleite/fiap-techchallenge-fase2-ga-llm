"""Camada de servico escalavel sobre o modelo congelado da avaliacao final."""

from .autoscaling import AutoscalerState, AutoscalingDecision, AutoscalingPolicy
from .model_server import FrozenModelServer, resolve_frozen_model
from .monitoring import PerformanceMonitor, latency_summary

__all__ = [
    "AutoscalerState",
    "AutoscalingDecision",
    "AutoscalingPolicy",
    "FrozenModelServer",
    "PerformanceMonitor",
    "latency_summary",
    "resolve_frozen_model",
]
