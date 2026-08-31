"""Benchmark de demanda variavel sobre o servico de inferencia congelado.

O enunciado pede recursos de escalabilidade automatica para lidar com
variacoes de demanda. Medir isso exige demanda que varie: o perfil abaixo
alterna vale, rajada e drenagem, e cada configuracao enfrenta exatamente a
mesma sequencia de chegadas. A comparacao contra um pool fixo minimo e o que
mostra o efeito do autoscaling, em vez de apenas afirma-lo.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd

from ..config import DEFAULT_DATA_PATH
from ..data import load_dataset, split_development_test
from ..genetic.serialization import save_json, stable_sha256
from .autoscaling import AutoscalerState, AutoscalingPolicy
from .model_server import FrozenModelServer, resolve_frozen_model
from .monitoring import PerformanceMonitor, latency_summary

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCALABILITY_ROOT = PROJECT_ROOT / "artifacts" / "scalability"
REPORT_NAME = "scalability_report.json"

# Vale, rajada sustentada e drenagem: cada item e o numero de lotes que chegam
# naquele ciclo. O perfil e fixo para que execucoes sejam comparaveis.
DEFAULT_DEMAND_PROFILE: tuple[int, ...] = (
    2, 2, 3, 2,
    18, 24, 30, 28, 22,
    8, 4, 2, 1,
)
# Um lote precisa custar mais que o overhead de despachar um worker, senao a
# medicao compara ruido de agendamento em vez de trabalho util. A varredura
# em SWEEP_BATCH_SIZES localiza esse limiar em vez de assumi-lo.
DEFAULT_BATCH_SIZE = 40000
SWEEP_BATCH_SIZES: tuple[int, ...] = (2000, 8000, 20000, 40000)
SWEEP_REQUESTS = 24


@dataclass(frozen=True)
class BenchmarkOutcome:
    """Resultado de uma configuracao enfrentando o perfil de demanda."""

    label: str
    autoscaling_enabled: bool
    total_requests: int
    total_records: int
    wall_seconds: float
    throughput_requests_per_second: float
    throughput_records_per_second: float
    latency: dict[str, float]
    worker_timeline: list[int]
    max_workers_used: int
    scaling_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "autoscaling_enabled": self.autoscaling_enabled,
            "total_requests": self.total_requests,
            "total_records": self.total_records,
            "wall_seconds": round(self.wall_seconds, 6),
            "throughput_requests_per_second": round(self.throughput_requests_per_second, 6),
            "throughput_records_per_second": round(self.throughput_records_per_second, 6),
            "latency": self.latency,
            "worker_timeline": self.worker_timeline,
            "max_workers_used": self.max_workers_used,
            "scaling_events": self.scaling_events,
        }


def build_request_frame(features: Any, batch_size: int) -> Any:
    """Monta um lote de exatamente `batch_size` linhas replicando o desenvolvimento.

    A replicacao e geracao de carga, nao dado novo: nenhuma metrica de modelo,
    selecao ou conclusao do estudo deriva deste frame. Ele existe para que o
    custo por pedido seja realista, ja que o desenvolvimento tem apenas 455
    linhas e um lote desse tamanho e dominado pelo overhead de agendamento.
    """

    if batch_size < 1:
        raise ValueError("batch_size precisa ser positivo.")
    repetitions = -(-batch_size // len(features))
    tiled = pd.concat([features] * repetitions, ignore_index=True)
    return tiled.iloc[:batch_size]


def _run_profile(
    *,
    label: str,
    server: FrozenModelServer,
    request_frame: Any,
    policy: AutoscalingPolicy,
    profile: Sequence[int],
    batch_size: int,
    autoscaling_enabled: bool,
    monitor: PerformanceMonitor,
) -> BenchmarkOutcome:
    """Executa o perfil completo com uma politica de dimensionamento.

    O pool e persistente e dimensionado pelo teto da politica; a concorrencia
    efetiva e limitada por um semaforo que acompanha a decisao do autoscaler.
    Criar um pool por ciclo pagaria criacao de threads a cada rajada e mediria
    o custo do agendador, nao o do servico.
    """

    autoscaler = AutoscalerState(policy=policy)
    latencies: list[float] = []
    worker_timeline: list[int] = []
    scaling_events = 0
    total_records = 0
    permits = threading.Semaphore(policy.min_workers)
    granted = policy.min_workers

    def serve(enqueued_at: float) -> tuple[float, int]:
        """Espera por um permit livre e so entao ocupa o modelo."""

        permits.acquire()
        try:
            outcome = server.predict_batch(request_frame)
        finally:
            permits.release()
        return (perf_counter() - enqueued_at) * 1000.0, outcome.batch_size

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=policy.max_workers) as pool:
        for cycle, arrivals in enumerate(profile):
            now = perf_counter() - started
            if autoscaling_enabled:
                decision = autoscaler.observe(backlog=arrivals, now=now)
                scaling_events += int(decision.changed)
                workers = decision.workers
            else:
                workers = policy.min_workers
            # Ajusta o semaforo ao tamanho decidido sem recriar o pool.
            while granted < workers:
                permits.release()
                granted += 1
            while granted > workers:
                permits.acquire()
                granted -= 1
            worker_timeline.append(workers)

            if not arrivals:
                continue
            cycle_started = perf_counter()
            enqueued_at = perf_counter()
            futures = [pool.submit(serve, enqueued_at) for _ in range(arrivals)]
            for future in futures:
                latency_ms, served_records = future.result()
                latencies.append(latency_ms)
                total_records += served_records
            monitor.record(
                "cycle_completed",
                scenario=label,
                cycle=cycle,
                arrivals=arrivals,
                workers=workers,
                cycle_latency_ms=round((perf_counter() - cycle_started) * 1000.0, 6),
            )

    wall_seconds = perf_counter() - started
    total_requests = sum(profile)
    return BenchmarkOutcome(
        label=label,
        autoscaling_enabled=autoscaling_enabled,
        total_requests=total_requests,
        total_records=total_records,
        wall_seconds=wall_seconds,
        throughput_requests_per_second=total_requests / wall_seconds if wall_seconds else 0.0,
        throughput_records_per_second=total_records / wall_seconds if wall_seconds else 0.0,
        latency=latency_summary(latencies),
        worker_timeline=worker_timeline,
        max_workers_used=max(worker_timeline) if worker_timeline else 0,
        scaling_events=scaling_events,
    )


def sweep_batch_sizes(
    *,
    server: FrozenModelServer,
    features: Any,
    policy: AutoscalingPolicy,
    sizes: Sequence[int] = SWEEP_BATCH_SIZES,
    requests: int = SWEEP_REQUESTS,
) -> list[dict[str, Any]]:
    """Mede onde adicionar replicas comeca a compensar.

    Escalar nao e util em qualquer regime: quando o pedido custa menos que o
    despacho, mais workers apenas somam contencao. A varredura torna esse
    limiar observavel em vez de deixa-lo implicito num unico numero.
    """

    rows: list[dict[str, Any]] = []
    for size in sizes:
        frame = build_request_frame(features, size)
        measured: dict[int, float] = {}
        for workers in (1, policy.max_workers):
            started = perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda _: server.predict_batch(frame), range(requests)))
            measured[workers] = (perf_counter() - started) * 1000.0
        serial, parallel = measured[1], measured[policy.max_workers]
        rows.append(
            {
                "batch_size": size,
                "requests": requests,
                "milliseconds_per_request_serial": round(serial / requests, 6),
                "elapsed_ms_one_worker": round(serial, 6),
                "elapsed_ms_max_workers": round(parallel, 6),
                "workers_at_ceiling": policy.max_workers,
                "speedup": round(serial / parallel, 6) if parallel else 0.0,
            }
        )
    return rows


def run_load_benchmark(
    *,
    artifact_root: Path = SCALABILITY_ROOT,
    policy: AutoscalingPolicy | None = None,
    profile: Sequence[int] = DEFAULT_DEMAND_PROFILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_path: Path = DEFAULT_DATA_PATH,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Compara pool fixo minimo e pool autoescalavel sob a mesma demanda."""

    policy = policy or AutoscalingPolicy()
    reference = resolve_frozen_model()
    server = FrozenModelServer(reference)
    server.warm_up()

    X, y = load_dataset(Path(data_path))
    split = split_development_test(X, y)
    features = split.X_development

    artifact_root = Path(artifact_root)
    monitor = PerformanceMonitor(
        log_path=artifact_root / "performance_events.jsonl" if write_artifacts else None
    )

    request_frame = build_request_frame(features, batch_size)
    sweep = sweep_batch_sizes(server=server, features=features, policy=policy)
    fixed = _run_profile(
        label="pool_fixo_minimo", server=server, request_frame=request_frame, policy=policy,
        profile=profile, batch_size=batch_size, autoscaling_enabled=False, monitor=monitor,
    )
    scaled = _run_profile(
        label="pool_autoescalavel", server=server, request_frame=request_frame, policy=policy,
        profile=profile, batch_size=batch_size, autoscaling_enabled=True, monitor=monitor,
    )

    speedup = (
        fixed.latency["p95_ms"] / scaled.latency["p95_ms"]
        if scaled.latency["p95_ms"]
        else 0.0
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "scalability_report",
        "demand_profile": list(profile),
        "batch_size": batch_size,
        "request_frame_is_replicated_development": True,
        "policy": policy.to_dict(),
        "frozen_model": reference.to_dict(),
        "model_load_seconds": round(server.load_seconds or 0.0, 6),
        "scenarios": [fixed.to_dict(), scaled.to_dict()],
        "batch_size_sweep": sweep,
        "comparison": {
            "p95_latency_reduction_factor": round(speedup, 6),
            "throughput_gain_factor": round(
                scaled.throughput_records_per_second
                / fixed.throughput_records_per_second
                if fixed.throughput_records_per_second
                else 0.0,
                6,
            ),
        },
        "scope_confirmations": {
            "new_training_performed": False,
            "selection_reopened": False,
            "threshold_changed": False,
            "individual_data_logged": False,
            "network_required": False,
        },
        "environment": {
            "available_cpus": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1),
            "blas_threads_per_worker": os.environ.get("OMP_NUM_THREADS", "unpinned"),
        },
        "measurement_is_environment_dependent": True,
    }
    report["signature"] = stable_sha256(
        {key: value for key, value in report.items() if key != "signature"}
    )
    if write_artifacts:
        save_json(report, artifact_root / REPORT_NAME)
    return report
