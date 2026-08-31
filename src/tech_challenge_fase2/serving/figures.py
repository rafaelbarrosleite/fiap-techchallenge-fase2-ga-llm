"""Figura de escalabilidade, no mesmo estilo visual das figuras finais."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..deliverable import PRESENTATION_FIGURE_ROOT, _chart_header, _plot_style, _save_figure  # noqa: E402
from .load_benchmark import REPORT_NAME, SCALABILITY_ROOT  # noqa: E402

FIGURE_NAME = "07_escalabilidade_automatica.png"

BLUE, BLUE_LIGHT, ORANGE = "#2563eb", "#93c5fd", "#d97706"
CHARCOAL, GREY = "#1f2937", "#9ca3af"


def generate_scalability_figure(
    *,
    artifact_root: Path = SCALABILITY_ROOT,
    figure_root: Path = PRESENTATION_FIGURE_ROOT,
) -> Path:
    """Desenha demanda, workers e o limiar em que escalar passa a compensar."""

    report: dict[str, Any] = json.loads(
        (Path(artifact_root) / REPORT_NAME).read_text(encoding="utf-8")
    )
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    _plot_style()

    scaled = next(s for s in report["scenarios"] if s["label"] == "pool_autoescalavel")
    demand = report["demand_profile"]
    cycles = np.arange(len(demand))
    sweep = report["batch_size_sweep"]

    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(10, 8.6), gridspec_kw={"height_ratios": [1.1, 1.0]}
    )

    upper.bar(cycles, demand, color=BLUE_LIGHT, edgecolor=CHARCOAL, label="Pedidos que chegam")
    workers_axis = upper.twinx()
    workers_axis.step(
        cycles, scaled["worker_timeline"], where="mid",
        color=ORANGE, linewidth=2.6, label="Workers ativos",
    )
    workers_axis.set_ylabel("Workers ativos", color=ORANGE)
    workers_axis.set_ylim(0, report["policy"]["max_workers"] + 0.6)
    workers_axis.grid(visible=False)
    workers_axis.tick_params(axis="y", colors=ORANGE)
    _chart_header(
        upper,
        "Escalabilidade automática acompanha a variação de demanda",
        f"Perfil de vale, rajada e drenagem; teto de {report['policy']['max_workers']} workers "
        f"em {report['environment']['available_cpus']} CPUs",
    )
    upper.set_ylabel("Pedidos por ciclo")
    upper.set_xlabel("Ciclo de observação")
    upper.grid(axis="x", visible=False)
    handles = upper.get_legend_handles_labels()[0] + workers_axis.get_legend_handles_labels()[0]
    labels = upper.get_legend_handles_labels()[1] + workers_axis.get_legend_handles_labels()[1]
    upper.legend(handles, labels, frameon=False, loc="upper left")

    sizes = [f"{row['batch_size']:,}".replace(",", ".") for row in sweep]
    speedups = [row["speedup"] for row in sweep]
    positions = np.arange(len(sweep))
    colors = [BLUE if value >= 1.0 else GREY for value in speedups]
    bars = lower.bar(positions, speedups, 0.55, color=colors, edgecolor=CHARCOAL)
    lower.axhline(1.0, color=ORANGE, linewidth=1.6, linestyle="--")
    lower.text(
        len(sweep) - 0.45, 1.02, "sem ganho", color=ORANGE, fontsize=10, ha="right", va="bottom"
    )
    _chart_header(
        lower,
        "Escalar réplicas só compensa acima de um custo mínimo por pedido",
        "Aceleração de 1 para o teto de workers, por tamanho de lote",
    )
    lower.set_ylabel(f"Aceleração com {report['policy']['max_workers']} workers")
    lower.set_xlabel("Registros por pedido")
    lower.set_xticks(positions, sizes)
    lower.set_ylim(0, max(speedups) * 1.25)
    lower.grid(axis="x", visible=False)
    for bar, row in zip(bars, sweep):
        lower.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(speedups) * 0.03,
            f"{row['speedup']:.2f}x\n{row['milliseconds_per_request_serial']:.1f} ms/pedido",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    path = figure_root / FIGURE_NAME
    _save_figure(fig, path)
    return path
