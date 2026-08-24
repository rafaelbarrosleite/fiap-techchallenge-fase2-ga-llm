"""Figuras estaticas da avaliacao final, geradas apenas apos a execucao unica."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT

MODEL_ORDER = ("logistic_regression", "random_forest", "knn")
MODEL_LABELS = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
    "knn": "KNN",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "ga": "GA",
    "random_search": "Busca aleatória",
}
COLORS = {"baseline": "#6B7280", "ga": "#0072B2", "random_search": "#D55E00"}


def generate_final_figures(
    results: dict[str, Any],
    predictions: dict[str, Any],
    comparisons: dict[str, Any],
    uncertainty: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    """Gera seis figuras com escala honesta, paleta acessivel e n explicito."""

    del predictions, comparisons  # resultados agregados bastam para os seis contratos.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir or PROJECT_ROOT / "reports" / "figures" / "final_evaluation")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    records = {record["candidate_id"]: record for record in results["candidate_results"]}
    paths: list[Path] = []

    def save(fig, filename: str) -> None:
        path = output / filename
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Figura vazia: {filename}")
        paths.append(path)

    # 1. Seis matrizes, baseline e GA lado a lado para as tres familias.
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for column, model in enumerate(MODEL_ORDER):
        for row, method in enumerate(("baseline", "ga")):
            matrix = np.asarray(records[f"{model}__{method}"]["metrics"]["confusion_matrix"])
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=72)
            for y in range(2):
                for x in range(2):
                    axis.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if matrix[y, x] > 36 else "#111827", fontsize=12)
            axis.set_xticks((0, 1), ("Benigno", "Maligno"))
            axis.set_yticks((0, 1), ("Benigno", "Maligno"))
            axis.set_xlabel("Predito")
            axis.set_ylabel("Real")
            axis.set_title(f"{MODEL_LABELS[model]} — {METHOD_LABELS[method]}")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Quantidade")
    fig.suptitle("Matrizes de confusão no teste final — baseline × GA (n=114)", y=1.01)
    save(fig, "01_matrizes_confusao_baseline_ga.png")

    # 2. Metricas em escala integral para nao exagerar diferencas pequenas.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    metric_specs = (
        ("recall_malignant", "Recall maligno"),
        ("f1_malignant", "F1 maligno"),
        ("roc_auc", "ROC-AUC"),
    )
    x = np.arange(3)
    for axis, (metric, label) in zip(axes, metric_specs, strict=True):
        for offset, method in enumerate(("baseline", "ga", "random_search")):
            values = [records[f"{model}__{method}"]["metrics"][metric] for model in MODEL_ORDER]
            positions = x + (offset - 1) * 0.13
            axis.scatter(positions, values, s=55, color=COLORS[method], label=METHOD_LABELS[method], zorder=3)
            for position, value in zip(positions, values, strict=True):
                axis.annotate(f"{value:.3f}", (position, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        axis.set_xticks(x, [MODEL_LABELS[model].replace(" ", "\n") for model in MODEL_ORDER])
        axis.set_ylim(0, 1.05)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Valor no teste")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Métricas dos candidatos predeterminados no teste final (n=114)", y=1.02)
    save(fig, "02_metricas_teste_final.png")

    # 3. Falsos negativos em barras iniciadas em zero.
    fig, axis = plt.subplots(figsize=(9, 4.4))
    width = 0.24
    for offset, method in enumerate(("baseline", "ga", "random_search")):
        values = [records[f"{model}__{method}"]["metrics"]["false_negatives"] for model in MODEL_ORDER]
        bars = axis.bar(x + (offset - 1) * width, values, width, color=COLORS[method], label=METHOD_LABELS[method])
        axis.bar_label(bars, padding=3)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_ylim(0, max(6, axis.get_ylim()[1]))
    axis.set_ylabel("Falsos negativos malignos")
    axis.set_title("Falsos negativos no teste final (42 casos malignos)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3)
    save(fig, "03_falsos_negativos.png")

    # 4. ROC por familia; origem compartilhada continua indicada no resultado.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        for method in ("baseline", "ga", "random_search"):
            record = records[f"{model}__{method}"]
            curve = record["roc_curve"]
            axis.plot(curve["false_positive_rate"], curve["true_positive_rate"], color=COLORS[method], linewidth=1.8, label=f"{METHOD_LABELS[method]} (AUC={record['metrics']['roc_auc']:.3f})")
        axis.plot((0, 1), (0, 1), linestyle="--", color="#9CA3AF", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Taxa de falsos positivos")
        axis.set_ylabel("Taxa de verdadeiros positivos")
        axis.set_title(MODEL_LABELS[model])
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="lower right")
    fig.suptitle("Curvas ROC dos candidatos predeterminados — teste final (n=114)", y=1.02)
    save(fig, "04_curvas_roc.png")

    # 5. CV e teste sao mostrados como estimativas distintas do recall.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        for method in ("baseline", "ga", "random_search"):
            record = records[f"{model}__{method}"]
            cv = record["cv_metrics"]["mean_recall_malignant"]
            test = record["metrics"]["recall_malignant"]
            axis.plot((0, 1), (cv, test), marker="o", color=COLORS[method], label=METHOD_LABELS[method])
        axis.set_xticks((0, 1), ("CV\n(desenvolvimento)", "Teste\nfinal"))
        axis.set_xlim(-0.2, 1.2)
        axis.set_ylim(0, 1.05)
        axis.set_title(MODEL_LABELS[model])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recall maligno")
    axes[-1].legend(frameon=False)
    fig.suptitle("Recall em validação cruzada e no teste — amostras distintas", y=1.02)
    save(fig, "05_recall_cv_vs_teste.png")

    # 6. Forest plot de Wilson para recall.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        for position, method in enumerate(("baseline", "ga", "random_search")):
            interval = uncertainty["candidate_intervals"][f"{model}__{method}"]["recall_malignant"]
            estimate = interval["estimate"]
            axis.errorbar(
                estimate,
                position,
                xerr=[[estimate - interval["lower"]], [interval["upper"] - estimate]],
                fmt="o",
                color=COLORS[method],
                capsize=4,
            )
            axis.annotate(f"{estimate:.3f}", (estimate, position), xytext=(5, 5), textcoords="offset points", fontsize=7)
        axis.set_yticks(range(3), [METHOD_LABELS[m] for m in ("baseline", "ga", "random_search")])
        axis.set_xlim(0, 1.02)
        axis.set_xlabel("Recall e IC95% de Wilson")
        axis.set_title(MODEL_LABELS[model])
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Incerteza do recall maligno no teste final (42 casos malignos)", y=1.02)
    save(fig, "06_intervalos_recall.png")

    if len(paths) != 6:
        raise RuntimeError("Eram esperadas exatamente seis figuras finais.")
    return paths
