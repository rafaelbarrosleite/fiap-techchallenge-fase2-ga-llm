"""Agregacao, congelamento e figuras estaticas sem dados do teste final."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import (
    MODEL_ORDER,
    best_ga_by_model,
    canonical_candidate_key,
    ga_candidate,
    select_best_candidate,
)
from .config import PROJECT_ROOT
from .genetic.serialization import save_json, stable_sha256

MODEL_LABELS = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
    "knn": "KNN",
}
CONFIG_COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
METHOD_COLORS = {
    "Baseline CV": "#7F7F7F",
    "Melhor GA": "#0072B2",
    "Busca aleatória": "#D55E00",
}


def _baseline_candidate(model_name: str, baseline: dict[str, Any]) -> dict[str, Any]:
    record = baseline["models"][model_name]
    return {
        "model": model_name,
        "origin": "baseline_cv",
        "parameters": record["parameters"],
        "metrics": record["metrics"],
        "canonical_key": canonical_candidate_key(model_name, record["parameters"]),
        "candidate_evaluations": record["candidate_evaluations"],
        "model_fits": record["model_fits"],
        "duration_seconds": record["metrics"]["evaluation_seconds"],
        "source_signature": baseline["signature"],
    }


def build_comparison_summary(
    official_artifacts: list[dict[str, Any]],
    baseline: dict[str, Any],
    randomized: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    ga_winners = best_ga_by_model(official_artifacts)
    models: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        baseline_candidate = _baseline_candidate(model_name, baseline)
        random_candidate = dict(randomized["models"][model_name]["winner"])
        random_candidate["canonical_key"] = canonical_candidate_key(
            model_name, random_candidate["parameters"]
        )
        experiments = [
            ga_candidate(artifact)
            for artifact in official_artifacts
            if artifact["run"]["model"] == model_name
        ]
        base_fitness = baseline_candidate["metrics"]["fitness"]
        ga_fitness = ga_winners[model_name]["metrics"]["fitness"]
        models[model_name] = {
            "baseline": baseline_candidate,
            "ga_experiments": experiments,
            "best_ga": ga_winners[model_name],
            "randomized_search": random_candidate,
            "ga_minus_baseline_absolute": ga_fitness - base_fitness,
            "ga_minus_baseline_relative": (
                (ga_fitness - base_fitness) / base_fitness
                if base_fitness != 0
                else None
            ),
        }
    payload = {
        "schema_version": "1.0",
        "artifact_type": "cv_methods_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": {"scope": "development_only", "holdout_used": False},
        "models": models,
    }
    payload["signature"] = stable_sha256(payload)
    save_json(
        payload,
        output_path
        or PROJECT_ROOT / "artifacts" / "comparison" / "methods_summary.json",
    )
    return payload


def freeze_candidates(
    official_artifacts: list[dict[str, Any]],
    baseline: dict[str, Any],
    randomized: dict[str, Any],
    *,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Congela vencedores por CV; nao ajusta nem avalia modelos no holdout."""

    ga_winners = best_ga_by_model(official_artifacts)
    winners: dict[str, dict[str, Any]] = {}
    considered: dict[str, list[dict[str, Any]]] = {}
    for model_name in MODEL_ORDER:
        random_candidate = dict(randomized["models"][model_name]["winner"])
        random_candidate["canonical_key"] = canonical_candidate_key(
            model_name, random_candidate["parameters"]
        )
        candidates = [
            _baseline_candidate(model_name, baseline),
            ga_winners[model_name],
            random_candidate,
        ]
        considered[model_name] = candidates
        winners[model_name] = select_best_candidate(candidates)
    global_winner = select_best_candidate(list(winners.values()))
    protocol = {
        "scope": "development_only",
        "holdout_used": False,
        "final_model_fitted": False,
        "split_seed": 42,
        "cv_seed": 42,
        "cv_splits": 5,
        "classification_threshold": 0.5,
        "fitness": "0.60*recall + 0.25*F1 + 0.15*ROC-AUC - 0.10*std(recall)",
        "selection_tolerance": 1e-12,
        "tie_breakers": [
            "fitness final",
            "recall maligno medio",
            "menor desvio do recall",
            "F1 maligno medio",
            "ROC-AUC medio",
            "menor complexidade objetiva da Random Forest",
            "chave canonica",
        ],
    }
    frozen = {
        "schema_version": "1.0",
        "artifact_type": "frozen_provisional_candidates",
        "freeze_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": baseline["dataset_sha256"],
        "protocol": protocol,
        "winners_by_model": winners,
        "global_provisional_winner": global_winner,
    }
    frozen["signature"] = stable_sha256(frozen)
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "selection_manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": baseline["dataset_sha256"],
        "protocol": protocol,
        "considered_candidates": considered,
        "source_signatures": {
            "baseline": baseline["signature"],
            "randomized_search": randomized["signature"],
            "official_ga": sorted(
                artifact["run"]["reproducibility_signature"]
                for artifact in official_artifacts
            ),
        },
        "frozen_candidates_signature": frozen["signature"],
    }
    manifest["signature"] = stable_sha256(manifest)
    save_json(
        frozen,
        output_path
        or PROJECT_ROOT / "artifacts" / "selection" / "frozen_candidates.json",
    )
    save_json(
        manifest,
        manifest_path
        or PROJECT_ROOT / "artifacts" / "selection" / "selection_manifest.json",
    )
    return frozen, manifest


def generate_report_figures(
    official_artifacts: list[dict[str, Any]],
    baseline: dict[str, Any],
    randomized: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    """Gera sete figuras com contrato visual explicito e somente metricas de CV."""

    import matplotlib.pyplot as plt

    output = Path(output_dir or PROJECT_ROOT / "reports" / "figures")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 130,
        }
    )
    by_model_config = {
        (artifact["run"]["model"], artifact["identity"]["config_label"]): artifact
        for artifact in official_artifacts
    }
    paths: list[Path] = []

    def save(fig, filename: str) -> None:
        path = output / filename
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    for field, title, ylabel, filename in (
        (
            "global_best_fitness",
            "Evolução do melhor fitness",
            "Melhor fitness acumulado",
            "01_evolucao_melhor_fitness.png",
        ),
        (
            "mean_fitness",
            "Evolução do fitness médio",
            "Fitness médio da população",
            "02_evolucao_fitness_medio.png",
        ),
        (
            "diversity_ratio",
            "Diversidade populacional",
            "Proporção de indivíduos únicos",
            "03_diversidade_populacional.png",
        ),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for axis, model_name in zip(axes, MODEL_ORDER, strict=True):
            for label in ("A", "B", "C"):
                history = by_model_config[(model_name, label)]["run"]["history"]
                axis.plot(
                    [record["generation"] for record in history],
                    [record[field] for record in history],
                    color=CONFIG_COLORS[label],
                    label=f"Configuração {label}",
                    linewidth=1.8,
                )
            axis.set_title(MODEL_LABELS[model_name])
            axis.set_xlabel("Geração")
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False)
        fig.suptitle(f"{title} — validação cruzada no desenvolvimento", y=1.03)
        save(fig, filename)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODEL_ORDER))
    offsets = (-0.16, 0.0, 0.16)
    abc_values: list[float] = []
    for offset, label in enumerate(("A", "B", "C")):
        values = [
            by_model_config[(model, label)]["run"]["best_individual"]["fitness"]["fitness"]
            for model in MODEL_ORDER
        ]
        abc_values.extend(values)
        positions = x + offsets[offset]
        axis.scatter(
            positions,
            values,
            label=f"Configuração {label}",
            color=CONFIG_COLORS[label],
            s=55,
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            axis.annotate(f"{value:.4f}", (position, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_ylabel("Fitness final")
    axis.set_title("Comparação A × B × C por modelo (CV)")
    axis.set_ylim(max(0.0, min(abc_values) - 0.006), min(1.0, max(abc_values) + 0.006))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3)
    save(fig, "04_comparacao_abc_por_modelo.png")

    ga_winners = best_ga_by_model(official_artifacts)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    methods = ("Baseline CV", "Melhor GA", "Busca aleatória")
    method_values: list[float] = []
    for offset, method in enumerate(methods):
        values = []
        for model in MODEL_ORDER:
            if method == "Baseline CV":
                metrics = baseline["models"][model]["metrics"]
            elif method == "Melhor GA":
                metrics = ga_winners[model]["metrics"]
            else:
                metrics = randomized["models"][model]["winner"]["metrics"]
            values.append(metrics["fitness"])
        method_values.extend(values)
        positions = x + offsets[offset]
        axis.scatter(
            positions,
            values,
            label=method,
            color=METHOD_COLORS[method],
            s=55,
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            axis.annotate(f"{value:.4f}", (position, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_ylabel("Fitness final")
    axis.set_title("Baseline CV × GA × RandomizedSearchCV")
    axis.set_ylim(max(0.0, min(method_values) - 0.006), min(1.0, max(method_values) + 0.006))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3)
    save(fig, "05_comparacao_metodos.png")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for offset, method in enumerate(methods):
        means, errors = [], []
        for model in MODEL_ORDER:
            if method == "Baseline CV":
                metrics = baseline["models"][model]["metrics"]
            elif method == "Melhor GA":
                metrics = ga_winners[model]["metrics"]
            else:
                metrics = randomized["models"][model]["winner"]["metrics"]
            means.append(metrics["mean_recall_malignant"])
            errors.append(metrics["std_recall_malignant"])
        axis.errorbar(
            x + (offset - 1) * 0.12,
            means,
            yerr=errors,
            fmt="o",
            capsize=4,
            label=method,
            color=METHOD_COLORS[method],
        )
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_ylabel("Recall maligno médio ± desvio-padrão")
    axis.set_title("Recall maligno e variabilidade entre as cinco dobras")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3)
    save(fig, "06_recall_com_variabilidade.png")

    labels = [
        f"{artifact['identity']['config_label']}-{MODEL_LABELS[artifact['run']['model']]}"
        for artifact in official_artifacts
    ]
    evaluations = [artifact["run"]["total_unique_evaluations"] for artifact in official_artifacts]
    durations = [artifact["run"]["total_seconds"] / 60 for artifact in official_artifacts]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = [CONFIG_COLORS[artifact["identity"]["config_label"]] for artifact in official_artifacts]
    axes[0].bar(np.arange(9), evaluations, color=colors)
    axes[0].set_ylabel("Avaliações únicas")
    axes[0].set_title("Custo dos nove experimentos oficiais")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(np.arange(9), durations, color=colors)
    axes[1].set_ylabel("Duração (minutos)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(np.arange(9), labels, rotation=35, ha="right")
    save(fig, "07_avaliacoes_e_duracao.png")

    if len(paths) != 7 or any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise RuntimeError("Falha ao produzir as sete figuras esperadas.")
    return paths
