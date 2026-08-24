"""Entradas de linha de comando para a terceira missao."""

from __future__ import annotations

import argparse
from pathlib import Path

from .comparison import (
    best_ga_by_model,
    load_official_artifacts,
    run_comparable_baselines,
    run_randomized_comparisons,
)
from .config import DEFAULT_DATA_PATH
from .genetic.official import (
    CONFIG_ORDER,
    MODEL_ORDER,
    create_execution_manifest,
    run_official_battery,
    run_official_experiment,
)
from .reporting import (
    build_comparison_summary,
    freeze_candidates,
    generate_report_figures,
)


def experiment_main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa ou retoma um experimento genetico oficial."
    )
    parser.add_argument("--model", required=True, choices=MODEL_ORDER)
    parser.add_argument("--config", required=True, choices=CONFIG_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    create_execution_manifest(data_path=args.data)
    artifact = run_official_experiment(
        model_name=args.model,
        config_label=args.config,
        seed=args.seed,
        data_path=args.data,
        artifact_root=args.artifact_root,
        force=args.force,
    )
    run = artifact["run"]
    metrics = run["best_individual"]["fitness"]
    print(
        f"Concluido {args.config}/{args.model}: fitness={metrics['fitness']:.6f}; "
        f"unicos={run['total_unique_evaluations']}; segundos={run['total_seconds']:.2f}"
    )
    print("O teste final nao foi usado.")


def battery_main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa ou retoma os nove experimentos oficiais em serie."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    manifest = create_execution_manifest(data_path=args.data)
    print(
        "Manifesto criado; estimativa baseada nos smoke tests: "
        f"{manifest['cost_estimate']['estimated_total_seconds_from_smoke'] / 60:.1f} min."
    )
    artifacts = run_official_battery(
        seed=args.seed,
        data_path=args.data,
        artifact_root=args.artifact_root,
    )
    print(f"Bateria concluida e validada: {len(artifacts)} experimentos.")
    print("O teste final nao foi usado.")


def analysis_main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara por CV, congela vencedores e gera figuras."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args()
    official = load_official_artifacts(artifact_root=args.artifact_root)
    baseline = run_comparable_baselines(data_path=args.data)
    ga_winners = best_ga_by_model(official)
    randomized = run_randomized_comparisons(
        ga_winners=ga_winners,
        data_path=args.data,
    )
    build_comparison_summary(official, baseline, randomized)
    frozen, _ = freeze_candidates(official, baseline, randomized)
    figures = generate_report_figures(
        official,
        baseline,
        randomized,
        output_dir=args.figures_dir,
    )
    winner = frozen["global_provisional_winner"]
    print(
        f"Analise concluida: vencedor global provisório={winner['model']} "
        f"({winner['origin']}); figuras={len(figures)}."
    )
    print("Nenhum modelo final foi treinado e o teste final nao foi usado.")
