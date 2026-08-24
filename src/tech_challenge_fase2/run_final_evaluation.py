"""Comandos da preparacao e da execucao confirmatoria final."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .final_evaluation import (
    FINAL_ROOT,
    FIGURE_ROOT,
    prepare_final_evaluation,
    run_final_evaluation,
)


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Valida e congela o plano sem predizer o holdout.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=FINAL_ROOT)
    args = parser.parse_args()
    preflight, plan = prepare_final_evaluation(
        data_path=args.data,
        final_root=args.artifact_dir,
        run_test_suite=True,
    )
    print(
        f"Preflight aprovado: testes={preflight['test_suite']['reported_test_count']}; "
        f"candidatos={len(plan['candidates'])}; treinos unicos={len(plan['unique_training_groups'])}."
    )
    print("Nenhum fit, predict, predict_proba ou score foi executado no holdout.")


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Executa ou carrega a avaliacao final unica.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=FINAL_ROOT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()
    result = run_final_evaluation(
        data_path=args.data,
        final_root=args.artifact_dir,
        figure_root=args.figures_dir,
    )
    print(
        f"Avaliacao final disponivel: origens={result['candidate_origins']}; "
        f"treinos unicos={result['unique_training_groups']}; limiar={result['classification_threshold']}."
    )
    print("Nenhuma nova otimizacao ou selecao foi realizada.")


if __name__ == "__main__":
    run_main()
