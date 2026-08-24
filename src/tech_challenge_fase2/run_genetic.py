"""Executa um smoke test genetico somente nos 80% de desenvolvimento."""

import argparse
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_PATH, PROJECT_ROOT
from .data import file_sha256, load_dataset, split_development_test
from .genetic.config import smoke_config
from .genetic.engine import GeneticAlgorithm
from .genetic.fitness import GeneticFitnessEvaluator
from .genetic.search_spaces import SPACES, get_search_space
from .genetic.serialization import (
    ARTIFACT_SCHEMA_VERSION,
    save_json,
    stable_sha256,
    validate_run_artifact,
)
from .logging_utils import configure_logging


def run_smoke_test(
    *,
    model_name: str,
    seed: int,
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Roda busca minima sem passar o teste final ao avaliador."""

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    if model_name not in SPACES:
        raise ValueError(f"Modelo invalido: {model_name}")
    output_path = output_path or (
        PROJECT_ROOT / "artifacts" / f"ga_smoke_{model_name}_seed_{seed}.json"
    )
    log_path = log_path or (
        PROJECT_ROOT / "logs" / f"ga_smoke_{model_name}_seed_{seed}.log"
    )
    logger = configure_logging(log_path)
    logger.info("Inicio smoke GA model=%s seed=%s", model_name, seed)

    X, y = load_dataset(data_path)
    split = split_development_test(X, y)
    configuration = smoke_config(seed)
    space = get_search_space(model_name)
    evaluator = GeneticFitnessEvaluator(
        space=space,
        X_development=split.X_development,
        y_development=split.y_development,
        cv_splits=configuration.cv_splits,
        cv_seed=configuration.cv_seed,
        estimator_seed=configuration.estimator_seed,
        instability_weight=configuration.instability_weight,
    )
    result = GeneticAlgorithm(
        space=space,
        evaluator=evaluator,
        configuration=configuration,
        logger=logger,
    ).run()
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "ga_smoke_test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Uso academico; nao substitui diagnostico medico.",
        "dataset": {
            "sha256": file_sha256(Path(data_path)),
            "source_rows": len(X),
            "feature_count": X.shape[1],
            "target_mapping": {"B": 0, "M": 1},
        },
        "data_scope": {
            "scope": "development_only",
            "development_rows": len(split.X_development),
            "held_out_rows": len(split.X_test),
            "holdout_used": False,
            "cv_splits": configuration.cv_splits,
            "cv_seed": configuration.cv_seed,
            "development_indices_sha256": stable_sha256(
                sorted(int(index) for index in evaluator.development_indices)
            ),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
        },
        "run": result.to_dict(),
    }
    validate_run_artifact(artifact)
    save_json(artifact, output_path)
    logger.info(
        "Fim smoke GA model=%s best=%.6f unique=%s seconds=%.3f output=%s",
        model_name,
        result.best_individual.fitness.fitness,
        result.total_unique_evaluations,
        result.total_seconds,
        output_path,
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(SPACES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    artifact = run_smoke_test(
        model_name=args.model,
        seed=args.seed,
        data_path=args.data,
        output_path=args.output,
        log_path=args.log_file,
    )
    run = artifact["run"]
    metrics = run["best_individual"]["fitness"]
    print(f"\nSmoke GA concluido: {run['model']}")
    print(f"fitness CV: {metrics['fitness']:.6f}")
    print(f"recall maligno CV: {metrics['mean_recall_malignant']:.6f}")
    print(f"F1 maligno CV: {metrics['mean_f1_malignant']:.6f}")
    print(f"ROC-AUC CV: {metrics['mean_roc_auc']:.6f}")
    print(f"assinatura: {run['reproducibility_signature']}")
    print("O teste final nao foi usado.")


if __name__ == "__main__":
    main()

