"""Executa a auditoria numerica da Fase 1 e o baseline corrigido."""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_RESULTS_PATH,
    RANDOM_STATE,
)
from .data import (
    file_sha256,
    load_dataset,
    reproduce_phase1_split,
    split_development_test,
)
from .evaluation import evaluate_classifier
from .logging_utils import configure_logging
from .models import build_models


def _class_counts(values: Any) -> dict[str, int]:
    counts = values.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _fit_and_evaluate(model: Any, X_train: Any, y_train: Any, X_eval: Any, y_eval: Any) -> dict[str, Any]:
    started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    metrics = evaluate_classifier(model, X_eval, y_eval)
    predict_seconds = perf_counter() - started
    return {
        **metrics,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": predict_seconds,
        "parameters": model.get_params(deep=True),
    }


def _phase1_reproduction(X: Any, y: Any, logger: Any) -> dict[str, Any]:
    split = reproduce_phase1_split(X, y)
    validation_results: dict[str, Any] = {}
    trained_models: dict[str, Any] = {}
    for name, model in build_models().items():
        logger.info("Reproduzindo Fase 1 na validacao: %s", name)
        validation_results[name] = _fit_and_evaluate(
            model,
            split.X_train,
            split.y_train,
            split.X_validation,
            split.y_validation,
        )
        trained_models[name] = model

    # Reproduz o notebook: o modelo escolhido permanece treinado somente nos 60%.
    logistic_test = evaluate_classifier(
        trained_models["logistic_regression"], split.X_test, split.y_test
    )
    return {
        "protocol": "60% treino, 20% validacao, 20% teste; sem reajuste final",
        "split": {
            "train_rows": len(split.X_train),
            "validation_rows": len(split.X_validation),
            "test_rows": len(split.X_test),
            "train_class_counts": _class_counts(split.y_train),
            "validation_class_counts": _class_counts(split.y_validation),
            "test_class_counts": _class_counts(split.y_test),
        },
        "validation_models": validation_results,
        "selected_logistic_regression_test": logistic_test,
    }


def _corrected_baseline(X: Any, y: Any, logger: Any) -> dict[str, Any]:
    split = split_development_test(X, y)
    results: dict[str, Any] = {}
    for name, model in build_models().items():
        logger.info("Executando baseline corrigido no teste final: %s", name)
        results[name] = _fit_and_evaluate(
            model,
            split.X_development,
            split.y_development,
            split.X_test,
            split.y_test,
        )
    return {
        "protocol": "80% desenvolvimento e 20% teste final estratificado",
        "split": {
            "development_rows": len(split.X_development),
            "test_rows": len(split.X_test),
            "development_class_counts": _class_counts(split.y_development),
            "test_class_counts": _class_counts(split.y_test),
        },
        "models": results,
    }


def run_baseline(data_path: Path, results_path: Path, log_path: Path) -> dict[str, Any]:
    """Executa os dois protocolos e persiste apenas agregados e metadados."""

    # Os modelos usam um unico processo. Isto tambem evita que ambientes restritos
    # consultem o hardware apenas para estimar paralelismo do joblib.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    logger = configure_logging(log_path)
    run_started = perf_counter()
    logger.info("Inicio do baseline com random_state=%s", RANDOM_STATE)
    X, y = load_dataset(data_path)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Uso academico; nao substitui diagnostico medico.",
        "random_state": RANDOM_STATE,
        "dataset": {
            "path": str(Path(data_path).resolve()),
            "sha256": file_sha256(Path(data_path)),
            "rows": len(X),
            "feature_count": X.shape[1],
            "target_mapping": {"B": 0, "M": 1},
            "class_counts": _class_counts(y),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
        },
        "phase1_reproduction": _phase1_reproduction(X, y, logger),
        "corrected_baseline": _corrected_baseline(X, y, logger),
    }
    result["total_seconds"] = perf_counter() - run_started
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Resultados gravados em %s", results_path)
    logger.info("Baseline concluido em %.3f segundos", result["total_seconds"])
    return result


def _print_summary(result: dict[str, Any]) -> None:
    print("\nBaseline corrigido - teste final")
    print("modelo                 accuracy precision recall   f1       roc_auc  FN")
    for name, metrics in result["corrected_baseline"]["models"].items():
        print(
            f"{name:22} "
            f"{metrics['accuracy']:.6f} "
            f"{metrics['precision_malignant']:.6f} "
            f"{metrics['recall_malignant']:.6f} "
            f"{metrics['f1_malignant']:.6f} "
            f"{metrics['roc_auc']:.6f} "
            f"{metrics['false_negatives_malignant']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()
    _print_summary(run_baseline(args.data, args.output, args.log_file))


if __name__ == "__main__":
    main()
