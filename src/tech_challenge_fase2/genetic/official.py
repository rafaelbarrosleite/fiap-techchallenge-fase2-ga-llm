"""Execucao oficial, retomavel e auditavel dos nove experimentos geneticos."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from .. import __version__
from ..config import DEFAULT_DATA_PATH, PROJECT_ROOT
from ..data import file_sha256, load_dataset, split_development_test
from ..logging_utils import configure_logging
from .config import EXPERIMENT_CONFIGS, GAConfig
from .engine import GeneticAlgorithm
from .fitness import GeneticFitnessEvaluator
from .search_spaces import SPACES, get_search_space
from .serialization import save_json, stable_sha256

OFFICIAL_SCHEMA_VERSION = "1.0"
OFFICIAL_SEED = 42
CONFIG_ALIASES = {
    "A": "A_small",
    "B": "B_balanced",
    "C": "C_exploratory",
}
MODEL_ORDER = ("logistic_regression", "random_forest", "knn")
CONFIG_ORDER = ("A", "B", "C")
RELEVANT_CODE_FILES = (
    "genetic/config.py",
    "genetic/genomes.py",
    "genetic/search_spaces.py",
    "genetic/fitness.py",
    "genetic/operators.py",
    "genetic/history.py",
    "genetic/engine.py",
    "genetic/serialization.py",
    "genetic/official.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def official_configuration(label: str, *, seed: int = OFFICIAL_SEED) -> GAConfig:
    try:
        return EXPERIMENT_CONFIGS[CONFIG_ALIASES[label.upper()]].with_seed(seed)
    except KeyError as error:
        raise ValueError(f"Configuracao oficial desconhecida: {label}") from error


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "project": __version__,
    }


def relevant_code_signature(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    source_root = Path(project_root) / "src" / "tech_challenge_fase2"
    files = {
        relative: file_sha256(source_root / relative)
        for relative in RELEVANT_CODE_FILES
    }
    return {"sha256": stable_sha256(files), "files": files}


def experiment_identity(
    *,
    model_name: str,
    config_label: str,
    configuration: GAConfig,
    dataset_sha256: str,
    development_indices_sha256: str,
    code_sha256: str,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "config_label": config_label,
        "configuration": configuration.to_dict(),
        "dataset_sha256": dataset_sha256,
        "development_indices_sha256": development_indices_sha256,
        "code_sha256": code_sha256,
        "project_version": __version__,
        "official_schema_version": OFFICIAL_SCHEMA_VERSION,
    }
    return {**payload, "sha256": stable_sha256(payload)}


def experiment_paths(
    model_name: str,
    config_label: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Path]:
    root = Path(artifact_root or PROJECT_ROOT / "artifacts" / "official")
    stem = f"ga_{config_label.lower()}_{model_name}"
    return {
        "artifact": root / "experiments" / f"{stem}.json",
        "status": root / "status" / f"{stem}.json",
        "checkpoint": root / "checkpoints" / f"{stem}.json",
        "log": PROJECT_ROOT / "logs" / "official" / f"{stem}.log",
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_official_artifact(
    payload: dict[str, Any],
    *,
    expected_identity_sha256: str | None = None,
) -> None:
    required = {
        "schema_version",
        "artifact_type",
        "generated_at_utc",
        "identity",
        "dataset",
        "data_scope",
        "software_versions",
        "source_signature",
        "run",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Campos ausentes no artefato oficial: {sorted(missing)}")
    if payload["schema_version"] != OFFICIAL_SCHEMA_VERSION:
        raise ValueError("Versao de schema oficial nao suportada.")
    if payload["artifact_type"] != "official_ga_experiment":
        raise ValueError("Tipo de artefato oficial inesperado.")
    identity = payload["identity"]
    identity_payload = {key: value for key, value in identity.items() if key != "sha256"}
    if identity.get("sha256") != stable_sha256(identity_payload):
        raise ValueError("Assinatura de identidade inconsistente.")
    if expected_identity_sha256 and identity["sha256"] != expected_identity_sha256:
        raise ValueError("Identidade do artefato nao coincide com a execucao solicitada.")
    scope = payload["data_scope"]
    if scope.get("scope") != "development_only" or scope.get("holdout_used") is not False:
        raise ValueError("O artefato oficial deve usar somente desenvolvimento.")
    if scope.get("cv_splits") != 5 or scope.get("classification_threshold") != 0.5:
        raise ValueError("Protocolo de CV ou limiar divergente.")
    run = payload["run"]
    required_run = {
        "model",
        "configuration",
        "best_individual",
        "history",
        "stop_reason",
        "total_seconds",
        "total_candidate_requests",
        "total_unique_evaluations",
        "cache_hits",
        "total_model_fits",
        "failure_count",
        "issue_count",
        "invalid_before_repair",
        "repaired_individuals",
        "reproducibility_signature",
    }
    missing_run = required_run.difference(run)
    if missing_run:
        raise ValueError(f"Campos ausentes na execucao oficial: {sorted(missing_run)}")
    if run["model"] not in SPACES:
        raise ValueError("Modelo oficial invalido.")
    history = run["history"]
    if not history or history[0]["generation"] != 0:
        raise ValueError("Historico oficial incompleto.")
    if history[-1]["generation"] + 1 != len(history):
        raise ValueError("Geracoes do historico nao sao contiguas.")
    planned = run["configuration"]["max_generations"]
    if run["stop_reason"] == "max_generations" and history[-1]["generation"] != planned:
        raise ValueError("Historico nao contem todas as geracoes planejadas.")
    metrics = run["best_individual"]["fitness"]
    if metrics["failure"] is not None or len(metrics["fold_metrics"]) != 5:
        raise ValueError("Melhor individuo oficial falhou ou nao possui cinco dobras.")
    if not -1.0 <= metrics["fitness"] <= 1.0:
        raise ValueError("Fitness oficial fora dos limites.")
    genome = run["best_individual"]["genome"]
    from .serialization import genome_from_dict

    SPACES[run["model"]].require_valid(genome_from_dict(genome))
    if run["cache_hits"] != (
        run["total_candidate_requests"] - run["total_unique_evaluations"]
    ):
        raise ValueError("Contagem de cache hits inconsistente.")
    if not isinstance(run["reproducibility_signature"], str) or len(
        run["reproducibility_signature"]
    ) != 64:
        raise ValueError("Assinatura de reprodutibilidade invalida.")


def completed_artifact_matches(path: Path, identity_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_json(path)
        validate_official_artifact(
            payload, expected_identity_sha256=identity_sha256
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def completed_experiment_matches(
    artifact_path: Path,
    status_path: Path,
    identity_sha256: str,
) -> bool:
    if not completed_artifact_matches(artifact_path, identity_sha256):
        return False
    if not status_path.is_file():
        return False
    try:
        status = _load_json(status_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        status.get("artifact_type") == "official_ga_status"
        and status.get("status") == "completed"
        and status.get("identity", {}).get("sha256") == identity_sha256
    )


def _status_payload(
    *,
    status: str,
    identity: dict[str, Any],
    started_at_utc: str,
    completed_generation: int | None = None,
    ended_at_utc: str | None = None,
    duration_seconds: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if status not in {"pending", "running", "completed", "failed"}:
        raise ValueError(f"Status invalido: {status}")
    return {
        "schema_version": "1.0",
        "artifact_type": "official_ga_status",
        "status": status,
        "identity": identity,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now(),
        "ended_at_utc": ended_at_utc,
        "duration_seconds": duration_seconds,
        "completed_generation": completed_generation,
        "reason": reason,
    }


def run_official_experiment(
    *,
    model_name: str,
    config_label: str,
    seed: int = OFFICIAL_SEED,
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_root: Path | None = None,
    force: bool = False,
    configuration_override: GAConfig | None = None,
) -> dict[str, Any]:
    """Executa ou retoma um experimento; nunca fornece o holdout ao fitness."""

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    config_label = config_label.upper()
    if model_name not in SPACES:
        raise ValueError(f"Modelo invalido: {model_name}")
    configuration = configuration_override or official_configuration(
        config_label, seed=seed
    )
    paths = experiment_paths(model_name, config_label, artifact_root=artifact_root)
    logger = configure_logging(paths["log"])
    X, y = load_dataset(data_path)
    split = split_development_test(X, y)
    development_indices_sha256 = stable_sha256(
        sorted(int(index) for index in split.X_development.index)
    )
    source_signature = relevant_code_signature()
    identity = experiment_identity(
        model_name=model_name,
        config_label=config_label,
        configuration=configuration,
        dataset_sha256=file_sha256(Path(data_path)),
        development_indices_sha256=development_indices_sha256,
        code_sha256=source_signature["sha256"],
    )
    if not force and completed_experiment_matches(
        paths["artifact"], paths["status"], identity["sha256"]
    ):
        logger.info("Experimento oficial valido ja concluido; pulando %s %s", config_label, model_name)
        return _load_json(paths["artifact"])

    resume_state: dict[str, Any] | None = None
    if not force and paths["checkpoint"].is_file():
        try:
            checkpoint = _load_json(paths["checkpoint"])
            if checkpoint.get("identity_sha256") == identity["sha256"]:
                resume_state = checkpoint["state"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            resume_state = None

    started_at = utc_now()
    wall_started = perf_counter()
    initial_generation = (
        int(resume_state["completed_generation"]) if resume_state is not None else None
    )
    save_json(
        _status_payload(
            status="running",
            identity=identity,
            started_at_utc=started_at,
            completed_generation=initial_generation,
            reason="resumed_from_checkpoint" if resume_state else "started",
        ),
        paths["status"],
    )
    logger.info(
        "Inicio experimento oficial config=%s model=%s seed=%s resume_generation=%s",
        config_label,
        model_name,
        seed,
        initial_generation,
    )
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

    def persist_checkpoint(state: dict[str, Any]) -> None:
        save_json(
            {
                "schema_version": "1.0",
                "artifact_type": "official_ga_checkpoint",
                "identity_sha256": identity["sha256"],
                "saved_at_utc": utc_now(),
                "state": state,
            },
            paths["checkpoint"],
        )
        save_json(
            _status_payload(
                status="running",
                identity=identity,
                started_at_utc=started_at,
                completed_generation=state["completed_generation"],
                duration_seconds=state["elapsed_seconds"],
                reason="checkpoint_saved",
            ),
            paths["status"],
        )

    try:
        result = GeneticAlgorithm(
            space=space,
            evaluator=evaluator,
            configuration=configuration,
            logger=logger,
        ).run(
            resume_state=resume_state,
            checkpoint_callback=persist_checkpoint,
        )
        artifact = {
            "schema_version": OFFICIAL_SCHEMA_VERSION,
            "artifact_type": "official_ga_experiment",
            "generated_at_utc": utc_now(),
            "disclaimer": "Uso academico; nao substitui diagnostico medico.",
            "identity": identity,
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
                "holdout_accessible_to_fitness": False,
                "cv_splits": configuration.cv_splits,
                "cv_seed": configuration.cv_seed,
                "split_seed": 42,
                "classification_threshold": 0.5,
                "development_indices_sha256": development_indices_sha256,
            },
            "software_versions": software_versions(),
            "source_signature": source_signature,
            "run": result.to_dict(),
        }
        validate_official_artifact(
            artifact, expected_identity_sha256=identity["sha256"]
        )
        save_json(artifact, paths["artifact"])
        save_json(
            _status_payload(
                status="completed",
                identity=identity,
                started_at_utc=started_at,
                ended_at_utc=utc_now(),
                duration_seconds=result.total_seconds,
                completed_generation=result.history[-1].generation,
                reason=result.stop_reason,
            ),
            paths["status"],
        )
        logger.info(
            "Fim experimento oficial config=%s model=%s fitness=%.6f unique=%s seconds=%.3f",
            config_label,
            model_name,
            result.best_individual.fitness.fitness,
            result.total_unique_evaluations,
            result.total_seconds,
        )
        return artifact
    except BaseException as error:
        save_json(
            _status_payload(
                status="failed",
                identity=identity,
                started_at_utc=started_at,
                ended_at_utc=utc_now(),
                duration_seconds=perf_counter() - wall_started,
                reason=f"{type(error).__name__}: {error}",
            ),
            paths["status"],
        )
        logger.exception("Falha no experimento oficial %s %s", config_label, model_name)
        raise


def create_execution_manifest(
    *,
    output_path: Path | None = None,
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    X, y = load_dataset(data_path)
    split = split_development_test(X, y)
    configs = {label: official_configuration(label).to_dict() for label in CONFIG_ORDER}
    maximum_candidates = sum(
        official_configuration(label).maximum_candidate_evaluations
        for label in CONFIG_ORDER
    )
    maximum_fits = maximum_candidates * 5 * len(MODEL_ORDER)
    smoke_seconds_per_candidate = {
        "logistic_regression": 0.2389595000004192 / 9,
        "random_forest": 14.679210375000366 / 9,
        "knn": 0.15582966599959036 / 9,
    }
    estimated_by_model = {
        model: maximum_candidates * seconds
        for model, seconds in smoke_seconds_per_candidate.items()
    }
    source_signature = relevant_code_signature()
    payload = {
        "schema_version": "1.0",
        "artifact_type": "official_execution_manifest",
        "generated_at_utc": utc_now(),
        "protocol_locked": True,
        "models": list(MODEL_ORDER),
        "configuration_order": list(CONFIG_ORDER),
        "configurations": configs,
        "seeds": {"split": 42, "cv": 42, "ga": 42, "estimator": 42},
        "fitness": {
            "recall_weight": 0.60,
            "f1_weight": 0.25,
            "roc_auc_weight": 0.15,
            "recall_instability_weight": 0.10,
        },
        "dataset": {
            "sha256": file_sha256(Path(data_path)),
            "source_rows": len(X),
            "development_rows": len(split.X_development),
            "held_out_rows": len(split.X_test),
            "development_indices_sha256": stable_sha256(
                sorted(int(index) for index in split.X_development.index)
            ),
        },
        "data_scope": {
            "scope": "development_only",
            "holdout_used": False,
            "holdout_accessible_to_fitness": False,
            "cv_splits": 5,
            "classification_threshold": 0.5,
        },
        "cost_estimate": {
            "maximum_candidate_requests_per_model": maximum_candidates,
            "maximum_model_fits_all_nine": maximum_fits,
            "estimated_seconds_by_model_from_smoke": estimated_by_model,
            "estimated_total_seconds_from_smoke": sum(estimated_by_model.values()),
            "execution_policy": "serial_n_jobs_1",
        },
        "software_versions": software_versions(),
        "source_signature": source_signature,
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    path = output_path or PROJECT_ROOT / "artifacts" / "official" / "execution_manifest.json"
    save_json(payload, path)
    return payload


def run_official_battery(
    *,
    seed: int = OFFICIAL_SEED,
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Executa em serie A, B e C; interrompe se qualquer validacao falhar."""

    results: list[dict[str, Any]] = []
    for label in CONFIG_ORDER:
        for model_name in MODEL_ORDER:
            artifact = run_official_experiment(
                model_name=model_name,
                config_label=label,
                seed=seed,
                data_path=data_path,
                artifact_root=artifact_root,
            )
            validate_official_artifact(artifact)
            results.append(artifact)
    return results
