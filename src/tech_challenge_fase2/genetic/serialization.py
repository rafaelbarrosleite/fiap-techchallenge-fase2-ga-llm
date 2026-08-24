"""Serializacao estavel de genomas e artefatos do algoritmo genetico."""

import json
import os
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from .genomes import Genome, KNNGenome, LogisticRegressionGenome, RandomForestGenome

ARTIFACT_SCHEMA_VERSION = "1.0"


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    return {"model": genome.model_name, **asdict(genome)}


def genome_from_dict(payload: dict[str, Any]) -> Genome:
    model = payload.get("model")
    values = {key: value for key, value in payload.items() if key != "model"}
    if model == LogisticRegressionGenome.model_name:
        return LogisticRegressionGenome(**values)
    if model == RandomForestGenome.model_name:
        return RandomForestGenome(**values)
    if model == KNNGenome.model_name:
        return KNNGenome(**values)
    raise ValueError(f"Modelo ausente ou desconhecido no genoma: {model}")


def genome_key(genome: Genome) -> str:
    return json.dumps(genome_to_dict(genome), sort_keys=True, separators=(",", ":"))


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Escreve JSON de forma atomica no mesmo sistema de arquivos."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_run_artifact(payload: dict[str, Any]) -> None:
    """Valida o schema minimo sem depender de biblioteca externa."""

    required_top_level = {
        "schema_version",
        "artifact_type",
        "generated_at_utc",
        "dataset",
        "data_scope",
        "software_versions",
        "run",
    }
    missing = required_top_level.difference(payload)
    if missing:
        raise ValueError(f"Campos ausentes no artefato: {sorted(missing)}")
    if payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Versao de schema nao suportada.")
    if payload["artifact_type"] != "ga_smoke_test":
        raise ValueError("Tipo de artefato inesperado.")
    scope = payload["data_scope"]
    if scope.get("holdout_used") is not False:
        raise ValueError("O artefato genetico nao pode usar o teste final.")
    if scope.get("cv_splits") != 5:
        raise ValueError("O artefato deve registrar cinco dobras de CV.")
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
        "total_model_fits",
        "unique_individuals",
        "failure_count",
        "issue_count",
        "reproducibility_signature",
    }
    missing_run = required_run.difference(run)
    if missing_run:
        raise ValueError(f"Campos ausentes em run: {sorted(missing_run)}")
    if run["model"] not in {"logistic_regression", "random_forest", "knn"}:
        raise ValueError("Modelo invalido no artefato.")
    if not run["history"]:
        raise ValueError("O historico nao pode ser vazio.")
    best = run["best_individual"]
    if set(best) != {"genome", "fitness"}:
        raise ValueError("Schema do melhor individuo e inconsistente.")
    required_fitness = {
        "fitness",
        "base_fitness",
        "mean_recall_malignant",
        "std_recall_malignant",
        "mean_f1_malignant",
        "mean_roc_auc",
        "evaluation_seconds",
        "fold_metrics",
        "issues",
        "failure",
    }
    missing_fitness = required_fitness.difference(best["fitness"])
    if missing_fitness:
        raise ValueError(
            f"Campos ausentes nas metricas: {sorted(missing_fitness)}"
        )
    genome = genome_from_dict(best["genome"])
    if genome.model_name != run["model"]:
        raise ValueError("Modelo do genoma difere do modelo da execucao.")
    from .search_spaces import get_search_space

    get_search_space(run["model"]).require_valid(genome)
    if best["fitness"]["failure"] is None and len(
        best["fitness"]["fold_metrics"]
    ) != 5:
        raise ValueError("Um melhor individuo valido deve registrar cinco dobras.")
    signature = run["reproducibility_signature"]
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("Assinatura de reprodutibilidade invalida.")
