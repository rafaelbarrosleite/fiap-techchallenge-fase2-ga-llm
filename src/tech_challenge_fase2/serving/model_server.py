"""Servico de inferencia sobre o pipeline congelado da avaliacao final.

O servidor nao treina, nao reabre selecao e nao altera o threshold. Ele carrega
uma unica vez o pipeline ja assinado, confere o hash contra o manifesto final e
responde a lotes de predicao. Carregar uma vez e o que torna o servico
escalavel: o custo por pedido passa a ser somente a predicao.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from ..genetic.serialization import stable_sha256
from ..llm.input_builder import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_ROOT = PROJECT_ROOT / "artifacts" / "final_evaluation"
PLAN_PATH = FINAL_ROOT / "final_evaluation_plan.json"
MANIFEST_PATH = FINAL_ROOT / "final_manifest.json"

# Vencedor global congelado antes do holdout; o servico nao reabre essa escolha.
SELECTED_CANDIDATE = "logistic_regression__random_search"
DECISION_THRESHOLD = 0.5


class ModelServerError(RuntimeError):
    """O modelo congelado nao pode ser servido com integridade."""


def _load_signed(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("artifact_type") != artifact_type or signature != stable_sha256(unsigned):
        raise ModelServerError(f"Artefato assinado invalido: {Path(path).name}.")
    return payload


@dataclass(frozen=True)
class FrozenModelReference:
    """Identidade auditavel do pipeline servido."""

    candidate_id: str
    training_group_id: str
    relative_path: str
    sha256: str
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "training_group_id": self.training_group_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "threshold": self.threshold,
        }


def resolve_frozen_model(candidate_id: str = SELECTED_CANDIDATE) -> FrozenModelReference:
    """Localiza o pipeline congelado e confirma o hash contra o manifesto."""

    plan = _load_signed(PLAN_PATH, "final_evaluation_plan")
    manifest = _load_signed(MANIFEST_PATH, "final_evaluation_manifest")
    candidate = next(
        (item for item in plan["candidates"] if item["candidate_id"] == candidate_id),
        None,
    )
    if candidate is None:
        raise ModelServerError(f"Candidato congelado ausente no plano: {candidate_id}.")
    group = candidate["training_group_id"]
    relative_path = f"artifacts/final_evaluation/models/pipeline_{group}.joblib"
    record = next(
        (item for item in manifest["files"] if item["relative_path"] == relative_path),
        None,
    )
    if record is None:
        raise ModelServerError("Modelo congelado nao consta do manifesto final.")
    model_path = PROJECT_ROOT / relative_path
    if not model_path.is_file():
        raise ModelServerError(f"Modelo congelado ausente em disco: {relative_path}.")
    if file_sha256(model_path) != record["sha256"]:
        raise ModelServerError("Hash do modelo congelado divergiu do manifesto.")
    return FrozenModelReference(
        candidate_id=candidate_id,
        training_group_id=group,
        relative_path=relative_path,
        sha256=record["sha256"],
        threshold=DECISION_THRESHOLD,
    )


class FrozenModelServer:
    """Carrega o pipeline uma vez e serve lotes de forma thread-safe.

    `predict_proba` do scikit-learn e reentrante para um estimador ja ajustado,
    entao os lotes correm em paralelo sem lock. O lock protege apenas a carga
    preguicosa, para que varios workers nao desserializem o mesmo arquivo.
    """

    def __init__(self, reference: FrozenModelReference | None = None) -> None:
        self.reference = reference or resolve_frozen_model()
        self._pipeline: Any | None = None
        self._lock = threading.Lock()
        self.load_seconds: float | None = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    started = perf_counter()
                    pipeline = joblib.load(PROJECT_ROOT / self.reference.relative_path)
                    if list(pipeline.named_steps) != ["scaler", "model"]:
                        raise ModelServerError("Pipeline congelado tem estrutura inesperada.")
                    self.load_seconds = perf_counter() - started
                    self._pipeline = pipeline
        return self._pipeline

    def warm_up(self) -> float:
        """Forca a carga fora do caminho de medicao de latencia."""

        self.pipeline
        return self.load_seconds or 0.0

    def predict_batch(self, features: Any) -> "BatchOutcome":
        """Classifica um lote e devolve somente contagens agregadas."""

        batch_size = int(len(features))
        if batch_size == 0:
            raise ModelServerError("Lote vazio nao pode ser servido.")
        started = perf_counter()
        probabilities = np.asarray(self.pipeline.predict_proba(features))[:, 1]
        positives = int(np.count_nonzero(probabilities >= self.reference.threshold))
        latency_ms = (perf_counter() - started) * 1000.0
        return BatchOutcome(
            batch_size=batch_size,
            positive_count=positives,
            latency_ms=latency_ms,
        )


@dataclass(frozen=True)
class BatchOutcome:
    """Saida agregada de um lote: nenhuma probabilidade por registro escapa."""

    batch_size: int
    positive_count: int
    latency_ms: float

    def to_event(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "positive_count": self.positive_count,
            "latency_ms": round(self.latency_ms, 6),
        }
