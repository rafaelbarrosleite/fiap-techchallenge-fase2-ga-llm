"""Leitura somente leitura dos artefatos que alimentam o painel.

O painel nao recalcula nada. Ele carrega artefatos ja assinados, confere as
assinaturas e organiza os campos para renderizacao. Se um artefato divergir da
propria assinatura, a construcao falha em vez de exibir numero nao confiavel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..genetic.serialization import stable_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = PROJECT_ROOT / "reports" / "figures" / "final_presentation"

FIGURES = (
    ("01_recall_baseline_vs_ga.png", "Recall maligno no teste final"),
    ("02_falsos_negativos_baseline_vs_ga.png", "Falsos negativos: baseline versus GA"),
    ("03_roc_auc_por_metodo.png", "ROC-AUC por método"),
    ("04_recall_cv_vs_holdout.png", "Recall na validação cruzada versus holdout"),
    ("05_intervalos_recall.png", "Intervalos de confiança do recall"),
    ("06_fitness_ga_vs_busca_aleatoria.png", "Fitness: GA versus busca aleatória"),
)
SCALABILITY_FIGURE = ("07_escalabilidade_automatica.png", "Escalabilidade automática")


class DashboardDataError(RuntimeError):
    """Um artefato exigido pelo painel esta ausente ou nao confere."""


def _load_signed(relative_path: str, artifact_type: str | None = None) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise DashboardDataError(f"Artefato ausente: {relative_path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature")
    if signature is not None:
        unsigned = {key: value for key, value in payload.items() if key != "signature"}
        if signature != stable_sha256(unsigned):
            raise DashboardDataError(f"Assinatura divergente: {relative_path}.")
    if artifact_type is not None and payload.get("artifact_type") != artifact_type:
        raise DashboardDataError(f"Tipo inesperado em {relative_path}.")
    return payload


@dataclass(frozen=True)
class DashboardData:
    """Visao consolidada e imutavel do que o painel exibe."""

    master_rows: list[dict[str, Any]]
    final_results: dict[str, Any]
    uncertainty: dict[str, Any]
    llm_output: dict[str, Any]
    llm_factuality: dict[str, Any]
    llm_safety: dict[str, Any]
    llm_evaluation: dict[str, Any]
    individual_output: dict[str, Any]
    individual_evaluation: dict[str, Any]
    delivery_manifest: dict[str, Any]
    scalability: dict[str, Any] | None
    figures: dict[str, bytes] = field(default_factory=dict)

    @property
    def selected_row(self) -> dict[str, Any]:
        """A linha do vencedor global, congelado antes do holdout."""

        marcadas = [r for r in self.master_rows if r.get("selection_status") == "global_selected"]
        if len(marcadas) != 1:
            raise DashboardDataError(
                f"Esperava exatamente um vencedor global; encontrei {len(marcadas)}."
            )
        row = marcadas[0]
        # O contrato da LLM nomeia o mesmo candidato; divergir aqui significaria
        # que a tabela mestre e a explicacao falam de modelos diferentes.
        contrato = self.llm_output["structured_output"]["modelo_selecionado"]
        if (contrato["model"], contrato["method"]) != (row["family"], row["method"]):
            raise DashboardDataError(
                "Tabela mestre e contrato da LLM discordam sobre o candidato congelado."
            )
        return row

    @property
    def false_negative_transitions(self) -> list[tuple[str, int, int]]:
        """Baseline -> GA em falsos negativos, por familia."""

        by_pair = {(r["family"], r["method"]): r for r in self.master_rows}
        families: list[tuple[str, int, int]] = []
        for family in ("logistic_regression", "random_forest", "knn"):
            baseline, ga = by_pair.get((family, "baseline")), by_pair.get((family, "ga"))
            if baseline and ga:
                families.append(
                    (baseline["family_label"], int(baseline["false_negatives"]), int(ga["false_negatives"]))
                )
        return families


def load_dashboard_data(*, include_scalability: bool = True) -> DashboardData:
    """Carrega e confere todos os artefatos exibidos pelo painel."""

    master = _load_signed("artifacts/final_summary/model_results.json")
    scalability: dict[str, Any] | None = None
    if include_scalability:
        try:
            scalability = _load_signed("artifacts/scalability/scalability_report.json")
        except DashboardDataError:
            # A medicao depende do hardware e pode nao ter sido executada; o
            # painel continua valido sem o cartao de escalabilidade.
            scalability = None

    figures: dict[str, bytes] = {}
    wanted = list(FIGURES) + ([SCALABILITY_FIGURE] if scalability else [])
    for name, _ in wanted:
        path = FIGURE_ROOT / name
        if not path.is_file():
            raise DashboardDataError(f"Figura ausente: {name}.")
        figures[name] = path.read_bytes()

    return DashboardData(
        master_rows=master["rows"],
        final_results=_load_signed("artifacts/final_evaluation/final_test_results.json"),
        uncertainty=_load_signed("artifacts/final_evaluation/uncertainty_results.json"),
        llm_output=_load_signed("artifacts/llm_evaluation/llm_output.json", "llm_output"),
        llm_factuality=_load_signed("artifacts/llm_evaluation/factuality_report.json"),
        llm_safety=_load_signed("artifacts/llm_evaluation/safety_report.json"),
        llm_evaluation=_load_signed("artifacts/llm_evaluation/evaluation_report.json"),
        individual_output=_load_signed(
            "artifacts/llm_individual_explanation/individual_output.json",
            "individual_explanation_output",
        ),
        individual_evaluation=_load_signed(
            "artifacts/llm_individual_explanation/evaluation_report.json"
        ),
        delivery_manifest=_load_signed("artifacts/final_summary/final_delivery_manifest.json"),
        scalability=scalability,
        figures=figures,
    )
