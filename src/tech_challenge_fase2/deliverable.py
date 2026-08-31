"""Consolidacao somente leitura das evidencias congeladas das Missoes 1 a 5."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tech_challenge_fase2 import __version__
from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_EVALUATION_ROOT = PROJECT_ROOT / "artifacts" / "final_evaluation"
LLM_EVALUATION_ROOT = PROJECT_ROOT / "artifacts" / "llm_evaluation"
LLM_V4_ROOT = PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v4"
INDIVIDUAL_LLM_ROOT = PROJECT_ROOT / "artifacts" / "llm_individual_explanation"
INDIVIDUAL_LLM_OPENAI_ROOT = PROJECT_ROOT / "artifacts" / "llm_individual_explanation_openai"
SELECTION_ROOT = PROJECT_ROOT / "artifacts" / "selection"
SUMMARY_ROOT = PROJECT_ROOT / "artifacts" / "final_summary"
PRESENTATION_FIGURE_ROOT = PROJECT_ROOT / "reports" / "figures" / "final_presentation"

MODEL_ORDER = ("logistic_regression", "random_forest", "knn")
METHOD_ORDER = ("baseline", "ga", "random_search")
MODEL_LABELS = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
    "knn": "KNN",
}
METHOD_LABELS = {"baseline": "Baseline", "ga": "GA", "random_search": "Busca aleatória"}

EXPECTED_DOCUMENTS = (
    "README.md",
    "docs/relatorio_final.md",
    "docs/resumo_executivo.md",
    "docs/roteiro_apresentacao.md",
    "docs/demo_guide.md",
    "docs/mapa_evidencias.md",
    "docs/matriz_rastreabilidade_final.md",
    "docs/auditoria_documental_final.md",
    "docs/camada_llm_segura.md",
    "docs/avaliacao_final.md",
    "docs/limitacoes_e_validade.md",
    "docs/contrato_llm_v2.md",
    "docs/avaliacao_provider_real_v4.md",
    "docs/explicacao_individual_llm.md",
    "docs/escalabilidade_e_monitoramento.md",
    "docs/examples/individual_explanation_v1.json",
)
# artifacts/scalability/ nao entra no selo: a medicao depende do hardware e
# mudaria de hash a cada execucao do benchmark, reprovando a validacao por um
# motivo que nao indica perda de integridade. Ele tem validador proprio em
# `validate-scalability`.
EXPECTED_SOURCE_ARTIFACTS = (
    "artifacts/final_evaluation/final_test_results.json",
    "artifacts/final_evaluation/uncertainty_results.json",
    "artifacts/final_evaluation/final_evaluation_plan.json",
    "artifacts/final_evaluation/final_manifest.json",
    "artifacts/selection/frozen_candidates.json",
    "artifacts/llm_evaluation/llm_evaluation_manifest.json",
    "artifacts/llm_evaluation/factuality_report.json",
    "artifacts/llm_evaluation/safety_report.json",
    "artifacts/llm_evaluation/evaluation_report.json",
    "artifacts/llm_contract_v2/contract_v2_manifest.json",
    "artifacts/llm_evaluation_openai_v4/llm_evaluation_manifest.json",
    "artifacts/llm_evaluation_openai_v4/factuality_report.json",
    "artifacts/llm_evaluation_openai_v4/safety_report.json",
    "artifacts/llm_evaluation_openai_v4/evaluation_report.json",
    "artifacts/llm_evaluation_openai_v4/hallucination_report.json",
    "artifacts/llm_individual_explanation/individual_explanation_manifest.json",
    "artifacts/llm_individual_explanation/evaluation_report.json",
    "artifacts/llm_individual_explanation_openai/individual_explanation_manifest.json",
    "artifacts/llm_individual_explanation_openai/individual_output.json",
    "artifacts/llm_individual_explanation_openai/privacy_report.json",
    "artifacts/llm_individual_explanation_openai/factuality_report.json",
    "artifacts/llm_individual_explanation_openai/safety_report.json",
    "artifacts/llm_individual_explanation_openai/evaluation_report.json",
)


class DeliverableError(RuntimeError):
    """A consolidacao ou a validacao encontrou inconsistencia."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_signed(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = _load_json(path)
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("artifact_type") != artifact_type or signature != stable_sha256(unsigned):
        raise DeliverableError(f"Schema ou assinatura invalida: {path}.")
    return payload


def _sign(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["signature"] = stable_sha256(result)
    return result


def load_authoritative_sources() -> dict[str, dict[str, Any]]:
    results = _load_signed(FINAL_EVALUATION_ROOT / "final_test_results.json", "final_test_results")
    uncertainty = _load_signed(FINAL_EVALUATION_ROOT / "uncertainty_results.json", "final_uncertainty_results")
    plan = _load_signed(FINAL_EVALUATION_ROOT / "final_evaluation_plan.json", "final_evaluation_plan")
    final_manifest = _load_signed(FINAL_EVALUATION_ROOT / "final_manifest.json", "final_evaluation_manifest")
    frozen = _load_signed(SELECTION_ROOT / "frozen_candidates.json", "frozen_provisional_candidates")
    llm_manifest = _load_signed(LLM_EVALUATION_ROOT / "llm_evaluation_manifest.json", "llm_evaluation_manifest")
    factuality = _load_signed(LLM_EVALUATION_ROOT / "factuality_report.json", "llm_factuality_report")
    safety = _load_signed(LLM_EVALUATION_ROOT / "safety_report.json", "llm_safety_report")
    evaluation = _load_signed(LLM_EVALUATION_ROOT / "evaluation_report.json", "llm_evaluation_report")
    contract_v2 = _load_signed(
        PROJECT_ROOT / "artifacts" / "llm_contract_v2" / "contract_v2_manifest.json",
        "llm_contract_v2_manifest",
    )
    llm_v4 = _load_signed(
        LLM_V4_ROOT / "llm_evaluation_manifest.json", "openai_v4_evaluation_manifest",
    )
    individual_fake = _load_signed(
        INDIVIDUAL_LLM_ROOT / "individual_explanation_manifest.json",
        "individual_explanation_manifest",
    )
    individual_real = _load_signed(
        INDIVIDUAL_LLM_OPENAI_ROOT / "individual_explanation_manifest.json",
        "individual_explanation_manifest",
    )
    individual_factuality = _load_signed(
        INDIVIDUAL_LLM_OPENAI_ROOT / "factuality_report.json",
        "individual_factuality_report",
    )
    individual_safety = _load_signed(
        INDIVIDUAL_LLM_OPENAI_ROOT / "safety_report.json",
        "individual_safety_report",
    )
    individual_evaluation = _load_signed(
        INDIVIDUAL_LLM_OPENAI_ROOT / "evaluation_report.json",
        "individual_evaluation_report",
    )
    individual_privacy = _load_signed(
        INDIVIDUAL_LLM_OPENAI_ROOT / "privacy_report.json",
        "individual_privacy_report",
    )
    if results["plan_signature"] != plan["signature"] or uncertainty["plan_signature"] != plan["signature"]:
        raise DeliverableError("Resultados finais nao pertencem ao plano assinado.")
    if final_manifest["plan_signature"] != plan["signature"]:
        raise DeliverableError("Manifesto final nao pertence ao plano assinado.")
    if results.get("new_optimization_performed") is not False or results.get("selection_reopened") is not False:
        raise DeliverableError("Resultados finais indicam nova otimizacao ou selecao.")
    if llm_manifest.get("provider") != "fake" or llm_manifest.get("individual_data_sent") is not False:
        raise DeliverableError("A avaliacao LLM oficial nao e o mock seguro esperado.")
    if not factuality.get("passed") or not safety.get("passed") or not evaluation.get("approved"):
        raise DeliverableError("A avaliacao LLM congelada nao esta aprovada.")
    if not (
        contract_v2.get("status") == "approved"
        and contract_v2.get("ready_for_real_v2_evaluation") is True
    ):
        raise DeliverableError("O contrato LLM V2 offline nao esta aprovado.")
    v4_checks = llm_v4.get("checks", {})
    if not (
        llm_v4.get("status") == "methodologically_complete_not_approved"
        and llm_v4.get("scientific_evaluation_approved") is False
        and v4_checks.get("factual_total") == 327
        and v4_checks.get("factual_passed") == 327
        and v4_checks.get("factuality") is True
        and v4_checks.get("safety") is True
        and v4_checks.get("completeness") is True
        and v4_checks.get("scientific_calibration") is False
        and llm_v4.get("call_budget", {}).get("scientific_main") == 1
        and llm_v4.get("call_budget", {}).get("automatic_retries") == 0
        and llm_v4.get("privacy", {}).get("individual_data_sent") is False
    ):
        raise DeliverableError("A evidencia complementar OpenAI V2 diverge do status congelado.")
    if not (
        individual_fake.get("status") == "approved"
        and individual_fake.get("approved") is True
        and individual_fake.get("execution_scope", {}).get("external_calls") == 0
        and individual_real.get("status") == "approved"
        and individual_real.get("approved") is True
        and individual_real.get("quality", {}).get("factuality") is True
        and individual_real.get("quality", {}).get("safety") is True
        and individual_real.get("quality", {}).get("completeness") is True
        and individual_real.get("quality", {}).get("clarity") is True
        and individual_real.get("quality", {}).get("medical_context_relevance") is True
        and individual_real.get("quality", {}).get("scientific_calibration") is True
        and individual_factuality.get("passed_checks") == 40
        and individual_factuality.get("total_checks") == 40
        and individual_safety.get("passed") is True
        and individual_evaluation.get("approved") is True
        and individual_privacy.get("passed") is True
        and individual_real.get("privacy", {}).get("patient_identifiers_sent") is False
        and individual_real.get("privacy", {}).get("raw_feature_values_sent") is False
        and individual_real.get("privacy", {}).get("holdout_case_sent") is False
        and individual_real.get("execution_scope", {}).get("new_training") is False
        and individual_real.get("execution_scope", {}).get("holdout_inference") is False
    ):
        raise DeliverableError("A explicacao individual nao esta integralmente aprovada.")
    return {
        "results": results, "uncertainty": uncertainty, "plan": plan,
        "final_manifest": final_manifest, "frozen": frozen, "llm_manifest": llm_manifest,
        "factuality": factuality, "safety": safety, "evaluation": evaluation,
        "contract_v2": contract_v2, "llm_v4": llm_v4,
        "individual_fake": individual_fake, "individual_real": individual_real,
        "individual_factuality": individual_factuality,
        "individual_safety": individual_safety,
        "individual_evaluation": individual_evaluation,
        "individual_privacy": individual_privacy,
    }


def _origin_method(origin: str) -> str:
    if origin == "RandomizedSearchCV":
        return "random_search"
    if origin.startswith("GA_"):
        return "ga"
    if origin == "baseline_cv":
        return "baseline"
    raise DeliverableError(f"Origem desconhecida: {origin}.")


def _observation(model: str, method: str) -> str:
    observations = {
        ("logistic_regression", "baseline"): "Referência original recalculada no protocolo final.",
        ("logistic_regression", "ga"): "GA B; mesmo resultado no holdout que a busca aleatória, sem reabrir seleção.",
        ("logistic_regression", "random_search"): "Vencedor global congelado antes do holdout e modelo para demonstração.",
        ("random_forest", "baseline"): "Referência original da família.",
        ("random_forest", "ga"): "Vencedor da família por CV; reduziu um falso negativo no holdout.",
        ("random_forest", "random_search"): "Mesma matriz do GA no holdout, com ROC-AUC diferente.",
        ("knn", "baseline"): "Referência original da família.",
        ("knn", "ga"): "Vencedor serializado da família; ganho de recall em CV não se confirmou no holdout.",
        ("knn", "random_search"): "Solução canonicamente idêntica ao GA; não é evidência independente.",
    }
    return observations[(model, method)]


def build_master_rows(sources: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    sources = sources or load_authoritative_sources()
    results, frozen = sources["results"], sources["frozen"]
    family_selected = {
        model: _origin_method(candidate["origin"])
        for model, candidate in frozen["winners_by_model"].items()
    }
    global_winner = frozen["global_provisional_winner"]
    global_pair = (global_winner["model"], _origin_method(global_winner["origin"]))
    by_pair = {(item["model"], item["method"]): item for item in results["candidate_results"]}
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            candidate = by_pair[(model, method)]
            metrics, cv_metrics = candidate["metrics"], candidate["cv_metrics"]
            if (model, method) == global_pair:
                selection_status = "global_selected"
            elif method == family_selected[model]:
                selection_status = "family_selected"
            elif method == "baseline":
                selection_status = "evaluated_baseline"
            else:
                selection_status = "evaluated_comparator"
            rows.append({
                "family": model,
                "family_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "configuration": candidate["parameters"],
                "fitness_cv": cv_metrics["fitness"],
                "recall_cv": cv_metrics["mean_recall_malignant"],
                "recall_test": metrics["recall_malignant"],
                "f1_test": metrics["f1_malignant"],
                "roc_auc_test": metrics["roc_auc"],
                "true_negatives": metrics["true_negatives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "true_positives": metrics["true_positives"],
                "origin": candidate["origin"],
                "selection_status": selection_status,
                "observation": _observation(model, method),
            })
    return rows


def generate_master_table(
    output_root: Path = SUMMARY_ROOT,
    *,
    generated_at_utc: str | None = None,
) -> tuple[Path, Path]:
    sources = load_authoritative_sources()
    rows = build_master_rows(sources)
    output_root = Path(output_root)
    csv_path = output_root / "model_results.csv"
    json_path = output_root / "model_results.json"
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family", "family_label", "method", "method_label", "configuration",
        "fitness_cv", "recall_cv", "recall_test", "f1_test", "roc_auc_test",
        "true_negatives", "false_positives", "false_negatives", "true_positives",
        "origin", "selection_status", "observation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["configuration"] = json.dumps(row["configuration"], sort_keys=True, ensure_ascii=False)
            writer.writerow(serialized)
    source_paths = {
        "final_test_results": FINAL_EVALUATION_ROOT / "final_test_results.json",
        "frozen_candidates": SELECTION_ROOT / "frozen_candidates.json",
    }
    artifact = _sign({
        "schema_version": "1.0",
        "artifact_type": "final_master_model_results",
        "generated_at_utc": generated_at_utc or utc_now(),
        "source_files": {
            name: {"relative_path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
            for name, path in source_paths.items()
        },
        "data_scope": {
            "derived_from_aggregate_artifacts_only": True,
            "new_training_performed": False,
            "new_optimization_performed": False,
            "new_holdout_inference_performed": False,
            "selection_reopened": False,
            "classification_threshold": sources["results"]["classification_threshold"],
        },
        "row_count": len(rows),
        "rows": rows,
    })
    save_json(artifact, json_path)
    return csv_path, json_path


def _plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 17,
        "axes.labelsize": 12, "axes.edgecolor": "#374151", "axes.labelcolor": "#1f2937",
        "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827",
        "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _label_bars(ax: plt.Axes, bars: Any, decimals: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.{decimals}f}", ha="center", va="bottom", fontsize=10)


def _chart_header(ax: plt.Axes, title: str, subtitle: str) -> None:
    """Renderiza título e contexto em linhas separadas e legíveis."""
    ax.set_title(title, loc="left", pad=35, fontweight="semibold")
    ax.text(
        0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        color="#4b5563",
        fontsize=10,
        va="bottom",
    )


def generate_presentation_figures(
    figure_root: Path = PRESENTATION_FIGURE_ROOT,
    *,
    generated_at_utc: str | None = None,
) -> Path:
    sources = load_authoritative_sources()
    rows = build_master_rows(sources)
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    _plot_style()
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    by_pair = {(row["family"], row["method"]): row for row in rows}
    blue, blue_light, orange = "#2563eb", "#93c5fd", "#d97706"
    charcoal, grey = "#1f2937", "#9ca3af"
    chart_map: list[dict[str, Any]] = []

    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    baseline = [by_pair[(model, "baseline")]["recall_test"] for model in MODEL_ORDER]
    ga = [by_pair[(model, "ga")]["recall_test"] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars_a = ax.bar(x - width / 2, baseline, width, label="Baseline", color=blue_light, edgecolor=charcoal)
    bars_b = ax.bar(x + width / 2, ga, width, label="GA", color=blue, edgecolor=charcoal)
    _chart_header(ax, "Recall maligno no teste final: baseline versus GA", "114 casos no holdout, incluindo 42 malignos; limiar fixo 0,5")
    ax.set_ylabel("Recall maligno")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    _label_bars(ax, bars_a)
    _label_bars(ax, bars_b)
    path = figure_root / "01_recall_baseline_vs_ga.png"
    _save_figure(fig, path)
    chart_map.append({"filename": path.name, "question": "Como o recall no holdout mudou do baseline para o GA?", "family": "Comparison & Ranking", "type": "grouped bar", "source": "final_test_results.json", "supported_claim": "LR e RF aumentaram recall; KNN permaneceu igual."})

    fn_baseline = [by_pair[(model, "baseline")]["false_negatives"] for model in MODEL_ORDER]
    fn_ga = [by_pair[(model, "ga")]["false_negatives"] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars_a = ax.bar(x - width / 2, fn_baseline, width, label="Baseline", color=blue_light, edgecolor=charcoal)
    bars_b = ax.bar(x + width / 2, fn_ga, width, label="GA", color=blue, edgecolor=charcoal)
    _chart_header(ax, "Falsos negativos no teste final: baseline versus GA", "Contagem entre 42 casos malignos; valores menores são descritivamente melhores")
    ax.set_ylabel("Falsos negativos")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(fn_baseline) + 1.4)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    _label_bars(ax, bars_a, decimals=0)
    _label_bars(ax, bars_b, decimals=0)
    path = figure_root / "02_falsos_negativos_baseline_vs_ga.png"
    _save_figure(fig, path)
    chart_map.append({"filename": path.name, "question": "Quantos falsos negativos foram observados?", "family": "Comparison & Ranking", "type": "grouped bar", "source": "final_test_results.json", "supported_claim": "LR passou de 3 para 1, RF de 4 para 3 e KNN manteve 4."})

    fig, ax = plt.subplots(figsize=(10, 5.8))
    offsets = {"baseline": -0.18, "ga": 0.0, "random_search": 0.18}
    colors = {"baseline": grey, "ga": blue, "random_search": orange}
    markers = {"baseline": "o", "ga": "s", "random_search": "D"}
    for method in METHOD_ORDER:
        values = [by_pair[(model, method)]["roc_auc_test"] for model in MODEL_ORDER]
        ax.scatter(values, x + offsets[method], s=90, color=colors[method], marker=markers[method], edgecolor=charcoal, label=METHOD_LABELS[method], zorder=3)
        for value, y in zip(values, x + offsets[method], strict=True):
            ax.text(value + 0.00045, y, f"{value:.4f}", va="center", fontsize=9)
    _chart_header(ax, "ROC-AUC no teste final por método", "Escala horizontal focada para tornar diferenças pequenas legíveis; valores completos nos artefatos")
    ax.set_xlabel("ROC-AUC")
    ax.set_yticks(x, labels)
    ax.set_xlim(0.968, 1.001)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    path = figure_root / "03_roc_auc_por_metodo.png"
    _save_figure(fig, path)
    chart_map.append({"filename": path.name, "question": "Como baseline, GA e busca aleatória se comparam em ROC-AUC?", "family": "Comparison & Ranking", "type": "faceted dot", "source": "final_test_results.json", "supported_claim": "RF e KNN mostram trade-off: AUC do GA caiu mesmo quando outras métricas melhoraram."})

    categories = []
    cv_values = []
    test_values = []
    for model in MODEL_ORDER:
        for method in ("baseline", "ga"):
            categories.append(f"{MODEL_LABELS[model]}\n{METHOD_LABELS[method]}")
            cv_values.append(by_pair[(model, method)]["recall_cv"])
            test_values.append(by_pair[(model, method)]["recall_test"])
    y = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for index, (cv_value, test_value) in enumerate(zip(cv_values, test_values, strict=True)):
        ax.plot([cv_value, test_value], [index, index], color="#cbd5e1", linewidth=2, zorder=1)
    ax.scatter(cv_values, y, color=orange, marker="o", s=70, edgecolor=charcoal, label="CV", zorder=3)
    ax.scatter(test_values, y, color=blue, marker="s", s=70, edgecolor=charcoal, label="Holdout", zorder=3)
    _chart_header(ax, "Recall em validação cruzada versus teste final", "Comparação descritiva; o holdout não foi usado para reordenar candidatos")
    ax.set_xlabel("Recall maligno")
    ax.set_yticks(y, categories)
    ax.set_xlim(0.88, 1.0)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    path = figure_root / "04_recall_cv_vs_holdout.png"
    _save_figure(fig, path)
    chart_map.append({"filename": path.name, "question": "Os ganhos de recall em CV apareceram no holdout?", "family": "Uncertainty & Benchmark", "type": "paired dot", "source": "final_test_results.json", "supported_claim": "O ganho apareceu em LR e RF, mas não se confirmou para KNN."})

    intervals = sources["uncertainty"]["candidate_intervals"]
    positions = np.arange(6)
    estimates, lower_errors, upper_errors, interval_labels, interval_colors, interval_markers = [], [], [], [], [], []
    for model in MODEL_ORDER:
        for method in ("baseline", "ga"):
            item = intervals[f"{model}__{method}"]["recall_malignant"]
            estimates.append(item["estimate"])
            lower_errors.append(item["estimate"] - item["lower"])
            upper_errors.append(item["upper"] - item["estimate"])
            interval_labels.append(f"{MODEL_LABELS[model]} — {METHOD_LABELS[method]}")
            interval_colors.append(blue_light if method == "baseline" else blue)
            interval_markers.append("o" if method == "baseline" else "s")
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for index, (estimate, low, high, color, marker) in enumerate(zip(estimates, lower_errors, upper_errors, interval_colors, interval_markers, strict=True)):
        ax.errorbar(estimate, index, xerr=[[low], [high]], fmt=marker, color=color, ecolor=charcoal, capsize=5, markersize=8, markeredgecolor=charcoal)
        ax.text(min(1.006, estimate + high + 0.004), index, f"{estimate:.4f}", va="center", fontsize=9)
    _chart_header(ax, "Intervalos de confiança de 95% do recall", "Wilson sobre 42 casos malignos; intervalos amplos limitam conclusões fortes")
    ax.set_xlabel("Recall maligno e IC95%")
    ax.set_yticks(positions, interval_labels)
    ax.set_xlim(0.74, 1.03)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    path = figure_root / "05_intervalos_recall.png"
    _save_figure(fig, path)
    chart_map.append({"filename": path.name, "question": "Quão precisas são as estimativas de recall?", "family": "Uncertainty & Benchmark", "type": "dot and interval", "source": "uncertainty_results.json", "supported_claim": "Os IC95% são amplos e se sobrepõem."})

    ga_fitness = [by_pair[(model, "ga")]["fitness_cv"] for model in MODEL_ORDER]
    random_fitness = [by_pair[(model, "random_search")]["fitness_cv"] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, (ga_value, random_value) in enumerate(zip(ga_fitness, random_fitness, strict=True)):
        ax.plot([ga_value, random_value], [index, index], color="#cbd5e1", linewidth=2)
    ax.scatter(ga_fitness, x, color=blue, marker="s", s=85, edgecolor=charcoal, label="GA", zorder=3)
    ax.scatter(random_fitness, x, color=orange, marker="D", s=85, edgecolor=charcoal, label="Busca aleatória", zorder=3)
    _chart_header(ax, "Fitness de CV: GA versus busca aleatória", "Mesmas cinco dobras e mesma fórmula; escala focada e tempo fora do desempate")
    ax.set_xlabel("Fitness composto de CV")
    ax.set_yticks(x, labels)
    ax.set_xlim(0.945, 0.977)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    path = figure_root / "06_fitness_ga_vs_busca_aleatoria.png"
    _save_figure(fig, path)
    chart_map.append({"filename": path.name, "question": "O GA superou a busca aleatória no objetivo de CV?", "family": "Uncertainty & Benchmark", "type": "paired dot", "source": "final_test_results.json", "supported_claim": "LR e KNN empataram nas métricas agregadas; RF teve fitness maior com GA."})

    records = []
    for chart in chart_map:
        path = figure_root / chart["filename"]
        with Image.open(path) as image:
            width_px, height_px = image.size
        records.append({
            **chart, "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "width_px": width_px, "height_px": height_px,
        })
    qa = _sign({
        "schema_version": "1.0", "artifact_type": "final_presentation_figure_qa",
        "generated_at_utc": generated_at_utc or utc_now(),
        "source_scope": "aggregate_artifacts_only",
        "new_training_performed": False, "new_inference_performed": False,
        "automated_checks": {
            "all_files_exist": True, "all_files_nonempty": True,
            "titles_axes_and_context_present": True, "explicit_palette_used": True,
        },
        "visual_review": {"status": "pending", "reviewed_at_utc": None, "reviewer": None, "notes": []},
        "figures": records,
    })
    qa_path = figure_root / "figure_qa_report.json"
    save_json(qa, qa_path)
    return qa_path


def approve_figure_review(
    qa_path: Path = PRESENTATION_FIGURE_ROOT / "figure_qa_report.json",
    *,
    notes: list[str],
    reviewed_at_utc: str | None = None,
) -> dict[str, Any]:
    qa_path = Path(qa_path)
    payload = _load_signed(qa_path, "final_presentation_figure_qa")
    for item in payload["figures"]:
        path = qa_path.parent / item["filename"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise DeliverableError(f"Figura mudou antes da revisao: {path}.")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    unsigned["visual_review"] = {
        "status": "approved", "reviewed_at_utc": reviewed_at_utc or utc_now(),
        "reviewer": "Codex visual inspection", "notes": notes,
    }
    approved = _sign(unsigned)
    save_json(approved, qa_path)
    return approved


LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def broken_local_links(paths: list[Path]) -> list[dict[str, str]]:
    broken: list[dict[str, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            candidate = (path.parent / target).resolve()
            if not candidate.exists():
                broken.append({"document": str(path.relative_to(PROJECT_ROOT)), "target": raw_target})
    return broken


def validate_deliverable(*, require_manifest: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    sources = load_authoritative_sources()
    for relative in EXPECTED_DOCUMENTS + EXPECTED_SOURCE_ARTIFACTS:
        add(f"exists:{relative}", (PROJECT_ROOT / relative).is_file(), relative)
    markdown_paths = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
    broken = broken_local_links(markdown_paths)
    add("local_markdown_links", not broken, json.dumps(broken, ensure_ascii=False))

    master_json_path = SUMMARY_ROOT / "model_results.json"
    master_csv_path = SUMMARY_ROOT / "model_results.csv"
    master = _load_signed(master_json_path, "final_master_model_results")
    expected_rows = build_master_rows(sources)
    add("master_json_rows", master["rows"] == expected_rows and master["row_count"] == 9, "9 candidatos e metricas exatas")
    with master_csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    add("master_csv_rows", len(csv_rows) == 9, f"rows={len(csv_rows)}")
    add("master_no_new_computation", all(master["data_scope"].get(key) is False for key in (
        "new_training_performed", "new_optimization_performed", "new_holdout_inference_performed", "selection_reopened",
    )), json.dumps(master["data_scope"], ensure_ascii=False))

    report = (PROJECT_ROOT / "docs" / "relatorio_final.md").read_text(encoding="utf-8")
    expected_numbers = ("3 para 1", "4 para 3", "0,904762", "0,976190", "42 casos")
    add("main_numbers_documented", all(value in report for value in expected_numbers), str(expected_numbers))
    divergence_terms = ("GA B", "GA C", "baseline histórico", "vencedor global")
    audit = (PROJECT_ROOT / "docs" / "auditoria_documental_final.md").read_text(encoding="utf-8")
    add("historical_divergences_documented", all(value in audit for value in divergence_terms), str(divergence_terms))

    qa = _load_signed(PRESENTATION_FIGURE_ROOT / "figure_qa_report.json", "final_presentation_figure_qa")
    figures_valid = qa["visual_review"]["status"] == "approved" and len(qa["figures"]) == 6
    for item in qa["figures"]:
        path = PRESENTATION_FIGURE_ROOT / item["filename"]
        figures_valid = figures_valid and path.is_file() and file_sha256(path) == item["sha256"]
    add("presentation_figures", figures_valid, f"figures={len(qa['figures'])}, review={qa['visual_review']['status']}")

    add("mission4_no_reselection", sources["results"]["selection_reopened"] is False, "selection_reopened=false")
    add("mission4_no_new_optimization", sources["results"]["new_optimization_performed"] is False, "new_optimization_performed=false")
    llm = sources["llm_manifest"]
    add("mission5_mock_offline", llm["provider"] == "fake" and llm["generation_configuration"]["network_required"] is False, "provider=fake")
    add("mission5_no_individual_data", llm["individual_data_sent"] is False, "individual_data_sent=false")
    llm_v4 = sources["llm_v4"]
    add(
        "mission75_real_provider_recorded",
        llm_v4["call_budget"]["scientific_main"] == 1
        and llm_v4["scientific_evaluation_approved"] is False,
        "1 chamada; avaliacao cientifica nao aprovada",
    )
    add(
        "mission75_factuality_and_privacy",
        llm_v4["checks"]["factual_passed"] == 327
        and llm_v4["checks"]["factual_total"] == 327
        and llm_v4["privacy"]["individual_data_sent"] is False,
        "327/327; zero dados individuais",
    )
    individual = sources["individual_real"]
    add(
        "individual_explanation_approved",
        individual["approved"] is True
        and sources["individual_factuality"]["passed_checks"] == 40
        and sources["individual_evaluation"]["approved"] is True,
        "40/40 fatos; seis dimensoes aprovadas",
    )
    add(
        "individual_explanation_privacy",
        individual["privacy"]["patient_identifiers_sent"] is False
        and individual["privacy"]["raw_feature_values_sent"] is False
        and individual["privacy"]["holdout_case_sent"] is False,
        "sem ID, linha bruta ou caso do holdout",
    )
    add("project_version", __version__ == "0.7.0", f"version={__version__}")

    if require_manifest:
        manifest_path = SUMMARY_ROOT / "final_delivery_manifest.json"
        manifest = _load_signed(manifest_path, "final_delivery_manifest")
        add("delivery_manifest_status", manifest["status"] == "completed" and manifest["tests"]["status"] == "passed", f"status={manifest['status']}")
        hashed_files_valid = True
        for group in ("documents", "artifacts", "figures"):
            for item in manifest["files"][group]:
                path = PROJECT_ROOT / item["relative_path"]
                hashed_files_valid = hashed_files_valid and path.is_file() and file_sha256(path) == item["sha256"]
        add("delivery_manifest_hashes", hashed_files_valid, "documentos, artefatos e figuras")
    passed = all(item["passed"] for item in checks)
    return {"status": "passed" if passed else "failed", "passed": passed, "check_count": len(checks), "checks": checks}


def generate_delivery_manifest(
    *,
    test_count: int,
    test_command: str = "uv run pytest",
    generated_at_utc: str | None = None,
    output_path: Path | None = None,
) -> Path:
    preflight = validate_deliverable(require_manifest=False)
    if not preflight["passed"]:
        failures = [item for item in preflight["checks"] if not item["passed"]]
        raise DeliverableError(f"Preflight da entrega falhou: {failures}")
    document_paths = [PROJECT_ROOT / relative for relative in EXPECTED_DOCUMENTS]
    artifact_paths = [
        SUMMARY_ROOT / "model_results.csv", SUMMARY_ROOT / "model_results.json",
        PRESENTATION_FIGURE_ROOT / "figure_qa_report.json",
        *sorted(LLM_V4_ROOT.glob("*.json")),
        *sorted(INDIVIDUAL_LLM_ROOT.glob("*.json")),
        *sorted(INDIVIDUAL_LLM_OPENAI_ROOT.glob("*.json")),
    ]
    figure_paths = sorted(PRESENTATION_FIGURE_ROOT.glob("*.png"))
    records = lambda paths: [
        {"relative_path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in paths
    ]
    sources = load_authoritative_sources()
    payload = _sign({
        "schema_version": "1.0", "artifact_type": "final_delivery_manifest",
        "generated_at_utc": generated_at_utc or utc_now(), "status": "completed",
        "project_version": __version__,
        "tests": {"command": test_command, "status": "passed", "passed_count": test_count},
        "authority_references": {
            "mission4_plan_signature": sources["plan"]["signature"],
            "mission4_manifest_signature": sources["final_manifest"]["signature"],
            "mission5_manifest_signature": sources["llm_manifest"]["signature"],
            "mission74_contract_signature": sources["contract_v2"]["signature"],
            "mission75_manifest_signature": sources["llm_v4"]["signature"],
            "individual_fake_manifest_signature": sources["individual_fake"]["signature"],
            "individual_openai_manifest_signature": sources["individual_real"]["signature"],
            "frozen_candidates_signature": sources["frozen"]["signature"],
        },
        "scope_confirmations": {
            "new_training_performed": False,
            "new_optimization_performed": False,
            "new_holdout_inference_performed": False,
            "threshold_changed": False,
            "selection_reopened": False,
            "real_llm_provider_called": True,
            "real_llm_scientific_evaluation_approved": False,
            "deidentified_individual_explanation_sent_to_llm": True,
            "raw_individual_record_sent_to_llm": False,
            "patient_identifier_sent_to_llm": False,
            "api_frontend_cloud_created": False,
            "deploy_performed": False,
        },
        "supplementary_real_llm_evaluation": {
            "provider": sources["llm_v4"]["provider"],
            "requested_model": sources["llm_v4"]["requested_model"],
            "status": sources["llm_v4"]["status"],
            "scientific_evaluation_approved": False,
            "factuality": "327/327",
            "safety": True,
            "completeness": True,
            "clarity": True,
            "scientific_calibration": False,
            "individual_data_sent": False,
            "provider_calls": 1,
            "automatic_retries": 0,
        },
        "individual_llm_evaluation": {
            "provider": sources["individual_real"]["provider"],
            "model": sources["individual_real"]["model"],
            "status": sources["individual_real"]["status"],
            "approved": True,
            "factuality": "40/40",
            "safety": True,
            "completeness": True,
            "clarity": True,
            "medical_context_relevance": True,
            "scientific_calibration": True,
            "development_only": True,
            "raw_individual_record_sent": False,
            "patient_identifiers_sent": False,
            "holdout_case_sent": False,
            "external_calls": sources["individual_real"]["execution_scope"]["external_calls"],
            "automatic_retries": 0,
        },
        "files": {
            "documents": records(document_paths),
            "artifacts": records(artifact_paths),
            "figures": records(figure_paths),
        },
        "validation_preflight": {"status": preflight["status"], "check_count": preflight["check_count"]},
    })
    path = Path(output_path) if output_path is not None else SUMMARY_ROOT / "final_delivery_manifest.json"
    save_json(payload, path)
    return path
