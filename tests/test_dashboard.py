"""Testes do painel estatico somente leitura."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tech_challenge_fase2.dashboard.model import DashboardDataError, load_dashboard_data
from tech_challenge_fase2.dashboard.privacy import (
    DashboardPrivacyError,
    assert_html_has_no_individual_data,
)
from tech_challenge_fase2.dashboard.render import render_dashboard
from tech_challenge_fase2.run_dashboard import build_dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def html() -> str:
    return render_dashboard(load_dashboard_data())


# --- privacidade -----------------------------------------------------------------


def test_rendered_dashboard_carries_no_individual_data(html: str) -> None:
    assert_html_has_no_individual_data(html)


@pytest.mark.parametrize(
    "trecho",
    [
        "<td>patient_id</td>",
        "<td>ground_truth: M</td>",
        "<td>observed_value: 0.4601</td>",
        "<td>17.9900, 10.3800, 122.8000, 1001.0000, 0.11840, 0.27760, 0.30010, 0.14710, 0.24190</td>",
    ],
)
def test_guard_rejects_individual_data(trecho: str) -> None:
    with pytest.raises(DashboardPrivacyError):
        assert_html_has_no_individual_data(trecho)


def test_guard_accepts_a_confirmation_that_nothing_was_sent() -> None:
    """`patient_identifier_sent_to_llm` afirma ausencia; bloquea-lo seria absurdo."""

    assert_html_has_no_individual_data(
        "<td>patient_identifier_sent_to_llm</td><td>não ocorreu</td>"
    )


def test_guard_accepts_a_named_signal_with_band_and_importance() -> None:
    """Nomear o sinal e o que torna a explicacao util; o valor medido e que nao vai."""

    assert_html_has_no_individual_data(
        "<td>maior simetria</td><td>elevada</td><td>55,86%</td>"
    )


# --- ausencia de rede e de inferencia --------------------------------------------


def test_dashboard_makes_no_network_request(html: str) -> None:
    for esquema in ("http://", "https://", "//cdn", "src=\"/"):
        assert esquema not in html, f"o painel referencia recurso externo: {esquema}"
    assert "data:image/png;base64," in html, "as figuras precisam estar embutidas"


def test_dashboard_source_never_trains_or_infers() -> None:
    root = PROJECT_ROOT / "src" / "tech_challenge_fase2" / "dashboard"
    fonte = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))
    for proibido in ("RandomizedSearchCV", "run_genetic", ".fit(", "predict_proba(", "urllib"):
        assert proibido not in fonte


# --- determinismo ----------------------------------------------------------------


def test_build_is_deterministic(tmp_path: Path) -> None:
    """Regerar o painel nao pode alterar bytes, ou reselar a entrega viraria rotina."""

    primeiro = build_dashboard(tmp_path / "a.html").read_bytes()
    segundo = build_dashboard(tmp_path / "b.html").read_bytes()
    assert primeiro == segundo


def test_dashboard_has_no_timestamp(html: str) -> None:
    assert not re.search(r"20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}", html)


# --- fidelidade ao artefato ------------------------------------------------------


def test_dashboard_shows_the_frozen_winner(html: str) -> None:
    data = load_dashboard_data()
    assert data.selected_row["family"] == "logistic_regression"
    assert data.selected_row["method"] == "random_search"
    assert "global_selected" in html


def test_false_negative_transitions_match_the_master_table() -> None:
    data = load_dashboard_data()
    assert data.false_negative_transitions == [
        ("Regressão Logística", 3, 1),
        ("Random Forest", 4, 3),
        ("KNN", 4, 4),
    ]


def test_all_factuality_checks_are_rendered(html: str) -> None:
    data = load_dashboard_data()
    total = len(data.llm_factuality["checks"])
    assert f"{total}/{total}" in html
    for check in data.llm_factuality["checks"][:5]:
        assert check["check"] in html


def test_mandatory_disclaimer_is_present(html: str) -> None:
    assert "não foram validados para uso clínico" in html
    assert "Aviso" in html


def test_scope_confirmations_reflect_the_repository(html: str) -> None:
    """As garantias protetivas continuam falsas; o que foi criado aparece como fato."""

    manifesto = json.loads(
        (PROJECT_ROOT / "artifacts" / "final_summary" / "final_delivery_manifest.json")
        .read_text(encoding="utf-8")
    )
    confirmacoes = manifesto["scope_confirmations"]
    for garantia in (
        "new_training_performed", "new_optimization_performed", "selection_reopened",
        "threshold_changed", "raw_individual_record_sent_to_llm",
        "patient_identifier_sent_to_llm", "cloud_resources_provisioned",
        "deploy_performed", "http_api_created",
    ):
        assert confirmacoes[garantia] is False, garantia
    # O repositorio contem configuracao de nuvem e este painel; negar isso seria falso.
    assert confirmacoes["cloud_configuration_created"] is True
    assert confirmacoes["read_only_dashboard_created"] is True


def test_missing_artifact_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tech_challenge_fase2.dashboard.model as model

    monkeypatch.setattr(model, "PROJECT_ROOT", tmp_path)
    with pytest.raises(DashboardDataError):
        model.load_dashboard_data()


def test_committed_dashboard_matches_a_fresh_build(tmp_path: Path) -> None:
    """O HTML versionado nao pode ficar defasado em relacao aos artefatos.

    O painel e versionado para que um avaliador o abra direto do clone. Sem esta
    verificacao, uma mudanca de artefato deixaria o documento comitado exibindo
    numeros antigos sem que nada falhasse -- o mesmo modo de falha que o guarda
    do notebook evita.
    """

    comitado = PROJECT_ROOT / "reports" / "dashboard" / "index.html"
    assert comitado.is_file(), "o painel versionado esta ausente; rode uv run build-dashboard"
    novo = build_dashboard(tmp_path / "novo.html")
    assert comitado.read_bytes() == novo.read_bytes(), (
        "o painel versionado divergiu dos artefatos; rode uv run build-dashboard"
    )
