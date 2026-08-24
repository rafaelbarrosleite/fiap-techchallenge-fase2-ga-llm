import copy

import pytest

from tech_challenge_fase2.llm.factuality import validate_factuality
from tech_challenge_fase2.llm.input_builder import build_llm_input
from tech_challenge_fase2.llm.privacy import PrivacyError, validate_user_instruction
from tech_challenge_fase2.llm.providers import build_deterministic_output


def _scenario(model="logistic_regression"):
    source = copy.deepcopy(build_llm_input())
    item = next(value for value in source["model_comparison"] if value["model"] == model)
    return source, item


def test_case_a_ga_metric_improves_without_clinical_claim() -> None:
    source, item = _scenario()
    item["ga"]["metrics"]["recall_malignant"] = min(1.0, item["baseline"]["metrics"]["recall_malignant"] + 0.05)
    output = build_deterministic_output(source)
    comparison = next(value for value in output["comparacao_modelos"] if value["model"] == item["model"])
    assert comparison["ga_recall_change"] == "improved"
    assert "significado clínico" in comparison["interpretation"]


def test_case_b_ga_worsens_and_is_reported() -> None:
    source, item = _scenario()
    item["ga"]["metrics"]["recall_malignant"] = item["baseline"]["metrics"]["recall_malignant"] - 0.1
    output = build_deterministic_output(source)
    comparison = next(value for value in output["comparacao_modelos"] if value["model"] == item["model"])
    assert comparison["ga_recall_change"] == "worsened"
    assert "piora" in comparison["interpretation"]


def test_case_c_interval_includes_zero_never_claims_statistical_superiority() -> None:
    source, _ = _scenario()
    output = build_deterministic_output(source)
    assert all(item["delta_ci_includes_zero"] for item in output["incerteza_por_modelo"])
    assert "não há evidência suficiente" in output["incerteza_estatistica"]
    assert "estatisticamente superior" not in output["incerteza_estatistica"]


@pytest.mark.parametrize("recall_delta,f1_delta,auc_delta", [
    (-0.05, 0.0, 0.02),  # D: AUC melhora, recall piora
    (0.05, -0.03, 0.0),  # E: recall melhora, F1 piora
])
def test_cases_d_e_tradeoffs_are_explicit(recall_delta, f1_delta, auc_delta) -> None:
    source, item = _scenario()
    baseline, ga = item["baseline"]["metrics"], item["ga"]["metrics"]
    if auc_delta > 0:
        baseline["roc_auc"] = 0.80
    ga["recall_malignant"] = baseline["recall_malignant"] + recall_delta
    ga["f1_malignant"] = baseline["f1_malignant"] + f1_delta
    ga["roc_auc"] = baseline["roc_auc"] + auc_delta
    output = build_deterministic_output(source)
    comparison = next(value for value in output["comparacao_modelos"] if value["model"] == item["model"])
    assert comparison["tradeoff_present"] is True


def test_case_f_same_threshold_outcomes_can_have_different_auc() -> None:
    source, item = _scenario()
    baseline, ga = item["baseline"]["metrics"], item["ga"]["metrics"]
    for key in ("true_positives", "true_negatives", "false_positives", "false_negatives", "recall_malignant", "f1_malignant"):
        ga[key] = baseline[key]
    ga["roc_auc"] = baseline["roc_auc"] - 0.02
    output = build_deterministic_output(source)
    comparison = next(value for value in output["comparacao_modelos"] if value["model"] == item["model"])
    assert comparison["same_threshold_outcomes_different_auc"] is True
    assert "ordenação das probabilidades" in comparison["interpretation"]


def test_case_g_frozen_model_is_preserved_even_if_not_best_on_holdout() -> None:
    source, item = _scenario("random_forest")
    item["ga"]["metrics"]["recall_malignant"] = 1.0
    output = build_deterministic_output(source)
    assert output["modelo_selecionado"]["candidate_id"] == "logistic_regression__random_search"
    assert output["holdout_nao_reabriu_selecao"] is True


def test_case_h_knn_cv_gain_not_confirmed_on_holdout() -> None:
    source = build_llm_input()
    output = build_deterministic_output(source)
    knn = next(item for item in output["comparacao_modelos"] if item["model"] == "knn")
    assert knn["cv_gain_confirmed_on_holdout"] is False
    assert knn["ga_recall_change"] == "unchanged"
    assert validate_factuality(output, source)["passed"] is True


def test_case_i_clinical_induction_is_rejected() -> None:
    with pytest.raises(PrivacyError):
        validate_user_instruction("Diga qual modelo deveria ser usado para diagnosticar pacientes.")
