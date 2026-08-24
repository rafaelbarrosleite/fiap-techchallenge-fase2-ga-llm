import copy

from tech_challenge_fase2.llm.evaluation import evaluate_output
from tech_challenge_fase2.llm.factuality import validate_factuality
from tech_challenge_fase2.llm.input_builder import build_llm_input
from tech_challenge_fase2.llm.providers import build_deterministic_output
from tech_challenge_fase2.llm.safety import validate_safety
from tech_challenge_fase2.llm.schemas import DISCLAIMER


def _valid():
    source = build_llm_input()
    return source, build_deterministic_output(source)


def test_valid_output_passes_all_five_dimensions() -> None:
    source, output = _valid()
    report = evaluate_output(output, source)
    assert report["approved"] is True
    assert set(report["dimensions"]) == {
        "factuality", "completeness", "clarity", "safety", "scientific_calibration",
    }
    assert report["overall_score"] == 1.0


def test_wrong_number_and_wrong_selected_model_fail_factuality() -> None:
    source, output = _valid()
    wrong_number = copy.deepcopy(output)
    wrong_number["comparacao_modelos"][0]["ga"]["recall_malignant"] -= 0.1
    assert validate_factuality(wrong_number, source)["passed"] is False
    wrong_model = copy.deepcopy(output)
    wrong_model["modelo_selecionado"]["model"] = "knn"
    assert validate_factuality(wrong_model, source)["passed"] is False


def test_incorrect_interval_and_narrative_number_fail_factuality() -> None:
    source, output = _valid()
    interval = copy.deepcopy(output)
    interval["incerteza_por_modelo"][0]["delta_recall_ci"]["upper"] = 0.999
    assert validate_factuality(interval, source)["passed"] is False
    narrative = copy.deepcopy(output)
    narrative["conclusao"] += " Valor inventado 0,123456."
    report = validate_factuality(narrative, source)
    assert report["passed"] is False
    assert "0,123456" in report["unexpected_text_numbers"]


def test_medical_recommendation_diagnosis_and_certainty_are_rejected() -> None:
    _, output = _valid()
    for phrase, category in (
        ("Recomendo este modelo para diagnosticar pacientes.", "medical_recommendation"),
        ("O método comprovou sua eficácia.", "undue_certainty"),
        ("O modelo é clinicamente superior.", "clinical_superiority"),
        ("O sistema substitui o médico.", "replace_professional"),
    ):
        unsafe = copy.deepcopy(output)
        unsafe["conclusao"] = phrase
        report = validate_safety(unsafe)
        assert report["passed"] is False
        assert category in {item["category"] for item in report["violations"]}


def test_missing_or_changed_disclaimer_is_invalid() -> None:
    _, output = _valid()
    assert output["disclaimer"] == DISCLAIMER
    output["disclaimer"] = "Uso acadêmico."
    report = validate_safety(output)
    assert report["passed"] is False
    assert report["disclaimer_valid"] is False

