from __future__ import annotations

import copy
import inspect
import json

import pytest

from tech_challenge_fase2.llm.safety import validate_safety
from tech_challenge_fase2.llm_individual import case_builder
from tech_challenge_fase2.llm_individual.case_builder import build_individual_input
from tech_challenge_fase2.llm_individual.engine import evaluate_existing, run
from tech_challenge_fase2.llm_individual.evaluation import evaluate
from tech_challenge_fase2.llm_individual.factuality import validate_factuality
from tech_challenge_fase2.llm_individual.privacy import validate_privacy
from tech_challenge_fase2.llm_individual.prompts import load_prompts
from tech_challenge_fase2.llm_individual.providers import (
    FakeIndividualProvider,
    OpenAIIndividualProvider,
    build_deterministic_output,
)
from tech_challenge_fase2.llm_individual.schemas import (
    IndividualSchemaError,
    output_json_schema,
    validate_input,
    validate_output,
)
from tech_challenge_fase2.llm.providers import LLMRequest


@pytest.fixture(scope="module")
def individual_input() -> dict:
    return build_individual_input()


@pytest.fixture(scope="module")
def individual_output(individual_input: dict) -> dict:
    return build_deterministic_output(individual_input)


def test_input_uses_only_development_and_is_deidentified(individual_input: dict) -> None:
    assert validate_input(individual_input) == individual_input
    assert validate_privacy(individual_input) == individual_input
    assert individual_input["case_context"]["source_scope"] == "development_only"
    assert individual_input["source_provenance"]["test_or_holdout_case_used"] is False
    assert individual_input["source_provenance"]["patient_identifiers_included"] is False


def test_input_contains_prediction_but_no_reconstructible_row(individual_input: dict) -> None:
    case = individual_input["case_context"]
    assert 0 <= case["probability_malignant"] <= 1
    assert case["raw_feature_values_included"] is False
    assert case["ground_truth_included"] is False
    assert case["source_record_reconstructible"] is False
    serialized = json.dumps(individual_input).lower()
    assert '"patient_id":' not in serialized
    assert '"dataset_index":' not in serialized
    assert '"final_predictions":' not in serialized


def test_exactly_five_typed_explanation_signals(individual_input: dict) -> None:
    signals = individual_input["explanation_signals"]
    assert len(signals) == 5
    assert [item["rank"] for item in signals] == [1, 2, 3, 4, 5]
    assert len({item["feature"] for item in signals}) == 5
    assert sum(item["relative_importance_percent"] for item in signals) == pytest.approx(100.0)


def test_builder_never_trains_or_reads_final_predictions() -> None:
    source = inspect.getsource(case_builder)
    assert ".fit(" not in source
    assert "final_predictions.json" not in source


def test_privacy_rejects_raw_features(individual_input: dict) -> None:
    unsafe = copy.deepcopy(individual_input)
    unsafe["raw_features"] = {"radius_mean": 10.0}
    with pytest.raises(ValueError):
        validate_privacy(unsafe)


def test_privacy_rejects_ground_truth_or_index_flags(individual_input: dict) -> None:
    unsafe = copy.deepcopy(individual_input)
    unsafe["case_context"]["ground_truth_included"] = True
    with pytest.raises(ValueError):
        validate_privacy(unsafe)
    unsafe = copy.deepcopy(individual_input)
    unsafe["source_provenance"]["original_row_index_included"] = True
    with pytest.raises(ValueError):
        validate_privacy(unsafe)


def test_output_schema_is_closed(individual_output: dict) -> None:
    assert validate_output(individual_output) == individual_output
    extra = copy.deepcopy(individual_output)
    extra["medical_diagnosis"] = "malignant"
    with pytest.raises(IndividualSchemaError):
        validate_output(extra)
    assert output_json_schema()["additionalProperties"] is False


def test_fake_provider_is_deterministic_and_offline(individual_input: dict) -> None:
    prompts = load_prompts()
    request = LLMRequest(
        input_payload=individual_input, system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text,
        model="deterministic-individual-explainer-v1",
    )
    first, second = FakeIndividualProvider(), FakeIndividualProvider()
    response_a, response_b = first.generate(request), second.generate(request)
    assert response_a.output == response_b.output
    assert response_a.usage["external_calls"] == 0
    assert response_a.usage["total_tokens"] == 0


def test_factuality_checks_all_individual_facts(individual_input: dict, individual_output: dict) -> None:
    report = validate_factuality(individual_output, individual_input)
    assert report["passed"] is True
    assert report["passed_checks"] == report["total_checks"] == 40


def test_factuality_rejects_changed_probability(individual_input: dict, individual_output: dict) -> None:
    wrong = copy.deepcopy(individual_output)
    wrong["classificacao_do_modelo"]["probability_malignant"] -= 0.1
    report = validate_factuality(wrong, individual_input)
    assert report["passed"] is False
    assert any(item["check"] == "classification.probability_malignant" and not item["passed"] for item in report["checks"])


def test_factuality_rejects_factor_assigned_to_wrong_rank(individual_input: dict, individual_output: dict) -> None:
    wrong = copy.deepcopy(individual_output)
    wrong["fatores_explicativos"][0]["feature"] = "radius_mean"
    report = validate_factuality(wrong, individual_input)
    assert report["passed"] is False


def test_safety_rejects_clinical_recommendation(individual_output: dict) -> None:
    unsafe = copy.deepcopy(individual_output)
    unsafe["conclusao"] = "Recomendo tratamento para este paciente."
    assert validate_safety(unsafe)["passed"] is False


def test_disclaimer_and_non_diagnosis_flags_are_mandatory(individual_output: dict) -> None:
    changed = copy.deepcopy(individual_output)
    changed["disclaimer"] = "Uso academico."
    with pytest.raises(IndividualSchemaError):
        validate_output(changed)
    changed = copy.deepcopy(individual_output)
    changed["predicao_nao_e_diagnostico"] = False
    with pytest.raises(IndividualSchemaError):
        validate_output(changed)


def test_quality_evaluation_covers_six_dimensions(individual_input: dict, individual_output: dict) -> None:
    report = evaluate(individual_output, individual_input)
    assert report["approved"] is True
    assert report["overall_score"] == 1.0
    assert set(report["dimensions"]) == {
        "factuality", "completeness", "clarity", "safety",
        "medical_context_relevance", "scientific_calibration",
    }


def test_calibration_accepts_explicit_non_causality_paraphrase(individual_input: dict, individual_output: dict) -> None:
    paraphrased = copy.deepcopy(individual_output)
    for factor in paraphrased["fatores_explicativos"]:
        factor["explanation"] = factor["explanation"].replace(
            "Essa associação não demonstra causa biológica.",
            "O peso local é matemático, sem indicar causalidade biológica.",
        )
    report = evaluate(paraphrased, individual_input)
    assert report["dimensions"]["scientific_calibration"]["checks"]["no_causal_claim"] is True
    assert report["approved"] is True


def test_prompt_engineering_is_versioned_and_medical(individual_input: dict) -> None:
    prompts = load_prompts()
    assert prompts.system_version == "system_individual_v1"
    assert prompts.explanation_version == "explanation_individual_v1"
    assert len(prompts.system_sha256) == len(prompts.explanation_sha256) == 64
    assert "CONTEXTO MEDICO" in prompts.system_text
    assert "diagnostico" in prompts.system_text.lower()


def test_openai_request_uses_structured_output_without_temperature(individual_input: dict) -> None:
    prompts = load_prompts()
    request = LLMRequest(
        input_payload=individual_input, system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model="gpt-5.5", max_output_tokens=5000,
    )
    body = OpenAIIndividualProvider.request_body(request)
    assert body["store"] is False
    assert "temperature" not in body
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["name"] == "individual_model_explanation_v1"


def test_openai_provider_extracts_text_from_typed_parser(monkeypatch, individual_input: dict) -> None:
    prompts = load_prompts()
    request = LLMRequest(
        input_payload=individual_input, system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model="gpt-5.5", max_output_tokens=5000,
    )
    expected = build_deterministic_output(individual_input)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "id": "resp_test", "status": "completed", "model": "gpt-5.5-test",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(expected)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            }).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    provider = OpenAIIndividualProvider(api_key="offline-test-key")
    response = provider.generate(request)
    assert response.output == expected
    assert response.response_id == "resp_test"
    assert response.usage["total_tokens"] == 30


def test_engine_generates_complete_offline_artifacts(tmp_path) -> None:
    root = tmp_path / "individual"
    provider = FakeIndividualProvider()
    result = run(artifact_root=root, provider=provider)
    assert result["approved"] is True
    assert provider.call_count == 1
    expected = {
        "individual_input_snapshot.json", "individual_output.json", "privacy_report.json",
        "factuality_report.json", "safety_report.json", "evaluation_report.json",
        "provider_usage.json", "individual_explanation_manifest.json",
    }
    assert expected == {path.name for path in root.iterdir()}
    reevaluation = evaluate_existing(artifact_root=root)
    assert reevaluation["approved"] is True
    assert reevaluation["manifest_valid"] is True
    assert reevaluation["privacy_valid"] is True


def test_engine_is_idempotent_without_second_provider_call(tmp_path) -> None:
    root = tmp_path / "individual"
    provider = FakeIndividualProvider()
    first = run(artifact_root=root, provider=provider)
    second = run(artifact_root=root, provider=provider)
    assert first == second
    assert provider.call_count == 1


def test_future_text_contract_is_ready_but_empty(individual_input: dict, individual_output: dict) -> None:
    future = individual_input["future_text_integration"]
    assert future["contract_ready"] is True
    assert future["text_data_included"] is False
    assert future["planned_fields"] == ["clinical_note_summary", "exam_report_summary"]
    assert individual_output["preparacao_modulo3"]["ready_for_future_text"] is True
    assert individual_output["preparacao_modulo3"]["current_text_data_used"] is False
