import inspect
import json
from pathlib import Path

from tech_challenge_fase2 import deliverable
from tech_challenge_fase2.deliverable import (
    MODEL_ORDER, build_master_rows, broken_local_links, generate_master_table,
    generate_delivery_manifest, generate_presentation_figures, validate_deliverable,
)
from tech_challenge_fase2.genetic.serialization import stable_sha256


def test_master_rows_match_all_nine_frozen_candidate_results() -> None:
    sources = deliverable.load_authoritative_sources()
    rows = build_master_rows(sources)
    assert len(rows) == 9
    assert {(row["family"], row["method"]) for row in rows} == {
        (model, method) for model in MODEL_ORDER for method in deliverable.METHOD_ORDER
    }
    source = {(item["model"], item["method"]): item for item in sources["results"]["candidate_results"]}
    for row in rows:
        candidate = source[(row["family"], row["method"])]
        assert row["recall_test"] == candidate["metrics"]["recall_malignant"]
        assert row["false_negatives"] == candidate["metrics"]["false_negatives"]
        assert row["recall_cv"] == candidate["cv_metrics"]["mean_recall_malignant"]


def test_master_table_serialization_and_selection_status(tmp_path: Path) -> None:
    csv_path, json_path = generate_master_table(tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    assert payload["signature"] == stable_sha256(unsigned)
    assert payload["row_count"] == 9
    assert payload["data_scope"]["new_holdout_inference_performed"] is False
    global_rows = [row for row in payload["rows"] if row["selection_status"] == "global_selected"]
    assert [(row["family"], row["method"]) for row in global_rows] == [("logistic_regression", "random_search")]
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 10


def test_presentation_figures_are_aggregate_and_complete(tmp_path: Path) -> None:
    qa_path = generate_presentation_figures(tmp_path)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["source_scope"] == "aggregate_artifacts_only"
    assert qa["new_training_performed"] is False
    assert qa["new_inference_performed"] is False
    assert len(qa["figures"]) == 6
    assert all((tmp_path / item["filename"]).stat().st_size > 20_000 for item in qa["figures"])


def test_deliverable_code_cannot_train_infer_search_or_call_network() -> None:
    source = inspect.getsource(deliverable)
    forbidden = (
        "import sklearn", "from sklearn", "RandomizedSearchCV(", ".fit(", ".predict(", ".predict_proba(",
        "final_predictions.json", "urllib", "requests.", "OpenAIResponsesProvider",
    )
    assert all(term not in source for term in forbidden)


def test_all_local_markdown_links_resolve() -> None:
    root = Path(__file__).parents[1]
    documents = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    assert broken_local_links(documents) == []


def test_deliverable_preflight_validates_frozen_evidence() -> None:
    report = validate_deliverable(require_manifest=False)
    assert report["passed"] is True, [item for item in report["checks"] if not item["passed"]]
    assert report["check_count"] >= 20


def test_final_delivery_manifest_is_signed_and_preserves_scope(tmp_path: Path) -> None:
    path = generate_delivery_manifest(test_count=182, output_path=tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    assert payload["signature"] == stable_sha256(unsigned)
    assert payload["status"] == "completed"
    assert payload["scope_confirmations"] == {
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
        "http_api_created": False,
        "cloud_configuration_created": True,
        "cloud_resources_provisioned": False,
        "read_only_dashboard_created": True,
        "deploy_performed": False,
    }
    assert payload["supplementary_real_llm_evaluation"] == {
        "provider": "openai_responses",
        "requested_model": "gpt-5.5",
        "status": "methodologically_complete_not_approved",
        "scientific_evaluation_approved": False,
        "factuality": "327/327",
        "safety": True,
        "completeness": True,
        "clarity": True,
        "scientific_calibration": False,
        "individual_data_sent": False,
        "provider_calls": 1,
        "automatic_retries": 0,
    }
    assert payload["individual_llm_evaluation"] == {
        "provider": "openai_responses",
        "model": "gpt-5.5-2026-04-23",
        "status": "approved",
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
        "external_calls": 2,
        "automatic_retries": 0,
    }
