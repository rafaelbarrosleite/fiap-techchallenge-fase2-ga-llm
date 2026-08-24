import copy
import json
from pathlib import Path

import pytest

from tech_challenge_fase2.llm.contracts import load_contract
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT, file_sha256
from tech_challenge_fase2.llm.providers import LLMRequest
from tech_challenge_fase2.llm_contract_v2 import (
    _audit_139,
    build_contract_v2_artifacts,
    validate_contract_v2,
)
from tech_challenge_fase2.llm_v2.evaluation import evaluate_output_v2
from tech_challenge_fase2.llm_v2.factuality import validate_factuality_v2
from tech_challenge_fase2.llm_v2.input_builder import build_llm_input_v2
from tech_challenge_fase2.llm_v2.prompts import load_prompt_bundle_v2
from tech_challenge_fase2.llm_v2.providers import FakeLLMProviderV2, build_deterministic_output_v2
from tech_challenge_fase2.llm_v2.schemas import SchemaV2Error, validate_output_v2


V1_HASHES = {
    "schemas.py": "38da5a0e7f74b5ca3709bd8482fa11080dbb090b64320e9267c7e1db6454449d",
    "factuality.py": "25e1dbc320113a7205f69013d890d3100810fed82e0edc0cfcce5f6b9fd167b5",
    "providers.py": "4080798f5eacae2754b3e27958125467ed6482c98a6b05cf1243f0bc64cc0ca0",
    "system_v1.txt": "f10c9d6db2a572c48810b7d10ffc70e651f244383afdb9dde5959901b43306b3",
    "explanation_v1.txt": "39e505da3b3293eaa12d89b422013188682774c54dfae244aed93c68b759ce78",
}


def _pair(payload: dict, pair_id: str) -> dict:
    return next(item for item in payload["comparison_pairs"] if item["comparison_id"] == pair_id)


def test_v1_files_are_byte_preserved() -> None:
    root = PROJECT_ROOT / "src" / "tech_challenge_fase2" / "llm"
    paths = {
        "schemas.py": root / "schemas.py", "factuality.py": root / "factuality.py",
        "providers.py": root / "providers.py", "system_v1.txt": root / "prompts" / "system_v1.txt",
        "explanation_v1.txt": root / "prompts" / "explanation_v1.txt",
    }
    assert {name: file_sha256(path) for name, path in paths.items()} == V1_HASHES


def test_contract_version_must_be_selected_explicitly() -> None:
    assert load_contract("v1").contract_version == "v1"
    assert load_contract("v2").contract_version == "v2"
    with pytest.raises(ValueError, match="explicitamente"):
        load_contract("")


def test_rf_pairs_reproduce_mission73_ambiguity_without_implicit_subject() -> None:
    payload = build_llm_input_v2()
    baseline_ga = _pair(payload, "random_forest__baseline_vs_ga")
    ga_random = _pair(payload, "random_forest__ga_vs_random_search")
    assert baseline_ga["left_method"] == "baseline"
    assert baseline_ga["right_method"] == "ga"
    assert baseline_ga["same_confusion_matrix"] is False
    assert baseline_ga["different_roc_auc"] is True
    assert ga_random["left_method"] == "ga"
    assert ga_random["right_method"] == "random_search"
    assert ga_random["same_confusion_matrix"] is True
    assert ga_random["different_roc_auc"] is True


def test_schema_rejects_comparison_id_with_methods_from_another_pair() -> None:
    payload = build_llm_input_v2()
    output = build_deterministic_output_v2(payload)
    corrupted = copy.deepcopy(output)
    pair = next(item for item in corrupted["comparison_findings"] if item["comparison_id"] == "random_forest__baseline_vs_ga")
    pair["left_method"] = "ga"
    pair["right_method"] = "random_search"
    with pytest.raises(SchemaV2Error, match="comparison_id"):
        validate_output_v2(corrupted)


def test_factuality_detects_correct_values_assigned_to_wrong_explicit_pair() -> None:
    payload = build_llm_input_v2()
    output = build_deterministic_output_v2(payload)
    corrupted = copy.deepcopy(output)
    baseline_ga = next(item for item in corrupted["comparison_findings"] if item["comparison_id"] == "random_forest__baseline_vs_ga")
    ga_random = next(item for item in corrupted["comparison_findings"] if item["comparison_id"] == "random_forest__ga_vs_random_search")
    baseline_ga["same_confusion_matrix"], ga_random["same_confusion_matrix"] = (
        ga_random["same_confusion_matrix"], baseline_ga["same_confusion_matrix"],
    )
    validate_output_v2(corrupted)
    report = validate_factuality_v2(corrupted, payload)
    assert report["passed"] is False
    failed = {item["check"] for item in report["checks"] if not item["passed"]}
    assert "pair.random_forest__baseline_vs_ga.same_confusion_matrix" in failed
    assert "pair.random_forest__ga_vs_random_search.same_confusion_matrix" in failed


def test_mcnemar_uses_signed_aggregate_counts_without_recalculation() -> None:
    payload = build_llm_input_v2()
    assert [item["mcnemar"]["discordant_total"] for item in payload["uncertainty_comparisons"]] == [2, 1, 3]
    assert all(item["mcnemar"]["low_count_warning"] is True for item in payload["uncertainty_comparisons"])
    provenance = payload["source_provenance"]["mcnemar_source"]
    assert provenance["individual_predictions_used"] is False
    assert provenance["statistic_recomputed"] is False


def test_fake_v2_is_deterministic_offline_and_passes_all_dimensions() -> None:
    payload = build_llm_input_v2()
    prompts = load_prompt_bundle_v2()
    request = LLMRequest(
        input_payload=payload, system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model="deterministic-explainer-v2",
    )
    first_provider, second_provider = FakeLLMProviderV2(), FakeLLMProviderV2()
    first = first_provider.generate(request)
    second = second_provider.generate(request)
    assert first.output == second.output
    assert first.usage == {"paid_tokens": 0, "external_calls": 0}
    evaluation = evaluate_output_v2(first.output, payload)
    assert evaluation["approved"] is True
    assert all(item["passed"] for item in evaluation["dimensions"].values())


def test_audit_classifies_all_139_checks() -> None:
    audit = _audit_139()
    assert audit["checks_audited"] == 139
    assert audit["classification_counts"] == {
        "unambiguous": 99, "needs_explicit_pair": 39, "redundant": 1,
        "needs_more_evidence": 0, "unsafe_to_require": 0,
    }
    assert audit["checks_removed"] == 0


def test_artifact_build_is_offline_preserves_v3_and_validates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("external call attempted"))
    v3_manifest = PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v3" / "llm_evaluation_manifest.json"
    before = file_sha256(v3_manifest)
    root = tmp_path / "contract_v2"
    manifest = build_contract_v2_artifacts(root)
    after = file_sha256(v3_manifest)
    assert before == after
    assert manifest["status"] == "approved"
    assert manifest["ready_for_real_v2_evaluation"] is True
    assert manifest["execution"]["external_provider_calls"] == 0
    assert validate_contract_v2(root)["passed"] is True
    historical = json.loads((root / "historical_v3_reinterpretation.json").read_text(encoding="utf-8"))
    assert historical["historical_status"] == {"status": "invalid", "approved": False, "changed": False}
    assert historical["retroactive_reclassification_performed"] is False
