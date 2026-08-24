import json
from pathlib import Path

import pytest

from tech_challenge_fase2.llm.engine import (
    EVALUATION_NAME, FACTUALITY_NAME, INPUT_NAME, MANIFEST_NAME, OUTPUT_NAME,
    SAFETY_NAME, evaluate_existing_output, run_evaluation,
)
from tech_challenge_fase2.llm.input_builder import FINAL_ROOT, file_sha256
from tech_challenge_fase2.llm.providers import FakeLLMProvider


def test_offline_execution_generates_signed_consistent_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    root = tmp_path / "llm"
    result = run_evaluation(artifact_root=root)
    assert result["approved"] is True
    expected = {INPUT_NAME, OUTPUT_NAME, FACTUALITY_NAME, SAFETY_NAME, EVALUATION_NAME, MANIFEST_NAME}
    assert expected.issubset({path.name for path in root.iterdir()})
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    assert manifest["individual_data_sent"] is False
    assert manifest["secrets_recorded"] is False
    assert manifest["ga_executed"] is False
    assert manifest["randomized_search_executed"] is False
    assert manifest["selection_reopened"] is False
    for item in manifest["files"]:
        assert file_sha256(root / item["filename"]) == item["sha256"]
    assert evaluate_existing_output(root)["approved"] is True


def test_same_identity_is_idempotent_and_does_not_call_provider_twice(tmp_path: Path) -> None:
    provider = FakeLLMProvider()
    root = tmp_path / "llm"
    first = run_evaluation(artifact_root=root, provider=provider)
    second = run_evaluation(artifact_root=root, provider=provider)
    assert first == second
    assert provider.call_count == 1


def test_different_identity_is_not_silently_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "llm"
    run_evaluation(artifact_root=root)
    from tech_challenge_fase2.llm.engine import ManualInterventionRequired
    with pytest.raises(ManualInterventionRequired, match="identidade"):
        run_evaluation(artifact_root=root, model="another-model")


def test_mission4_artifacts_remain_byte_identical(tmp_path: Path) -> None:
    names = ("final_test_results.json", "uncertainty_results.json", "final_manifest.json", "final_evaluation_plan.json")
    before = {name: file_sha256(FINAL_ROOT / name) for name in names}
    run_evaluation(artifact_root=tmp_path / "llm")
    after = {name: file_sha256(FINAL_ROOT / name) for name in names}
    assert before == after


def test_llm_source_does_not_import_training_or_search_components() -> None:
    root = Path(__file__).parents[1] / "src" / "tech_challenge_fase2" / "llm"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    forbidden = ("RandomizedSearchCV", "run_genetic", "fit(", "final_predictions.json", "sklearn")
    assert all(term not in source for term in forbidden)
