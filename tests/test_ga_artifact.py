import json

from tech_challenge_fase2.run_genetic import run_smoke_test
from tech_challenge_fase2.genetic.serialization import validate_run_artifact


def test_smoke_command_writes_valid_development_only_json(tmp_path) -> None:
    output = tmp_path / "smoke.json"
    artifact = run_smoke_test(
        model_name="logistic_regression",
        seed=42,
        output_path=output,
        log_path=tmp_path / "smoke.log",
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    validate_run_artifact(loaded)
    assert loaded == artifact
    assert loaded["data_scope"]["holdout_used"] is False
    assert loaded["data_scope"]["cv_splits"] == 5
    assert len(loaded["run"]["history"]) == 3
    assert loaded["run"]["total_unique_evaluations"] <= 12
    assert loaded["run"]["total_model_fits"] == (
        loaded["run"]["total_unique_evaluations"] * 5
    )
    serialized = json.dumps(loaded).lower()
    assert "test_accuracy" not in serialized
    assert "test_recall" not in serialized
