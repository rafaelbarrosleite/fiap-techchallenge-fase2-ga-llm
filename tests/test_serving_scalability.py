"""Testes da camada de escalabilidade, monitoramento e servico congelado."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_challenge_fase2.serving.autoscaling import (
    COOLDOWN,
    HOLD,
    SCALE_DOWN,
    SCALE_UP,
    AutoscalerState,
    AutoscalingError,
    AutoscalingPolicy,
)
from tech_challenge_fase2.serving.model_server import (
    DECISION_THRESHOLD,
    SELECTED_CANDIDATE,
    FrozenModelServer,
    resolve_frozen_model,
)
from tech_challenge_fase2.serving.monitoring import (
    FORBIDDEN_EVENT_KEYS,
    MonitoringError,
    PerformanceMonitor,
    assert_event_is_aggregate,
    latency_summary,
    percentile,
)
from tech_challenge_fase2.serving.validation import validate_scalability_report

POLICY = AutoscalingPolicy(min_workers=1, max_workers=4, target_backlog_per_worker=4)


# --- politica de escalabilidade -------------------------------------------------


def test_policy_scales_up_on_burst_and_down_after_drain() -> None:
    state = AutoscalerState(policy=POLICY)
    assert state.workers == 1

    burst = state.observe(backlog=40, now=1.0)
    assert burst.action == SCALE_UP and state.workers == POLICY.max_workers

    drained = state.observe(backlog=2, now=2.0)
    assert drained.action == SCALE_DOWN and state.workers == POLICY.min_workers


def test_policy_holds_inside_the_dead_band() -> None:
    """A faixa morta evita trocar de tamanho a cada oscilacao do backlog."""

    state = AutoscalerState(policy=POLICY)
    state.observe(backlog=40, now=0.0)
    workers = state.workers
    # 4 workers com backlog 16 => 4 por worker: entre a margem de descida (2)
    # e a de subida (6), logo nada deve mudar.
    held = state.observe(backlog=16, now=1.0)
    assert held.action == HOLD and state.workers == workers


def test_cooldown_defers_a_change_without_losing_the_intent() -> None:
    state = AutoscalerState(policy=AutoscalingPolicy(min_workers=1, max_workers=4, cooldown_seconds=30.0))
    state.observe(backlog=40, now=0.0)
    blocked = state.observe(backlog=0, now=1.0)
    assert blocked.action == COOLDOWN
    assert blocked.desired_workers < blocked.workers
    assert state.workers == 4


def test_policy_never_exceeds_its_bounds() -> None:
    state = AutoscalerState(policy=POLICY)
    state.observe(backlog=10_000, now=0.0)
    assert state.workers == POLICY.max_workers
    state.observe(backlog=0, now=1.0)
    assert state.workers == POLICY.min_workers


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_workers": 0},
        {"min_workers": 4, "max_workers": 2},
        {"target_backlog_per_worker": 0},
        {"scale_up_backlog_per_worker": 2, "scale_down_backlog_per_worker": 4},
        {"cooldown_seconds": -1.0},
    ],
)
def test_incoherent_policies_are_rejected(kwargs: dict) -> None:
    with pytest.raises(AutoscalingError):
        AutoscalingPolicy(**kwargs)


def test_decision_history_is_auditable() -> None:
    state = AutoscalerState(policy=POLICY)
    state.observe(backlog=40, now=0.0)
    state.observe(backlog=0, now=1.0)
    assert [record["action"] for record in state.history] == [SCALE_UP, SCALE_DOWN]
    assert all("backlog" in record for record in state.history)


# --- monitoramento sem dado individual ------------------------------------------


def test_monitor_accepts_aggregate_events() -> None:
    monitor = PerformanceMonitor()
    monitor.record("cycle_completed", workers=4, arrivals=12, cycle_latency_ms=8.5)
    assert monitor.of_type("cycle_completed")[0]["workers"] == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"probability": 0.97},
        {"predictions": [1, 0, 1]},
        {"patient_id": "abc"},
        {"nested": {"diagnosis": "M"}},
        {"batches": [{"target": 1}]},
    ],
)
def test_monitor_rejects_individual_data(payload: dict) -> None:
    monitor = PerformanceMonitor()
    with pytest.raises(MonitoringError):
        monitor.record("cycle_completed", **payload)


def test_monitor_rejects_long_series_that_could_rebuild_per_record_output() -> None:
    monitor = PerformanceMonitor()
    with pytest.raises(MonitoringError):
        monitor.record("cycle_completed", serie=list(range(200)))


def test_forbidden_keys_cover_identity_target_and_per_record_output() -> None:
    for key in ("patient_id", "diagnosis", "target", "features", "probability", "predictions"):
        assert key in FORBIDDEN_EVENT_KEYS


def test_monitor_writes_one_json_line_per_event(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    monitor = PerformanceMonitor(log_path=log_path)
    monitor.record("cycle_completed", workers=1)
    monitor.record("cycle_completed", workers=2)
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [item["workers"] for item in lines] == [1, 2]
    for line in lines:
        assert_event_is_aggregate(line)


# --- estatisticas de latencia ---------------------------------------------------


def test_percentiles_interpolate_and_respect_order() -> None:
    values = [10, 20, 30, 40]
    assert percentile(values, 0.0) == 10
    assert percentile(values, 1.0) == 40
    assert percentile(values, 0.5) == 25


def test_latency_summary_is_monotonic_across_percentiles() -> None:
    summary = latency_summary([5, 10, 15, 20, 100])
    assert summary["min_ms"] <= summary["p50_ms"] <= summary["p95_ms"] <= summary["p99_ms"]
    assert summary["p99_ms"] <= summary["max_ms"]
    assert summary["count"] == 5


def test_empty_latency_sample_is_rejected() -> None:
    with pytest.raises(MonitoringError):
        latency_summary([])


# --- servico sobre o modelo congelado -------------------------------------------


def test_server_serves_the_frozen_winner_without_reopening_selection() -> None:
    reference = resolve_frozen_model()
    assert reference.candidate_id == SELECTED_CANDIDATE
    assert reference.threshold == DECISION_THRESHOLD


def test_server_returns_only_aggregate_counts() -> None:
    from tech_challenge_fase2.config import DEFAULT_DATA_PATH
    from tech_challenge_fase2.data import load_dataset, split_development_test

    X, y = load_dataset(Path(DEFAULT_DATA_PATH))
    features = split_development_test(X, y).X_development.iloc[:16]
    outcome = FrozenModelServer().predict_batch(features)

    assert outcome.batch_size == 16
    assert 0 <= outcome.positive_count <= 16
    # A saida do servico nao expoe probabilidade ou classe por registro.
    assert_event_is_aggregate(outcome.to_event())


def test_serving_layer_never_trains_or_reaches_the_network() -> None:
    root = Path(__file__).parents[1] / "src" / "tech_challenge_fase2" / "serving"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    for forbidden in ("RandomizedSearchCV", "run_genetic", ".fit(", "urllib", "requests."):
        assert forbidden not in source


# --- relatorio de escalabilidade ------------------------------------------------


def test_validation_reports_absence_instead_of_raising(tmp_path: Path) -> None:
    result = validate_scalability_report(tmp_path)
    assert result["passed"] is False
    assert result["checks"][0]["check"] == "report_present"


def test_committed_scalability_report_is_valid_and_in_scope() -> None:
    result = validate_scalability_report()
    failed = [check for check in result["checks"] if not check["passed"]]
    assert not failed, failed
