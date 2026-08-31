"""Validacao somente leitura do relatorio de escalabilidade.

O relatorio de desempenho depende do hardware e por isso nao pode ser comparado
a numeros congelados. O que se pode verificar sem ambiguidade e a integridade:
assinatura, coerencia interna e as confirmacoes de escopo que garantem que
medir desempenho nao reabriu treino, selecao nem threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..genetic.serialization import stable_sha256
from .load_benchmark import REPORT_NAME, SCALABILITY_ROOT

REQUIRED_SCOPE_CONFIRMATIONS = (
    "new_training_performed",
    "selection_reopened",
    "threshold_changed",
    "individual_data_logged",
    "network_required",
)


def validate_scalability_report(artifact_root: Path = SCALABILITY_ROOT) -> dict[str, Any]:
    """Confere o relatorio sem executar nova medicao."""

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    path = Path(artifact_root) / REPORT_NAME
    if not path.is_file():
        check("report_present", False, f"ausente: {path}")
        return {"passed": False, "checks": checks}
    check("report_present", True, str(path))

    report = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in report.items() if key != "signature"}
    check(
        "signature",
        report.get("artifact_type") == "scalability_report"
        and report.get("signature") == stable_sha256(unsigned),
    )

    scenarios = {item["label"]: item for item in report.get("scenarios", [])}
    check("both_scenarios_present", set(scenarios) == {"pool_fixo_minimo", "pool_autoescalavel"})

    if len(scenarios) == 2:
        fixed = scenarios["pool_fixo_minimo"]
        scaled = scenarios["pool_autoescalavel"]
        check(
            "same_demand_served",
            fixed["total_requests"] == scaled["total_requests"]
            and fixed["total_records"] == scaled["total_records"],
            {"requests": fixed["total_requests"], "records": fixed["total_records"]},
        )
        check("fixed_pool_stayed_minimal", fixed["max_workers_used"] == report["policy"]["min_workers"])
        check("autoscaling_changed_size", scaled["scaling_events"] > 0, scaled["scaling_events"])
        check(
            "autoscaling_respected_ceiling",
            scaled["max_workers_used"] <= report["policy"]["max_workers"],
            {"used": scaled["max_workers_used"], "ceiling": report["policy"]["max_workers"]},
        )

    confirmations = report.get("scope_confirmations", {})
    check(
        "scope_confirmations_all_false",
        all(confirmations.get(name) is False for name in REQUIRED_SCOPE_CONFIRMATIONS),
        confirmations,
    )
    check(
        "measurement_declared_environment_dependent",
        report.get("measurement_is_environment_dependent") is True,
    )

    events_path = Path(artifact_root) / "performance_events.jsonl"
    if events_path.is_file():
        from .monitoring import assert_event_is_aggregate

        violations: list[str] = []
        for number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                assert_event_is_aggregate(json.loads(line))
            except Exception as error:  # noqa: BLE001 - detalhe vai para o relatorio
                violations.append(f"linha {number}: {error}")
        check("performance_log_has_no_individual_data", not violations, violations or "limpo")

    return {"passed": all(item["passed"] for item in checks), "checks": checks}
