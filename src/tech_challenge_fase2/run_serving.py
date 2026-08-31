"""CLIs da camada de escalabilidade e monitoramento.

As variaveis de ambiente do BLAS sao fixadas antes de qualquer importacao que
carregue NumPy. Sem isso, a biblioteca de algebra linear paraleliza cada
predicao internamente, satura as CPUs com um unico worker e faz o
dimensionamento por workers parecer inutil. Uma thread de BLAS por worker e a
configuracao usual de servico: o paralelismo passa a vir das replicas, que sao
o recurso que o autoscaling controla.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from .serving.autoscaling import AutoscalingPolicy  # noqa: E402
from .serving.load_benchmark import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    REPORT_NAME,
    SCALABILITY_ROOT,
    run_load_benchmark,
)
from .serving.validation import validate_scalability_report  # noqa: E402


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(
        description="Mede o servico congelado sob demanda variavel, com e sem autoscaling."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--artifact-root", type=Path, default=SCALABILITY_ROOT)
    arguments = parser.parse_args()

    policy_kwargs: dict[str, int] = {"min_workers": arguments.min_workers}
    if arguments.max_workers is not None:
        policy_kwargs["max_workers"] = arguments.max_workers

    report = run_load_benchmark(
        artifact_root=arguments.artifact_root,
        policy=AutoscalingPolicy(**policy_kwargs),
        batch_size=arguments.batch_size,
    )
    summary = {
        "artifact": str(Path(arguments.artifact_root) / REPORT_NAME),
        "available_cpus": report["environment"]["available_cpus"],
        "scenarios": {
            scenario["label"]: {
                "p95_ms": scenario["latency"]["p95_ms"],
                "records_per_second": scenario["throughput_records_per_second"],
                "max_workers_used": scenario["max_workers_used"],
            }
            for scenario in report["scenarios"]
        },
        "comparison": report["comparison"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Confere assinatura, escopo e coerencia do relatorio de escalabilidade."
    )
    parser.add_argument("--artifact-root", type=Path, default=SCALABILITY_ROOT)
    arguments = parser.parse_args()
    result = validate_scalability_report(arguments.artifact_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
