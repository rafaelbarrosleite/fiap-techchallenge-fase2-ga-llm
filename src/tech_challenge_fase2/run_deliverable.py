"""CLI da consolidacao academica final."""

from __future__ import annotations

import argparse
import json

from .deliverable import (
    PRESENTATION_FIGURE_ROOT, SUMMARY_ROOT, generate_delivery_manifest,
    generate_master_table, generate_presentation_figures, validate_deliverable,
)


def build_main() -> None:
    parser = argparse.ArgumentParser(description="Gera tabela e figuras somente de artefatos agregados.")
    parser.add_argument("--test-count", type=int, default=None)
    args = parser.parse_args()
    final_manifest = SUMMARY_ROOT / "final_delivery_manifest.json"
    if final_manifest.is_file():
        validation = validate_deliverable(require_manifest=True)
        if not validation["passed"]:
            raise SystemExit("Entrega existente divergiu do manifesto; revise sem sobrescrever artefatos.")
        print(json.dumps({
            "status": "reused_completed",
            "model_results_csv": str(SUMMARY_ROOT / "model_results.csv"),
            "model_results_json": str(SUMMARY_ROOT / "model_results.json"),
            "figure_qa": str(PRESENTATION_FIGURE_ROOT / "figure_qa_report.json"),
            "manifest": str(final_manifest),
        }, ensure_ascii=False))
        return
    csv_path, json_path = generate_master_table()
    qa_path = generate_presentation_figures()
    result = {"model_results_csv": str(csv_path), "model_results_json": str(json_path), "figure_qa": str(qa_path)}
    if args.test_count is not None:
        result["manifest"] = str(generate_delivery_manifest(test_count=args.test_count))
    print(json.dumps(result, ensure_ascii=False))


def validate_main() -> None:
    result = validate_deliverable()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
