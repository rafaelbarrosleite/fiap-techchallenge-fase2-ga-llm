"""CLI da Missao 5; mock offline por padrao e provider real somente opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm.engine import configured_real_model, evaluate_existing_output, prepare_evaluation, run_evaluation


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("fake", "openai_responses"), default="fake")
    parser.add_argument("--model", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)


def _values(args: argparse.Namespace) -> dict:
    provider = args.provider
    model = args.model
    if provider == "fake":
        model = model or "deterministic-explainer-v1"
    elif model is None:
        model = configured_real_model()
    result = {"provider_name": provider, "model": model}
    if args.artifact_dir is not None:
        result["artifact_root"] = args.artifact_dir
    return result


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepara entrada agregada e sanitizada da avaliacao LLM.")
    _common(parser)
    args = parser.parse_args()
    result = prepare_evaluation(**_values(args))
    print(json.dumps({"status": "prepared", "run_identity": result["run_identity"]}, ensure_ascii=False))


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Executa explicacao LLM protegida; mock offline por padrao.")
    _common(parser)
    args = parser.parse_args()
    result = run_evaluation(**_values(args))
    print(json.dumps({"approved": result["approved"], "provider": result["provider"], "model": result["model"]}, ensure_ascii=False))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Revalida factualidade e seguranca sem chamar provider.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()
    kwargs = {"artifact_root": args.artifact_dir} if args.artifact_dir is not None else {}
    result = evaluate_existing_output(**kwargs)
    print(json.dumps({"approved": result["approved"], "overall_score": result["overall_score"]}, ensure_ascii=False))
