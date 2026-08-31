"""Comandos da explicacao individual desidentificada."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_individual.engine import OPENAI_ROOT, evaluate_existing, prepare, revalidate_existing, run


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepara caso individual desidentificado sem chamar provider.")
    parser.add_argument("--provider", choices=("fake", "openai_responses"), default="fake")
    parser.add_argument("--model", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()
    result = prepare(provider_name=args.provider, model=args.model, artifact_root=args.artifact_dir)
    print(json.dumps({"status": "prepared", "run_identity": result["run_identity"], "privacy": result["privacy"]}, ensure_ascii=False))


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Gera explicacao individual; fake offline por padrao.")
    parser.add_argument("--provider", choices=("fake", "openai_responses"), default="fake")
    parser.add_argument("--model", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run(provider_name=args.provider, model=args.model, artifact_root=args.artifact_dir)
    print(json.dumps({"approved": result["approved"], "provider": result["provider"], "model": result["model"]}, ensure_ascii=False))


def run_openai_main() -> None:
    result = run(provider_name="openai_responses", artifact_root=OPENAI_ROOT)
    print(json.dumps({"approved": result["approved"], "provider": result["provider"], "model": result["model"]}, ensure_ascii=False))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Reavalia uma explicacao individual sem rede.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()
    kwargs = {"artifact_root": args.artifact_dir} if args.artifact_dir else {}
    result = evaluate_existing(**kwargs)
    print(json.dumps({"approved": result["approved"], "overall_score": result["overall_score"]}, ensure_ascii=False))


def revalidate_main() -> None:
    parser = argparse.ArgumentParser(description="Revalida offline uma saida individual preservada.")
    parser.add_argument("--artifact-dir", type=Path, default=OPENAI_ROOT)
    args = parser.parse_args()
    result = revalidate_existing(artifact_root=args.artifact_dir)
    print(json.dumps({"approved": result["approved"], "provider_called": False, "overall_score": result["overall_score"]}, ensure_ascii=False))
