"""Missao 7.4: versao semantica V2, auditada e executada somente offline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tech_challenge_fase2.genetic.serialization import save_json, stable_sha256
from tech_challenge_fase2.llm.input_builder import PROJECT_ROOT, file_sha256
from tech_challenge_fase2.llm.providers import LLMRequest
from tech_challenge_fase2.llm_v2.evaluation import evaluate_output_v2
from tech_challenge_fase2.llm_v2.input_builder import build_llm_input_v2
from tech_challenge_fase2.llm_v2.prompts import load_prompt_bundle_v2
from tech_challenge_fase2.llm_v2.providers import FakeLLMProviderV2
from tech_challenge_fase2.llm_v2.schemas import output_json_schema_v2, validate_input_v2, validate_output_v2
from tech_challenge_fase2.provider_real_evaluation import utc_now

CONTRACT_V2_ROOT = PROJECT_ROOT / "artifacts" / "llm_contract_v2"
CONTRACT_NAME = "contract_v2.json"
AUDIT_NAME = "factual_checks_audit.json"
MCNEMAR_NAME = "mcnemar_evidence_report.json"
COMPARISON_NAME = "v1_vs_v2_comparison.json"
HISTORICAL_NAME = "historical_v3_reinterpretation.json"
FAKE_NAME = "fake_v2_evaluation.json"
MANIFEST_NAME = "contract_v2_manifest.json"

HISTORICAL_ROOTS = {
    "mission5": PROJECT_ROOT / "artifacts" / "llm_evaluation",
    "mission7": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai",
    "mission71": PROJECT_ROOT / "artifacts" / "openai_integration_diagnosis",
    "mission72": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v2",
    "mission721": PROJECT_ROOT / "artifacts" / "openai_response_parsing_diagnosis",
    "mission73": PROJECT_ROOT / "artifacts" / "llm_evaluation_openai_v3",
}
V1_FILES = {
    "schemas_v1": PROJECT_ROOT / "src" / "tech_challenge_fase2" / "llm" / "schemas.py",
    "factuality_v1": PROJECT_ROOT / "src" / "tech_challenge_fase2" / "llm" / "factuality.py",
    "providers_v1": PROJECT_ROOT / "src" / "tech_challenge_fase2" / "llm" / "providers.py",
    "system_v1": PROJECT_ROOT / "src" / "tech_challenge_fase2" / "llm" / "prompts" / "system_v1.txt",
    "explanation_v1": PROJECT_ROOT / "src" / "tech_challenge_fase2" / "llm" / "prompts" / "explanation_v1.txt",
}


class ContractV2Error(RuntimeError):
    """A construcao V2 violou preservacao, contrato ou avaliacao offline."""


def _sign(payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["signature"] = stable_sha256(signed)
    return signed


def _load_signed(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("artifact_type") != artifact_type or signature != stable_sha256(unsigned):
        raise ContractV2Error(f"Artefato V2 invalido: {path.name}.")
    return payload


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ContractV2Error(f"Evidencia historica ausente: {root}.")
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _historical_hashes() -> dict[str, dict[str, str]]:
    return {name: _tree_hashes(root) for name, root in HISTORICAL_ROOTS.items()}


def _v1_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in V1_FILES.items()}


def _classify_v1_check(name: str) -> tuple[str, str, str]:
    if name == "no_unexpected_narrative_numbers":
        return (
            "redundant", "retained_as_defense_in_depth",
            "Duplica parcialmente os checks numericos estruturados, mas ainda detecta numeros inventados em narrativa.",
        )
    pair_suffixes = (
        ".ga_recall_change", ".ga_f1_change", ".ga_auc_change", ".tradeoff",
        ".same_threshold_different_auc", ".cv_gain_confirmed", ".uncertainty_present",
        ".delta_recall", ".mcnemar_p_value", ".delta_ci_includes_zero",
    )
    if name.endswith(pair_suffixes) or ".delta_recall_ci." in name:
        return (
            "needs_explicit_pair", "replaced_by_comparison_id_scoped_check",
            "O V1 depende de baseline_vs_ga implicito; o V2 fixa comparison_id, left_method e right_method.",
        )
    return (
        "unambiguous", "retained_or_versioned_equivalent",
        "O sujeito, metodo ou campo verificado ja e explicito no nome e na estrutura.",
    )


def _audit_139() -> dict[str, Any]:
    report_path = HISTORICAL_ROOTS["mission73"] / "factuality_report.json"
    historical = _load_signed(report_path, "openai_factuality_report")
    checks = []
    for item in historical["checks"]:
        classification, action, rationale = _classify_v1_check(item["check"])
        checks.append({
            "v1_check": item["check"], "v1_passed_in_mission73": item["passed"],
            "classification": classification, "v2_action": action, "rationale": rationale,
        })
    if len(checks) != 139:
        raise ContractV2Error(f"Auditoria esperava 139 checks e encontrou {len(checks)}.")
    counts = Counter(item["classification"] for item in checks)
    for classification in ("unambiguous", "needs_explicit_pair", "needs_more_evidence", "redundant", "unsafe_to_require"):
        counts.setdefault(classification, 0)
    return {
        "schema_version": "1.0", "artifact_type": "llm_v1_factual_checks_audit",
        "generated_at_utc": utc_now(), "checks_audited": len(checks),
        "classification_counts": dict(counts), "checks_changed": counts["needs_explicit_pair"],
        "checks_removed": 0,
        "supplementary_requirements": [{
            "requirement": "Declare low McNemar power because discordances are few.",
            "v1_classification": "needs_more_evidence",
            "v1_issue": "The V1 LLM payload exposed p-values but omitted aggregate discordance counts.",
            "v2_resolution": "Strategy A: signed aggregate counts and low_count_warning are included per explicit pair.",
            "unsafe_to_require_without_v2_evidence": True,
        }],
        "checks": checks,
    }


def _historical_reinterpretation(v2_input: dict[str, Any]) -> dict[str, Any]:
    root = HISTORICAL_ROOTS["mission73"]
    manifest = _load_signed(root / "llm_evaluation_manifest.json", "openai_llm_evaluation_manifest")
    output_artifact = _load_signed(root / "llm_output.json", "openai_llm_output")
    factuality = _load_signed(root / "factuality_report.json", "openai_factuality_report")
    if manifest.get("status") != "invalid" or manifest.get("approved") is not False:
        raise ContractV2Error("Missao 7.3 nao preserva status historico invalido.")
    rf_pairs = {
        item["comparison_id"]: item
        for item in v2_input["comparison_pairs"] if item["model"] == "random_forest"
    }
    historical_rf = next(
        item for item in output_artifact["structured_output"]["comparacao_modelos"]
        if item["model"] == "random_forest"
    )
    return {
        "schema_version": "1.0", "artifact_type": "mission73_historical_reinterpretation",
        "generated_at_utc": utc_now(), "analysis_only": True, "provider_called": False,
        "historical_status": {"status": manifest["status"], "approved": manifest["approved"], "changed": False},
        "v1_result": {
            "checks_passed": sum(item["passed"] for item in factuality["checks"]),
            "checks_total": len(factuality["checks"]),
            "failed_field": "random_forest.same_threshold_different_auc",
            "historical_value": historical_rf["same_threshold_outcomes_different_auc"],
            "v1_expected_value": False,
            "v1_implicit_pair": "random_forest__baseline_vs_ga",
        },
        "v2_explicit_pairs": {
            pair_id: {
                "left_method": pair["left_method"], "right_method": pair["right_method"],
                "same_confusion_matrix": pair["same_confusion_matrix"],
                "different_roc_auc": pair["different_roc_auc"],
            }
            for pair_id, pair in rf_pairs.items()
        },
        "interpretation_assessment": {
            "historical_boolean_is_correct_for": "random_forest__ga_vs_random_search",
            "historical_boolean_is_incorrect_for": "random_forest__baseline_vs_ga",
            "numerical_hallucination": False,
            "semantic_ambiguity_confirmed": True,
        },
        "still_valid": [
            "All structured metrics and uncertainty values that passed 138 V1 checks.",
            "Frozen selection, non-clinical status, disclaimer, safety and scientific calibration.",
            "The RF narrative about GA and random_search is reusable only when attached to that explicit V2 comparison_id.",
        ],
        "not_reusable_as_v2": [
            "The V1 output object as a whole, because it does not satisfy schema 2.0.",
            "The unscoped same_threshold_outcomes_different_auc field.",
            "Any pairwise narrative that omits comparison_id, left_method or right_method.",
        ],
        "retroactive_reclassification_performed": False,
    }


def build_contract_v2_artifacts(artifact_root: Path = CONTRACT_V2_ROOT) -> dict[str, Any]:
    """Gera a bateria V2 offline e falha diante de diretorio parcial."""

    root = Path(artifact_root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        return _load_signed(manifest_path, "llm_contract_v2_manifest")
    if root.exists() and any(root.iterdir()):
        raise ContractV2Error("Diretorio llm_contract_v2 parcial; revisao manual obrigatoria.")
    historical_before = _historical_hashes()
    v1_before = _v1_hashes()
    v2_input = build_llm_input_v2()
    validate_input_v2(v2_input)
    prompts = load_prompt_bundle_v2()
    output_schema = output_json_schema_v2()
    contract_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_contract_v2_definition",
        "generated_at_utc": utc_now(), "contract_version": "v2",
        "input_schema_version": "2.0", "output_schema_version": "2.0",
        "selection_must_be_explicit": True, "silent_migration": False,
        "input_sha256": stable_sha256(v2_input), "input_contract": v2_input,
        "output_json_schema_sha256": stable_sha256(output_schema), "output_json_schema": output_schema,
        "prompts": {
            "system_version": prompts.system_version, "explanation_version": prompts.explanation_version,
            "system_sha256": prompts.system_sha256, "explanation_sha256": prompts.explanation_sha256,
        },
        "privacy": {
            "aggregate_only": True, "individual_data_included": False,
            "individual_predictions_included": False, "final_predictions_included": False,
        },
    })
    audit_artifact = _sign(_audit_139())
    mcnemar_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_v2_mcnemar_evidence_report",
        "generated_at_utc": utc_now(), "strategy": "A", "strategy_preferred": True,
        "source": v2_input["source_provenance"]["mcnemar_source"],
        "statistic_recomputed": False, "dataset_read": False, "individual_predictions_read": False,
        "comparisons": [
            {
                "comparison_id": item["comparison_id"], "model": item["model"],
                "left_method": item["left_method"], "right_method": item["right_method"],
                **item["mcnemar"],
            }
            for item in v2_input["uncertainty_comparisons"]
        ],
        "low_power_statement_supported": all(
            item["mcnemar"]["low_count_warning"] and item["mcnemar"]["discordant_total"] < 10
            for item in v2_input["uncertainty_comparisons"]
        ),
    })
    comparison_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_contract_v1_vs_v2_comparison",
        "generated_at_utc": utc_now(),
        "v1": {
            "schema_version": "1.0", "prompt_versions": ["system_v1", "explanation_v1"],
            "pair_semantics": "implicit in derived field names", "mcnemar_discordances_in_payload": False,
            "historical_reproduction_supported": True, "files_sha256": v1_before,
        },
        "v2": {
            "schema_version": "2.0", "prompt_versions": ["system_v2", "explanation_v2"],
            "pair_semantics": "comparison_id plus left_method plus right_method",
            "comparison_pairs": 9, "delta_direction": "right_minus_left",
            "mcnemar_discordances_in_payload": True, "silent_default": False,
        },
        "breaking_changes_are_explicit": True, "v1_modified": False,
    })
    historical_artifact = _sign(_historical_reinterpretation(v2_input))
    provider = FakeLLMProviderV2()
    request = LLMRequest(
        input_payload=v2_input, system_prompt=prompts.system_text,
        explanation_prompt=prompts.explanation_text, model="deterministic-explainer-v2",
        temperature=0.0, max_output_tokens=5000,
    )
    response = provider.generate(request)
    validate_output_v2(response.output)
    evaluation = evaluate_output_v2(response.output, v2_input)
    fake_artifact = _sign({
        "schema_version": "1.0", "artifact_type": "llm_fake_v2_evaluation",
        "generated_at_utc": utc_now(), "status": "approved" if evaluation["approved"] else "invalid",
        "provider": response.provider, "model": response.model, "response_id": response.response_id,
        "contract_version": "v2", "provider_calls": provider.call_count,
        "external_calls": 0, "paid_tokens": 0, "deterministic": True,
        "output": response.output, "evaluation": evaluation,
        "factual_checks": len(evaluation["factuality"]["checks"]),
    })
    for name, artifact in (
        (CONTRACT_NAME, contract_artifact), (AUDIT_NAME, audit_artifact),
        (MCNEMAR_NAME, mcnemar_artifact), (COMPARISON_NAME, comparison_artifact),
        (HISTORICAL_NAME, historical_artifact), (FAKE_NAME, fake_artifact),
    ):
        save_json(artifact, root / name)
    historical_after = _historical_hashes()
    v1_after = _v1_hashes()
    rf_pairs = {
        item["comparison_id"]: item for item in v2_input["comparison_pairs"]
        if item["model"] == "random_forest"
    }
    ready = all((
        evaluation["approved"], audit_artifact["checks_audited"] == 139,
        rf_pairs["random_forest__baseline_vs_ga"]["same_confusion_matrix"] is False,
        rf_pairs["random_forest__ga_vs_random_search"]["same_confusion_matrix"] is True,
        rf_pairs["random_forest__ga_vs_random_search"]["different_roc_auc"] is True,
        mcnemar_artifact["low_power_statement_supported"] is True,
        historical_before == historical_after, v1_before == v1_after,
    ))
    files = [
        {"filename": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.glob("*.json")) if path.name != MANIFEST_NAME
    ]
    manifest = _sign({
        "schema_version": "1.0", "artifact_type": "llm_contract_v2_manifest",
        "generated_at_utc": utc_now(), "status": "approved" if ready else "invalid",
        "contract_version": "v2", "ready_for_real_v2_evaluation": ready,
        "checks_audited": audit_artifact["checks_audited"],
        "ambiguous_checks": audit_artifact["classification_counts"]["needs_explicit_pair"],
        "checks_removed": audit_artifact["checks_removed"],
        "fake_v2_approved": evaluation["approved"],
        "fake_v2_factual_checks": len(evaluation["factuality"]["checks"]),
        "mcnemar_strategy": "A", "mcnemar_recomputed": False,
        "historical_status_preserved": historical_before == historical_after,
        "v1_files_preserved": v1_before == v1_after,
        "mission73_scientific_evaluation_approved": False,
        "execution": {
            "external_provider_calls": 0, "ga_executed": False,
            "randomized_search_executed": False, "training_performed": False,
            "new_holdout_inference_performed": False, "individual_data_used": False,
            "deploy_performed": False, "automatic_commit_performed": False,
            "push_performed": False,
        },
        "historical_hashes_before": historical_before, "v1_hashes": v1_before,
        "files": files,
    })
    save_json(manifest, manifest_path)
    return manifest


def validate_contract_v2(artifact_root: Path = CONTRACT_V2_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    required = (CONTRACT_NAME, AUDIT_NAME, MCNEMAR_NAME, COMPARISON_NAME, HISTORICAL_NAME, FAKE_NAME, MANIFEST_NAME)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for name in required:
        add(f"exists:{name}", (root / name).is_file(), name)
    if not all((root / name).is_file() for name in required):
        return {"passed": False, "checks": checks}
    manifest = _load_signed(root / MANIFEST_NAME, "llm_contract_v2_manifest")
    artifact_types = {
        CONTRACT_NAME: "llm_contract_v2_definition", AUDIT_NAME: "llm_v1_factual_checks_audit",
        MCNEMAR_NAME: "llm_v2_mcnemar_evidence_report", COMPARISON_NAME: "llm_contract_v1_vs_v2_comparison",
        HISTORICAL_NAME: "mission73_historical_reinterpretation", FAKE_NAME: "llm_fake_v2_evaluation",
    }
    for item in manifest["files"]:
        path = root / item["filename"]
        add(f"hash:{item['filename']}", path.is_file() and file_sha256(path) == item["sha256"], item["filename"])
        _load_signed(path, artifact_types[item["filename"]])
    contract = _load_signed(root / CONTRACT_NAME, artifact_types[CONTRACT_NAME])
    audit = _load_signed(root / AUDIT_NAME, artifact_types[AUDIT_NAME])
    mcnemar = _load_signed(root / MCNEMAR_NAME, artifact_types[MCNEMAR_NAME])
    historical = _load_signed(root / HISTORICAL_NAME, artifact_types[HISTORICAL_NAME])
    fake = _load_signed(root / FAKE_NAME, artifact_types[FAKE_NAME])
    validate_input_v2(contract["input_contract"])
    validate_output_v2(fake["output"])
    rf_pairs = {
        item["comparison_id"]: item for item in contract["input_contract"]["comparison_pairs"]
        if item["model"] == "random_forest"
    }
    add("manifest_approved", manifest["status"] == "approved", manifest["status"])
    add("ready", manifest["ready_for_real_v2_evaluation"] is True, manifest["ready_for_real_v2_evaluation"])
    add("audit_139", audit["checks_audited"] == 139, audit["checks_audited"])
    add("explicit_pair_count", len(contract["input_contract"]["comparison_pairs"]) == 9, len(contract["input_contract"]["comparison_pairs"]))
    add("rf_baseline_vs_ga", rf_pairs["random_forest__baseline_vs_ga"]["same_confusion_matrix"] is False, rf_pairs["random_forest__baseline_vs_ga"])
    add("rf_ga_vs_random", rf_pairs["random_forest__ga_vs_random_search"]["same_confusion_matrix"] is True and rf_pairs["random_forest__ga_vs_random_search"]["different_roc_auc"] is True, rf_pairs["random_forest__ga_vs_random_search"])
    add("mcnemar_strategy_a", mcnemar["strategy"] == "A" and mcnemar["statistic_recomputed"] is False, mcnemar["strategy"])
    add("mcnemar_counts", [item["discordant_total"] for item in mcnemar["comparisons"]] == [2, 1, 3], [item["discordant_total"] for item in mcnemar["comparisons"]])
    add("fake_v2", fake["status"] == "approved" and fake["evaluation"]["approved"] is True, fake["status"])
    add("historical_invalid", historical["historical_status"] == {"status": "invalid", "approved": False, "changed": False}, historical["historical_status"])
    add("historical_hashes", manifest["historical_hashes_before"] == _historical_hashes(), "preserved")
    add("v1_hashes", manifest["v1_hashes"] == _v1_hashes(), "preserved")
    add("zero_external_calls", manifest["execution"]["external_provider_calls"] == 0, manifest["execution"])
    add("scope", not any(manifest["execution"].values()), manifest["execution"])
    content = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
    add("no_secrets", all(token not in content for token in ("Authorization", "Bearer ", '"OPENAI_API_KEY"', '".env"')), "safe")
    add("no_individual_fields", all(token not in content for token in ('"patient_id"', '"features"', '"prediction"', '"probability"')), "aggregate_only")
    return {
        "passed": all(item["passed"] for item in checks), "check_count": len(checks),
        "ready_for_real_v2_evaluation": manifest["ready_for_real_v2_evaluation"], "checks": checks,
    }
