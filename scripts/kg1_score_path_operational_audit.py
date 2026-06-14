#!/usr/bin/env python3
"""CPU-only audit for the active KG1 score path.

This gate is intentionally conservative: it does not train, package, submit to
Kaggle, call external APIs, or read secrets. It verifies that the active
bit/equation LoRA-transfer path still matches the score-facing contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    OFFICIAL_INFERENCE_CONFIG,
    PROMPT_SUFFIX,
    ensure_prompt_suffix,
    extract_closed_boxed_answers,
    extract_final_answer,
    has_unclosed_boxed_answer,
    verify_answer,
)


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "score_path_operational_audit"
V1243_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft"
V1243_DRYRUN_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft_tokenize_dryrun"
SAFE_LORA_MODULES = [
    "down_proj",
    "in_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
SAFE_LORA_MODULE_STRING = ",".join(SAFE_LORA_MODULES)
FULL_ADAPTER_LORA_MODULES = [
    "down_proj",
    "in_proj",
    "k_proj",
    "lm_head",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
FULL_ADAPTER_LORA_MODULE_STRING = ",".join(FULL_ADAPTER_LORA_MODULES)
INIT_ADAPTER_REPO = "felipesp1983/kg1-recovered-v291-v290-checkpoint6-submit086"
INIT_ADAPTER_REVISION = "f4134a6d223249d27be2f1c5d94ed59d118d1ce5"
INIT_ADAPTER_CONFIG_SHA256 = "a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d"
INIT_ADAPTER_WEIGHTS_SHA256 = "0a7b6144231d9358ae73a5e57d8778b32be1520fa47e3041414b3e025aaa1aa1"
CANONICAL_FULL947_SOLUTION_SHA256 = "84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935"
EXPECTED_FULL947_FAMILY_COUNTS = {
    "bit_manipulation": 160,
    "equation_transform": 155,
    "gravity_constant": 159,
    "numeral_system": 157,
    "text_encryption": 157,
    "unit_conversion": 159,
}
EXPECTED_OFFICIAL_CONFIG = {
    "max_lora_rank": 32,
    "max_tokens": 7680,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
}
EXPECTED_V1243 = {
    "bit_specialist": {
        "path": "v1243_bit_specialist_train.jsonl",
        "rows": 724,
        "family_counts": {
            "bit_manipulation": 540,
            "gravity_constant": 46,
            "numeral_system": 46,
            "text_encryption": 46,
            "unit_conversion": 46,
        },
        "source_role_counts": {
            "v1243_bit_solver_verified": 540,
            "v1243_protected_replay": 184,
        },
        "weight_by_family": {
            "bit_manipulation": 729.0,
            "gravity_constant": 92.0,
            "numeral_system": 92.0,
            "text_encryption": 92.0,
            "unit_conversion": 92.0,
        },
    },
    "equation_specialist": {
        "path": "v1243_equation_specialist_train.jsonl",
        "rows": 544,
        "family_counts": {
            "equation_transform": 360,
            "gravity_constant": 46,
            "numeral_system": 46,
            "text_encryption": 46,
            "unit_conversion": 46,
        },
        "source_role_counts": {
            "v1243_equation_solver_verified": 360,
            "v1243_protected_replay": 184,
        },
        "weight_by_family": {
            "equation_transform": 594.0,
            "gravity_constant": 92.0,
            "numeral_system": 92.0,
            "text_encryption": 92.0,
            "unit_conversion": 92.0,
        },
    },
    "protected_replay": {
        "path": "v1243_protected_replay_train.jsonl",
        "rows": 184,
        "family_counts": {
            "gravity_constant": 46,
            "numeral_system": 46,
            "text_encryption": 46,
            "unit_conversion": 46,
        },
        "source_role_counts": {
            "v1243_protected_anchor": 184,
        },
        "weight_by_family": {
            "gravity_constant": 92.0,
            "numeral_system": 92.0,
            "text_encryption": 92.0,
            "unit_conversion": 92.0,
        },
    },
    "micro_consolidation": {
        "path": "v1243_micro_consolidation_train.jsonl",
        "rows": 1084,
        "family_counts": {
            "bit_manipulation": 540,
            "equation_transform": 360,
            "gravity_constant": 46,
            "numeral_system": 46,
            "text_encryption": 46,
            "unit_conversion": 46,
        },
        "source_role_counts": {
            "v1243_bit_solver_verified": 540,
            "v1243_equation_solver_verified": 360,
            "v1243_protected_replay": 184,
        },
        "weight_by_family": {
            "bit_manipulation": 567.0,
            "equation_transform": 432.0,
            "gravity_constant": 92.0,
            "numeral_system": 92.0,
            "text_encryption": 92.0,
            "unit_conversion": 92.0,
        },
    },
    "val170": {
        "path": "v1243_val170.jsonl",
        "rows": 170,
        "family_counts": {
            "bit_manipulation": 90,
            "equation_transform": 60,
            "gravity_constant": 5,
            "numeral_system": 5,
            "text_encryption": 5,
            "unit_conversion": 5,
        },
        "source_role_counts": {
            "v1243_holdout": 170,
        },
        "weight_by_family": {
            "bit_manipulation": 0.0,
            "equation_transform": 0.0,
            "gravity_constant": 0.0,
            "numeral_system": 0.0,
            "text_encryption": 0.0,
            "unit_conversion": 0.0,
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def last_message_content(row: dict[str, Any], role: str) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or first_message_content(row, "user"))


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def rounded_weight_by_family(rows: list[dict[str, Any]]) -> dict[str, float]:
    weights: Counter[str] = Counter()
    for row in rows:
        try:
            weight = float(row.get("row_loss_weight", row.get("loss_weight", 1.0)) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        weights[str(row.get("family") or "")] += weight
    return {key: round(value, 6) for key, value in sorted(weights.items())}


def audit_official_config() -> dict[str, Any]:
    errors: list[str] = []
    prescore = load_module("kg1_official_metric_prescore_for_audit", ROOT / "scripts" / "kg1_official_metric_prescore.py")
    prescore_defaults = getattr(prescore, "OFFICIAL_VISIBLE_SCORE_DEFAULTS", {})
    for key, expected in EXPECTED_OFFICIAL_CONFIG.items():
        expect(errors, OFFICIAL_INFERENCE_CONFIG.get(key) == expected, f"OFFICIAL_INFERENCE_CONFIG[{key}] mismatch")
        expect(errors, prescore_defaults.get(key) == expected, f"prescore default {key} mismatch")
    expect(errors, getattr(prescore, "OFFICIAL_PROMPT_SUFFIX", None) == PROMPT_SUFFIX, "prescore prompt suffix mismatch")
    return {
        "errors": errors,
        "expected": EXPECTED_OFFICIAL_CONFIG,
        "local_official_inference_config": {
            key: OFFICIAL_INFERENCE_CONFIG.get(key)
            for key in EXPECTED_OFFICIAL_CONFIG
        },
        "prescore_visible_score_defaults": {
            key: prescore_defaults.get(key)
            for key in EXPECTED_OFFICIAL_CONFIG
        },
    }


class _AuditTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return messages[0]["content"]


class _AuditQuestions:
    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    def itertuples(self, index: bool = False) -> list[SimpleNamespace]:
        del index
        return [SimpleNamespace(prompt=prompt) for prompt in self.prompts]


def audit_prompt_rendering() -> dict[str, Any]:
    errors: list[str] = []
    suffix_body = PROMPT_SUFFIX.strip()
    cases = {
        "plain": "Solve 2+2.",
        "already_suffixed": "Solve 2+2." + PROMPT_SUFFIX,
        "empty_suffix_custom": "Solve 2+2.",
    }
    expect(errors, ensure_prompt_suffix(cases["plain"]).count(suffix_body) == 1, "plain prompt did not receive one suffix")
    expect(
        errors,
        ensure_prompt_suffix(cases["already_suffixed"]).count(suffix_body) == 1,
        "already-suffixed prompt was duplicated",
    )
    expect(errors, ensure_prompt_suffix(cases["empty_suffix_custom"], "") == cases["empty_suffix_custom"], "empty suffix changed prompt")

    evaluator = load_module("evaluate_lora_adapter_for_audit", ROOT / "scripts" / "evaluate_lora_adapter.py")
    rendered = evaluator.render_prompts(
        _AuditTokenizer(),
        _AuditQuestions([cases["plain"], cases["already_suffixed"]]),
        {"prompt_suffix": PROMPT_SUFFIX, "enable_thinking": True},
    )
    for index, prompt in enumerate(rendered, start=1):
        expect(errors, prompt.count(suffix_body) == 1, f"render_prompts case {index} suffix count != 1")
    return {
        "errors": errors,
        "rendered_suffix_counts": [prompt.count(suffix_body) for prompt in rendered],
    }


def audit_v1243_dataset(name: str, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    row_errors: list[str] = []
    rows = load_jsonl(path)
    ids = [str(row.get("id", "")) for row in rows]
    family_counts = count_values(rows, "family")
    source_role_counts = count_values(rows, "source_role")
    weight_by_family = rounded_weight_by_family(rows)
    prompt_hashes: set[str] = set()

    expect(errors, len(rows) == expected["rows"], f"{name}: rows {len(rows)} != {expected['rows']}")
    expect(errors, len(ids) == len(set(ids)), f"{name}: duplicate ids")
    expect(errors, family_counts == expected["family_counts"], f"{name}: family_counts mismatch")
    expect(errors, source_role_counts == expected["source_role_counts"], f"{name}: source_role_counts mismatch")
    expect(errors, weight_by_family == expected["weight_by_family"], f"{name}: weight_by_family mismatch")

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"row_{index}"))
        user_text = first_message_content(row, "user")
        assistant_text = last_message_content(row, "assistant")
        prompt = prompt_text(row)
        answer = row.get("answer")
        prompt_hashes.add(sha256_text(prompt))
        local: list[str] = []
        if prompt != user_text:
            local.append("prompt != user message")
        if prompt.count(PROMPT_SUFFIX.strip()) != 1:
            local.append("prompt suffix count != 1")
        if PROMPT_SUFFIX not in prompt:
            local.append("prompt suffix byte sequence missing")
        if not assistant_text.strip().startswith("</think>\n\\boxed{"):
            local.append("assistant target prefix mismatch")
        if assistant_text.count("\\boxed{") != 1:
            local.append("assistant target boxed marker count != 1")
        if has_unclosed_boxed_answer(assistant_text):
            local.append("unclosed boxed answer")
        boxed = extract_closed_boxed_answers(assistant_text)
        if len(boxed) != 1:
            local.append(f"closed boxed answer count {len(boxed)} != 1")
        elif not verify_answer(answer, boxed[0]):
            local.append("closed boxed payload does not verify")
        if not verify_answer(answer, extract_final_answer(assistant_text)):
            local.append("official final extracted answer does not verify")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in prompt + assistant_text):
            local.append("control character in prompt/target")
        if name == "val170":
            try:
                val_weight = float(row.get("row_loss_weight", row.get("loss_weight", 1.0)) or 0.0)
            except (TypeError, ValueError):
                val_weight = 1.0
            if val_weight != 0.0:
                local.append("validation row has non-zero loss weight")
        if local and len(row_errors) < 30:
            row_errors.append(f"{name}:{index}:{row_id}: " + "; ".join(local))

    errors.extend(row_errors)
    return {
        "errors": errors,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "family_counts": family_counts,
        "source_role_counts": source_role_counts,
        "weight_by_family": weight_by_family,
        "prompt_hashes": len(prompt_hashes),
    }


def audit_v1243_artifacts() -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = V1243_DIR / "kg1_v1243_solver_to_lora_graft_manifest.json"
    env_path = V1243_DIR / "v1243_hf_env_preview.json"
    manifest = load_json(manifest_path)
    env_preview = load_json(env_path)
    datasets: dict[str, Any] = {}

    expect(errors, manifest.get("decision") == "pass_v1243_cpu_dataset_graft_no_gpu_no_submit", "V1243 manifest decision mismatch")
    blocked = set(manifest.get("algorithm", {}).get("not_authorized", []))
    for item in ("paid_gpu_launch", "adapter_package", "kaggle_submit", "score_claim"):
        expect(errors, item in blocked, f"V1243 manifest missing not_authorized={item}")

    commands = manifest.get("gate_commands", {})
    for name, command in sorted(commands.items()):
        command_text = str(command)
        if name == "full947_086_baseline_readiness":
            expect(
                errors,
                "kg1_v1241_full947_baseline_readiness_gate.py" in command_text,
                f"{name}: wrong readiness gate script",
            )
            expect(errors, "--solution-csv" in command_text, f"{name}: missing --solution-csv")
            expect(
                errors,
                "--natural-baseline-predictions" in command_text,
                f"{name}: missing natural 086 predictions",
            )
            expect(
                errors,
                "--answer-only-baseline-predictions" in command_text,
                f"{name}: missing answer-only 086 predictions",
            )
            expect(errors, "--candidate-predictions" not in command_text, f"{name}: readiness must not require candidate predictions")
        else:
            expect(errors, "--baseline-predictions" in command_text, f"{name}: missing --baseline-predictions")
        if name.startswith("baseline_identity_probe"):
            expect(errors, "--baseline-identity-probe" in command_text, f"{name}: missing --baseline-identity-probe")
            expect(errors, "--profile full947_089" in command_text, f"{name}: missing full947_089 profile")
            expect(errors, "--candidate-predictions" not in command_text, f"{name}: probe must not require candidate predictions")
        elif name != "full947_086_baseline_readiness":
            expect(errors, "--candidate-predictions" in command_text, f"{name}: missing --candidate-predictions")
        expect(errors, " compare " not in f" {command_text} ", f"{name}: obsolete compare subcommand")
        expect(errors, "--baseline-csv" not in command_text, f"{name}: obsolete --baseline-csv")
        expect(errors, "--candidate-csv" not in command_text, f"{name}: obsolete --candidate-csv")
        if name in {
            "full947_086_baseline_readiness",
            "baseline_identity_probe_086_natural",
            "baseline_identity_probe_086_answer_only",
            "full947_089",
        }:
            expect(errors, "official_train_seed42_stratified10_val.csv" in command_text, f"{name}: missing canonical full947 solution path")
            expect(errors, "path\\to\\full947_solution.csv" not in command_text, f"{name}: still uses full947 solution placeholder")

    expect(errors, "full947_086_baseline_readiness" in commands, "manifest missing full947 086 readiness command")
    expect(errors, "baseline_identity_probe_086_natural" in commands, "manifest missing natural 086 probe command")
    expect(errors, "baseline_identity_probe_086_answer_only" in commands, "manifest missing answer-only 086 probe command")

    for name, expected in EXPECTED_V1243.items():
        path = V1243_DIR / expected["path"]
        dataset_report = audit_v1243_dataset(name, path, expected)
        datasets[name] = dataset_report
        errors.extend(dataset_report["errors"])
        summary = manifest.get("summaries", {}).get(name, {})
        expect(errors, summary.get("rows") == expected["rows"], f"manifest summary {name}: rows mismatch")
        expect(errors, summary.get("source_role_counts") == expected["source_role_counts"], f"manifest summary {name}: source roles mismatch")
        expect(errors, summary.get("weight_by_family") == expected["weight_by_family"], f"manifest summary {name}: weights mismatch")

    audit = manifest.get("audit", {})
    expect(errors, audit.get("train_val_prompt_overlap") == 0, "manifest train/val overlap must be zero")
    full947_judge = audit.get("full947_judge", {})
    expect(errors, isinstance(full947_judge, dict), "manifest missing full947 judge leak audit")
    expect(errors, full947_judge.get("leak_check_pass") is True, "full947 judge leak check not pass")
    expect(errors, full947_judge.get("canonical_full947_solution_rows") == 947, "full947 judge row count mismatch")
    expect(
        errors,
        full947_judge.get("canonical_full947_solution_sha256") == CANONICAL_FULL947_SOLUTION_SHA256,
        "full947 judge sha256 mismatch",
    )
    expect(
        errors,
        full947_judge.get("canonical_full947_solution_family_counts") == EXPECTED_FULL947_FAMILY_COUNTS,
        "full947 judge family counts mismatch",
    )
    expect(errors, full947_judge.get("train_full947_prompt_overlap") == 0, "V1243 train overlaps full947 judge")
    expect(errors, full947_judge.get("val_full947_prompt_overlap") == 0, "V1243 val overlaps full947 judge")
    inputs = manifest.get("inputs", {})
    canonical_solution = Path(str(inputs.get("canonical_full947_solution_csv", "")))
    expect(errors, canonical_solution.exists(), "manifest canonical full947 solution CSV missing")
    if canonical_solution.exists():
        expect(
            errors,
            sha256_file(canonical_solution) == CANONICAL_FULL947_SOLUTION_SHA256,
            "manifest canonical full947 solution sha256 mismatch",
        )
    expect(
        errors,
        inputs.get("canonical_full947_solution_sha256") == CANONICAL_FULL947_SOLUTION_SHA256,
        "manifest canonical full947 solution sha256 not pinned",
    )
    expect(
        errors,
        inputs.get("expected_canonical_full947_solution_sha256") == CANONICAL_FULL947_SOLUTION_SHA256,
        "manifest expected canonical full947 solution sha256 mismatch",
    )
    adapter_contract = manifest.get("algorithm", {}).get("adapter_contract", {})
    expect(errors, adapter_contract.get("init_adapter_repo") == INIT_ADAPTER_REPO, "manifest init adapter repo mismatch")
    expect(
        errors,
        adapter_contract.get("init_adapter_revision") == INIT_ADAPTER_REVISION,
        "manifest init adapter revision mismatch",
    )
    expect(
        errors,
        adapter_contract.get("init_adapter_config_sha256") == INIT_ADAPTER_CONFIG_SHA256,
        "manifest init adapter config sha256 mismatch",
    )
    expect(
        errors,
        adapter_contract.get("init_adapter_weights_sha256") == INIT_ADAPTER_WEIGHTS_SHA256,
        "manifest init adapter weights sha256 mismatch",
    )

    required_env = {
        "DRY_RUN_VALIDATE_ONLY": "1",
        "TOKENIZE_ONLY_DRY_RUN": "1",
        "UPLOAD_TO_HF": "0",
        "REQUIRE_BOXED_PAYLOAD_WEIGHT": "1",
        "REQUIRE_OFFSET_MASK": "1",
        "LORA_R": "32",
        "LORA_ALPHA": "32",
        "LORA_TARGET_MODULES": FULL_ADAPTER_LORA_MODULE_STRING,
        "TRAINABLE_LORA_MODULES": SAFE_LORA_MODULE_STRING,
        "INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "INIT_ADAPTER_REVISION": INIT_ADAPTER_REVISION,
        "EXPECTED_INIT_ADAPTER_CONFIG_SHA256": INIT_ADAPTER_CONFIG_SHA256,
        "EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256": INIT_ADAPTER_WEIGHTS_SHA256,
        "INIT_ADAPTER_LOAD_MODE": "manual",
        "PEFT_MANUAL_LOAD_METHOD": "direct",
        "SCORE_CONTRACT_RUNTIME_CHECK": "1",
        "REQUIRE_SCORE_CONTRACT_PASS": "1",
        "SCORE_CONTRACT_TARGET_ACCURACY": "0.89",
        "SCORE_CONTRACT_FULL_ROWS": "947",
        "SCORE_CONTRACT_BASELINE_CORRECT": "823",
        "SCORE_CONTRACT_EXPECTED_TARGET_MODULES": FULL_ADAPTER_LORA_MODULE_STRING,
        "SCORE_CONTRACT_EXPECTED_TRAINABLE_MODULES": SAFE_LORA_MODULE_STRING,
        "SCORE_CONTRACT_REQUIRE_TRAINABLE_FILTER": "1",
        "SCORE_CONTRACT_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "SCORE_PROXY_EVAL_CHECK": "1",
        "SCORE_PROXY_EVAL_MAX_EXAMPLES": "170",
        "SCORE_TRAJECTORY_CHECK": "1",
        "REQUIRE_SCORE_TRAJECTORY_PASS": "0",
        "REQUIRE_SCORE_TRAJECTORY_FINAL_ONLY": "0",
        "SCORE_TRAJECTORY_MIN_WEAK_EXACT_DELTA": "0.0",
        "SCORE_TRAJECTORY_MAX_PROTECTED_EXACT_DROP": "0.0",
        "SCORE_TRAJECTORY_MAX_OVERALL_EXACT_DROP": "0.0",
        "SCORE_TRAJECTORY_MAX_BOXED_LOSS_REGRESSION": "0.0",
        "ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA": "-1.0",
        "BASELINE_EVAL_BEFORE_TRAIN": "1",
        "EVAL_MAX_EXAMPLES": "170",
        "FRIENDLY_REALTIME_LOGS": "1",
        "FRIENDLY_LOG_SCORE_HINTS": "1",
        "SAMPLING_MODE": "weighted_replacement",
    }
    env_errors: list[str] = []
    for phase in ("bit_specialist", "equation_specialist", "micro_consolidation"):
        env = env_preview.get(phase, {})
        for key, expected in required_env.items():
            expect(env_errors, str(env.get(key)) == expected, f"{phase}: {key} mismatch")
    errors.extend(env_errors)

    return {
        "errors": errors,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "env_preview_path": str(env_path),
        "env_preview_sha256": sha256_file(env_path),
        "datasets": datasets,
        "env_errors": env_errors,
    }


def audit_dryrun_report(
    phase: str,
    expected_rows: int,
    expected_shares: dict[str, float],
) -> dict[str, Any]:
    errors: list[str] = []
    path = V1243_DRYRUN_DIR / phase / "dry_run_model_recipe_report.json"
    report = load_json(path)
    decision = report.get("decision", {})
    data = report.get("data", {})
    lora = report.get("lora", {})
    training = report.get("training", {})
    loss = training.get("train_loss_weighting", {})
    val_loss = training.get("validation_loss_weighting", {})
    shares = training.get("sampling", {}).get("weighted_share_by_subcategory", {})
    dataset_by_phase = {
        "bit": "bit_specialist",
        "equation": "equation_specialist",
        "micro_consolidation": "micro_consolidation",
    }
    dataset_name = dataset_by_phase[phase]
    expected_dataset_path = V1243_DIR / EXPECTED_V1243[dataset_name]["path"]
    expected_val_path = V1243_DIR / EXPECTED_V1243["val170"]["path"]

    expect(errors, decision.get("tokenization_contract_passed") is True, f"{phase}: tokenization contract not passed")
    expect(errors, decision.get("full_training_allowed") is False, f"{phase}: dry-run must not authorize full training")
    expect(errors, data.get("train_records") == expected_rows, f"{phase}: train_records mismatch")
    expect(errors, data.get("tokenized_train_records") == expected_rows, f"{phase}: tokenized_train_records mismatch")
    expect(errors, data.get("validation_records") == 170, f"{phase}: validation_records mismatch")
    expect(errors, data.get("tokenized_validation_records") == 170, f"{phase}: tokenized_validation_records mismatch")
    expect(errors, data.get("train_file_sha256") == sha256_file(expected_dataset_path), f"{phase}: train file sha mismatch")
    expect(errors, data.get("validation_file_sha256") == sha256_file(expected_val_path), f"{phase}: validation file sha mismatch")
    expect(errors, Path(str(data.get("train_file", ""))).name == expected_dataset_path.name, f"{phase}: train file name mismatch")
    expect(errors, Path(str(data.get("validation_file", ""))).name == expected_val_path.name, f"{phase}: validation file name mismatch")
    expect(errors, lora.get("r") == 32, f"{phase}: LoRA rank mismatch")
    expect(errors, lora.get("alpha") == 32, f"{phase}: LoRA alpha mismatch")
    expect(errors, lora.get("parsed_target_modules") == FULL_ADAPTER_LORA_MODULES, f"{phase}: parsed target modules mismatch")
    expect(errors, lora.get("target_modules") == FULL_ADAPTER_LORA_MODULE_STRING, f"{phase}: target_modules mismatch")
    expect(errors, lora.get("init_adapter_repo") == INIT_ADAPTER_REPO, f"{phase}: init adapter repo mismatch")
    expect(errors, lora.get("init_adapter_revision") == INIT_ADAPTER_REVISION, f"{phase}: init adapter revision mismatch")
    trainable_filter = lora.get("trainable_lora_module_filter", {})
    expect(errors, trainable_filter.get("enabled") is True, f"{phase}: trainable filter disabled")
    expect(errors, trainable_filter.get("requested") == SAFE_LORA_MODULES, f"{phase}: trainable filter requested modules mismatch")
    expect(errors, loss.get("rows") == expected_rows, f"{phase}: train loss rows mismatch")
    expect(errors, loss.get("rows_with_boxed_payload_weight") == expected_rows, f"{phase}: boxed payload rows mismatch")
    expect(errors, loss.get("rows_without_boxed_payload_weight") == 0, f"{phase}: boxed payload missing rows")
    expect(errors, val_loss.get("rows") == 170, f"{phase}: validation loss rows mismatch")
    expect(errors, val_loss.get("rows_with_boxed_payload_weight") == 170, f"{phase}: validation boxed payload rows mismatch")
    for expected_share_key, expected_share in expected_shares.items():
        actual_share = round(float(shares.get(expected_share_key, 0.0)), 6)
        expect(
            errors,
            actual_share == expected_share,
            f"{phase}: weighted share mismatch for {expected_share_key}: {actual_share} != {expected_share}",
        )
    return {
        "errors": errors,
        "path": str(path),
        "sha256": sha256_file(path),
        "decision": decision,
        "data": {
            "train_records": data.get("train_records"),
            "tokenized_train_records": data.get("tokenized_train_records"),
            "validation_records": data.get("validation_records"),
            "tokenized_validation_records": data.get("tokenized_validation_records"),
        },
        "lora": {
            "target_modules": lora.get("target_modules"),
            "parsed_target_modules": lora.get("parsed_target_modules"),
        },
        "weighted_share_by_subcategory": shares,
    }


def audit_existing_gate_reports() -> dict[str, Any]:
    errors: list[str] = []
    reports: dict[str, Any] = {}

    def add_report(name: str, path: Path) -> Any:
        payload = load_json(path)
        reports[name] = {"path": str(path), "sha256": sha256_file(path), "payload": payload}
        return payload

    official = add_report("official_metric_prescore_gate", ROOT / "artifacts" / "official_metric_prescore_gate" / "report.json")
    expect(errors, official.get("decision") == "pass", "official metric prescore gate did not pass")
    expect(errors, official.get("blockers") == [], "official metric prescore gate has blockers")
    expect(errors, official.get("function_parity_failures") == [], "official metric function parity failures")
    expect(errors, official.get("local_utils_parity_failures") == [], "local utils parity failures")

    contract = add_report(
        "v1243_graft_trainer_contract_gate",
        ROOT / "artifacts" / "v1243_solver_to_lora_graft_contract_gate" / "kg1_v1243_graft_trainer_contract_gate.json",
    )
    expect(
        errors,
        contract.get("decision") == "pass_v1243_graft_trainer_contract_no_gpu_no_submit",
        "V1243 trainer contract gate did not pass",
    )
    expect(errors, contract.get("errors") == [], "V1243 trainer contract gate has errors")
    expect(errors, contract.get("train_val_prompt_overlap") == 0, "V1243 trainer contract train/val overlap not zero")

    solve_rate = add_report("solve_rate_gate_self_test", ROOT / "artifacts" / "solve_rate_gate" / "self_test_report.json")
    expect(errors, solve_rate.get("prediction_only_blocked") is True, "solve_rate self-test did not block prediction-only false positive")
    expect(errors, solve_rate.get("multi_box_blocked") is True, "solve_rate self-test did not block multi-box false positive")
    expect(errors, solve_rate.get("truncation_blocked") is True, "solve_rate self-test did not block truncation false positive")

    v1241 = add_report(
        "v1241_transfer_gate_selftest",
        ROOT / "artifacts" / "v1241_bit_equation_transfer_gate_selftest" / "kg1_v1241_bit_equation_transfer_gate_selftest.json",
    )
    expect(errors, v1241.get("decision") == "pass_v1241_self_test", "V1241 self-test did not pass")
    expect(errors, all(case.get("expected_pass") == case.get("observed_pass") for case in v1241.get("cases", [])), "V1241 self-test case mismatch")
    v1241_cases = {str(case.get("case")): case for case in v1241.get("cases", [])}
    expect(errors, "pass_full947_clean_baseline_identity" in v1241_cases, "V1241 self-test missing clean full947 baseline identity case")
    dirty_identity = v1241_cases.get("fail_full947_dirty_baseline_identity", {})
    expect(errors, bool(dirty_identity), "V1241 self-test missing dirty full947 baseline identity case")
    expect(
        errors,
        any("baseline_strict_clean_identity_unverified" in str(blocker) for blocker in dirty_identity.get("blockers", [])),
        "V1241 dirty baseline identity case did not block on strict-clean identity",
    )
    expect(
        errors,
        "probe_pass_full947_clean_baseline_identity" in v1241_cases,
        "V1241 self-test missing baseline identity probe pass case",
    )
    probe_dirty_identity = v1241_cases.get("probe_fail_full947_dirty_baseline_identity", {})
    expect(errors, bool(probe_dirty_identity), "V1241 self-test missing baseline identity probe dirty case")
    expect(
        errors,
        any("baseline_strict_clean_identity_not_pass" in str(blocker) for blocker in probe_dirty_identity.get("blockers", [])),
        "V1241 dirty baseline identity probe did not block on strict-clean identity",
    )

    v1241_legacy = add_report(
        "v1241_transfer_gate_legacy_alias_selftest",
        ROOT / "artifacts" / "v1241_bit_equation_transfer_gate_selftest_legacy_alias" / "kg1_v1241_bit_equation_transfer_gate_selftest.json",
    )
    expect(errors, v1241_legacy.get("decision") == "pass_v1241_self_test", "V1241 legacy alias self-test did not pass")
    expect(
        errors,
        all(case.get("expected_pass") == case.get("observed_pass") for case in v1241_legacy.get("cases", [])),
        "V1241 legacy alias self-test case mismatch",
    )
    v1241_legacy_cases = {str(case.get("case")): case for case in v1241_legacy.get("cases", [])}
    expect(
        errors,
        "fail_full947_dirty_baseline_identity" in v1241_legacy_cases,
        "V1241 legacy alias self-test missing dirty full947 baseline identity case",
    )
    expect(
        errors,
        "probe_fail_full947_dirty_baseline_identity" in v1241_legacy_cases,
        "V1241 legacy alias self-test missing dirty baseline identity probe case",
    )

    return {"errors": errors, "reports": reports}


def audit_score_logic_fixtures() -> dict[str, Any]:
    errors: list[str] = []
    evaluator = load_module("evaluate_lora_adapter_score_logic_audit", ROOT / "scripts" / "evaluate_lora_adapter.py")
    solve_rate = load_module("solve_rate_gate_score_logic_audit", ROOT / "scripts" / "solve_rate_gate.py")
    import pandas as pd

    truncation_cases = [
        ("length", 1, 7680, True),
        ("max_tokens", 1, 7680, True),
        ("max_output_tokens", 1, 7680, True),
        ("token_limit", 1, 7680, True),
        ("truncated", 1, 7680, True),
        ("stop", 7680, 7680, True),
        ("stop", 7679, 7680, False),
        ("", "", 7680, False),
    ]
    observed_truncation = []
    for finish_reason, completion_tokens, max_tokens, expected in truncation_cases:
        observed = bool(evaluator.is_truncated_completion(finish_reason, completion_tokens, max_tokens))
        observed_truncation.append(
            {
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "max_tokens": max_tokens,
                "observed": observed,
                "expected": expected,
            }
        )
        expect(
            errors,
            observed == expected,
            f"evaluate_lora_adapter truncation fixture mismatch: {finish_reason}/{completion_tokens}",
        )

    solution = pd.DataFrame(
        [
            {
                "id": "logic_one",
                "prompt": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
                "answer": "42",
                "family_gate": "bit_manipulation",
            }
        ]
    )
    baseline_predictions = pd.DataFrame(
        [{"id": "logic_one", "raw_output": r"\boxed{42}", "finish_reason": "stop", "completion_tokens": "8"}]
    )
    multi_box_predictions = pd.DataFrame(
        [{"id": "logic_one", "raw_output": r"\boxed{0}\n\boxed{42}", "finish_reason": "stop", "completion_tokens": "8"}]
    )
    truncated_predictions = pd.DataFrame(
        [
            {
                "id": "logic_one",
                "raw_output": r"\boxed{42}",
                "finish_reason": "stop",
                "completion_tokens": str(EXPECTED_OFFICIAL_CONFIG["max_tokens"]),
            }
        ]
    )
    max_tokens = int(EXPECTED_OFFICIAL_CONFIG["max_tokens"])
    baseline = solve_rate.score_predictions(solution, baseline_predictions, "baseline", max_tokens=max_tokens)
    multi_box = solve_rate.score_predictions(solution, multi_box_predictions, "candidate", max_tokens=max_tokens)
    truncated = solve_rate.score_predictions(solution, truncated_predictions, "candidate", max_tokens=max_tokens)
    expect(errors, bool(multi_box["correct"].iloc[0]) is True, "multi-box fixture should be public-metric correct")
    expect(errors, bool(multi_box["exact_one_closed_boxed"].iloc[0]) is False, "multi-box fixture should not be exact-one boxed")
    expect(errors, bool(truncated["correct"].iloc[0]) is True, "truncation fixture should be public-metric correct")
    expect(errors, bool(truncated["truncated_detected"].iloc[0]) is True, "truncation fixture not detected")
    multi_approved, multi_reasons, _ = solve_rate.compare(
        baseline,
        multi_box,
        family_regression_tolerance=0.0,
        min_net_gain=-1.0,
        min_boxed_rate=1.0,
        max_candidate_truncated=0,
    )
    trunc_approved, trunc_reasons, _ = solve_rate.compare(
        baseline,
        truncated,
        family_regression_tolerance=0.0,
        min_net_gain=-1.0,
        min_boxed_rate=1.0,
        max_candidate_truncated=0,
    )
    expect(
        errors,
        not multi_approved and any("boxed_rate" in reason for reason in multi_reasons),
        "solve_rate did not reject multi-box fixture",
    )
    expect(
        errors,
        not trunc_approved and any("truncated" in reason for reason in trunc_reasons),
        "solve_rate did not reject truncation fixture",
    )

    return {
        "errors": errors,
        "truncation_cases": observed_truncation,
        "multi_box_reasons": multi_reasons,
        "truncation_reasons": trunc_reasons,
    }


def audit_sources() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sources = {
        "competition_utils": ROOT / "src" / "competition_utils.py",
        "evaluate_lora_adapter": ROOT / "scripts" / "evaluate_lora_adapter.py",
        "evaluate_lora_adapters_batch": ROOT / "scripts" / "evaluate_lora_adapters_batch.py",
        "active_gate_registry_audit": ROOT / "scripts" / "kg1_active_gate_registry_audit.py",
        "hf_job_train_v90": ROOT / "scripts" / "hf_job_train_v90.py",
        "graft_builder": ROOT / "scripts" / "kg1_v1243_solver_to_lora_graft_builder.py",
        "graft_contract_gate": ROOT / "scripts" / "kg1_v1243_graft_trainer_contract_gate.py",
        "colab_launcher": ROOT / "scripts" / "kg1_colab_v1243_launcher.py",
        "colab_realtime_runner": ROOT / "scripts" / "kg1_colab_realtime_runner.py",
        "live_log_common": ROOT / "scripts" / "kg1_live_log_common.py",
        "v1241_transfer_gate": ROOT / "scripts" / "kg1_v1241_bit_equation_transfer_gate.py",
        "full947_baseline_readiness_gate": ROOT / "scripts" / "kg1_v1241_full947_baseline_readiness_gate.py",
        "solve_rate_gate": ROOT / "scripts" / "solve_rate_gate.py",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}

    expect(errors, "def ensure_prompt_suffix(" in text["competition_utils"], "missing ensure_prompt_suffix helper")
    expect(
        errors,
        'user_content = ensure_prompt_suffix(getattr(row, "prompt"), prompt_suffix)' in text["evaluate_lora_adapter"],
        "evaluate_lora_adapter is not using ensure_prompt_suffix",
    )
    expect(
        errors,
        'user_content = str(getattr(row, "prompt")) + prompt_suffix' not in text["evaluate_lora_adapter"],
        "evaluate_lora_adapter still has unconditional suffix append",
    )
    expect(errors, "def is_truncated_completion(" in text["evaluate_lora_adapter"], "evaluate_lora_adapter missing robust truncation helper")
    expect(errors, "token_count >= int(max_tokens)" in text["evaluate_lora_adapter"], "evaluate_lora_adapter does not detect max-token hits")
    expect(errors, "is_truncated_completion" in text["evaluate_lora_adapters_batch"], "batch evaluator missing shared truncation helper")

    match = re.search(r"DEFAULT_LORA_TARGET_MODULES\s*=\s*\(\s*\"([^\"]+)\"\s*\)", text["hf_job_train_v90"], re.S)
    expect(errors, bool(match), "hf_job_train_v90 default LoRA modules block not found")
    if match:
        expect(errors, match.group(1) == SAFE_LORA_MODULE_STRING, "hf_job_train_v90 default LoRA modules mismatch")
    expect(errors, "prompt_suffix_count = prompt.count(PROMPT_SUFFIX.strip())" in text["graft_builder"], "graft builder missing prompt suffix count audit")
    expect(errors, "def ps_quote(" in text["graft_builder"], "graft builder missing PowerShell command quoting helper")
    expect(errors, "--baseline-predictions" in text["graft_builder"], "graft builder missing canonical baseline flag")
    expect(errors, "--candidate-predictions" in text["graft_builder"], "graft builder missing canonical candidate flag")
    expect(errors, "SCORE_CONTRACT_RUNTIME_CHECK" in text["graft_builder"], "graft builder missing score contract env")
    expect(errors, "def score_contract_runtime_report(" in text["hf_job_train_v90"], "trainer missing score contract runtime report")
    expect(errors, "KG1_SCORE_CONTRACT_STATUS=" in text["hf_job_train_v90"], "trainer missing score contract status log")
    expect(errors, '"score_contract_runtime"' in text["hf_job_train_v90"], "trainer manifest missing score contract runtime")
    expect(errors, "SCORE_PROXY_EVAL_CHECK" in text["graft_builder"], "graft builder missing score proxy env")
    expect(errors, "SCORE_TRAJECTORY_CHECK" in text["graft_builder"], "graft builder missing score trajectory env")
    expect(errors, "REQUIRE_SCORE_TRAJECTORY_FINAL_ONLY" in text["graft_builder"], "graft builder missing final-only score trajectory env")
    expect(
        errors,
        "def enforce_micro_consolidation_real_train_guards(" in text["colab_launcher"],
        "launcher missing enforced micro real_train score trajectory guard",
    )
    expect(
        errors,
        'args.phase != "micro_consolidation" or args.run_mode != "real_train"' in text["colab_launcher"],
        "launcher micro real_train guard is not phase/run-mode scoped",
    )
    expect(errors, "def evaluate_score_proxy(" in text["hf_job_train_v90"], "trainer missing score proxy eval")
    expect(errors, "KG1_SCORE_PROXY_STATUS=" in text["hf_job_train_v90"], "trainer missing score proxy status log")
    expect(errors, "def score_trajectory_report(" in text["hf_job_train_v90"], "trainer missing score trajectory report")
    expect(errors, "REQUIRE_SCORE_TRAJECTORY_FINAL_ONLY" in text["hf_job_train_v90"], "trainer missing final-only trajectory guard")
    expect(errors, 'str(label).lower() == "final"' in text["hf_job_train_v90"], "trainer final-only trajectory guard is not label-gated")
    expect(errors, "weak_families_individually_improved" in text["hf_job_train_v90"], "trainer trajectory can average-mask bit/equation regression")
    expect(errors, "bit_and_equation_improved_without_global_or_protected_regression" in text["hf_job_train_v90"], "trainer trajectory missing individual bit/equation success reason")
    expect(errors, "KG1_SCORE_TRAJECTORY_STATUS=" in text["hf_job_train_v90"], "trainer missing score trajectory status log")
    expect(errors, "score_trajectory_alignment" in text["hf_job_train_v90"], "trainer missing score trajectory alignment flag")
    expect(errors, "TRAJECTORY_RE" in text["live_log_common"], "live log parser missing score trajectory regex")
    expect(errors, "score_trajectory" in text["live_log_common"], "live log state missing score trajectory")
    expect(errors, "KG1_WATCHDOG_STALE_SECONDS" in text["colab_realtime_runner"], "realtime runner missing stale watchdog env")
    expect(errors, "KG1_WATCHDOG_MAX_RUNTIME_SECONDS" in text["colab_realtime_runner"], "realtime runner missing max runtime watchdog env")
    expect(errors, "KG1_WATCHDOG_STOP" in text["colab_realtime_runner"], "realtime runner missing watchdog stop marker")
    expect(errors, "health_stop" in text["colab_realtime_runner"], "realtime runner missing health watchdog")
    expect(errors, "KG1_WATCHDOG_STOP" in text["live_log_common"], "live log parser missing watchdog fatal marker")
    expect(errors, '"boxed_tail_exact_rate"' in text["hf_job_train_v90"], "trainer missing boxed-tail exact proxy")
    expect(errors, '"score_proxy"' in text["hf_job_train_v90"], "trainer manifest missing score proxy report")
    expect(errors, "FRIENDLY_REALTIME_LOGS" in text["graft_builder"], "graft builder missing friendly log env")
    expect(errors, "def kg1_teach_card(" in text["hf_job_train_v90"], "trainer missing friendly real-time log cards")
    expect(errors, "[KG1-TEACH]" in text["hf_job_train_v90"], "trainer missing KG1-TEACH marker")
    expect(errors, "O que e:" in text["hf_job_train_v90"], "trainer missing friendly what field")
    expect(errors, "Por que importa:" in text["hf_job_train_v90"], "trainer missing friendly why field")
    expect(errors, "audit_manifest_commands" in text["graft_contract_gate"], "contract gate missing manifest command audit")
    expect(errors, '"--baseline-predictions", "--baseline-csv"' in text["v1241_transfer_gate"], "V1241 transfer gate missing legacy baseline alias")
    expect(errors, 'if argv and argv[0] == "compare":' in text["v1241_transfer_gate"], "V1241 transfer gate missing legacy compare compatibility")
    expect(errors, '"baseline_identity": baseline_identity' in text["v1241_transfer_gate"], "V1241 transfer gate missing baseline identity report")
    expect(errors, "baseline_public_identity_unverified" in text["v1241_transfer_gate"], "V1241 transfer gate missing public baseline identity blocker")
    expect(errors, "baseline_strict_clean_identity_unverified" in text["v1241_transfer_gate"], "V1241 transfer gate missing strict-clean baseline blocker")
    expect(errors, "fail_full947_dirty_baseline_identity" in text["v1241_transfer_gate"], "V1241 transfer gate missing dirty full947 identity self-test")
    expect(errors, "--baseline-identity-probe" in text["v1241_transfer_gate"], "V1241 transfer gate missing baseline identity probe CLI")
    expect(errors, "def run_baseline_identity_probe(" in text["v1241_transfer_gate"], "V1241 transfer gate missing baseline identity probe function")
    expect(errors, "probe_fail_full947_dirty_baseline_identity" in text["v1241_transfer_gate"], "V1241 transfer gate missing dirty baseline identity probe self-test")
    expect(
        errors,
        "kg1_v1241_full947_baseline_readiness_gate.py" in text["graft_builder"],
        "graft builder missing full947 baseline readiness gate command",
    )
    expect(
        errors,
        "official_train_seed42_stratified10_val.csv" in text["graft_builder"],
        "graft builder missing canonical full947 solution path",
    )
    expect(
        errors,
        "def audit_canonical_full947_overlap(" in text["graft_builder"],
        "graft builder missing canonical full947 leak-check helper",
    )
    expect(
        errors,
        '"train_full947_prompt_overlap"' in text["graft_builder"],
        "graft builder missing train/full947 overlap field",
    )
    expect(
        errors,
        '"val_full947_prompt_overlap"' in text["graft_builder"],
        "graft builder missing val/full947 overlap field",
    )
    expect(
        errors,
        CANONICAL_FULL947_SOLUTION_SHA256 in text["graft_builder"],
        "graft builder missing canonical full947 solution sha256",
    )
    expect(
        errors,
        "--natural-baseline-predictions" in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate missing natural 086 input",
    )
    expect(
        errors,
        "--answer-only-baseline-predictions" in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate missing answer-only 086 input",
    )
    expect(
        errors,
        "answer_only_086_baseline_identity_probe_not_pass" in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate must block on answer-only probe failure",
    )
    expect(
        errors,
        CANONICAL_FULL947_SOLUTION_SHA256 in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate missing canonical solution sha256",
    )
    expect(
        errors,
        '"target": ">=0.89"' in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate missing >=0.89 objective",
    )
    expect(
        errors,
        '"target_correct": TARGET_089_CORRECT' in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate missing 843/947 target math",
    )
    expect(
        errors,
        "natural_probe_passed" in text["full947_baseline_readiness_gate"],
        "full947 baseline readiness gate self-test must preserve natural diagnostic probe status",
    )
    expect(errors, "exact_one_closed_boxed" in text["solve_rate_gate"], "solve_rate_gate missing exact boxed gate")
    expect(errors, "truncated_detected" in text["solve_rate_gate"], "solve_rate_gate missing truncation gate")
    expect(errors, "--max-candidate-truncated" in text["solve_rate_gate"], "solve_rate_gate missing truncation threshold")
    expect(
        errors,
        "pass_active_gate_registry_no_train_no_paid_no_submit" in text["active_gate_registry_audit"],
        "active gate registry audit missing no-train/no-paid decision",
    )

    stale_9_module_matches: list[str] = []
    for path in (ROOT / "scripts").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md"}:
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "lm_head,o_proj,out_proj" not in body:
            continue
        has_current_v1243_contract = (
            "SCORE_CONTRACT_EXPECTED_TRAINABLE_MODULES" in body
            or "TRAINABLE_DELTA_MODULES" in body
            or "SAFE_TRAINABLE_LORA_MODULES" in body
        )
        if not has_current_v1243_contract:
            stale_9_module_matches.append(str(path.relative_to(ROOT)))
    if stale_9_module_matches:
        warnings.append(f"historical_9_module_lora: stale references found in {stale_9_module_matches[:8]}")

    stale_patterns = [
        ("historical_stale_max_num_seqs_128", "max_num_seqs 128", ROOT / "docs"),
        ("historical_stale_max_tokens_3500", "max_tokens 3500", ROOT / "docs"),
    ]
    for label, pattern, base in stale_patterns:
        matches: list[str] = []
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".md"}:
                continue
            if path.name == Path(__file__).name:
                continue
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if pattern in body:
                matches.append(str(path.relative_to(ROOT)))
        if matches:
            warnings.append(f"{label}: historical stale references found in {matches[:8]}")

    return {
        "errors": errors,
        "warnings": warnings,
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
    }


def run(args: argparse.Namespace) -> int:
    print("[score-path-audit] START", flush=True)
    print(f"[score-path-audit] root={ROOT}", flush=True)
    output_dir = args.output_dir.resolve()

    checks: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for name, func in [
        ("official_config", audit_official_config),
        ("prompt_rendering", audit_prompt_rendering),
        ("v1243_artifacts", audit_v1243_artifacts),
        (
            "dryrun_bit",
            lambda: audit_dryrun_report("bit", 724, {"v1243_bit_solver_verified": 0.664540}),
        ),
        (
            "dryrun_equation",
            lambda: audit_dryrun_report("equation", 544, {"v1243_equation_solver_verified": 0.617464}),
        ),
        (
            "dryrun_micro_consolidation",
            lambda: audit_dryrun_report(
                "micro_consolidation",
                1084,
                {
                    "v1243_bit_solver_verified": 0.414777,
                    "v1243_equation_solver_verified": 0.316020,
                    "v1243_protected_replay": 0.269203,
                },
            ),
        ),
        ("existing_gate_reports", audit_existing_gate_reports),
        ("score_logic_fixtures", audit_score_logic_fixtures),
        ("sources", audit_sources),
    ]:
        print(f"[score-path-audit] check={name}", flush=True)
        try:
            check = func()
        except Exception as exc:  # pragma: no cover - gate surface
            check = {"errors": [f"{name}: exception: {type(exc).__name__}: {exc}"]}
        checks[name] = check
        errors.extend(str(error) for error in check.get("errors", []))
        warnings.extend(str(warning) for warning in check.get("warnings", []))

    report = {
        "schema_version": "kg1_score_path_operational_audit_v2",
        "generated_at_utc": utc_now(),
        "decision": "pass_score_path_operational_audit_no_gpu_no_submit" if not errors else "fail",
        "root": str(ROOT),
        "not_authorized": ["paid_gpu_launch", "adapter_package", "kaggle_submit", "score_claim"],
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }
    report_path = output_dir / "kg1_score_path_operational_audit.json"
    write_json(report_path, report)

    print(f"[score-path-audit] decision={report['decision']}", flush=True)
    print(f"[score-path-audit] errors={len(errors)} warnings={len(warnings)}", flush=True)
    print(f"[score-path-audit] report={report_path}", flush=True)
    print("[score-path-audit] END", flush=True)
    if errors:
        for error in errors[:50]:
            print(f"[score-path-audit] ERROR {error}", flush=True)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
