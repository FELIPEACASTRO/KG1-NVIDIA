#!/usr/bin/env python3
"""KG1 V1243 GRAFT trainer contract gate.

CPU-only audit for the solver-to-LoRA transfer artifacts. It verifies that the
V1243 datasets, trainer hooks, and dry-run environment preview are consistent
before any GPU job, adapter package, or Kaggle submission is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    extract_closed_boxed_answers,
    extract_final_answer,
    has_unclosed_boxed_answer,
    verify_answer,
)


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft_contract_gate"
DEFAULT_TRAINER = ROOT / "scripts" / "hf_job_train_v90.py"

EXPECTED_DATASETS = {
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
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            rows.append(item)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def resolve_artifact_file(artifact_dir: Path, value: object) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    normalized = raw.replace("\\", "/")
    if "\\" in raw or ":" in raw[:3]:
        normalized = Path(normalized).name
    return (artifact_dir / normalized).resolve()


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def audit_dataset(name: str, rows: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    row_errors: list[str] = []
    ids = [str(row.get("id", "")) for row in rows]
    family_counts = count_values(rows, "family")
    source_role_counts = count_values(rows, "source_role")
    weight_by_family: Counter[str] = Counter()
    prompt_hashes: set[str] = set()
    boxed_payload_hashes: set[str] = set()
    total_boxed_payload_chars = 0

    expect(errors, len(rows) == expected["rows"], f"{name}: rows {len(rows)} != {expected['rows']}")
    expect(errors, family_counts == expected["family_counts"], f"{name}: unexpected family_counts={family_counts}")
    if expected.get("source_role_counts") is not None:
        expect(
            errors,
            source_role_counts == expected["source_role_counts"],
            f"{name}: unexpected source_role_counts={source_role_counts}",
        )
    for row in rows:
        try:
            row_weight = float(row.get("row_loss_weight", row.get("loss_weight", 1.0)) or 0.0)
        except (TypeError, ValueError):
            row_weight = 0.0
        weight_by_family[str(row.get("family") or "")] += row_weight
    rounded_weight_by_family = {
        key: round(value, 6)
        for key, value in sorted(weight_by_family.items())
    }
    if expected.get("weight_by_family") is not None:
        expect(
            errors,
            rounded_weight_by_family == expected["weight_by_family"],
            f"{name}: unexpected weight_by_family={rounded_weight_by_family}",
        )
    expect(errors, len(ids) == len(set(ids)), f"{name}: duplicate ids detected")

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"row_{index}"))
        answer = row.get("answer")
        user_text = first_message_content(row, "user")
        assistant_text = last_message_content(row, "assistant")
        prompt = prompt_text(row)
        prompt_hashes.add(sha256_text(prompt))

        local_errors: list[str] = []
        if answer is None or str(answer).strip() == "":
            local_errors.append("missing top-level answer")
        if not isinstance(row.get("messages"), list) or not row.get("messages"):
            local_errors.append("missing messages")
        if not user_text:
            local_errors.append("missing user message")
        if not assistant_text:
            local_errors.append("missing assistant message")
        if prompt != user_text:
            local_errors.append("prompt differs from user message")
        prompt_suffix_count = prompt.count(PROMPT_SUFFIX.strip())
        if prompt_suffix_count != 1:
            local_errors.append(f"official prompt suffix count {prompt_suffix_count} != 1")
        if PROMPT_SUFFIX not in prompt:
            local_errors.append("official prompt suffix missing")
        if any((ord(char) < 32 and char not in "\n\r\t") for char in prompt + assistant_text):
            local_errors.append("control character in prompt or assistant target")
        if has_unclosed_boxed_answer(assistant_text):
            local_errors.append("unclosed boxed answer")

        boxed_answers = extract_closed_boxed_answers(assistant_text)
        if len(boxed_answers) != 1:
            local_errors.append(f"boxed answer count {len(boxed_answers)} != 1")
        else:
            boxed_payload = boxed_answers[0]
            boxed_payload_hashes.add(sha256_text(boxed_payload))
            total_boxed_payload_chars += len(boxed_payload)
            if not verify_answer(answer, boxed_payload):
                local_errors.append("boxed payload does not verify against answer")

        stripped_assistant = assistant_text.strip()
        if not stripped_assistant.startswith("</think>\n\\boxed{"):
            local_errors.append("assistant target does not start with '</think>\\n\\\\boxed{'")
        if stripped_assistant.count("\\boxed{") != 1:
            local_errors.append("assistant target contains multiple boxed markers")
        if not verify_answer(answer, extract_final_answer(assistant_text)):
            local_errors.append("final extracted answer does not verify against answer")

        if local_errors and len(row_errors) < 30:
            row_errors.append(f"{name}:{index}:{row_id}: " + "; ".join(local_errors))

    errors.extend(row_errors)
    return {
        "name": name,
        "rows": len(rows),
        "family_counts": family_counts,
        "source_role_counts": source_role_counts,
        "weight_by_family": rounded_weight_by_family,
        "prompt_hashes": len(prompt_hashes),
        "boxed_payload_hashes": len(boxed_payload_hashes),
        "avg_boxed_payload_chars": round(total_boxed_payload_chars / max(1, len(rows)), 4),
        "errors": errors,
    }


def audit_env_preview(artifact_dir: Path, env_preview: dict[str, Any], datasets: dict[str, Path]) -> dict[str, Any]:
    errors: list[str] = []
    full_adapter_target_modules = "down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj"
    trainable_delta_modules = "down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj"
    init_adapter_repo = "felipesp1983/kg1-recovered-v291-v290-checkpoint6-submit086"
    init_adapter_revision = "f4134a6d223249d27be2f1c5d94ed59d118d1ce5"
    init_adapter_config_sha256 = "a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d"
    init_adapter_weights_sha256 = "0a7b6144231d9358ae73a5e57d8778b32be1520fa47e3041414b3e025aaa1aa1"
    required_common = {
        "DRY_RUN_VALIDATE_ONLY": "1",
        "TOKENIZE_ONLY_DRY_RUN": "1",
        "UPLOAD_TO_HF": "0",
        "REQUIRE_BOXED_PAYLOAD_WEIGHT": "1",
        "REQUIRE_OFFSET_MASK": "1",
        "SCORE_CONTRACT_RUNTIME_CHECK": "1",
        "REQUIRE_SCORE_CONTRACT_PASS": "1",
        "SCORE_CONTRACT_TARGET_ACCURACY": "0.89",
        "SCORE_CONTRACT_FULL_ROWS": "947",
        "SCORE_CONTRACT_BASELINE_CORRECT": "823",
        "SCORE_CONTRACT_EXPECTED_TARGET_MODULES": full_adapter_target_modules,
        "SCORE_CONTRACT_EXPECTED_TRAINABLE_MODULES": trainable_delta_modules,
        "SCORE_CONTRACT_REQUIRE_TRAINABLE_FILTER": "1",
        "SCORE_CONTRACT_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "SCORE_PROXY_EVAL_CHECK": "1",
        "SCORE_PROXY_EVAL_MAX_EXAMPLES": "170",
        "SCORE_TRAJECTORY_CHECK": "1",
        "REQUIRE_SCORE_TRAJECTORY_PASS": "0",
        "SCORE_TRAJECTORY_MIN_WEAK_EXACT_DELTA": "0.0",
        "SCORE_TRAJECTORY_MAX_PROTECTED_EXACT_DROP": "0.0",
        "SCORE_TRAJECTORY_MAX_OVERALL_EXACT_DROP": "0.0",
        "SCORE_TRAJECTORY_MAX_BOXED_LOSS_REGRESSION": "0.0",
        "BASELINE_EVAL_BEFORE_TRAIN": "1",
        "EVAL_MAX_EXAMPLES": "170",
        "FRIENDLY_REALTIME_LOGS": "1",
        "FRIENDLY_LOG_SCORE_HINTS": "1",
        "MIN_VAL_EXAMPLES": "170",
        "MIN_TOKENIZED_VAL_EXAMPLES": "170",
        "SAMPLING_MODE": "weighted_replacement",
        "LORA_R": "32",
        "LORA_ALPHA": "32",
        "LORA_TARGET_MODULES": full_adapter_target_modules,
        "TRAINABLE_LORA_MODULES": trainable_delta_modules,
        "REQUIRE_INIT_ADAPTER": "1",
        "REQUIRE_INIT_ADAPTER_REVISION": "1",
        "INIT_ADAPTER_REPO": init_adapter_repo,
        "INIT_ADAPTER_REVISION": init_adapter_revision,
        "EXPECTED_INIT_ADAPTER_CONFIG_SHA256": init_adapter_config_sha256,
        "EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256": init_adapter_weights_sha256,
        "INIT_ADAPTER_LOAD_MODE": "manual",
        "PEFT_MANUAL_LOAD_METHOD": "direct",
    }
    expected_train_rows = {"bit_specialist": "724", "equation_specialist": "544"}

    for phase in ("bit_specialist", "equation_specialist"):
        env = env_preview.get(phase)
        if not isinstance(env, dict):
            errors.append(f"missing env preview phase: {phase}")
            continue
        for key, value in required_common.items():
            expect(errors, str(env.get(key)) == value, f"{phase}: {key}={env.get(key)!r} != {value!r}")
        try:
            boxed_weight = float(env.get("BOXED_PAYLOAD_LOSS_WEIGHT", "0"))
        except ValueError:
            boxed_weight = 0.0
        expect(errors, boxed_weight > 1.0, f"{phase}: BOXED_PAYLOAD_LOSS_WEIGHT must be > 1.0")
        expect(
            errors,
            str(env.get("MIN_TRAIN_EXAMPLES")) == expected_train_rows[phase],
            f"{phase}: MIN_TRAIN_EXAMPLES mismatch",
        )
        expect(
            errors,
            str(env.get("MIN_TOKENIZED_TRAIN_EXAMPLES")) == expected_train_rows[phase],
            f"{phase}: MIN_TOKENIZED_TRAIN_EXAMPLES mismatch",
        )

        data_path = resolve_artifact_file(artifact_dir, env.get("DATA_FILE", ""))
        val_path = resolve_artifact_file(artifact_dir, env.get("VAL_FILE", ""))
        expect(errors, data_path == datasets[phase].resolve(), f"{phase}: DATA_FILE points to {data_path}")
        expect(errors, val_path == datasets["val170"].resolve(), f"{phase}: VAL_FILE points to {val_path}")
        if data_path.exists():
            expect(
                errors,
                str(env.get("EXPECTED_TRAIN_SHA256", "")).lower() == sha256_file(data_path).lower(),
                f"{phase}: EXPECTED_TRAIN_SHA256 mismatch",
            )
        if val_path.exists():
            expect(
                errors,
                str(env.get("EXPECTED_VAL_SHA256", "")).lower() == sha256_file(val_path).lower(),
                f"{phase}: EXPECTED_VAL_SHA256 mismatch",
            )

    return {"errors": errors}


def audit_manifest_commands(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    commands = manifest.get("gate_commands", {})
    if not isinstance(commands, dict):
        return {"errors": ["manifest gate_commands must be a dict"]}
    expected_profiles = {
        "tiny": "--profile tiny",
        "val170": "--profile val170",
        "full947_089": "--profile full947_089",
    }
    for name, profile_flag in expected_profiles.items():
        command = str(commands.get(name, ""))
        expect(errors, bool(command), f"manifest gate command missing: {name}")
        expect(errors, "kg1_v1241_bit_equation_transfer_gate.py" in command, f"{name}: wrong gate script")
        expect(errors, profile_flag in command, f"{name}: missing {profile_flag}")
        expect(errors, "--solution-csv" in command, f"{name}: missing --solution-csv")
        expect(errors, "--baseline-predictions" in command, f"{name}: missing --baseline-predictions")
        expect(errors, "--candidate-predictions" in command, f"{name}: missing --candidate-predictions")
        expect(errors, " compare " not in f" {command} ", f"{name}: obsolete compare subcommand present")
        expect(errors, "--baseline-csv" not in command, f"{name}: obsolete --baseline-csv present")
        expect(errors, "--candidate-csv" not in command, f"{name}: obsolete --candidate-csv present")
    return {"errors": errors, "commands": commands}


def audit_trainer_source(trainer_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    source_bytes = trainer_path.read_bytes()
    source = source_bytes.decode("utf-8")
    required_snippets = [
        'BOXED_PAYLOAD_LOSS_WEIGHT = env_float("BOXED_PAYLOAD_LOSS_WEIGHT", 1.0)',
        'REQUIRE_BOXED_PAYLOAD_WEIGHT = env_bool("REQUIRE_BOXED_PAYLOAD_WEIGHT", False)',
        'DEFAULT_LORA_TARGET_MODULES = (',
        '"down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj"',
        "def apply_boxed_payload_loss_weight(",
        "PEFT_IMPORT_ERROR",
        "def require_peft() -> None:",
        "return input_ids, loss_mask, True, False, boxed_weighted_tokens",
        'if REQUIRE_BOXED_PAYLOAD_WEIGHT:',
        '"boxed_payload_weighted_tokens"',
        "def row_sampling_weight(",
        '"row_loss_weight": row_sampling_weight(ex)',
        "weight = row_sampling_weight(item)",
        "def loss_weighting_report(",
        "loss_mask = torch.tensor(loss_mask_batch, dtype=torch.float32",
        "shift_mask = loss_mask[..., 1:].contiguous().float()",
        'SCORE_CONTRACT_RUNTIME_CHECK = env_bool("SCORE_CONTRACT_RUNTIME_CHECK", True)',
        'REQUIRE_SCORE_CONTRACT_PASS = env_bool("REQUIRE_SCORE_CONTRACT_PASS", False)',
        'SCORE_CONTRACT_EXPECTED_TRAINABLE_MODULES = env_str(',
        '"expected_trainable_lora_modules": expected_trainable_modules',
        "def score_contract_runtime_report(",
        "KG1_SCORE_CONTRACT_RUNTIME_JSON_BEGIN",
        "KG1_SCORE_CONTRACT_STATUS=",
        '"score_contract_runtime"',
        'SCORE_PROXY_EVAL_CHECK = env_bool("SCORE_PROXY_EVAL_CHECK", True)',
        "def evaluate_score_proxy(",
        "KG1_SCORE_PROXY_EVAL_JSON_BEGIN",
        "KG1_SCORE_PROXY_STATUS=",
        '"boxed_tail_exact_rate"',
        '"score_proxy"',
        'SCORE_TRAJECTORY_CHECK = env_bool("SCORE_TRAJECTORY_CHECK", True)',
        "def score_trajectory_report(",
        "KG1_SCORE_TRAJECTORY_JSON_BEGIN",
        "KG1_SCORE_TRAJECTORY_STATUS=",
        '"score_trajectory"',
        'FRIENDLY_REALTIME_LOGS = env_bool("FRIENDLY_REALTIME_LOGS", True)',
        "def kg1_teach_card(",
        "[KG1-TEACH]",
        "O que e:",
        "Por que importa:",
        '"tokenization_contract_passed": True',
        '"full_training_allowed": False',
    ]
    for snippet in required_snippets:
        expect(errors, snippet in source, f"trainer missing snippet: {snippet}")

    bad_control_bytes = [
        (index, byte)
        for index, byte in enumerate(source_bytes)
        if byte < 32 and byte not in {9, 10, 13}
    ]
    expect(errors, not bad_control_bytes, f"trainer has suspicious control bytes: {bad_control_bytes[:5]}")
    return {
        "path": str(trainer_path),
        "sha256": sha256_file(trainer_path),
        "required_snippets": len(required_snippets),
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    trainer_path = args.trainer.resolve()
    print("[v1243-contract] START", flush=True)
    print(f"[v1243-contract] artifact_dir={artifact_dir}", flush=True)
    print(f"[v1243-contract] trainer={trainer_path}", flush=True)

    manifest_path = artifact_dir / "kg1_v1243_solver_to_lora_graft_manifest.json"
    env_path = artifact_dir / "v1243_hf_env_preview.json"
    for path in (manifest_path, env_path, trainer_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_json(manifest_path)
    env_preview = load_json(env_path)
    datasets = {
        name: (artifact_dir / spec["path"]).resolve()
        for name, spec in EXPECTED_DATASETS.items()
    }
    for path in datasets.values():
        if not path.exists():
            raise FileNotFoundError(path)

    dataset_reports: dict[str, Any] = {}
    all_errors: list[str] = []
    loaded_rows: dict[str, list[dict[str, Any]]] = {}
    for name, spec in EXPECTED_DATASETS.items():
        rows = load_jsonl(datasets[name])
        loaded_rows[name] = rows
        report = audit_dataset(name, rows, spec)
        dataset_reports[name] = report
        all_errors.extend(report["errors"])

    train_prompt_hashes = set()
    for name in ("bit_specialist", "equation_specialist", "protected_replay", "micro_consolidation"):
        train_prompt_hashes.update(sha256_text(prompt_text(row)) for row in loaded_rows[name])
    val_prompt_hashes = {sha256_text(prompt_text(row)) for row in loaded_rows["val170"]}
    overlap = sorted(train_prompt_hashes & val_prompt_hashes)
    expect(all_errors, not overlap, f"train/val prompt overlap count={len(overlap)}")

    expect(
        all_errors,
        manifest.get("decision") == "pass_v1243_cpu_dataset_graft_no_gpu_no_submit",
        f"unexpected manifest decision: {manifest.get('decision')}",
    )
    expect(
        all_errors,
        "paid_gpu_launch" in set(manifest.get("algorithm", {}).get("not_authorized", [])),
        "manifest must explicitly block paid_gpu_launch",
    )
    expect(
        all_errors,
        "kaggle_submit" in set(manifest.get("algorithm", {}).get("not_authorized", [])),
        "manifest must explicitly block kaggle_submit",
    )

    env_report = audit_env_preview(artifact_dir, env_preview, datasets)
    manifest_commands_report = audit_manifest_commands(manifest)
    trainer_report = audit_trainer_source(trainer_path)
    all_errors.extend(env_report["errors"])
    all_errors.extend(manifest_commands_report["errors"])
    all_errors.extend(trainer_report["errors"])

    report = {
        "schema_version": "kg1_v1243_graft_trainer_contract_gate_v1",
        "generated_at_utc": utc_now(),
        "decision": "pass_v1243_graft_trainer_contract_no_gpu_no_submit" if not all_errors else "fail",
        "artifact_dir": str(artifact_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "env_preview_sha256": sha256_file(env_path),
        "datasets": dataset_reports,
        "train_prompt_hashes": len(train_prompt_hashes),
        "val_prompt_hashes": len(val_prompt_hashes),
        "train_val_prompt_overlap": len(overlap),
        "env_preview": env_report,
        "manifest_commands": manifest_commands_report,
        "trainer": trainer_report,
        "not_authorized": ["paid_gpu_launch", "adapter_package", "kaggle_submit", "score_claim"],
        "errors": all_errors,
    }
    report_path = output_dir / "kg1_v1243_graft_trainer_contract_gate.json"
    write_json(report_path, report)

    print(f"[v1243-contract] decision={report['decision']}", flush=True)
    print(f"[v1243-contract] wrote {report_path}", flush=True)
    print("[v1243-contract] END", flush=True)
    if all_errors:
        for error in all_errors[:40]:
            print(f"[v1243-contract] ERROR {error}", flush=True)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
