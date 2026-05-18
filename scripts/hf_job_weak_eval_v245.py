#!/usr/bin/env python3
"""Run a guarded weak-family LoRA eval inside Hugging Face Jobs.

This runner is intentionally narrow: it validates the V245 weak CSV bridge
artifact before spending vLLM/model-load time, evaluates one or more adapters,
and uploads the eval outputs back to an existing HF model repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import PROMPT_SUFFIX, canonical_family, classify_puzzle  # noqa: E402


EXPECTED_WEAK_COUNTS = {"bit_manipulation": 160, "equation_transform": 155}
EXPECTED_WEAK_ROWS = 315
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
DEFAULT_DATA_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_WEAK_CSV_FILE = (
    "runtime_artifacts/v245_weak_eval_bridge/"
    "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_WEAK_MANIFEST_FILE = (
    "runtime_artifacts/v245_weak_eval_bridge/"
    "v245-weak-bridge-hfonly-20260510T1950Z/v245_weak_eval_bridge_manifest.json"
)
DEFAULT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures"
DEFAULT_ADAPTER_SUBFOLDER = "final"
DEFAULT_OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures"
DEFAULT_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
DEFAULT_PROMPT_SUFFIX = PROMPT_SUFFIX


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    raw = env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    raw = env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw!r}") from exc


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_str(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "y", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def log_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label} = {json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)}", flush=True)


def configure_text_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def torch_status() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": str(getattr(torch.version, "cuda", "")),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(props.name if props else ""),
        "gpu_total_gib": float(props.total_memory / 1024**3 if props else 0.0),
    }


def validate_gpu() -> None:
    status = torch_status()
    log_json("torch_gpu_status", status)
    cuda_runtime = str(status.get("cuda") or "")
    cuda_major = int(cuda_runtime.split(".", 1)[0]) if cuda_runtime[:1].isdigit() else 0
    flavor = env_str("KG1_HF_FLAVOR", "")
    if (
        "a100" in flavor.lower()
        and cuda_major >= 13
        and not env_bool("KG1_ALLOW_CUDA13_ON_A100", False)
    ):
        raise RuntimeError(
            "Blocked CUDA 13 runtime on HF A100 for weak vLLM eval. "
            "Use H200 for this vLLM image or a CUDA 12-compatible image."
        )
    if env_bool("KG1_REQUIRE_CUDA", True) and not status["cuda_available"]:
        raise RuntimeError("CUDA is required for weak vLLM eval.")
    min_gib = float(env_str("KG1_MIN_GPU_TOTAL_GIB", "79"))
    if status["gpu_total_gib"] < min_gib:
        raise RuntimeError(f"GPU memory below required floor: {status['gpu_total_gib']:.2f} < {min_gib:.2f}")
    required_regex = env_str("KG1_REQUIRED_GPU_NAME_REGEX", "")
    if required_regex:
        import re

        if not re.search(required_regex, status["gpu_name"], re.IGNORECASE):
            raise RuntimeError(f"GPU name {status['gpu_name']!r} does not match {required_regex!r}")


def validate_repo_commit() -> str:
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected = env_str("KG1_EXPECTED_COMMIT", "")
    print("repo_commit =", observed, flush=True)
    print("expected_repo_commit =", expected, flush=True)
    if expected and observed != expected:
        raise RuntimeError(f"repo commit mismatch: expected {expected}, got {observed}")
    return observed


def validate_weak_csv(path: Path, expected_csv_sha: str, expected_contract: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    observed_csv_sha = sha256_file(path)
    if expected_csv_sha and observed_csv_sha != expected_csv_sha:
        raise RuntimeError(f"weak CSV sha mismatch: expected {expected_csv_sha}, got {observed_csv_sha}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "prompt", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("weak CSV missing required columns: " + json.dumps(missing))
    if "family" not in frame.columns:
        if "type" in frame.columns:
            frame["family"] = frame["type"]
        else:
            frame["family"] = frame["prompt"].map(classify_puzzle)
    frame["family"] = frame["family"].map(canonical_family)
    inferred = frame["prompt"].map(classify_puzzle)
    mismatch = frame["family"].ne("unknown") & inferred.ne("unknown") & frame["family"].ne(inferred)
    if mismatch.any():
        sample = frame.loc[mismatch, ["id", "family", "prompt"]].head(5).to_dict(orient="records")
        raise RuntimeError("weak CSV family/prompt mismatch: " + json.dumps(sample, sort_keys=True))
    if frame["id"].fillna("").astype(str).eq("").any():
        raise RuntimeError("weak CSV contains empty ids")
    if frame["answer"].fillna("").astype(str).eq("").any():
        raise RuntimeError("weak CSV contains empty answers")
    counts = {str(k): int(v) for k, v in frame["family"].value_counts().sort_index().to_dict().items()}
    if counts != EXPECTED_WEAK_COUNTS:
        raise RuntimeError(f"weak family counts mismatch: expected {EXPECTED_WEAK_COUNTS}, got {counts}")
    if len(frame) != EXPECTED_WEAK_ROWS:
        raise RuntimeError(f"weak row count mismatch: expected {EXPECTED_WEAK_ROWS}, got {len(frame)}")
    if int(frame["id"].duplicated().sum()):
        raise RuntimeError("weak CSV has duplicate ids")
    frame["prompt_sha256"] = frame["prompt"].map(sha256_text)
    records = {
        str(row.id): (str(row.family), str(row.answer), str(row.prompt_sha256))
        for row in frame.itertuples(index=False)
    }
    digest_payload = "\n".join(
        f"{row_id}\t{family}\t{answer}\t{prompt_hash}"
        for row_id, (family, answer, prompt_hash) in sorted(records.items())
    )
    observed_contract = sha256_text(digest_payload)
    if expected_contract and observed_contract != expected_contract:
        raise RuntimeError(f"weak row contract mismatch: expected {expected_contract}, got {observed_contract}")
    return {
        "path": str(path),
        "sha256": observed_csv_sha,
        "observed_shared_row_contract_sha256": observed_contract,
        "rows": int(len(frame)),
        "family_counts": counts,
    }


def validate_weak_manifest(path: Path, expected_contract: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    observed = str(payload.get("canonical_weak_csv", {}).get("observed_shared_row_contract_sha256", ""))
    if expected_contract and observed != expected_contract:
        raise RuntimeError(f"weak manifest contract mismatch: expected {expected_contract}, got {observed}")
    return payload


def validate_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    config = read_json(config_path)
    expected_r = env_int("KG1_EXPECTED_LORA_R", 32)
    expected_alpha = env_int("KG1_EXPECTED_LORA_ALPHA", 32)
    if int(config.get("r", -1)) != expected_r:
        raise RuntimeError(f"adapter r mismatch: expected {expected_r}, got {config.get('r')}")
    if int(config.get("lora_alpha", -1)) != expected_alpha:
        raise RuntimeError(f"adapter alpha mismatch: expected {expected_alpha}, got {config.get('lora_alpha')}")
    modules_to_save = config.get("modules_to_save") or []
    if modules_to_save:
        raise RuntimeError(
            "adapter modules_to_save must be empty for KG1 adapter-only submit path: "
            + json.dumps(modules_to_save, sort_keys=True)
        )
    expected_base = env_str(
        "KG1_EXPECTED_ADAPTER_BASE_MODEL_NAME_OR_PATH",
        env_str("KG1_MODEL_NAME", DEFAULT_MODEL_NAME),
    ).rstrip("/")
    adapter_base = str(config.get("base_model_name_or_path") or "").rstrip("/")
    if expected_base and adapter_base and adapter_base != expected_base:
        raise RuntimeError(f"adapter base model mismatch: expected {expected_base}, got {adapter_base}")
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_config": str(config_path),
        "adapter_weights": str(weights_path),
        "adapter_weights_bytes": int(weights_path.stat().st_size),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "modules_to_save": modules_to_save,
        "target_modules": config.get("target_modules"),
        "target_parameters": config.get("target_parameters"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
    }


def parse_adapter_specs(adapter_repo: str, adapter_subfolders_raw: str) -> list[dict[str, str]]:
    specs_raw = env_str("KG1_ADAPTER_SPECS_JSON", "")
    if specs_raw:
        try:
            parsed = json.loads(specs_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("KG1_ADAPTER_SPECS_JSON must be valid JSON") from exc
        if not isinstance(parsed, list) or not parsed:
            raise RuntimeError("KG1_ADAPTER_SPECS_JSON must be a non-empty list")
        specs: list[dict[str, str]] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise RuntimeError(f"adapter spec at index {index} must be an object")
            repo = str(item.get("repo", "")).strip()
            subfolder = str(item.get("subfolder", "")).strip().strip("/")
            name = str(item.get("name", "")).strip()
            if not repo:
                raise RuntimeError(f"adapter spec at index {index} missing repo")
            if not name:
                suffix = subfolder.replace("/", "_") if subfolder else "root"
                name = f"candidate_{index}_{suffix}"
            specs.append({"repo": repo, "subfolder": subfolder, "name": name})
        return specs

    if adapter_subfolders_raw:
        adapter_subfolders = [part.strip().strip("/") for part in adapter_subfolders_raw.split(",") if part.strip()]
    else:
        adapter_subfolders = [env_str("KG1_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER).strip("/")]
    if not adapter_subfolders:
        raise RuntimeError("At least one adapter subfolder is required")

    candidate_names_raw = env_str("KG1_CANDIDATE_NAMES", "")
    candidate_names = [part.strip() for part in candidate_names_raw.split(",") if part.strip()] if candidate_names_raw else []
    if candidate_names and len(candidate_names) != len(adapter_subfolders):
        raise RuntimeError(
            f"KG1_CANDIDATE_NAMES count must match adapter count: {len(candidate_names)} != {len(adapter_subfolders)}"
        )

    specs = []
    for index, subfolder in enumerate(adapter_subfolders):
        name = (
            candidate_names[index]
            if candidate_names
            else env_str("KG1_CANDIDATE_NAME", "v244_" + (subfolder.replace("/", "_") or "adapter"))
        )
        specs.append({"repo": adapter_repo, "subfolder": subfolder, "name": name})
    return specs


def weak_promotion_gate(summary: dict[str, Any]) -> dict[str, Any]:
    total_min = env_int("KG1_WEAK_PROMOTE_TOTAL_MIN", 196)
    equation_min = env_int("KG1_WEAK_PROMOTE_EQUATION_MIN", 60)
    bit_min = env_int("KG1_WEAK_PROMOTE_BIT_MIN", 136)
    trunc_max = env_int("KG1_WEAK_PROMOTE_TRUNC_MAX", 0)
    # Official-like weak eval needs long reasoning for the current V290 baseline
    # itself. Keep length as an optional drift diagnostic, not a default
    # promotion blocker; truncation and protected-row guards remain mandatory.
    avg_completion_tokens_max = env_int("KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX", 0)
    max_completion_tokens_max = env_int("KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX", 0)
    label_aware_delta_max = env_int("KG1_WEAK_PROMOTE_LABEL_AWARE_DELTA_MAX", 0)
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status_ok = str(row.get("status", "")).lower() == "ok"
        correct = int(row.get("correct", 0) or 0)
        equation = int(row.get("equation_transform_correct", 0) or 0)
        bit = int(row.get("bit_manipulation_correct", 0) or 0)
        truncated = int(row.get("truncated", 0) or 0)
        label_aware_correct = int(row.get("label_aware_debug_correct", correct) or 0)
        label_aware_delta = max(0, label_aware_correct - correct)
        completion_tokens = int(row.get("completion_tokens", 0) or 0)
        candidate_rows = int(row.get("rows", summary.get("weak_rows", EXPECTED_WEAK_ROWS)) or EXPECTED_WEAK_ROWS)
        avg_completion_tokens = float(row.get("avg_completion_tokens", 0) or 0)
        if avg_completion_tokens <= 0 and candidate_rows > 0:
            avg_completion_tokens = completion_tokens / candidate_rows
        max_completion_tokens = int(row.get("max_completion_tokens", 0) or 0)
        avg_completion_tokens_blocked = (
            avg_completion_tokens_max > 0 and avg_completion_tokens > avg_completion_tokens_max
        )
        max_completion_tokens_blocked = (
            max_completion_tokens_max > 0 and max_completion_tokens > max_completion_tokens_max
        )
        passed = (
            status_ok
            and correct >= total_min
            and equation >= equation_min
            and bit >= bit_min
            and truncated <= trunc_max
            and not avg_completion_tokens_blocked
            and not max_completion_tokens_blocked
            and label_aware_delta <= label_aware_delta_max
        )
        candidates.append(
            {
                "name": str(row.get("name", "")),
                "status_ok": status_ok,
                "correct": correct,
                "equation_transform_correct": equation,
                "bit_manipulation_correct": bit,
                "truncated": truncated,
                "label_aware_debug_correct": label_aware_correct,
                "label_aware_minus_label_free_correct": label_aware_delta,
                "completion_tokens": completion_tokens,
                "avg_completion_tokens": avg_completion_tokens,
                "max_completion_tokens": max_completion_tokens,
                "passed": passed,
                "blocking_reasons": [
                    reason
                    for reason, blocked in [
                        ("status_not_ok", not status_ok),
                        (f"correct_lt_{total_min}", correct < total_min),
                        (f"equation_lt_{equation_min}", equation < equation_min),
                        (f"bit_lt_{bit_min}", bit < bit_min),
                        (f"truncated_gt_{trunc_max}", truncated > trunc_max),
                        (
                            f"avg_completion_tokens_gt_{avg_completion_tokens_max}",
                            avg_completion_tokens_blocked,
                        ),
                        (
                            f"max_completion_tokens_gt_{max_completion_tokens_max}",
                            max_completion_tokens_blocked,
                        ),
                        (
                            f"label_aware_delta_gt_{label_aware_delta_max}",
                            label_aware_delta > label_aware_delta_max,
                        ),
                    ]
                    if blocked
                ],
            }
        )
    passed_candidates = [row for row in candidates if row["passed"]]
    diagnostic_only = env_bool("KG1_WEAK_EVAL_DIAGNOSTIC_ONLY", False)
    enforced = env_bool("KG1_ENFORCE_WEAK_PROMOTION_GATE", not diagnostic_only)
    return {
        "enforced": enforced,
        "diagnostic_only": diagnostic_only,
        "thresholds": {
            "correct_min": total_min,
            "equation_transform_min": equation_min,
            "bit_manipulation_min": bit_min,
            "truncated_max": trunc_max,
            "avg_completion_tokens_max": avg_completion_tokens_max,
            "max_completion_tokens_max": max_completion_tokens_max,
            "label_aware_delta_max": label_aware_delta_max,
        },
        "candidate_count": len(candidates),
        "passed_candidate_count": len(passed_candidates),
        "passed_candidates": [row["name"] for row in passed_candidates],
        "candidates": candidates,
        "decision": "weak_promotion_gate_passed" if passed_candidates else "weak_promotion_gate_blocked",
    }


def catastrophic_eval_guard(summary: dict[str, Any]) -> dict[str, Any]:
    """Fail-fast on candidates that are clearly broken, not merely weak.

    This is a FinOps/crisis-mode guard.  A candidate with near-zero weak ACC and
    mass truncation should not continue through later checkpoints just because
    the normal consecutive-failure counter was disabled for exploration.
    """

    enabled = env_bool("KG1_CATASTROPHIC_EVAL_GUARD", True)
    allow_continue = env_bool("KG1_ALLOW_CATASTROPHIC_EVAL_CONTINUE", False)
    correct_max = env_int("KG1_CATASTROPHIC_CORRECT_MAX", 10)
    truncation_rate_min = float(env_str("KG1_CATASTROPHIC_TRUNCATION_RATE_MIN", "0.50"))
    truncated_min = env_int("KG1_CATASTROPHIC_TRUNCATED_MIN", 100)
    min_rows = env_int("KG1_CATASTROPHIC_MIN_ROWS", 100)
    bit_correct_max = env_int("KG1_CATASTROPHIC_BIT_CORRECT_MAX", 5)
    candidates: list[dict[str, Any]] = []
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status_ok = str(row.get("status", "")).lower() == "ok"
        candidate_rows = int(row.get("rows", summary.get("weak_rows", EXPECTED_WEAK_ROWS)) or EXPECTED_WEAK_ROWS)
        if not status_ok or candidate_rows < min_rows:
            continue
        correct = int(row.get("correct", 0) or 0)
        bit = int(row.get("bit_manipulation_correct", 0) or 0)
        truncated = int(row.get("truncated", 0) or 0)
        truncation_rate = float(row.get("truncation_rate", 0) or 0)
        if truncation_rate <= 0 and candidate_rows > 0:
            truncation_rate = truncated / candidate_rows
        near_zero_with_truncation = (
            correct <= correct_max
            and truncated >= truncated_min
            and truncation_rate >= truncation_rate_min
        )
        bit_family_destroyed = bit <= bit_correct_max and truncated >= truncated_min
        if near_zero_with_truncation or bit_family_destroyed:
            candidates.append(
                {
                    "name": str(row.get("name", "")),
                    "correct": correct,
                    "bit_manipulation_correct": bit,
                    "equation_transform_correct": int(row.get("equation_transform_correct", 0) or 0),
                    "rows": candidate_rows,
                    "truncated": truncated,
                    "truncation_rate": truncation_rate,
                    "reasons": [
                        reason
                        for reason, blocked in [
                            ("near_zero_acc_with_mass_truncation", near_zero_with_truncation),
                            ("bit_family_destroyed", bit_family_destroyed),
                        ]
                        if blocked
                    ],
                }
            )
    should_stop = bool(enabled and candidates and not allow_continue)
    return {
        "schema_version": "kg1_v245_catastrophic_eval_guard_v1",
        "enabled": enabled,
        "allow_continue": allow_continue,
        "thresholds": {
            "correct_max": correct_max,
            "bit_correct_max": bit_correct_max,
            "truncated_min": truncated_min,
            "truncation_rate_min": truncation_rate_min,
            "min_rows": min_rows,
        },
        "candidate_count": len(candidates),
        "catastrophic_candidates": candidates,
        "should_stop": should_stop,
        "decision": "catastrophic_eval_stop" if should_stop else "catastrophic_eval_pass",
    }


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def protected_row_backfire_guard(summary: dict[str, Any]) -> dict[str, Any]:
    """Run row-level regression guards on candidate prediction CSVs.

    Family totals are not enough for promotion. A candidate can preserve the
    aggregate threshold while regressing a known protected row. This guard runs
    after prediction CSVs are produced and before the weak promotion decision is
    accepted.
    """

    enforced = env_bool("KG1_PROTECTED_ROW_GUARD", env_bool("KG1_CRISIS_MODE_BACKFIRE_GUARD", False))
    guard = {
        "schema_version": "kg1_v245_protected_row_backfire_guard_v1",
        "enforced": enforced,
        "baseline_csv": "",
        "protected_id_answers": [],
        "candidate_count": 0,
        "passed_candidates": [],
        "blocked_candidates": [],
        "candidate_reports": [],
        "blockers": [],
        "passed": True,
    }
    if not enforced:
        return guard

    baseline_csv = _resolve_repo_path(
        env_str("KG1_PROTECTED_BASELINE_CSV", "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv")
    )
    protected_id_answers = [
        part.strip()
        for part in env_str("KG1_PROTECTED_ID_ANSWERS", "8740ed31=01101000,59bee375=10010101").split(",")
        if part.strip()
    ]
    guard["baseline_csv"] = str(baseline_csv)
    guard["protected_id_answers"] = protected_id_answers
    if not baseline_csv.exists():
        guard["blockers"].append(f"protected_baseline_missing:{baseline_csv}")
        guard["passed"] = False
        return guard
    if not protected_id_answers:
        guard["blockers"].append("protected_id_answers_empty")
        guard["passed"] = False
        return guard

    from scripts.kg1_weak_backfire_row_guard import audit as audit_backfire_rows  # noqa: E402

    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if str(row.get("status", "")).lower() != "ok":
            continue
        guard["candidate_count"] += 1
        report_json_raw = str(row.get("report_json", "")).strip()
        if not report_json_raw:
            blocker = f"candidate_report_json_missing:{name}"
            guard["blockers"].append(blocker)
            guard["blocked_candidates"].append(name)
            guard["candidate_reports"].append({"name": name, "passed": False, "blockers": [blocker]})
            continue
        report_json = Path(report_json_raw)
        if not report_json.exists():
            blocker = f"candidate_report_json_not_found:{name}:{report_json}"
            guard["blockers"].append(blocker)
            guard["blocked_candidates"].append(name)
            guard["candidate_reports"].append({"name": name, "passed": False, "blockers": [blocker]})
            continue
        report_payload = read_json(report_json)
        predictions_csv_raw = str(report_payload.get("outputs", {}).get("predictions_csv", "")).strip()
        if not predictions_csv_raw:
            blocker = f"candidate_predictions_csv_missing:{name}"
            guard["blockers"].append(blocker)
            guard["blocked_candidates"].append(name)
            guard["candidate_reports"].append({"name": name, "passed": False, "blockers": [blocker]})
            continue
        predictions_csv = Path(predictions_csv_raw)
        if not predictions_csv.exists():
            blocker = f"candidate_predictions_csv_not_found:{name}:{predictions_csv}"
            guard["blockers"].append(blocker)
            guard["blocked_candidates"].append(name)
            guard["candidate_reports"].append({"name": name, "passed": False, "blockers": [blocker]})
            continue
        output_json = report_json.with_name(f"{name}_protected_row_backfire_guard.json")
        audit_args = argparse.Namespace(
            baseline_csv=baseline_csv,
            candidate_csv=predictions_csv,
            protected_id_answer=list(protected_id_answers),
            baseline_prediction_column="prediction",
            candidate_prediction_column="prediction",
            baseline_correct_column="correct",
            candidate_correct_column="correct",
            output_json=output_json,
            allow_blocked=True,
        )
        candidate_report = audit_backfire_rows(audit_args)
        candidate_report["name"] = name
        candidate_report["output_json"] = str(output_json)
        guard["candidate_reports"].append(candidate_report)
        if candidate_report.get("passed"):
            guard["passed_candidates"].append(name)
        else:
            guard["blocked_candidates"].append(name)
            guard["blockers"].extend([f"{name}:{item}" for item in candidate_report.get("blockers", [])])

    guard["blocked_candidates"] = sorted(set(str(name) for name in guard["blocked_candidates"] if name))
    guard["passed_candidates"] = sorted(set(str(name) for name in guard["passed_candidates"] if name))
    guard["passed"] = not guard["blockers"]
    return guard


def apply_protected_row_guard(
    promotion_gate: dict[str, Any], protected_guard: dict[str, Any]
) -> dict[str, Any]:
    if not protected_guard.get("enforced"):
        promotion_gate["protected_row_backfire_guard_enforced"] = False
        return promotion_gate
    blocked = set(str(name) for name in protected_guard.get("blocked_candidates", []))
    if protected_guard.get("blockers") and not blocked:
        blocked = set(str(row.get("name", "")) for row in promotion_gate.get("candidates", []) if row.get("name"))
    updated_candidates = []
    for candidate in promotion_gate.get("candidates", []):
        candidate = dict(candidate)
        if str(candidate.get("name", "")) in blocked:
            candidate["passed"] = False
            reasons = list(candidate.get("blocking_reasons", []))
            if "protected_row_backfire_guard_failed" not in reasons:
                reasons.append("protected_row_backfire_guard_failed")
            candidate["blocking_reasons"] = reasons
        updated_candidates.append(candidate)
    passed_candidates = [row for row in updated_candidates if row.get("passed")]
    promotion_gate["protected_row_backfire_guard_enforced"] = True
    promotion_gate["protected_row_backfire_guard_passed"] = bool(protected_guard.get("passed"))
    promotion_gate["protected_row_backfire_guard_blocked_candidates"] = sorted(blocked)
    promotion_gate["candidates"] = updated_candidates
    promotion_gate["passed_candidate_count"] = len(passed_candidates)
    promotion_gate["passed_candidates"] = [str(row.get("name", "")) for row in passed_candidates]
    promotion_gate["decision"] = "weak_promotion_gate_passed" if passed_candidates else "weak_promotion_gate_blocked"
    return promotion_gate


def ensure_import(module_name: str) -> None:
    __import__(module_name)
    print(f"import_ok = {module_name}", flush=True)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, timeout_s: int) -> int:
    printable = " ".join(str(part) for part in cmd)
    print("--- COMMAND START ---", flush=True)
    print("cwd =", cwd, flush=True)
    print("+", printable, flush=True)
    print("timeout_s =", timeout_s, flush=True)
    print("log_path =", log_path, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        rc = proc.wait(timeout=timeout_s)
    print("returncode =", rc, flush=True)
    print("--- COMMAND END ---", flush=True)
    if rc:
        raise RuntimeError(f"command failed rc={rc}: {printable}")
    return rc


def replace_cmd_arg(cmd: list[str], arg_name: str, value: str) -> list[str]:
    updated = list(cmd)
    try:
        index = updated.index(arg_name)
    except ValueError as exc:
        raise RuntimeError(f"internal command is missing {arg_name}") from exc
    if index + 1 >= len(updated):
        raise RuntimeError(f"internal command has no value after {arg_name}")
    updated[index + 1] = value
    return updated


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V245 HF WEAK EVAL JOB START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    validate_gpu()
    repo_commit = validate_repo_commit()
    ensure_import("vllm")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = env_str("HF_TOKEN", "")
    data_repo = env_str("KG1_DATA_REPO", DEFAULT_DATA_REPO)
    weak_csv_file = env_str("KG1_WEAK_CSV_FILE", DEFAULT_WEAK_CSV_FILE)
    weak_manifest_file = env_str("KG1_WEAK_MANIFEST_FILE", DEFAULT_WEAK_MANIFEST_FILE)
    adapter_repo = env_str("KG1_ADAPTER_REPO", DEFAULT_ADAPTER_REPO)
    adapter_subfolders_raw = env_str("KG1_ADAPTER_SUBFOLDERS", "")
    adapter_specs = parse_adapter_specs(adapter_repo, adapter_subfolders_raw)
    output_repo = env_str("KG1_OUTPUT_REPO", DEFAULT_OUTPUT_REPO)
    run_id = env_str("KG1_RUN_ID", "v245-hf-weak-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output_dir = Path(env_str("KG1_OUTPUT_DIR", "/tmp/kg1_v245_weak_eval")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("data_repo =", data_repo, flush=True)
    print("weak_csv_file =", weak_csv_file, flush=True)
    print("weak_manifest_file =", weak_manifest_file, flush=True)
    print("adapter_repo =", adapter_repo, flush=True)
    print("adapter_specs =", json.dumps(adapter_specs, indent=2, sort_keys=True), flush=True)
    print("output_repo =", output_repo, flush=True)
    print("run_id =", run_id, flush=True)
    print("output_dir =", output_dir, flush=True)

    weak_csv = Path(hf_hub_download(repo_id=data_repo, repo_type="dataset", filename=weak_csv_file, token=token or None))
    weak_manifest = Path(
        hf_hub_download(repo_id=data_repo, repo_type="dataset", filename=weak_manifest_file, token=token or None)
    )
    expected_csv_sha = env_str("KG1_EXPECTED_WEAK_CSV_SHA256", EXPECTED_WEAK_CSV_SHA256)
    expected_contract = env_str("KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256", EXPECTED_SHARED_ROW_CONTRACT_SHA256)
    weak_meta = validate_weak_csv(weak_csv, expected_csv_sha, expected_contract)
    manifest_meta = validate_weak_manifest(weak_manifest, expected_contract)
    log_json("weak_csv_gate", weak_meta)
    log_json(
        "weak_manifest_gate",
        {
            "schema_version": manifest_meta.get("schema_version"),
            "path_in_repo": manifest_meta.get("path_in_repo"),
            "canonical_weak_csv": manifest_meta.get("canonical_weak_csv"),
        },
    )

    adapter_cache_dir = Path(env_str("KG1_ADAPTER_CACHE_DIR", "/tmp/kg1_v245_adapter_snapshots")) / run_id
    specs_by_repo: dict[str, list[dict[str, str]]] = {}
    for spec in adapter_specs:
        specs_by_repo.setdefault(spec["repo"], []).append(spec)

    repo_roots: dict[str, Path] = {}
    for repo, specs in specs_by_repo.items():
        subfolders = sorted({spec["subfolder"] for spec in specs})
        allow_patterns = [f"{subfolder}/*" for subfolder in subfolders if subfolder] or ["*"]
        repo_cache_dir = adapter_cache_dir / hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]
        print("snapshot_adapter_repo =", repo, flush=True)
        print("snapshot_allow_patterns =", json.dumps(allow_patterns), flush=True)
        repo_roots[repo] = Path(
            snapshot_download(
                repo_id=repo,
                repo_type="model",
                allow_patterns=allow_patterns,
                local_dir=str(repo_cache_dir),
                token=token or None,
            )
        )

    adapter_metas: list[dict[str, Any]] = []
    candidate_payload: list[dict[str, str]] = []
    for spec in adapter_specs:
        adapter_root = repo_roots[spec["repo"]]
        subfolder = spec["subfolder"]
        adapter_dir = adapter_root / subfolder if subfolder else adapter_root
        adapter_meta = validate_adapter(adapter_dir)
        adapter_meta["repo"] = spec["repo"]
        adapter_meta["subfolder"] = subfolder
        candidate_name = spec["name"]
        adapter_meta["candidate_name"] = candidate_name
        adapter_metas.append(adapter_meta)
        candidate_payload.append(
            {
                "name": candidate_name,
                "adapter": str(adapter_dir),
                "source_kind": "hf_model_repo",
            }
        )
    log_json("adapter_gates", {"count": len(adapter_metas), "adapters": adapter_metas})

    candidate_json = output_dir / "v245_weak_eval_candidates.json"
    candidate_json.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True), encoding="utf-8")
    eval_out = output_dir / "eval"
    disable_thinking = env_bool("KG1_DISABLE_THINKING", False)
    no_prompt_suffix = env_bool("KG1_NO_PROMPT_SUFFIX", False)
    prompt_suffix = os.environ.get("KG1_PROMPT_SUFFIX", DEFAULT_PROMPT_SUFFIX)
    log_json(
        "eval_prompt_controls",
        {
            "disable_thinking": disable_thinking,
            "no_prompt_suffix": no_prompt_suffix,
            "prompt_suffix": "" if no_prompt_suffix else prompt_suffix,
        },
    )
    eval_limit = env_int("KG1_EVAL_LIMIT", 0)
    if env_bool("KG1_REQUIRE_SMOKE_EVAL_LIMIT", False) and not (1 <= eval_limit <= 8):
        raise RuntimeError(
            "KG1_REQUIRE_SMOKE_EVAL_LIMIT is enabled; KG1_EVAL_LIMIT must be between 1 and 8 "
            f"for a FinOps-safe smoke eval, got {eval_limit}."
        )
    eval_gpu_memory_utilization = env_float(
        "KG1_VLLM_GPU_MEMORY_UTILIZATION",
        env_float("KG1_GPU_MEMORY_UTILIZATION", 0.0),
    )
    eval_runtime_controls = {
        "eval_limit": eval_limit,
        "gpu_memory_utilization": eval_gpu_memory_utilization,
        "generation_timeout_s": env_int("KG1_GENERATION_TIMEOUT_S", 0),
        "llm_init_timeout_s": env_int("KG1_LLM_INIT_TIMEOUT_S", 0),
        "vllm_enable_prefix_caching": env_str("KG1_VLLM_ENABLE_PREFIX_CACHING", ""),
        "vllm_enable_chunked_prefill": env_str("KG1_VLLM_ENABLE_CHUNKED_PREFILL", ""),
        "vllm_enforce_eager": env_str("KG1_VLLM_ENFORCE_EAGER", ""),
        "vllm_use_v1": env_str("VLLM_USE_V1", ""),
    }
    log_json("eval_runtime_controls", eval_runtime_controls)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_lora_adapters_batch.py"),
        "--solution-csv",
        str(weak_csv),
        "--questions-csv",
        str(weak_csv),
        "--candidates-json",
        str(candidate_json),
        "--base-model-path",
        env_str("KG1_MODEL_NAME", DEFAULT_MODEL_NAME),
        "--label-prefix",
        env_str("KG1_LABEL_PREFIX", "v245_hf_weak"),
        "--seed",
        env_str("KG1_SEED", "42"),
        "--limit",
        str(eval_limit),
        "--output-dir",
        str(eval_out),
        "--max-tokens",
        str(env_int("KG1_MAX_TOKENS", 7680)),
        "--max-model-len",
        str(env_int("KG1_MAX_MODEL_LEN", 8192)),
        "--max-num-seqs",
        str(env_int("KG1_MAX_NUM_SEQS", 64)),
        "--gpu-memory-utilization",
        str(eval_gpu_memory_utilization),
        "--llm-init-timeout-s",
        str(env_int("KG1_LLM_INIT_TIMEOUT_S", 0)),
        "--generation-timeout-s",
        str(env_int("KG1_GENERATION_TIMEOUT_S", 0)),
        "--warmup-rows",
        "0",
        "--continue-on-error",
    ]
    if disable_thinking:
        cmd.append("--disable-thinking")
    if no_prompt_suffix:
        cmd.append("--no-prompt-suffix")
    elif prompt_suffix:
        cmd.extend(["--prompt-suffix", prompt_suffix])
    candidate_by_candidate = env_bool("KG1_EVAL_CANDIDATE_BY_CANDIDATE", len(candidate_payload) > 1)
    incremental_manifests: list[str] = []
    if not candidate_by_candidate:
        run_cmd(
            cmd,
            cwd=ROOT,
            log_path=output_dir / "v245_hf_weak_eval.log",
            timeout_s=env_int("KG1_EVAL_TIMEOUT_S", 1800),
        )
        summary_path = eval_out / "batch_candidate_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
    else:
        print("candidate_by_candidate_eval = true", flush=True)
        stop_after_failures = env_int("KG1_STOP_AFTER_CONSECUTIVE_FAILED_CANDIDATES", 2)
        upload_incremental = env_bool("KG1_UPLOAD_INCREMENTAL_EVAL_DIAGNOSTICS", env_bool("KG1_UPLOAD_TO_HF", True))
        incremental_api = HfApi(token=token or None) if upload_incremental else None
        incremental_path_in_repo = env_str("KG1_OUTPUT_PATH_IN_REPO", f"evals/{run_id}")
        all_rows: list[dict[str, Any]] = []
        consecutive_failures = 0
        stopped_early = False
        for index, candidate in enumerate(candidate_payload, start=1):
            single_name = f"candidate_{index:02d}_{hashlib.sha256(candidate['name'].encode('utf-8')).hexdigest()[:8]}"
            single_json = output_dir / f"{single_name}.json"
            single_out = eval_out / single_name
            single_json.write_text(json.dumps([candidate], indent=2, sort_keys=True), encoding="utf-8")
            single_cmd = replace_cmd_arg(cmd, "--candidates-json", str(single_json))
            single_cmd = replace_cmd_arg(single_cmd, "--output-dir", str(single_out))
            run_cmd(
                single_cmd,
                cwd=ROOT,
                log_path=output_dir / f"{single_name}_weak_eval.log",
                timeout_s=env_int("KG1_EVAL_TIMEOUT_S", 1800),
            )
            single_summary_path = single_out / "batch_candidate_summary.json"
            if not single_summary_path.exists():
                raise FileNotFoundError(single_summary_path)
            single_summary = read_json(single_summary_path)
            single_rows = single_summary.get("rows", [])
            if not isinstance(single_rows, list):
                single_rows = []
            all_rows.extend(row for row in single_rows if isinstance(row, dict))
            current_summary = {
                "generated_at_utc": utc_now(),
                "mode": "candidate_by_candidate_incremental",
                "candidate_index": index,
                "candidate_count": len(candidate_payload),
                "current_candidate": candidate["name"],
                "rows": list(all_rows),
            }
            current_gate = weak_promotion_gate(current_summary)
            current_protected_guard = protected_row_backfire_guard(current_summary)
            current_gate = apply_protected_row_guard(current_gate, current_protected_guard)
            current_catastrophic_guard = catastrophic_eval_guard(current_summary)
            incremental_manifest = {
                "schema_version": "kg1_v245_incremental_weak_eval_manifest_v1",
                "generated_at_utc": utc_now(),
                "run_id": run_id,
                "candidate_index": index,
                "candidate_count": len(candidate_payload),
                "current_candidate": candidate["name"],
                "candidate_summary": current_summary,
                "weak_promotion_gate": current_gate,
                "protected_row_backfire_guard": current_protected_guard,
                "catastrophic_eval_guard": current_catastrophic_guard,
            }
            incremental_path = output_dir / f"{single_name}_incremental_manifest.json"
            incremental_path.write_text(json.dumps(incremental_manifest, indent=2, sort_keys=True), encoding="utf-8")
            incremental_manifests.append(str(incremental_path))
            log_json("incremental_weak_promotion_gate", current_gate)
            log_json("incremental_catastrophic_eval_guard", current_catastrophic_guard)
            if upload_incremental and incremental_api is not None:
                print("incremental_upload_folder_repo =", output_repo, flush=True)
                print("incremental_upload_folder_path_in_repo =", incremental_path_in_repo, flush=True)
                incremental_api.upload_folder(
                    repo_id=output_repo,
                    repo_type="model",
                    folder_path=str(output_dir),
                    path_in_repo=incremental_path_in_repo,
                    commit_message=f"Add {run_id} weak eval diagnostics through {candidate['name']}",
                    ignore_patterns=["adapter_snapshot/**"],
                )
            if current_gate["passed_candidates"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if current_catastrophic_guard.get("should_stop"):
                print(
                    "candidate_by_candidate_stop = "
                    + json.dumps(
                        {
                            "reason": "catastrophic_eval_guard",
                            "last_candidate": candidate["name"],
                            "catastrophic_candidates": current_catastrophic_guard.get("catastrophic_candidates", []),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                stopped_early = True
                break
            if stop_after_failures > 0 and consecutive_failures >= stop_after_failures:
                print(
                    "candidate_by_candidate_stop = "
                    + json.dumps(
                        {
                            "reason": "consecutive_failed_candidates",
                            "stop_after_failures": stop_after_failures,
                            "consecutive_failures": consecutive_failures,
                            "last_candidate": candidate["name"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                stopped_early = True
                break
        summary_path = eval_out / "batch_candidate_summary.json"
        summary = {
            "generated_at_utc": utc_now(),
            "base_model_path": env_str("KG1_MODEL_NAME", DEFAULT_MODEL_NAME),
            "solution_csv": str(weak_csv),
            "questions_csv": str(weak_csv),
            "candidates_json": str(candidate_json),
            "mode": "candidate_by_candidate",
            "stopped_early": stopped_early,
            "incremental_manifests": incremental_manifests,
            "rows": all_rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log_json("candidate_summary_payload", summary)
    promotion_gate = weak_promotion_gate(summary)
    protected_guard = protected_row_backfire_guard(summary)
    catastrophic_guard = catastrophic_eval_guard(summary)
    log_json("protected_row_backfire_guard", protected_guard)
    promotion_gate = apply_protected_row_guard(promotion_gate, protected_guard)
    log_json("weak_promotion_gate", promotion_gate)
    log_json("catastrophic_eval_guard", catastrophic_guard)

    final_manifest = {
        "schema_version": "kg1_v245_hf_weak_eval_manifest_v1",
        "generated_at_utc": utc_now(),
        "repo_commit": repo_commit,
        "run_id": run_id,
        "weak_csv": weak_meta,
        "adapters": adapter_metas,
        "eval_summary_json": str(summary_path),
        "candidate_summary": summary,
        "weak_promotion_gate": promotion_gate,
        "protected_row_backfire_guard": protected_guard,
        "catastrophic_eval_guard": catastrophic_guard,
        "candidate_by_candidate_eval": candidate_by_candidate,
        "incremental_manifests": incremental_manifests,
        "eval_prompt_controls": {
            "disable_thinking": disable_thinking,
            "no_prompt_suffix": no_prompt_suffix,
            "prompt_suffix": "" if no_prompt_suffix else prompt_suffix,
        },
        "blocked_actions": ["full_eval", "package", "kaggle_submit"],
    }
    final_manifest_path = output_dir / "v245_hf_weak_eval_manifest.json"
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("final_manifest_path =", final_manifest_path, flush=True)

    if env_bool("KG1_UPLOAD_TO_HF", True):
        api = HfApi(token=token or None)
        path_in_repo = env_str("KG1_OUTPUT_PATH_IN_REPO", f"evals/{run_id}")
        final_manifest["path_in_repo"] = path_in_repo
        final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("upload_folder_repo =", output_repo, flush=True)
        print("upload_folder_path_in_repo =", path_in_repo, flush=True)
        upload_info = api.upload_folder(
            repo_id=output_repo,
            repo_type="model",
            folder_path=str(output_dir),
            path_in_repo=path_in_repo,
            commit_message=f"Add {run_id} weak eval outputs",
            ignore_patterns=["adapter_snapshot/**"],
        )
        final_manifest["upload_info"] = str(upload_info)
        final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("upload_info =", upload_info, flush=True)

    if catastrophic_guard.get("should_stop"):
        raise RuntimeError(
            "Catastrophic eval guard stopped/blocked candidates after uploading diagnostics: "
            + json.dumps(catastrophic_guard.get("thresholds", {}), sort_keys=True)
        )
    if promotion_gate["enforced"] and not promotion_gate["passed_candidates"]:
        raise RuntimeError(
            "Weak promotion gate blocked all candidates after uploading diagnostics: "
            + json.dumps(promotion_gate["thresholds"], sort_keys=True)
        )

    print("=== V245 HF WEAK EVAL JOB END ===", flush=True)
    return final_manifest


def self_test() -> int:
    print("=== V245 HF WEAK EVAL SELF TEST START ===", flush=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "kg1_v245_weak_eval_self_test"
    tmp.mkdir(parents=True, exist_ok=True)
    csv_path = tmp / "weak.csv"
    frame = pd.DataFrame(
        [
            {"id": "b", "prompt": "Perform bit manipulation on 8-bit binary 00000000.", "answer": "00000000", "type": "bit_manipulation"},
            {"id": "e", "prompt": "Apply the transformation rule a -> b.", "answer": "b", "type": "equation_transform"},
        ]
    )
    frame.to_csv(csv_path, index=False)
    try:
        validate_weak_csv(csv_path, "", "")
    except RuntimeError as exc:
        if "weak family counts mismatch" not in str(exc):
            raise
    else:
        raise RuntimeError("self-test expected small weak CSV count failure")

    adapter_ok = tmp / "adapter_ok"
    adapter_ok.mkdir(parents=True, exist_ok=True)
    (adapter_ok / "adapter_model.safetensors").write_bytes(b"kg1-self-test")
    (adapter_ok / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": DEFAULT_MODEL_NAME,
                "lora_alpha": 32,
                "modules_to_save": None,
                "r": 32,
                "target_modules": ["q_proj"],
                "target_parameters": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    adapter_meta = validate_adapter(adapter_ok)
    if adapter_meta.get("modules_to_save") != []:
        raise RuntimeError("self-test expected empty modules_to_save in adapter meta")

    adapter_modules_to_save = tmp / "adapter_modules_to_save"
    adapter_modules_to_save.mkdir(parents=True, exist_ok=True)
    (adapter_modules_to_save / "adapter_model.safetensors").write_bytes(b"kg1-self-test")
    (adapter_modules_to_save / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": DEFAULT_MODEL_NAME,
                "lora_alpha": 32,
                "modules_to_save": ["lm_head"],
                "r": 32,
                "target_modules": ["q_proj"],
                "target_parameters": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    try:
        validate_adapter(adapter_modules_to_save)
    except RuntimeError as exc:
        if "modules_to_save must be empty" not in str(exc):
            raise
    else:
        raise RuntimeError("self-test expected modules_to_save adapter failure")

    adapter_wrong_base = tmp / "adapter_wrong_base"
    adapter_wrong_base.mkdir(parents=True, exist_ok=True)
    (adapter_wrong_base / "adapter_model.safetensors").write_bytes(b"kg1-self-test")
    (adapter_wrong_base / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "wrong/base",
                "lora_alpha": 32,
                "modules_to_save": None,
                "r": 32,
                "target_modules": ["q_proj"],
                "target_parameters": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    try:
        validate_adapter(adapter_wrong_base)
    except RuntimeError as exc:
        if "adapter base model mismatch" not in str(exc):
            raise
    else:
        raise RuntimeError("self-test expected adapter base model mismatch failure")

    old_env = {
        name: os.environ.get(name)
        for name in [
            "KG1_ENFORCE_WEAK_PROMOTION_GATE",
            "KG1_WEAK_EVAL_DIAGNOSTIC_ONLY",
            "KG1_PROTECTED_ROW_GUARD",
            "KG1_PROTECTED_BASELINE_CSV",
            "KG1_PROTECTED_ID_ANSWERS",
            "KG1_CRISIS_MODE_BACKFIRE_GUARD",
            "KG1_CATASTROPHIC_EVAL_GUARD",
            "KG1_ALLOW_CATASTROPHIC_EVAL_CONTINUE",
            "KG1_CATASTROPHIC_CORRECT_MAX",
            "KG1_CATASTROPHIC_TRUNCATION_RATE_MIN",
            "KG1_CATASTROPHIC_TRUNCATED_MIN",
            "KG1_CATASTROPHIC_MIN_ROWS",
            "KG1_CATASTROPHIC_BIT_CORRECT_MAX",
        ]
    }
    os.environ.pop("KG1_ENFORCE_WEAK_PROMOTION_GATE", None)
    os.environ.pop("KG1_WEAK_EVAL_DIAGNOSTIC_ONLY", None)
    blocked = weak_promotion_gate(
        {
            "rows": [
                {
                    "name": "bad",
                    "status": "ok",
                    "correct": 192,
                    "equation_transform_correct": 56,
                    "bit_manipulation_correct": 136,
                    "truncated": 0,
                }
            ]
        }
    )
    assert blocked["decision"] == "weak_promotion_gate_blocked"
    assert blocked["enforced"] is True
    os.environ["KG1_WEAK_EVAL_DIAGNOSTIC_ONLY"] = "1"
    diagnostic = weak_promotion_gate({"rows": []})
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["enforced"] is False
    os.environ.pop("KG1_WEAK_EVAL_DIAGNOSTIC_ONLY", None)
    passed = weak_promotion_gate(
        {
            "rows": [
                {
                    "name": "good",
                    "status": "ok",
                    "correct": 196,
                    "equation_transform_correct": 60,
                    "bit_manipulation_correct": 136,
                    "truncated": 0,
                    "label_aware_debug_correct": 196,
                    "rows": EXPECTED_WEAK_ROWS,
                    "completion_tokens": 3150,
                    "avg_completion_tokens": 10,
                    "max_completion_tokens": 40,
                }
            ]
        }
    )
    assert passed["decision"] == "weak_promotion_gate_passed"
    os.environ["KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX"] = "512"
    os.environ["KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX"] = "2048"
    runaway_blocked = weak_promotion_gate(
        {
            "rows": [
                {
                    "name": "runaway",
                    "status": "ok",
                    "correct": 196,
                    "equation_transform_correct": 60,
                    "bit_manipulation_correct": 136,
                    "truncated": 0,
                    "label_aware_debug_correct": 196,
                    "rows": EXPECTED_WEAK_ROWS,
                    "completion_tokens": 1500000,
                    "avg_completion_tokens": 4761.9,
                    "max_completion_tokens": 7680,
                }
            ]
        }
    )
    assert runaway_blocked["decision"] == "weak_promotion_gate_blocked"
    assert any(
        reason.startswith("avg_completion_tokens_gt_")
        for reason in runaway_blocked["candidates"][0]["blocking_reasons"]
    )
    os.environ.pop("KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX", None)
    os.environ.pop("KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX", None)
    label_aware_delta_blocked = weak_promotion_gate(
        {
            "rows": [
                {
                    "name": "expected_aware_only_gain",
                    "status": "ok",
                    "correct": 196,
                    "label_aware_debug_correct": 197,
                    "equation_transform_correct": 60,
                    "bit_manipulation_correct": 136,
                    "truncated": 0,
                    "rows": EXPECTED_WEAK_ROWS,
                    "completion_tokens": 3150,
                    "avg_completion_tokens": 10,
                    "max_completion_tokens": 40,
                }
            ]
        }
    )
    assert label_aware_delta_blocked["decision"] == "weak_promotion_gate_blocked"
    assert any(
        reason.startswith("label_aware_delta_gt_")
        for reason in label_aware_delta_blocked["candidates"][0]["blocking_reasons"]
    )
    catastrophic = catastrophic_eval_guard(
        {
            "rows": [
                {
                    "name": "v548_like_collapse",
                    "status": "ok",
                    "correct": 3,
                    "equation_transform_correct": 3,
                    "bit_manipulation_correct": 0,
                    "truncated": 288,
                    "truncation_rate": 288 / EXPECTED_WEAK_ROWS,
                    "rows": EXPECTED_WEAK_ROWS,
                }
            ]
        }
    )
    assert catastrophic["decision"] == "catastrophic_eval_stop"
    assert catastrophic["should_stop"] is True
    assert catastrophic["catastrophic_candidates"][0]["name"] == "v548_like_collapse"
    os.environ["KG1_ALLOW_CATASTROPHIC_EVAL_CONTINUE"] = "1"
    catastrophic_allowed = catastrophic_eval_guard(
        {
            "rows": [
                {
                    "name": "diagnostic_collapse",
                    "status": "ok",
                    "correct": 3,
                    "equation_transform_correct": 3,
                    "bit_manipulation_correct": 0,
                    "truncated": 288,
                    "truncation_rate": 288 / EXPECTED_WEAK_ROWS,
                    "rows": EXPECTED_WEAK_ROWS,
                }
            ]
        }
    )
    assert catastrophic_allowed["decision"] == "catastrophic_eval_pass"
    assert catastrophic_allowed["should_stop"] is False
    os.environ.pop("KG1_ALLOW_CATASTROPHIC_EVAL_CONTINUE", None)
    baseline_csv = tmp / "baseline_predictions.csv"
    good_predictions_csv = tmp / "good_predictions.csv"
    bad_predictions_csv = tmp / "bad_predictions.csv"
    report_good = tmp / "good_report.json"
    report_bad = tmp / "bad_report.json"
    baseline_csv.write_text(
        "id,family,answer,prediction,correct\n"
        "8740ed31,bit_manipulation,01101000,01101000,True\n"
        "59bee375,bit_manipulation,10010101,10010101,True\n",
        encoding="utf-8",
    )
    good_predictions_csv.write_text(
        "id,family,answer,prediction,correct\n"
        "8740ed31,bit_manipulation,01101000,01101000,True\n"
        "59bee375,bit_manipulation,10010101,10010101,True\n",
        encoding="utf-8",
    )
    bad_predictions_csv.write_text(
        "id,family,answer,prediction,correct\n"
        "8740ed31,bit_manipulation,01101000,01111000,False\n"
        "59bee375,bit_manipulation,10010101,2,False\n",
        encoding="utf-8",
    )
    report_good.write_text(
        json.dumps({"outputs": {"predictions_csv": str(good_predictions_csv)}}),
        encoding="utf-8",
    )
    report_bad.write_text(
        json.dumps({"outputs": {"predictions_csv": str(bad_predictions_csv)}}),
        encoding="utf-8",
    )
    os.environ["KG1_PROTECTED_ROW_GUARD"] = "1"
    os.environ["KG1_PROTECTED_BASELINE_CSV"] = str(baseline_csv)
    os.environ["KG1_PROTECTED_ID_ANSWERS"] = "8740ed31=01101000,59bee375=10010101"
    guard_summary = {
        "rows": [
            {"name": "good", "status": "ok", "report_json": str(report_good)},
            {"name": "bad", "status": "ok", "report_json": str(report_bad)},
        ]
    }
    protected_guard = protected_row_backfire_guard(guard_summary)
    assert protected_guard["enforced"] is True
    assert "good" in protected_guard["passed_candidates"]
    assert "bad" in protected_guard["blocked_candidates"]
    gate = weak_promotion_gate(
        {
            "rows": [
                {
                    "name": "bad",
                    "status": "ok",
                    "correct": 196,
                    "equation_transform_correct": 60,
                    "bit_manipulation_correct": 136,
                    "truncated": 0,
                    "rows": EXPECTED_WEAK_ROWS,
                    "completion_tokens": 3150,
                    "avg_completion_tokens": 10,
                    "max_completion_tokens": 40,
                }
            ]
        }
    )
    guarded_gate = apply_protected_row_guard(gate, protected_guard)
    assert guarded_gate["decision"] == "weak_promotion_gate_blocked"
    global_blocked_gate = apply_protected_row_guard(
        weak_promotion_gate(
            {
                "rows": [
                    {
                        "name": "otherwise_good",
                        "status": "ok",
                        "correct": 196,
                        "equation_transform_correct": 60,
                        "bit_manipulation_correct": 136,
                        "truncated": 0,
                        "rows": EXPECTED_WEAK_ROWS,
                        "completion_tokens": 3150,
                        "avg_completion_tokens": 10,
                        "max_completion_tokens": 40,
                    }
                ]
            }
        ),
        {
            "enforced": True,
            "passed": False,
            "blocked_candidates": [],
            "blockers": ["protected_baseline_missing:/tmp/missing.csv"],
        },
    )
    assert global_blocked_gate["decision"] == "weak_promotion_gate_blocked"
    for name, value in old_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    print("v245_hf_weak_eval_self_test=ok", flush=True)
    print("=== V245 HF WEAK EVAL SELF TEST END ===", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_text_streams()
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    run_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
