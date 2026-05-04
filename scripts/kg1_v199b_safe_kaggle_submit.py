#!/usr/bin/env python3
"""Audit and safely submit the V199B baseline-gated adapter to Kaggle.

This is the final production guard for the V199B run. It validates the ZIP
contents, training manifest, posttrain report, preflight report, doublecheck
report, stages the exact candidate as submission.zip, and submits through the
Kaggle Python API only when --submit is explicitly passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kg1_v198_final_submit_doublecheck import (  # noqa: E402
    EXPECTED_ROOT_ENTRIES,
    get_adapter_model_path,
    inspect_config,
    inspect_tensors,
)
from kg1_v198_safe_kaggle_submit import (  # noqa: E402
    COMPETITION,
    REQUIRED_SUBMISSION_BASENAME,
    exception_details,
    summarize_latest_submissions,
)


EXPECTED_RUN_ID = "v199b-h100-baseline-gated-v194-rank19-10s"
EXPECTED_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
EXPECTED_MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
EXPECTED_TRAIN_SHA256 = "6d2742616300818eb50c54d36019551b24f5b71c607a2b28feda7461a709def0"
EXPECTED_VAL_SHA256 = "e59c907c6545e5e587097a64762e3e874508e8cd74d85d5c7c79354ebe56e73c"
V194_RANK19_ADAPTER_SHA256 = "01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f"
V198_REGRESSION_FINAL_ZIP_SHA256 = "52c585c7f075a1a9735d23c16905e535d1ebbf51246b03a50ac3d07c3768a3a9"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def prepare_kaggle_named_zip(candidate_zip: Path, output_json: Path) -> tuple[Path, dict[str, Any]]:
    """Stage the exact ZIP with Kaggle's required submission.zip basename."""

    if candidate_zip.name == REQUIRED_SUBMISSION_BASENAME:
        return candidate_zip, {
            "required_basename": REQUIRED_SUBMISSION_BASENAME,
            "mode": "original",
            "path": str(candidate_zip),
        }

    preferred_stage = Path("/content/kg1_v199b/kaggle_submit_stage")
    stage_dir = preferred_stage if preferred_stage.parent.exists() else output_json.parent / "kaggle_submit_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_zip = stage_dir / REQUIRED_SUBMISSION_BASENAME
    if staged_zip.exists() or staged_zip.is_symlink():
        staged_zip.unlink()

    try:
        os.symlink(candidate_zip, staged_zip)
        mode = "symlink"
    except OSError:
        shutil.copy2(candidate_zip, staged_zip)
        mode = "copy"

    if sha256_file(staged_zip) != sha256_file(candidate_zip):
        raise RuntimeError("Staged submission.zip SHA mismatch")

    return staged_zip, {
        "required_basename": REQUIRED_SUBMISSION_BASENAME,
        "mode": mode,
        "source": str(candidate_zip),
        "path": str(staged_zip),
    }


def default_paths(candidate_zip: Path) -> dict[str, Path]:
    cursor = candidate_zip.resolve()
    gate_root = None
    for parent in cursor.parents:
        if parent.name == "posttrain_kaggle_gate":
            gate_root = parent
            break
    if gate_root is None:
        gate_root = cursor.parent.parent.parent
    output_root = gate_root.parent
    return {
        "posttrain_report": gate_root / "v199_posttrain_gate_report.json",
        "preflight_report": output_root / "final_preflight.json",
        "doublecheck_json": output_root / "final_submit_doublecheck.json",
        "manifest_json": output_root / "final_adapter" / "v90_training_manifest.json",
    }


def inspect_zip_strict(candidate_zip: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if not candidate_zip.exists():
        return {"ok": False, "path": str(candidate_zip), "reasons": ["candidate_zip_missing"]}
    try:
        with zipfile.ZipFile(candidate_zip) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            dirs = [info.filename.replace("\\", "/") for info in archive.infolist() if info.is_dir()]
            entries = [info.filename.replace("\\", "/") for info in infos]
            sizes = {info.filename.replace("\\", "/"): info.file_size for info in infos}
            compressed_sizes = {info.filename.replace("\\", "/"): info.compress_size for info in infos}
            adapter_config = (
                json.loads(archive.read("adapter_config.json").decode("utf-8"))
                if "adapter_config.json" in entries
                else {}
            )
    except zipfile.BadZipFile:
        return {"ok": False, "path": str(candidate_zip), "reasons": ["bad_zip_file"]}
    except json.JSONDecodeError:
        return {"ok": False, "path": str(candidate_zip), "reasons": ["adapter_config_invalid_json"]}

    if set(entries) != EXPECTED_ROOT_ENTRIES or len(entries) != len(EXPECTED_ROOT_ENTRIES):
        reasons.append("zip_entries_not_exactly_adapter_config_and_safetensors")
    if len(set(entries)) != len(entries):
        reasons.append("duplicate_zip_entries")
    if dirs:
        reasons.append("directory_entries_present")
    nested = [entry for entry in entries if "/" in entry.strip("/")]
    if nested:
        reasons.append("nested_entries_present")
    unsafe_paths = [
        entry for entry in entries
        if entry.startswith("/") or ".." in Path(entry).parts or "\\" in entry
    ]
    if unsafe_paths:
        reasons.append("unsafe_zip_paths")
    blocked_names = [
        entry for entry in entries
        if Path(entry).name.lower() in {"kaggle.json", ".env", "optimizer.pt", "scheduler.pt", "trainer_state.json"}
    ]
    if blocked_names:
        reasons.append("blocked_training_or_secret_files_present")
    if sizes.get("adapter_config.json", 0) < 100:
        reasons.append("adapter_config_too_small")
    if sizes.get("adapter_model.safetensors", 0) < 4_000_000_000:
        reasons.append("adapter_model_too_small")
    zip_sha = sha256_file(candidate_zip)
    adapter_member_sha = (
        sha256_zip_member(candidate_zip, "adapter_model.safetensors")
        if "adapter_model.safetensors" in entries
        else None
    )
    if zip_sha == V198_REGRESSION_FINAL_ZIP_SHA256:
        reasons.append("candidate_is_known_v198_regression_zip")
    if adapter_member_sha == V194_RANK19_ADAPTER_SHA256:
        reasons.append("candidate_adapter_is_unchanged_v194_baseline")
    return {
        "ok": not reasons,
        "path": str(candidate_zip),
        "basename": candidate_zip.name,
        "sha256": zip_sha,
        "adapter_model_member_sha256": adapter_member_sha,
        "entries": entries,
        "directory_entries": dirs,
        "sizes": sizes,
        "compressed_sizes": compressed_sizes,
        "adapter_config": adapter_config,
        "required_kaggle_basename": REQUIRED_SUBMISSION_BASENAME,
        "needs_kaggle_basename_stage": candidate_zip.name != REQUIRED_SUBMISSION_BASENAME,
        "reasons": reasons,
    }


def inspect_manifest(path: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "path": str(path), "reasons": ["manifest_missing"]}
    manifest = read_json(path)
    training = manifest.get("training") or {}
    baseline_gate = training.get("baseline_gate") or {}
    baseline_eval = training.get("baseline_eval_loss")
    final_eval = training.get("final_eval_loss")
    best_eval = training.get("best_eval_loss")
    if manifest.get("run_id") != EXPECTED_RUN_ID:
        reasons.append("unexpected_run_id")
    if manifest.get("model_name") != EXPECTED_MODEL:
        reasons.append("unexpected_model_name")
    if manifest.get("model_revision") != EXPECTED_MODEL_REVISION:
        reasons.append("unexpected_model_revision")
    if manifest.get("train_file_sha256") != EXPECTED_TRAIN_SHA256:
        reasons.append("unexpected_train_sha256")
    if manifest.get("val_file_sha256") != EXPECTED_VAL_SHA256:
        reasons.append("unexpected_val_sha256")
    if training.get("max_steps") != 10 or training.get("final_step") != 10:
        reasons.append("unexpected_step_count")
    if training.get("learning_rate") != 1e-6:
        reasons.append("unexpected_learning_rate")
    if training.get("final_learning_rate") != 3e-7:
        reasons.append("unexpected_final_learning_rate")
    if baseline_gate.get("baseline_eval_before_train") is not True:
        reasons.append("baseline_eval_before_train_not_recorded")
    if baseline_gate.get("require_final_eval_lte_baseline") is not True:
        reasons.append("final_lte_baseline_gate_not_recorded")
    if baseline_eval is None or final_eval is None:
        reasons.append("missing_baseline_or_final_eval")
    elif float(final_eval) > float(baseline_eval):
        reasons.append("final_eval_regressed_vs_baseline")
    if baseline_eval is not None and best_eval is not None and float(best_eval) > float(baseline_eval):
        reasons.append("best_eval_not_better_than_baseline")
    return {
        "ok": not reasons,
        "path": str(path),
        "run_id": manifest.get("run_id"),
        "baseline_eval_loss": baseline_eval,
        "final_eval_loss": final_eval,
        "best_eval_loss": best_eval,
        "final_minus_baseline": (
            float(final_eval) - float(baseline_eval)
            if baseline_eval is not None and final_eval is not None
            else None
        ),
        "reasons": reasons,
    }


def inspect_posttrain_report(path: Path, candidate_zip: Path, zip_sha: str) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "path": str(path), "reasons": ["posttrain_report_missing"]}
    report = read_json(path)
    decision = report.get("decision") or {}
    if decision.get("ready") is not True:
        reasons.append("posttrain_not_ready")
    if decision.get("primary_label") != "final":
        reasons.append("primary_label_not_final")
    if Path(str(decision.get("primary_zip") or "")).name != candidate_zip.name:
        reasons.append("primary_zip_name_mismatch")
    final = next((item for item in report.get("candidates", []) if item.get("label") == "final"), None)
    if not final:
        reasons.append("final_candidate_missing_in_posttrain")
    else:
        if not final.get("decision", {}).get("ready"):
            reasons.append("final_candidate_not_ready")
        if final.get("zip", {}).get("sha256") != zip_sha:
            reasons.append("final_zip_sha_mismatch")
    return {
        "ok": not reasons,
        "path": str(path),
        "primary_label": decision.get("primary_label"),
        "primary_zip": decision.get("primary_zip"),
        "reasons": reasons,
    }


def inspect_preflight_report(path: Path, candidate_zip: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "path": str(path), "reasons": ["preflight_report_missing"]}
    report = read_json(path)
    decision = report.get("decision") or {}
    if decision.get("production_ready") is not True:
        reasons.append("preflight_not_production_ready")
    if decision.get("reasons"):
        reasons.append("preflight_has_reasons")
    if Path(str(report.get("adapter_zip") or "")).name != candidate_zip.name:
        reasons.append("preflight_zip_name_mismatch")
    return {
        "ok": not reasons,
        "path": str(path),
        "production_ready": decision.get("production_ready"),
        "decision_reasons": decision.get("reasons"),
        "reasons": reasons,
    }


def inspect_doublecheck(path: Path, candidate_zip: Path, zip_sha: str) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "path": str(path), "reasons": ["doublecheck_missing"]}
    report = read_json(path)
    decision = report.get("decision") or {}
    if decision.get("submit_ready") is not True:
        reasons.append("doublecheck_not_submit_ready")
    if decision.get("reasons"):
        reasons.append("doublecheck_has_reasons")
    if Path(str(report.get("candidate_zip") or "")).name != candidate_zip.name:
        reasons.append("doublecheck_zip_name_mismatch")
    if report.get("zip", {}).get("sha256") != zip_sha:
        reasons.append("doublecheck_zip_sha_mismatch")
    return {
        "ok": not reasons,
        "path": str(path),
        "submit_ready": decision.get("submit_ready"),
        "decision_reasons": decision.get("reasons"),
        "reasons": reasons,
    }


def ensure_kaggle_credentials(kaggle_json: Path | None) -> dict[str, Any]:
    home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
    candidates = []
    if kaggle_json:
        candidates.append(kaggle_json)
    candidates.extend([
        home_kaggle,
        Path("/content/drive/MyDrive/kaggle.json"),
        Path("/content/kaggle.json"),
        Path.cwd() / "kaggle.json",
    ])
    source = next((path for path in candidates if path.exists()), None)
    if source is not None:
        home_kaggle.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != home_kaggle.resolve():
            shutil.copy2(source, home_kaggle)
    else:
        username = key = None
        try:
            from google.colab import userdata  # type: ignore

            username = userdata.get("KAGGLE_USERNAME")
            key = userdata.get("KAGGLE_KEY")
        except Exception:
            username = key = None
        if not username or not key:
            raise FileNotFoundError(
                "Kaggle credentials missing. Provide /content/drive/MyDrive/kaggle.json "
                "or Colab secrets KAGGLE_USERNAME and KAGGLE_KEY."
            )
        home_kaggle.parent.mkdir(parents=True, exist_ok=True)
        home_kaggle.write_text(
            json.dumps({"username": username, "key": key}), encoding="utf-8"
        )
        source = Path("colab_userdata:KAGGLE_USERNAME/KAGGLE_KEY")
    home_kaggle.chmod(stat.S_IRUSR | stat.S_IWUSR)
    os.environ["KAGGLE_CONFIG_DIR"] = str(home_kaggle.parent)
    creds = read_json(home_kaggle)
    if not creds.get("username") or not creds.get("key"):
        raise ValueError(f"Invalid Kaggle credentials file: {home_kaggle}")
    return {"path": str(home_kaggle), "source": str(source), "username": creds.get("username")}


def build_audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    candidate_zip = args.candidate_zip.resolve()
    paths = default_paths(candidate_zip)
    posttrain_report = args.posttrain_report or paths["posttrain_report"]
    preflight_report = args.preflight_report or paths["preflight_report"]
    doublecheck_json = args.doublecheck_json or paths["doublecheck_json"]
    manifest_json = args.manifest_json or paths["manifest_json"]

    zip_report = inspect_zip_strict(candidate_zip)
    if args.expected_sha256 and zip_report.get("sha256") != args.expected_sha256:
        zip_report.setdefault("reasons", []).append("expected_zip_sha_mismatch")
        zip_report["ok"] = False

    config_report = inspect_config(zip_report.get("adapter_config") or {})
    if zip_report.get("adapter_model_member_sha256"):
        adapter_model, tmp = get_adapter_model_path(candidate_zip)
        try:
            tensor_report = inspect_tensors(adapter_model)
        finally:
            if tmp is not None:
                tmp.cleanup()
        if tensor_report.get("sha256") != zip_report.get("adapter_model_member_sha256"):
            tensor_report.setdefault("reasons", []).append("nearby_adapter_sha_mismatch_vs_zip_member")
            tensor_report["ok"] = False
    else:
        tensor_report = {
            "ok": False,
            "path": str(candidate_zip),
            "reasons": ["adapter_model_member_missing"],
        }

    posttrain = inspect_posttrain_report(posttrain_report, candidate_zip, str(zip_report.get("sha256") or ""))
    preflight = inspect_preflight_report(preflight_report, candidate_zip)
    doublecheck = inspect_doublecheck(doublecheck_json, candidate_zip, str(zip_report.get("sha256") or ""))
    manifest = inspect_manifest(manifest_json)

    sections = {
        "zip": zip_report,
        "config": config_report,
        "tensors": tensor_report,
        "manifest": manifest,
        "posttrain_report": posttrain,
        "preflight_report": preflight,
        "doublecheck": doublecheck,
    }
    reasons: list[str] = []
    for name, section in sections.items():
        if not section.get("ok"):
            reasons.extend(f"{name}:{reason}" for reason in section.get("reasons", []))

    audit = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "competition": COMPETITION,
        "candidate_zip": str(candidate_zip),
        "sections": sections,
        "decision": {
            "submit_ready": not reasons,
            "reasons": reasons,
            "requires_kaggle_basename_stage": candidate_zip.name != REQUIRED_SUBMISSION_BASENAME,
        },
    }
    return audit, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-zip", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--posttrain-report", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--doublecheck-json", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--kaggle-json", type=Path)
    parser.add_argument("--message", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit, reasons = build_audit(args)
    if reasons:
        audit["submitted"] = False
        write_json(args.output_json, audit)
        print("=== V199B SAFE SUBMIT AUDIT ===")
        print("submit_ready: False")
        for reason in reasons:
            print("  -", reason)
        print("report:", args.output_json)
        return 2

    creds = ensure_kaggle_credentials(args.kaggle_json)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError("kaggle package is required. Run: pip install -q --upgrade kaggle==2.0.2") from exc

    api = KaggleApi()
    api.authenticate()
    before = summarize_latest_submissions(api, limit=8)
    candidate_zip = args.candidate_zip.resolve()
    submit_zip, submit_stage = prepare_kaggle_named_zip(candidate_zip, args.output_json)
    audit["credentials"] = creds
    audit["message"] = args.message
    audit["submit_stage"] = submit_stage
    audit["submissions_before"] = before

    print("=== V199B SAFE KAGGLE SUBMIT ===")
    print("submit_ready: True")
    print("candidate_zip:", candidate_zip)
    print("submit_zip:", submit_zip)
    print("submit_stage_mode:", submit_stage["mode"])
    print("zip_sha256:", audit["sections"]["zip"]["sha256"])
    print("adapter_model_sha256:", audit["sections"]["zip"]["adapter_model_member_sha256"])
    print("baseline_eval_loss:", audit["sections"]["manifest"]["baseline_eval_loss"])
    print("final_eval_loss:", audit["sections"]["manifest"]["final_eval_loss"])
    print("kaggle_user:", creds["username"])

    if not args.submit:
        audit["submitted"] = False
        audit["submit_skipped_reason"] = "missing_--submit"
        write_json(args.output_json, audit)
        print("DRY AUDIT ONLY: pass --submit to upload to Kaggle.")
        print("report:", args.output_json)
        return 0

    try:
        response = api.competition_submit(str(submit_zip), args.message, COMPETITION)
    except Exception as exc:
        audit["submitted"] = False
        audit["submit_error"] = exception_details(exc)
        audit["submissions_after"] = summarize_latest_submissions(api, limit=8)
        write_json(args.output_json, audit)
        print("submit_failed: True")
        print("submit_error:", audit["submit_error"].get("message"))
        print("report:", args.output_json)
        raise

    audit["submitted"] = True
    audit["submit_response"] = str(response)
    if args.poll_seconds > 0:
        print(f"Waiting {args.poll_seconds}s before polling submissions...")
        time.sleep(args.poll_seconds)
    audit["submissions_after"] = summarize_latest_submissions(api, limit=8)
    write_json(args.output_json, audit)
    print("submitted: True")
    print("submit_response:", response)
    print("report:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
