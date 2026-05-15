#!/usr/bin/env python3
"""V454 CPU bit guardrail decision.

This script consolidates the known bit-manipulation solver/teacher evidence and
the adapter-transfer attempts. It is intentionally CPU-only and does not train,
submit, or authorize GPU by itself.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "v454_bit_guardrail_decision" / "20260515T_cpu_gate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_metric(text: str, pattern: str, default: int | None = None) -> int | None:
    match = re.search(pattern, text, re.I)
    if not match:
        return default
    return int(match.group(1))


def latest_glob(pattern: str) -> Path:
    paths = sorted(REPO_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[0]


def parse_v359(text: str) -> dict[str, Any]:
    return {
        "status": "rejected",
        "weak_total": extract_metric(text, r"Result:\s*`?(\d+)/315"),
        "equation_transform": extract_metric(text, r"`equation_transform`:\s*`?(\d+)/155"),
        "bit_manipulation": extract_metric(text, r"`bit_manipulation`:\s*`?(\d+)/160"),
        "truncated": extract_metric(text, r"Truncated:\s*`?(\d+)"),
    }


def parse_v368(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows") or []
    row = rows[0] if rows else {}
    return {
        "status": row.get("status", ""),
        "weak_total": int(row.get("correct", 0) or 0),
        "equation_transform": int(row.get("equation_transform_correct", 0) or 0),
        "bit_manipulation": int(row.get("bit_manipulation_correct", 0) or 0),
        "truncated": int(row.get("truncated", 0) or 0),
        "adapter": row.get("adapter", ""),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V454 BIT GUARDRAIL DECISION START ===", flush=True)
    print("output_dir =", args.output_dir, flush=True)

    v296_path = Path(args.v296_summary)
    v333_path = Path(args.v333_manifest)
    v357_path = Path(args.v357_summary_md)
    v366_path = Path(args.v366_summary_md)
    v359_path = Path(args.v359_summary_md)
    v368_path = Path(args.v368_eval_summary_json)

    for path in (v296_path, v333_path, v357_path, v366_path, v359_path, v368_path):
        print("input =", path, "exists =", path.exists(), flush=True)
        if not path.exists():
            raise FileNotFoundError(path)

    v296 = read_json(v296_path)
    v333 = read_json(v333_path)
    v357_text = read_text(v357_path)
    v366_text = read_text(v366_path)
    v359_text = read_text(v359_path)
    v368 = read_json(v368_path)

    v333_weak = v333["weak_gate"]
    v333_train = v333["train_bit_comparison"]
    v359 = parse_v359(v359_text)
    v368_result = parse_v368(v368)

    evidence = {
        "v296_stride_train": {
            "bit_rows": v296.get("bit_rows"),
            "stride_correct": v296.get("stride_correct"),
            "current_correct": v296.get("current_correct"),
            "stride_gain_vs_current": v296.get("stride_gain_vs_current"),
            "stride_loss_vs_current": v296.get("stride_loss_vs_current"),
            "decision": "diagnostic_only_lossy",
        },
        "v333_tong_train": {
            "rows": v333_train.get("rows"),
            "tong_correct": v333_train.get("tong_correct"),
            "current_solver_correct": v333_train.get("current_solver_correct"),
            "tong_gain_vs_current": v333_train.get("tong_gain_vs_current"),
            "tong_loss_vs_current": v333_train.get("tong_loss_vs_current"),
            "decision": "teacher_strong_but_lossy_on_weak_replace",
        },
        "v333_tong_weak_replace": {
            "baseline_total": v333_weak["baseline_summary"]["correct"],
            "baseline_bit": v333_weak["baseline_summary"]["family"]["bit_manipulation"]["correct"],
            "tong_total": v333_weak["tong_bit_replace_summary"]["correct"],
            "tong_bit": v333_weak["tong_bit_replace_summary"]["family"]["bit_manipulation"]["correct"],
            "tong_gains_vs_baseline": v333_weak["tong_gains_vs_baseline"],
            "tong_losses_vs_baseline": v333_weak["tong_losses_vs_baseline"],
            "decision": v333["decision"]["decision"],
        },
        "v357_teacher_cpu": {
            "weak_total": extract_metric(v357_text, r"Weak total\s*\|\s*`\d+/315`\s*\|\s*`(\d+)/315`"),
            "bit_manipulation": extract_metric(v357_text, r"`bit_manipulation`\s*\|\s*`\d+/160`\s*\|\s*`(\d+)/160`"),
            "losses": extract_metric(v357_text, r"Losses\s*\|\s*`0`\s*\|\s*`(\d+)`"),
            "decision": "teacher_cpu_only",
        },
        "v366_teacher_cpu": {
            "weak_total": extract_metric(v366_text, r"Saida V366:\s*`(\d+)/315`"),
            "equation_transform": extract_metric(v366_text, r"Saida V366:.*?equation_transform=(\d+)/155"),
            "bit_manipulation": extract_metric(v366_text, r"Saida V366:.*?bit_manipulation=(\d+)/160"),
            "accepted_losses": extract_metric(v366_text, r"Accepted losses:\s*`?(\d+)"),
            "decision": "teacher_cpu_only",
        },
        "v359_adapter_transfer": v359 | {"decision": "rejected_adapter_transfer"},
        "v368_adapter_transfer": v368_result | {"decision": "rejected_adapter_transfer"},
    }

    train_teacher_bit = int(evidence["v333_tong_train"]["tong_correct"] or 0)
    train_teacher_rows = int(evidence["v333_tong_train"]["rows"] or 0)
    weak_teacher_best_bit = max(
        int(evidence["v357_teacher_cpu"]["bit_manipulation"] or 0),
        int(evidence["v366_teacher_cpu"]["bit_manipulation"] or 0),
    )
    adapter_best_bit = max(
        int(v359.get("bit_manipulation") or 0),
        int(v368_result.get("bit_manipulation") or 0),
        136,
    )
    transfer_gap = weak_teacher_best_bit - adapter_best_bit

    hf_gpu_allowed = False
    decision = {
        "decision": "bit_gpu_blocked_until_new_transfer_evidence",
        "hf_gpu_allowed": hf_gpu_allowed,
        "reason": (
            "Bit teacher signal is strong, but direct weak replacement is lossy "
            "and adapter-transfer attempts regressed below bit>=136."
        ),
        "train_teacher_bit": train_teacher_bit,
        "train_teacher_rows": train_teacher_rows,
        "weak_teacher_best_bit": weak_teacher_best_bit,
        "adapter_best_bit_submit_safe": adapter_best_bit,
        "transfer_gap": transfer_gap,
        "next_action": (
            "Do not run bit-only GPU. Build V455 equation target-audit CPU and only "
            "include bit as replay/guardrail if equation CPU gate proves gain."
        ),
    }

    manifest = {
        "schema_version": "kg1_v454_bit_guardrail_decision_v1",
        "generated_at_utc": utc_now(),
        "baseline_submit_safe": {
            "weak_total": 192,
            "equation_transform": 56,
            "bit_manipulation": 136,
            "truncated": 0,
        },
        "inputs": {
            "v296_summary": str(v296_path),
            "v333_manifest": str(v333_path),
            "v357_summary_md": str(v357_path),
            "v366_summary_md": str(v366_path),
            "v359_summary_md": str(v359_path),
            "v368_eval_summary_json": str(v368_path),
        },
        "evidence": evidence,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "v454_bit_guardrail_decision_manifest.json"
    report_path = args.output_dir / "V454_BIT_GUARDRAIL_DECISION.md"
    manifest["outputs"] = {
        "manifest_json": str(manifest_path),
        "report_md": str(report_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# V454 Bit Guardrail Decision",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Submit-safe baseline bit | `{manifest['baseline_submit_safe']['bit_manipulation']}/160` |",
        f"| Tong train teacher bit signal | `{train_teacher_bit}/{train_teacher_rows}` |",
        f"| Best weak CPU teacher bit signal | `{weak_teacher_best_bit}/160` |",
        f"| Best adapter-transfer bit after V359/V368 | `{adapter_best_bit}/160` |",
        f"| Transfer gap indicator | `{transfer_gap}` |",
        f"| `hf_gpu_allowed` | `{str(hf_gpu_allowed).lower()}` |",
        "",
        "## Evidence",
        "",
        "| Source | Finding | Decision |",
        "|---|---|---|",
        (
            f"| V296 stride | train `{v296.get('stride_correct')}/{v296.get('bit_rows')}`, "
            f"gains `{v296.get('stride_gain_vs_current')}`, losses `{v296.get('stride_loss_vs_current')}` | lossy diagnostic |"
        ),
        (
            f"| V333 Tong weak replace | total `{v333_weak['tong_bit_replace_summary']['correct']}/315`, "
            f"bit `{v333_weak['tong_bit_replace_summary']['family']['bit_manipulation']['correct']}/160`, "
            f"gains `{v333_weak['tong_gains_vs_baseline']}`, losses `{v333_weak['tong_losses_vs_baseline']}` | not deployable |"
        ),
        (
            f"| V366 CPU teacher | total `{evidence['v366_teacher_cpu']['weak_total']}/315`, "
            f"bit `{evidence['v366_teacher_cpu']['bit_manipulation']}/160`, losses `{evidence['v366_teacher_cpu']['accepted_losses']}` | teacher only |"
        ),
        (
            f"| V359 adapter transfer | total `{v359.get('weak_total')}/315`, "
            f"bit `{v359.get('bit_manipulation')}/160`, trunc `{v359.get('truncated')}` | rejected |"
        ),
        (
            f"| V368 adapter transfer | total `{v368_result.get('weak_total')}/315`, "
            f"bit `{v368_result.get('bit_manipulation')}/160`, trunc `{v368_result.get('truncated')}` | rejected |"
        ),
        "",
        "## Decision",
        "",
        decision["reason"],
        "",
        "Do not launch another bit-only HF GPU job. Bit remains a guardrail/replay family.",
        "The next active route is equation CPU target audit; bit is included only to prevent regression.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("manifest_json =", manifest_path, flush=True)
    print("report_md =", report_path, flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("=== V454 BIT GUARDRAIL DECISION END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--v296-summary",
        default=REPO_ROOT
        / "artifacts"
        / "v296_bit_stride_solver_audit"
        / "20260512T0450Z"
        / "v296_bit_stride_solver_audit_summary.json",
    )
    parser.add_argument(
        "--v333-manifest",
        default=latest_glob("artifacts/v333_tong_bit_reasoner_gate/*/v333_tong_bit_reasoner_gate_manifest.json"),
    )
    parser.add_argument(
        "--v357-summary-md",
        default=REPO_ROOT / "artifacts" / "v357_bit_global_ternary_gate" / "V357_RESULT_SUMMARY.md",
    )
    parser.add_argument(
        "--v366-summary-md",
        default=REPO_ROOT / "artifacts" / "v366_bit_fullbyte_ternary_op_gate" / "V366_RESULT_SUMMARY.md",
    )
    parser.add_argument(
        "--v359-summary-md",
        default=REPO_ROOT / "artifacts" / "v359_hf_a100_v358_bit_ternary_launch" / "V359_RESULT_SUMMARY.md",
    )
    parser.add_argument(
        "--v368-eval-summary-json",
        default=REPO_ROOT
        / "artifacts"
        / "v368_hf_a100_v367_bit_ternary_launch"
        / "eval_checkpoint1"
        / "batch_candidate_summary.json",
    )
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
