#!/usr/bin/env python3
"""V524 quota/token objective audit for KG1 targeted datasets.

Rows are not the same as optimization weight in causal LM SFT. If one family
has much longer assistant traces, token-level CE gives that family more loss
mass than row counts imply. V524 computes row share and loss-token share against
the V522 no-loss reference-gain mix before any paid train.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v524_quota_token_objective_audit"
DEFAULT_V523_MANIFEST = ROOT / "artifacts/v523_targeted_source_trace_pack/20260516T235821Z/v523_targeted_source_trace_pack_manifest.json"
DEFAULT_V286_MANIFEST = ROOT / "artifacts/v523_targeted_source_trace_pack/20260516T235821Z/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
DEFAULT_V522_MANIFEST = ROOT / "artifacts/v522_source_target_alignment_audit/v522_source_target_alignment_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def family_loss_token_mass(v286_split: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    family_summary = v286_split.get("family_summary", {})
    for family, summary in family_summary.items():
        rows = int(summary.get("rows", 0))
        p50 = int(summary.get("loss_token_p50", 0))
        out[family] = rows * p50
    return out


def share(value: float, total: float) -> float:
    return round(value / total, 6) if total else 0.0


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    v523 = read_json(args.v523_manifest_json)
    v286 = read_json(args.v286_manifest_json)
    v522 = read_json(args.v522_manifest_json)
    gains = (v522.get("reference_signal_summary") or {}).get("gain_family_counts", {})
    bit_gain = int(gains.get("bit_manipulation", 0))
    eq_gain = int(gains.get("equation_transform", 0))
    gain_total = bit_gain + eq_gain
    target_bit_share = share(bit_gain, gain_total)

    train_family_counts = v523["train_summary"]["family_counts"]
    train_rows = int(v523["train_summary"]["rows"])
    row_bit_share = share(int(train_family_counts.get("bit_manipulation", 0)), train_rows)
    token_train = (v286.get("tokenization") or {}).get("train") or v286.get("tokenization_train")
    if not isinstance(token_train, dict):
        raise RuntimeError("missing V286 train tokenization summary")
    token_mass = family_loss_token_mass(token_train)
    token_total = sum(token_mass.values())
    token_bit_share = share(token_mass.get("bit_manipulation", 0), token_total)

    findings: list[dict[str, str]] = []
    if token_bit_share > target_bit_share + 0.12:
        findings.append(
            {
                "severity": "warning",
                "code": "bit_token_share_above_reference_gain_share",
                "detail": (
                    f"V523 row bit share={row_bit_share:.3f}, token bit share={token_bit_share:.3f}, "
                    f"reference gain bit share={target_bit_share:.3f}. Use row-normalized loss, "
                    "family weights, or shorter bit traces before GPU."
                ),
            }
        )
    if row_bit_share < target_bit_share - 0.12:
        findings.append(
            {
                "severity": "warning",
                "code": "bit_row_share_below_reference_gain_share",
                "detail": f"row bit share={row_bit_share:.3f}; target from V522={target_bit_share:.3f}",
            }
        )
    if not findings:
        findings.append(
            {
                "severity": "info",
                "code": "quota_within_reference_gain_band",
                "detail": "row/token shares are within the configured tolerance band",
            }
        )

    decision = {
        "gpu_allowed": False,
        "status": "objective_adjustment_required" if any(f["severity"] == "warning" for f in findings) else "quota_ok_cpu_only",
        "reason": "Rows and loss-token mass differ materially; paid GPU must use an objective that prevents token-length bias.",
        "next_action": (
            "Before GPU, either enable row/family-normalized loss or build V525 shorter bit traces, then rerun V286/V513/V524."
        ),
    }
    manifest = {
        "version": "V524",
        "schema_version": "kg1_v524_quota_token_objective_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reference_gain_mix": {
            "bit_manipulation": bit_gain,
            "equation_transform": eq_gain,
            "target_bit_share": target_bit_share,
        },
        "v523_train_mix": {
            "row_family_counts": train_family_counts,
            "row_bit_share": row_bit_share,
            "loss_token_mass": token_mass,
            "loss_token_bit_share": token_bit_share,
        },
        "findings": findings,
        "outputs": {
            "manifest_json": str(output_dir / "v524_quota_token_objective_manifest.json"),
            "summary_md": str(output_dir / "KG1_V524_QUOTA_TOKEN_OBJECTIVE_AUDIT.md"),
        },
    }
    write_json(output_dir / "v524_quota_token_objective_manifest.json", manifest)
    write_summary(output_dir / "KG1_V524_QUOTA_TOKEN_OBJECTIVE_AUDIT.md", manifest)
    return manifest


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# V524 Quota Token Objective Audit",
        "",
        "## Decision",
        "",
        f"- GPU allowed: `{manifest['decision']['gpu_allowed']}`",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Reason: {manifest['decision']['reason']}",
        f"- Next action: {manifest['decision']['next_action']}",
        "",
        "## Calculation",
        "",
        f"- Reference gain bit share: `{manifest['reference_gain_mix']['target_bit_share']}`",
        f"- V523 row bit share: `{manifest['v523_train_mix']['row_bit_share']}`",
        f"- V523 loss-token bit share: `{manifest['v523_train_mix']['loss_token_bit_share']}`",
        f"- Loss-token mass: `{json.dumps(manifest['v523_train_mix']['loss_token_mass'], sort_keys=True)}`",
        "",
        "## Findings",
        "",
    ]
    for finding in manifest["findings"]:
        lines.append(f"- `{finding['severity']}` `{finding['code']}`: {finding['detail']}")
    lines.extend(
        [
            "",
            "## Literature Mapping",
            "",
            "- Class-balanced loss supports reweighting when row counts do not reflect useful signal.",
            "- Focal/hard-example ideas support emphasizing hard residual classes only when labels are verified.",
            "- Curriculum learning supports short verified traces before harder residual traces.",
            "- Scaling/mixture laws warn that loss movement can be dominated by mixture/token mass rather than target ACC.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    if share(2, 4) != 0.5:
        raise SystemExit("self-test failed")
    print("audit_v524_quota_token_objective_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v523-manifest-json", type=Path, default=DEFAULT_V523_MANIFEST)
    parser.add_argument("--v286-manifest-json", type=Path, default=DEFAULT_V286_MANIFEST)
    parser.add_argument("--v522-manifest-json", type=Path, default=DEFAULT_V522_MANIFEST)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    manifest = audit(args)
    print("v524_manifest =", manifest["outputs"]["manifest_json"], flush=True)
    print("v524_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("v524_mix =", json.dumps(manifest["v523_train_mix"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
