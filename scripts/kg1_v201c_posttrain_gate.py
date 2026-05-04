#!/usr/bin/env python3
"""Convert and gate one V201C multi-candidate output directory.

The script packages only local training artifacts. It never submits to Kaggle.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kg1_v198_posttrain_gate import convert_candidate, maybe_manifest, utc_now, write_json  # noqa: E402


SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/kg1_v199"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SAFE_LABEL_RE.match(args.candidate_label):
        raise ValueError(f"Unsafe candidate label: {args.candidate_label!r}")

    root = args.root.resolve()
    output_root = args.output_root.resolve()
    gate_dir = output_root / "posttrain_kaggle_gate"
    candidates = [
        (
            "final",
            output_root / "final_adapter",
            gate_dir / "final",
            f"v201c-{args.candidate_label}-final",
        ),
    ]
    for checkpoint_dir in sorted(output_root.glob("checkpoint-*")):
        label = checkpoint_dir.name.replace("-", "")
        candidates.append(
            (
                label,
                checkpoint_dir,
                gate_dir / label,
                f"v201c-{args.candidate_label}-{label}",
            )
        )

    results = []
    for label, source_dir, out_dir, run_id in candidates:
        result = convert_candidate(root, source_dir, out_dir, run_id)
        result["label"] = label
        result["candidate_label"] = args.candidate_label
        result["training_manifest"] = maybe_manifest(source_dir)
        results.append(result)

    ready = [result for result in results if result["decision"]["ready"]]
    primary = next((result for result in ready if result["label"] == "final"), None)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root": str(root),
        "output_root": str(output_root),
        "candidate_label": args.candidate_label,
        "candidates": results,
        "decision": {
            "ready_candidate_count": len(ready),
            "primary_label": primary["label"] if primary else None,
            "primary_zip": primary["zip"]["path"] if primary else None,
            "do_not_submit_without_explicit_authorization": True,
            "production_baseline_ref": 52275052,
            "production_baseline_public_score": "0.86",
            "production_baseline_rank": "19/2613",
            "ready": bool(primary),
        },
    }
    report_path = gate_dir / "v201c_posttrain_gate_report.json"
    write_json(report_path, report)

    print(f"\n=== V201C POSTTRAIN GATE: {args.candidate_label} ===")
    for result in results:
        status = "READY" if result["decision"]["ready"] else "BLOCKED"
        print(f"{result['label']}: {status}")
        if result.get("available") and result.get("zip"):
            print(f"  zip: {result['zip']['path']}")
            print(f"  zip_sha256: {result['zip']['sha256']}")
        if result["decision"]["reasons"]:
            print(f"  reasons: {result['decision']['reasons']}")
    print(f"report: {report_path}")
    if primary:
        print(f"PRIMARY_CANDIDATE_ZIP={primary['zip']['path']}")
    else:
        print("PRIMARY_CANDIDATE_ZIP=NONE")
    if args.fail_on_block and not primary:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
