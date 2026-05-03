#!/usr/bin/env python3
"""Convert and gate V199 post-training adapters.

V199 is a short conservative continuation from the submitted V198 final
adapter. This script reuses the proven V198 conversion/gate primitives, but
checks the V199 candidate set: final, checkpoint-20, and checkpoint-10.
It never submits to Kaggle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kg1_v198_posttrain_gate import convert_candidate, maybe_manifest, utc_now, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/kg1_v199"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    gate_dir = output_root / "posttrain_kaggle_gate"
    candidates = [
        ("final", output_root / "final_adapter", gate_dir / "final", "v199-conservative-final"),
        ("checkpoint20", output_root / "checkpoint-20", gate_dir / "checkpoint20", "v199-conservative-checkpoint20"),
        ("checkpoint10", output_root / "checkpoint-10", gate_dir / "checkpoint10", "v199-conservative-checkpoint10"),
    ]

    results = []
    for label, source_dir, out_dir, run_id in candidates:
        result = convert_candidate(root, source_dir, out_dir, run_id)
        result["label"] = label
        result["training_manifest"] = maybe_manifest(source_dir)
        results.append(result)

    ready = [result for result in results if result["decision"]["ready"]]
    primary = next((result for result in ready if result["label"] == "final"), ready[0] if ready else None)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root": str(root),
        "output_root": str(output_root),
        "candidates": results,
        "decision": {
            "ready_candidate_count": len(ready),
            "primary_label": primary["label"] if primary else None,
            "primary_zip": primary["zip"]["path"] if primary else None,
            "do_not_submit_without_explicit_authorization": True,
            "ready": bool(primary),
        },
    }
    report_path = gate_dir / "v199_posttrain_gate_report.json"
    write_json(report_path, report)

    print("\n=== V199 POSTTRAIN GATE ===")
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
