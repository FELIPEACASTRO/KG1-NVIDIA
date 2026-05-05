#!/usr/bin/env python3
"""Convert and gate one V202D-selected output directory.

This is offline and non-submitting. It packages only the selected local
``final_adapter`` into Kaggle adapter layout after the strict promotion eval
has selected a candidate.
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
    parser.add_argument("--root", type=Path, default=Path("/content/kg1_v202d"))
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
    gate_dir = output_root / "posttrain_kaggle_gate_v202d"
    result = convert_candidate(
        root,
        output_root / "final_adapter",
        gate_dir / "final",
        f"v202d-{args.candidate_label}-final",
    )
    result["label"] = "final"
    result["candidate_label"] = args.candidate_label
    result["training_manifest"] = maybe_manifest(output_root / "final_adapter")

    ready = bool(result["decision"]["ready"])
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root": str(root),
        "output_root": str(output_root),
        "candidate_label": args.candidate_label,
        "candidates": [result],
        "decision": {
            "ready_candidate_count": 1 if ready else 0,
            "primary_label": "final" if ready else None,
            "primary_zip": result["zip"]["path"] if ready and result.get("zip") else None,
            "do_not_submit_without_explicit_authorization": True,
            "production_baseline_ref": 52275052,
            "production_baseline_public_score": "0.86",
            "production_baseline_rank": "19/2613",
            "ready": ready,
        },
    }
    report_path = gate_dir / "v202d_posttrain_gate_report.json"
    write_json(report_path, report)

    print(f"\n=== V202D POSTTRAIN GATE: {args.candidate_label} ===")
    status = "READY" if ready else "BLOCKED"
    print(f"final: {status}")
    if result.get("available") and result.get("zip"):
        print(f"  zip: {result['zip']['path']}")
        print(f"  zip_sha256: {result['zip']['sha256']}")
    if result["decision"]["reasons"]:
        print(f"  reasons: {result['decision']['reasons']}")
    print(f"report: {report_path}")
    if ready:
        print(f"PRIMARY_CANDIDATE_ZIP={result['zip']['path']}")
    else:
        print("PRIMARY_CANDIDATE_ZIP=NONE")
    if args.fail_on_block and not ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
