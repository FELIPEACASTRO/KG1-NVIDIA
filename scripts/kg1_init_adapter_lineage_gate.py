#!/usr/bin/env python3
"""Block unsafe Colab training lineage before execution.

This gate is intentionally narrow: it prevents the exact regression that caused
V198 to score 0.84 by allowing a training run to start from unconfirmed V195 or
V198-derived adapters. New continuation notebooks must start from the exact
V194 rank-19 adapter SHA or another explicitly allowlisted init adapter.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


APPROVED_INIT_ADAPTER_MODEL_SHA256 = {
    # Exact V194 rank-19 adapter: 0.86 public score, rank 19/2613.
    "01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f",
}

APPROVED_INIT_ADAPTER_CONFIG_SHA256 = {
    "e5499f128fde60d32d0595d427e4fe84d8abe6dbde1d80886c970e8184e4b743",
}

APPROVED_RANK19_ZIP_SHA256 = {
    "49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8",
}

APPROVED_COMPONENT_SHA256 = {
    # aaitdads component only; not sufficient as the init adapter for V199.
    "3d16ba908a5c8808624f1abd8fdc2b29f92723f5c874761161c894d7e5759f21",
    # Historical submission.zip backing submission 51997779.
    "a3b64b154a6690a58f2338ba1c405422eadc6e1c1357f662eecb187463dfdeee",
    # Canonical Huikang default/20 adapter converted by the original Tinker notebook for submission 51997779.
    "559fd024f5ffcaff0caceddeaf25c3801009d6cabf247fc8dfccbfaf2addd916",
}

KNOWN_BAD_INIT_ADAPTER_MODEL_SHA256 = {
    # V198 final adapter submitted as ref 52301667 and scored 0.84.
    "dd718b0d416fd9cd6ed928e90e185c131fee9d4cb956f57e59b7d00c3266dafa",
}

FORBIDDEN_INIT_PATTERNS = [
    r"INIT_ADAPTER\s*=\s*V198_DRIVE_ROOT\s*/\s*['\"]output_v198/final_adapter['\"]",
    r"INIT_ADAPTER\s*=\s*.*KG1_NVIDIA_V198.*/output_v198/final_adapter",
    r"os\.environ\[['\"]INIT_ADAPTER_DIR['\"]\]\s*=\s*str\(V198",
    r"V198_FINAL_ADAPTER_SHA256",
    r"submitted V198 final adapter",
    r"INIT_ADAPTER\s*=\s*DRIVE_ROOT\s*/\s*['\"]init_adapter_0p86_aaitdads['\"]",
    r"RUN_ID['\"]\]\s*=\s*['\"]v199-conservative-v198-final-20s['\"]",
    r"V195_OUT\s*/\s*['\"]checkpoint-55['\"]",
    r"V195_OUT\s*/\s*['\"]checkpoint-75['\"]",
    r"V195_OUT\s*/\s*['\"]checkpoint-110['\"]",
    r"V195_OUT\s*/\s*['\"]final_adapter['\"]",
    r"best adapter found in Drive",
    r"felipe1983/tinker-adapter-to-ready-to-submit-adapter",
    r"download_lineage_kernel_output",
    r"kernels_output",
    r"KaggleApi",
    r"kaggle['\"]?,\s*['\"]kernels['\"]?,\s*['\"]output",
]


@dataclass
class Finding:
    severity: str
    code: str
    detail: str


def read_notebook_source(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    source_parts: list[str] = []
    for cell in data.get("cells", []):
        source_parts.append("".join(cell.get("source") or []))
    return "\n".join(source_parts)


def read_source(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return read_notebook_source(path)
    return path.read_text(encoding="utf-8")


def add(findings: list[Finding], severity: str, code: str, detail: str) -> None:
    findings.append(Finding(severity=severity, code=code, detail=detail))


def gate_source(source: str, findings: list[Finding]) -> None:
    for pattern in FORBIDDEN_INIT_PATTERNS:
        if re.search(pattern, source, flags=re.IGNORECASE):
            add(findings, "error", "forbidden_init_adapter_lineage", pattern)

    for sha in KNOWN_BAD_INIT_ADAPTER_MODEL_SHA256:
        if sha in source:
            add(findings, "error", "known_bad_init_adapter_sha", sha)

    approved_model_present = any(sha in source for sha in APPROVED_INIT_ADAPTER_MODEL_SHA256)
    approved_config_present = any(sha in source for sha in APPROVED_INIT_ADAPTER_CONFIG_SHA256)
    approved_rank19_zip_present = any(sha in source for sha in APPROVED_RANK19_ZIP_SHA256)
    if not approved_model_present:
        add(
            findings,
            "error",
            "missing_approved_init_adapter_model_sha",
            "training notebook must pin the exact V194 rank-19 init adapter model SHA",
        )
    if not approved_config_present:
        add(
            findings,
            "error",
            "missing_approved_init_adapter_config_sha",
            "training notebook must pin an approved init adapter config SHA",
        )
    if not approved_rank19_zip_present:
        add(
            findings,
            "error",
            "missing_rank19_zip_sha",
            "training notebook must pin the V194 rank-19 submission.zip SHA",
        )

    required_runtime_guards = [
        "BEST_RANKING_BASELINE_RULE",
        "BEST_RANKING_BASELINE",
        "always_start_from_best_known_kaggle_ranking_submission",
        "V194_RANK19_ADAPTER_MODEL_SHA256",
        "V194_RANK19_ZIP_SHA256",
        "V194_RANK19_PUBLIC_SCORE",
        "V194_RANK19_RANK",
        "assert BEST_RANKING_BASELINE['rank'] == '19/2613'",
        "assert BEST_RANKING_BASELINE['adapter_model_sha256'] == V194_RANK19_ADAPTER_MODEL_SHA256",
        "assert BEST_RANKING_BASELINE['zip_sha256'] == V194_RANK19_ZIP_SHA256",
        "def adapter_ready(path, min_model_bytes=1_000_000)",
        "adapter_ready(AAITDADS_ADAPTER, min_model_bytes=4_000_000_000)",
        "adapter_ready(LINEAGE_ADAPTER, min_model_bytes=3_000_000_000)",
        "adapter_ready(INIT_ADAPTER, min_model_bytes=4_000_000_000)",
        "def install_kagglehub_runtime():",
        "kagglesdk==0.1.23",
        "assert hasattr(ke, 'get_web_endpoint')",
        "purge_modules('kagglehub', 'kagglesdk')",
        "LINEAGE_HUIKANG_MODEL_HANDLE",
        "huikang/nemotron-adapter/transformers/default/20",
        "LINEAGE_51997779_ADAPTER_MODEL_SHA256",
        "LINEAGE_51997779_ADAPTER_CONFIG_SHA256",
        "def locate_adapter_dir(root)",
        "def patch_tinker_cookbook_merge()",
        "weights.build_lora_adapter",
        "kagglehub.model_download(LINEAGE_HUIKANG_MODEL_HANDLE",
        "assert lineage_cfg_sha == LINEAGE_51997779_ADAPTER_CONFIG_SHA256",
        "assert lineage_model_sha == LINEAGE_51997779_ADAPTER_MODEL_SHA256",
        "assert sha256_path(model) == V194_RANK19_ADAPTER_MODEL_SHA256",
        "assert sha256_path(cfg) == V194_RANK19_ADAPTER_CONFIG_SHA256",
        "assert manifest.get('output_adapter_sha256') == V194_RANK19_ADAPTER_MODEL_SHA256",
        "assert manifest.get('output_zip_sha256') == V194_RANK19_ZIP_SHA256",
        "FORBIDDEN_INIT_PATH_FRAGMENTS",
        "assert not any(fragment in init_path_text",
    ]
    for fragment in required_runtime_guards:
        if fragment not in source:
            add(findings, "error", "missing_runtime_lineage_guard", fragment)

    if "kagglehub.dataset_download('aaitdads/my-0p86-adapter'" not in source:
        add(
            findings,
            "error",
            "missing_aaitdads_component_download",
            "notebook must download or reuse aaitdads/my-0p86-adapter as a V194 component",
        )

    extra_required = [
        "kagglehub==1.0.1",
        "'--upgrade', '--force-reinstall', 'kagglesdk==0.1.23', 'kagglehub==1.0.1'",
        "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly",
        "A._merge_fused_projections = patched_merge_fused_projections",
        "559fd024f5ffcaff0caceddeaf25c3801009d6cabf247fc8dfccbfaf2addd916",
        "'--primary-weight', '0.985'",
        "'--other-weight', '0.015'",
        "'--include-key-regex'",
        "in_proj|out_proj|q_proj|k_proj|v_proj|o_proj",
        "lora_A",
        "v194 attention-only micro-merge aaitdads98p5 lineage1p5",
    ]
    for fragment in extra_required:
        if fragment not in source:
            add(findings, "error", "missing_rank19_reconstruction_guard", fragment)

    forbidden_kaggle_invocations = [
        "sys.executable, '-m', 'kaggle'",
        'sys.executable, "-m", "kaggle"',
        "kaggle kernels output",
        "api.kernels_output(",
        "from kaggle.api.kaggle_api_extended import KaggleApi",
    ]
    for fragment in forbidden_kaggle_invocations:
        if fragment in source:
            add(
                findings,
                "error",
                "forbidden_private_kernel_lineage_fetch",
                "Do not depend on private Kaggle kernel output. Rebuild the 51997779 lineage from Huikang default/20 and block on the canonical adapter SHA.",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []
    for path in args.paths:
        if not path.exists():
            add(findings, "error", "path_missing", str(path))
            continue
        try:
            gate_source(read_source(path), findings)
        except Exception as exc:
            add(findings, "error", "read_failed", f"{path}: {exc}")

    payload = {
        "decision": "PASS" if not any(item.severity == "error" for item in findings) else "FAIL",
        "paths": [str(path) for path in args.paths],
        "findings": [item.__dict__ for item in findings],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
