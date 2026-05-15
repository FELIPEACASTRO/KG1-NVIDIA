#!/usr/bin/env python3
"""Launch V416 V221-contract weak eval for rawstyle transfer checkpoints."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_EVAL = REPO_ROOT / "artifacts/v413_hf_h200_solver_first_transfer_launch/launch_v413_hf_weak_eval.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("v413_base_eval", BASE_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base eval launcher: {BASE_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.VERSION = "v416_rawstyle_transfer_v221_weak_eval"
base.EXPECTED_COMMIT = "f0dff1caaaceb18bbc5978e41dd4ff1e25ab7a50"
base.RUN_ID = "v416-h200-v221contract-rawstyle-transfer-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base.ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v416-h200-rawstyle-transfer-v290ckpt6"
base.REQUESTED_ADAPTERS = [
    ("checkpoint-2", "v416_rawstyle_transfer_checkpoint_2_v221_contract"),
    ("checkpoint-4", "v416_rawstyle_transfer_checkpoint_4_v221_contract"),
]
base.OUTPUT_REPO = base.ADAPTER_REPO
base.OUTPUT_PATH_IN_REPO = f"evals/{base.RUN_ID}"

_base_main = base.main


def main() -> int:
    print("=== V416 WEAK EVAL LAUNCH START ===", flush=True)
    print("adapter_repo =", base.ADAPTER_REPO, flush=True)
    print("requested_adapters =", json.dumps(base.REQUESTED_ADAPTERS), flush=True)
    rc = _base_main()
    print("=== V416 WEAK EVAL LAUNCH END ===", flush=True)
    return rc


base.main = main


if __name__ == "__main__":
    raise SystemExit(base.main())
