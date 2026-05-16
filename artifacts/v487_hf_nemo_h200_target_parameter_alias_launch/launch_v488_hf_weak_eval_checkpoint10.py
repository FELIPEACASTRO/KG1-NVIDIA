#!/usr/bin/env python3
"""Launch focused V488 V221-contract weak eval for V487 checkpoint-10 only."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_v487_hf_weak_eval.py")


def load_base_launcher():
    spec = importlib.util.spec_from_file_location("launch_v487_hf_weak_eval_base", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    launcher = load_base_launcher()
    launcher.VERSION = "v488_target_parameter_alias_checkpoint10_focused_weak_eval"
    launcher.RUN_ID = "v488-h200-v221contract-checkpoint10-focused-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    launcher.REQUESTED_ADAPTERS = [
        ("checkpoint-10", "v488_target_parameter_alias_checkpoint_10_v221_contract"),
    ]
    launcher.OUTPUT_PATH_IN_REPO = f"evals/{launcher.RUN_ID}"
    return int(launcher.main())


if __name__ == "__main__":
    raise SystemExit(main())
