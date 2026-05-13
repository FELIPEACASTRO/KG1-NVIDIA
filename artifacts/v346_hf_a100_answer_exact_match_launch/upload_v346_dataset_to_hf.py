#!/usr/bin/env python3
"""Upload V346 answer-exact-match dataset to the KG1 HF dataset repo."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = REPO_ROOT / "artifacts/v346_answer_exact_match_dataset/20260513T_cpu_gate"
MANIFEST = DATASET_DIR / "v346_answer_exact_match_manifest.json"
TOKENIZATION_GATE = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
PATH_IN_REPO = "data/v346_answer_exact_match/20260513T_cpu_gate"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V346 dataset upload.")
    manifest = read_json(MANIFEST)
    gate = read_json(TOKENIZATION_GATE)
    if manifest.get("schema_version") != "kg1_v346_answer_exact_match_dataset_v1":
        raise RuntimeError("Unexpected V346 dataset schema.")
    if gate.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V346 tokenization gate did not pass.")
    outputs = manifest.get("outputs", {})
    required = [
        Path(outputs["train_jsonl"]),
        Path(outputs["val_jsonl"]),
        MANIFEST,
        TOKENIZATION_GATE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add KG1 V346 answer-exact-match transfer dataset",
    )
    out = {
        "version": "v346_answer_exact_match_hf_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo": DATASET_REPO,
        "dataset_dir": str(DATASET_DIR),
        "path_in_repo": PATH_IN_REPO,
        "upload": str(info),
        "manifest_json": str(MANIFEST),
        "tokenization_gate_manifest": str(TOKENIZATION_GATE),
        "hashes": {
            "train_sha256": outputs.get("train_sha256"),
            "val_sha256": outputs.get("val_sha256"),
        },
    }
    out_path = RUN_DIR / "v346_answer_exact_match_hf_upload_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest =", out_path, flush=True)
    print("upload_url =", info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
