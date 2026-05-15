#!/usr/bin/env python3
"""Collect V291/V290 adapter raw outputs on V435B prompt-only probes.

This script is inference-only. It does not train, score, package, or submit.
The input pack must not contain labels; the output stores adapter generations
so V435 can later build adapter-level hard-negative evidence without using
weak/full labels for training construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import (  # noqa: E402
    MODEL_NAME,
    OFFICIAL_INFERENCE_CONFIG,
    classify_puzzle,
    extract_final_answer,
)
from scripts.evaluate_lora_adapter import (  # noqa: E402
    _sampling_params,
    apply_vllm_runtime_safety_settings,
    render_prompts,
    resolve_base_model_path,
    resolve_model_revision,
    validate_adapter_dir,
)


DEFAULT_PROMPT_PACK = (
    REPO_ROOT
    / "artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/"
    / "v435b_adapter_probe_prompt_pack_prompts.jsonl"
)
DEFAULT_PROMPT_PACK_MANIFEST = (
    REPO_ROOT
    / "artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/"
    / "v435b_adapter_probe_prompt_pack_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v435c_adapter_probe_raw_outputs"
DEFAULT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
DEFAULT_ADAPTER_SUBFOLDER = "checkpoint-6"

OUTPUT_COLUMNS = [
    "id",
    "family",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "prompt",
    "rendered_prompt_sha256",
    "raw_output",
    "prediction",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "adapter_repo",
    "adapter_subfolder",
    "adapter_revision",
    "adapter_identity",
    "decode_config_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not a JSON object")
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def select_prompt_rows(rows: list[dict[str, Any]], *, max_equation: int, max_bit: int, limit: int) -> list[dict[str, Any]]:
    caps = {"equation_transform": max_equation, "bit_manipulation": max_bit}
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        prompt = str(row.get("prompt", ""))
        family = str(row.get("family") or classify_puzzle(prompt))
        if family not in caps:
            continue
        if counts[family] >= caps[family]:
            continue
        out = dict(row)
        out["id"] = str(out.get("id", "")).strip()
        out["family"] = family
        out["prompt"] = prompt
        out["prompt_sha256"] = str(out.get("prompt_sha256") or sha256_text(prompt.replace("\r\n", "\n")))
        out["prompt_normalized_sha256"] = str(out.get("prompt_normalized_sha256") or sha256_text(normalize_prompt(prompt)))
        selected.append(out)
        counts[family] += 1
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def assert_prompt_pack_has_no_answers(rows: list[dict[str, Any]]) -> None:
    forbidden = {"answer", "label", "target", "correct", "is_correct", "solution"}
    offenders: list[str] = []
    for row in rows:
        present = sorted(forbidden.intersection(row.keys()))
        if present:
            offenders.append(f"{row.get('id', '<missing-id>')}:{','.join(present)}")
    if offenders:
        raise RuntimeError("prompt pack contains forbidden answer-like columns: " + "; ".join(offenders[:10]))


def materialize_adapter(adapter_repo: str, adapter_subfolder: str, adapter_revision: str) -> Path:
    from huggingface_hub import snapshot_download

    print("adapter_download_repo =", adapter_repo, flush=True)
    print("adapter_download_subfolder =", adapter_subfolder, flush=True)
    print("adapter_download_revision =", adapter_revision or "main", flush=True)
    allow_patterns = None
    if adapter_subfolder:
        allow_patterns = [f"{adapter_subfolder.rstrip('/')}/*"]
    local_root = Path(
        snapshot_download(
            repo_id=adapter_repo,
            revision=adapter_revision or None,
            allow_patterns=allow_patterns,
        )
    )
    adapter_dir = local_root / adapter_subfolder if adapter_subfolder else local_root
    return validate_adapter_dir(adapter_dir)


def decode_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(OFFICIAL_INFERENCE_CONFIG)
    config.update(
        {
            "max_tokens": int(args.max_tokens),
            "max_model_len": int(args.max_model_len),
            "max_num_seqs": int(args.max_num_seqs),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "warmup_rows": int(args.warmup_rows),
        }
    )
    if args.disable_thinking:
        config["enable_thinking"] = False
    if args.no_prompt_suffix:
        config["prompt_suffix"] = ""
    elif args.prompt_suffix:
        config["prompt_suffix"] = args.prompt_suffix
    return config


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V435C ADAPTER PROBE RAW OUTPUTS START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("prompt_pack_jsonl =", args.prompt_pack_jsonl, flush=True)
    print("prompt_pack_manifest_json =", args.prompt_pack_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("max_equation =", args.max_equation, flush=True)
    print("max_bit =", args.max_bit, flush=True)
    print("limit =", args.limit, flush=True)
    print("adapter_repo =", args.adapter_repo, flush=True)
    print("adapter_subfolder =", args.adapter_subfolder, flush=True)
    print("adapter_revision =", args.adapter_revision or "main", flush=True)

    if not args.prompt_pack_jsonl.is_file():
        raise FileNotFoundError(args.prompt_pack_jsonl)
    if not args.prompt_pack_manifest_json.is_file():
        raise FileNotFoundError(args.prompt_pack_manifest_json)
    prompt_manifest = read_json(args.prompt_pack_manifest_json)
    prompt_rows = read_jsonl(args.prompt_pack_jsonl)
    assert_prompt_pack_has_no_answers(prompt_rows)
    selected = select_prompt_rows(prompt_rows, max_equation=args.max_equation, max_bit=args.max_bit, limit=args.limit)
    if not selected:
        raise RuntimeError("no prompt rows selected for V435C")
    family_counts = Counter(str(row.get("family", "")) for row in selected)
    print("selected_rows =", len(selected), flush=True)
    print("selected_family_counts =", json.dumps(dict(sorted(family_counts.items())), sort_keys=True), flush=True)

    config = decode_config(args)
    config_sha = sha256_text(json.dumps(config, sort_keys=True))
    print("decode_config =", json.dumps(config, indent=2, sort_keys=True), flush=True)
    print("decode_config_sha256 =", config_sha, flush=True)
    runtime_settings = apply_vllm_runtime_safety_settings()
    print("vllm_runtime_safety_settings =", json.dumps(runtime_settings, indent=2, sort_keys=True), flush=True)

    adapter_dir = materialize_adapter(args.adapter_repo, args.adapter_subfolder, args.adapter_revision)
    adapter_identity = {
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "adapter_revision": args.adapter_revision or "main",
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
    }
    for candidate in ("adapter_model.safetensors", "adapter_model.bin"):
        model_path = adapter_dir / candidate
        if model_path.exists():
            adapter_identity["adapter_model_file"] = candidate
            adapter_identity["adapter_model_sha256"] = sha256_file(model_path)
            adapter_identity["adapter_model_bytes"] = model_path.stat().st_size
            break
    adapter_identity_json = json.dumps(adapter_identity, sort_keys=True)
    print("adapter_identity =", json.dumps(adapter_identity, indent=2, sort_keys=True), flush=True)

    base_model_path = resolve_base_model_path(args.base_model_path)
    print("base_model_path =", base_model_path, flush=True)
    print("model_name =", MODEL_NAME, flush=True)
    print("seed =", args.seed, flush=True)

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm_kwargs = {
        "model": str(base_model_path),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_num_seqs": int(config.get("max_num_seqs", 64)),
        "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.85)),
        "dtype": config.get("dtype", "auto"),
        "max_model_len": int(config.get("max_model_len", 8192)),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "enable_lora": True,
        "max_lora_rank": int(config.get("max_lora_rank", 32)),
        "enable_prefix_caching": bool(config.get("enable_prefix_caching", True)),
        "enable_chunked_prefill": bool(config.get("enable_chunked_prefill", True)),
    }
    model_revision = resolve_model_revision(str(base_model_path), config)
    if model_revision:
        llm_kwargs["revision"] = model_revision
        llm_kwargs["tokenizer_revision"] = model_revision
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    print("llm_kwargs =", json.dumps({k: str(v) for k, v in llm_kwargs.items()}, indent=2, sort_keys=True), flush=True)

    load_start = time.time()
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print("vllm_load_elapsed_s =", round(time.time() - load_start, 3), flush=True)

    questions = pd.DataFrame(selected)[["id", "prompt"]].copy()
    rendered = render_prompts(tokenizer, questions, config)
    sampling_params = _sampling_params(config, int(args.seed))
    lora_request = LoRARequest("adapter", 1, str(adapter_dir))

    warmup_rows = min(int(config.get("warmup_rows", 4)), len(rendered))
    if warmup_rows > 0:
        print("warmup_rows =", warmup_rows, flush=True)
        warmup_start = time.time()
        _ = llm.generate(rendered[:warmup_rows], sampling_params=sampling_params, lora_request=lora_request)
        print("warmup_elapsed_s =", round(time.time() - warmup_start, 3), flush=True)
    else:
        print("warmup_rows = 0", flush=True)

    print("generation_rows =", len(rendered), flush=True)
    gen_start = time.time()
    outputs = llm.generate(rendered, sampling_params=sampling_params, lora_request=lora_request)
    gen_elapsed = time.time() - gen_start
    print("generation_elapsed_s =", round(gen_elapsed, 3), flush=True)

    output_rows: list[dict[str, Any]] = []
    for row, rendered_prompt, output in zip(selected, rendered, outputs):
        completion = output.outputs[0]
        raw_output = str(completion.text or "")
        output_rows.append(
            {
                "id": str(row["id"]),
                "family": str(row["family"]),
                "prompt_sha256": str(row["prompt_sha256"]),
                "prompt_normalized_sha256": str(row["prompt_normalized_sha256"]),
                "prompt": str(row["prompt"]),
                "rendered_prompt_sha256": sha256_text(rendered_prompt),
                "raw_output": raw_output,
                "prediction": extract_final_answer(raw_output),
                "prompt_tokens": len(getattr(output, "prompt_token_ids", []) or []),
                "completion_tokens": len(getattr(completion, "token_ids", []) or []),
                "finish_reason": completion.finish_reason or "",
                "adapter_repo": args.adapter_repo,
                "adapter_subfolder": args.adapter_subfolder,
                "adapter_revision": args.adapter_revision or "main",
                "adapter_identity": adapter_identity_json,
                "decode_config_sha256": config_sha,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.label}_raw_outputs.csv"
    jsonl_path = args.output_dir / f"{args.label}_raw_outputs.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_csv(csv_path, output_rows)
    write_jsonl(jsonl_path, output_rows)
    finish_counts = Counter(str(row.get("finish_reason", "")) for row in output_rows)
    token_total = sum(int(row.get("completion_tokens", 0) or 0) for row in output_rows)
    manifest = {
        "schema_version": "kg1_v435c_adapter_probe_raw_outputs_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "answers_input": False,
            "answers_exported": False,
            "weak_or_full_rows_input": False,
            "purpose": "Collect adapter raw outputs on permitted prompt-only probes for adapter-level hard-negative mining.",
        },
        "inputs": {
            "prompt_pack_jsonl": str(args.prompt_pack_jsonl),
            "prompt_pack_jsonl_sha256": sha256_file(args.prompt_pack_jsonl),
            "prompt_pack_manifest_json": str(args.prompt_pack_manifest_json),
            "prompt_pack_manifest_sha256": sha256_file(args.prompt_pack_manifest_json),
            "prompt_pack_schema": prompt_manifest.get("schema_version", ""),
        },
        "selection": {
            "available_rows": len(prompt_rows),
            "selected_rows": len(output_rows),
            "max_equation": int(args.max_equation),
            "max_bit": int(args.max_bit),
            "limit": int(args.limit),
            "family_counts": dict(sorted(family_counts.items())),
        },
        "adapter_identity": adapter_identity,
        "model": {
            "base_model_path": str(base_model_path),
            "model_name": MODEL_NAME,
            "model_revision": config.get("model_revision", ""),
        },
        "decode_config": config,
        "decode_config_sha256": config_sha,
        "runtime_settings": runtime_settings,
        "generation": {
            "seed": int(args.seed),
            "finish_reason_counts": dict(sorted(finish_counts.items())),
            "completion_tokens": int(token_total),
            "generation_elapsed_s": float(gen_elapsed),
            "tokens_per_second": float(token_total / gen_elapsed) if gen_elapsed > 0 else 0.0,
        },
        "outputs": {
            "raw_outputs_csv": str(csv_path),
            "raw_outputs_csv_sha256": sha256_file(csv_path),
            "raw_outputs_jsonl": str(jsonl_path),
            "raw_outputs_jsonl_sha256": sha256_file(jsonl_path),
            "manifest_json": str(manifest_path),
        },
        "next_action": "Rerun V435 adapter-level pair gate using these raw outputs; only then consider V436 transfer generation.",
    }
    write_json(manifest_path, manifest)
    print("raw_outputs_csv =", csv_path, flush=True)
    print("raw_outputs_jsonl =", jsonl_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("manifest_sha256 =", sha256_file(manifest_path), flush=True)
    print("=== V435C ADAPTER PROBE RAW OUTPUTS END ===", flush=True)
    return manifest


def self_test() -> None:
    rows = [
        {"id": "e1", "family": "equation_transform", "prompt": "Find the transformation rule."},
        {"id": "b1", "family": "bit_manipulation", "prompt": "8-bit binary bit manipulation"},
        {"id": "e2", "family": "equation_transform", "prompt": "transformation rules"},
        {"id": "x1", "family": "unit_conversion", "prompt": "unit conversion"},
    ]
    selected = select_prompt_rows(rows, max_equation=1, max_bit=1, limit=0)
    assert [row["id"] for row in selected] == ["e1", "b1"], selected
    selected_limited = select_prompt_rows(rows, max_equation=2, max_bit=1, limit=2)
    assert [row["id"] for row in selected_limited] == ["e1", "b1"], selected_limited
    assert extract_final_answer("abc \\boxed{1010}") == "1010"
    try:
        assert_prompt_pack_has_no_answers([{"id": "bad", "prompt": "x", "answer": "1"}])
    except RuntimeError:
        pass
    else:
        raise AssertionError("forbidden answer column was not rejected")
    print("v435c_self_test=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-pack-jsonl", type=Path, default=DEFAULT_PROMPT_PACK)
    parser.add_argument("--prompt-pack-manifest-json", type=Path, default=DEFAULT_PROMPT_PACK_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v435c_adapter_probe_raw_outputs")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--adapter-subfolder", default=DEFAULT_ADAPTER_SUBFOLDER)
    parser.add_argument("--adapter-revision", default="")
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--max-equation", type=int, default=200)
    parser.add_argument("--max-bit", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--warmup-rows", type=int, default=2)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--no-prompt-suffix", action="store_true")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
