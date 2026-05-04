#!/usr/bin/env python3
"""Streaming KG1 LoRA update-space soup writer.

This variant keeps memory bounded enough to preserve the aaitdads
`lm_head.base_layer.weight` tensor and can emit keys in the primary adapter's
namespace. That matters for Kaggle runtimes that expect `backbone` keys.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
DTYPE_TORCH = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class TensorRecord:
    out_key: str
    kind: str
    canonical_a_key: str | None
    raw_key: str | None
    dtype: str
    shape: list[int]
    start: int
    end: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_key(key: str) -> str:
    return key.replace("base_model.model.backbone.", "base_model.model.model.")


def is_lora_key(key: str) -> bool:
    return ".lora_A." in key or ".lora_B." in key


def lora_pair_key(a_key: str) -> str:
    return a_key.replace(".lora_A.", ".lora_B.")


def safe_primary_extra_key(key: str) -> bool:
    return canonical_key(key) == "base_model.model.lm_head.base_layer.weight"


def key_map(path: Path) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    non_lora: list[str] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for raw in handle.keys():
            if not is_lora_key(raw):
                non_lora.append(raw)
                continue
            key = canonical_key(raw)
            if key in mapping:
                raise RuntimeError(f"canonical key collision: {raw} -> {key}")
            mapping[key] = raw
    return mapping, non_lora


def dtype_name(raw: str) -> str:
    raw = str(raw)
    if raw.startswith("torch."):
        raw = raw.removeprefix("torch.")
    return {
        "float16": "F16",
        "bfloat16": "BF16",
        "float32": "F32",
        "float64": "F64",
        "int64": "I64",
        "int32": "I32",
        "int16": "I16",
        "int8": "I8",
        "uint8": "U8",
        "bool": "BOOL",
        "F16": "F16",
        "BF16": "BF16",
        "F32": "F32",
        "F64": "F64",
        "I64": "I64",
        "I32": "I32",
        "I16": "I16",
        "I8": "I8",
        "U8": "U8",
        "BOOL": "BOOL",
    }[raw]


def nbytes(dtype: str, shape: list[int]) -> int:
    size = DTYPE_BYTES[dtype]
    for dim in shape:
        size *= int(dim)
    return size


def tensor_to_file(tensor: torch.Tensor, handle) -> None:
    arr = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    arr.tofile(handle)


def safetensors_metadata(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    data_start = 8 + header_len
    metadata: dict[str, dict[str, Any]] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        metadata[key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "start": data_start + int(start),
            "end": data_start + int(end),
        }
    return metadata


def copy_file_span(src_handle, dst_handle, start: int, end: int, chunk_size: int = 16 * 1024 * 1024) -> None:
    src_handle.seek(start)
    remaining = end - start
    while remaining:
        chunk = src_handle.read(min(chunk_size, remaining))
        if not chunk:
            raise EOFError(f"unexpected EOF while copying raw tensor span {start}:{end}")
        dst_handle.write(chunk)
        remaining -= len(chunk)


def read_tensor_span(src_handle, meta: dict[str, Any]) -> torch.Tensor:
    start = int(meta["start"])
    end = int(meta["end"])
    src_handle.seek(start)
    payload = bytearray(src_handle.read(end - start))
    if len(payload) != end - start:
        raise EOFError(f"unexpected EOF while reading tensor span {start}:{end}")
    tensor = torch.frombuffer(payload, dtype=DTYPE_TORCH[meta["dtype"]])
    return tensor.reshape([int(dim) for dim in meta["shape"]]).clone()


def compress_weighted_sum(
    a_primary: torch.Tensor,
    b_primary: torch.Tensor,
    a_other: torch.Tensor,
    b_other: torch.Tensor,
    w_primary: float,
    w_other: float,
    rank: int,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    parts_b: list[torch.Tensor] = []
    parts_a: list[torch.Tensor] = []
    for weight, b, a in ((w_primary, b_primary, a_primary), (w_other, b_other, a_other)):
        if abs(weight) < 1e-12:
            continue
        sign = 1.0 if weight >= 0 else -1.0
        scale = math.sqrt(abs(weight))
        parts_b.append(b * scale)
        parts_a.append(a * (sign * scale))

    b_cat = torch.cat(parts_b, dim=1)
    a_cat = torch.cat(parts_a, dim=0)
    q_b, r_b = torch.linalg.qr(b_cat, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_cat.T, mode="reduced")
    u_mid, s, vh_mid = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)

    use_rank = min(rank, int(s.numel()))
    u = q_b @ u_mid[:, :use_rank]
    vh = vh_mid[:use_rank, :] @ q_a.T
    s_used = s[:use_rank].clamp_min(0)
    sroot = torch.sqrt(s_used)
    b_new = (u * sroot.unsqueeze(0)).to(dtype_b).contiguous()
    a_new = (sroot.unsqueeze(1) * vh).to(dtype_a).contiguous()

    if use_rank < rank:
        b_new = torch.cat([b_new, torch.zeros((b_new.shape[0], rank - use_rank), dtype=dtype_b)], dim=1).contiguous()
        a_new = torch.cat([a_new, torch.zeros((rank - use_rank, a_new.shape[1]), dtype=dtype_a)], dim=0).contiguous()

    total_energy = float((s * s).sum().item()) if s.numel() else 0.0
    kept_energy = float((s_used * s_used).sum().item()) if s_used.numel() else 0.0
    return a_new, b_new, {
        "singular_values": float(s.numel()),
        "kept_rank": float(use_rank),
        "energy_keep_ratio": kept_energy / total_energy if total_energy > 0 else 1.0,
    }


def build_records(
    primary_weights: Path,
    primary_map: dict[str, str],
    primary_non_lora: list[str],
    rank: int,
    include_safe_primary_non_lora: bool,
    include_key_regex: str | None = None,
    exclude_key_regex: str | None = None,
) -> tuple[list[TensorRecord], list[str]]:
    records: list[TensorRecord] = []
    offset = 0
    copied_non_lora: list[str] = []
    include_re = re.compile(include_key_regex) if include_key_regex else None
    exclude_re = re.compile(exclude_key_regex) if exclude_key_regex else None

    def should_merge_pair(canonical_a_key: str) -> bool:
        if include_re and not include_re.search(canonical_a_key):
            return False
        if exclude_re and exclude_re.search(canonical_a_key):
            return False
        return True

    with safe_open(str(primary_weights), framework="pt", device="cpu") as hp:
        if include_safe_primary_non_lora:
            for raw_key in sorted(primary_non_lora):
                if not safe_primary_extra_key(raw_key):
                    raise RuntimeError(f"unsafe primary non-LoRA tensor: {raw_key}")
                sl = hp.get_slice(raw_key)
                dtype = dtype_name(sl.get_dtype())
                shape = list(sl.get_shape())
                size = nbytes(dtype, shape)
                records.append(TensorRecord(raw_key, "primary_raw", None, raw_key, dtype, shape, offset, offset + size))
                offset += size
                copied_non_lora.append(raw_key)

        for a_key in sorted(k for k in primary_map if ".lora_A." in k):
            b_key = lora_pair_key(a_key)
            merge_pair = should_merge_pair(a_key)
            for canonical, raw_key, kind in ((a_key, primary_map[a_key], "lora_A"), (b_key, primary_map[b_key], "lora_B")):
                sl = hp.get_slice(raw_key)
                dtype = dtype_name(sl.get_dtype())
                shape = list(sl.get_shape())
                size = nbytes(dtype, shape)
                record_kind = kind if merge_pair else "primary_raw"
                records.append(TensorRecord(raw_key, record_kind, a_key, raw_key, dtype, shape, offset, offset + size))
                offset += size
    return records, copied_non_lora


def write_streaming_safetensors(
    out_path: Path,
    records: list[TensorRecord],
    primary_weights: Path,
    other_weights: Path,
    primary_map: dict[str, str],
    other_map: dict[str, str],
    primary_weight: float,
    other_weight: float,
    rank: int,
) -> dict[str, float]:
    header = {
        rec.out_key: {"dtype": rec.dtype, "shape": rec.shape, "data_offsets": [rec.start, rec.end]}
        for rec in records
    }
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding

    energy_min = 1.0
    energy_sum = 0.0
    energy_count = 0
    pair_cache: tuple[str, torch.Tensor, torch.Tensor, dict[str, float]] | None = None
    primary_meta = safetensors_metadata(primary_weights)
    other_meta = safetensors_metadata(other_weights)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out, primary_weights.open("rb") as primary_raw, other_weights.open("rb") as other_raw:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for idx, rec in enumerate(records, 1):
            if rec.kind == "primary_raw":
                start = int(primary_meta[rec.raw_key]["start"])
                end = int(primary_meta[rec.raw_key]["end"])
                copy_file_span(primary_raw, out, start, end)
            else:
                assert rec.canonical_a_key is not None
                if pair_cache is None or pair_cache[0] != rec.canonical_a_key:
                    a_key = rec.canonical_a_key
                    b_key = lora_pair_key(a_key)
                    a_p_raw = read_tensor_span(primary_raw, primary_meta[primary_map[a_key]])
                    b_p_raw = read_tensor_span(primary_raw, primary_meta[primary_map[b_key]])
                    a_o_raw = read_tensor_span(other_raw, other_meta[other_map[a_key]])
                    b_o_raw = read_tensor_span(other_raw, other_meta[other_map[b_key]])
                    a_new, b_new, info = compress_weighted_sum(
                        a_p_raw.float(),
                        b_p_raw.float(),
                        a_o_raw.float(),
                        b_o_raw.float(),
                        primary_weight,
                        other_weight,
                        rank,
                        a_p_raw.dtype,
                        b_p_raw.dtype,
                    )
                    pair_cache = (a_key, a_new, b_new, info)
                    energy_min = min(energy_min, info["energy_keep_ratio"])
                    energy_sum += info["energy_keep_ratio"]
                    energy_count += 1
                    del a_p_raw, b_p_raw, a_o_raw, b_o_raw
                tensor_to_file(pair_cache[1] if rec.kind == "lora_A" else pair_cache[2], out)

            if idx % 500 == 0:
                gc.collect()
                print(f"wrote_tensors={idx}/{len(records)} energy_min={energy_min:.6f}", flush=True)

    return {
        "energy_keep_ratio_min": energy_min,
        "energy_keep_ratio_mean": energy_sum / energy_count if energy_count else 1.0,
        "lora_pair_count": energy_count,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_dir
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    primary_weights = args.primary_adapter / "adapter_model.safetensors"
    other_weights = args.other_adapter / "adapter_model.safetensors"
    config_source = args.config_source or args.primary_adapter / "adapter_config.json"
    for path in (primary_weights, other_weights, config_source):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = read_json(config_source)
    rank = int(args.rank or cfg.get("r", 32))
    if rank > 32:
        raise ValueError(f"rank exceeds KG1 limit: {rank}")

    primary_map, primary_non_lora = key_map(primary_weights)
    other_map, other_non_lora = key_map(other_weights)
    if set(primary_map) != set(other_map):
        raise RuntimeError("canonical LoRA key sets do not match")

    records, copied_non_lora = build_records(
        primary_weights,
        primary_map,
        primary_non_lora,
        rank,
        args.copy_safe_primary_non_lora,
        args.include_key_regex,
        args.exclude_key_regex,
    )

    out_weights = adapter_dir / "adapter_model.safetensors"
    stats = write_streaming_safetensors(
        out_weights,
        records,
        primary_weights,
        other_weights,
        primary_map,
        other_map,
        args.primary_weight,
        args.other_weight,
        rank,
    )
    shutil.copy2(config_source, adapter_dir / "adapter_config.json")

    # Validate before zipping.
    with safe_open(str(out_weights), framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())

    zip_path = out_dir / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(adapter_dir / "adapter_config.json", "adapter_config.json")
        archive.write(out_weights, "adapter_model.safetensors")

    manifest = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "streaming_update_space_weighted_svd_soup",
        "primary_adapter": str(args.primary_adapter),
        "other_adapter": str(args.other_adapter),
        "config_source": str(config_source),
        "primary_weight": args.primary_weight,
        "other_weight": args.other_weight,
        "rank": rank,
        "tensor_count": tensor_count,
        "records": len(records),
        "merged_lora_tensor_count": sum(1 for rec in records if rec.kind in {"lora_A", "lora_B"}),
        "copied_primary_tensor_count": sum(1 for rec in records if rec.kind == "primary_raw"),
        "include_key_regex": args.include_key_regex,
        "exclude_key_regex": args.exclude_key_regex,
        "primary_non_lora_skipped": primary_non_lora,
        "primary_non_lora_copied": copied_non_lora,
        "other_non_lora_skipped": other_non_lora,
        "output_key_namespace": "primary",
        "output_adapter_sha256": sha256_file(out_weights),
        "output_zip_sha256": sha256_file(zip_path),
        "output_adapter_bytes": out_weights.stat().st_size,
        "output_zip_bytes": zip_path.stat().st_size,
        **stats,
    }
    write_json(out_dir / "update_space_soup_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-adapter", type=Path, required=True)
    parser.add_argument("--other-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-source", type=Path)
    parser.add_argument("--primary-weight", type=float, default=0.98)
    parser.add_argument("--other-weight", type=float, default=0.02)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--copy-safe-primary-non-lora", action="store_true")
    parser.add_argument("--include-key-regex", help="Only merge LoRA pairs whose canonical lora_A key matches this regex.")
    parser.add_argument("--exclude-key-regex", help="Copy primary LoRA pairs whose canonical lora_A key matches this regex.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total = args.primary_weight + args.other_weight
    if total == 0:
        raise ValueError("weights sum to zero")
    args.primary_weight /= total
    args.other_weight /= total
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
