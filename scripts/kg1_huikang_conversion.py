#!/usr/bin/env python3
"""Tinker -> Kaggle adapter converter (huikang recipe, 0.85 LB verified).

Faithfully reproduces the conversion pipeline from
``huikang/tinker-submission-notebook`` cell 24 and
``asalhi/tinker-adapter-to-ready-to-submit-adapter`` (top kernels tricks #1).

The Tinker trainer checkpoint and the Kaggle-expected PEFT layout differ on:

1. **Key namespace**: Tinker saves as ``base_model.model.model.layers.*`` but
   Nemotron on HF uses ``base_model.model.backbone.layers.*``.
2. **Fused projections**: Tinker stores ``gate_proj`` and ``x_proj`` as separate
   LoRA pairs; Kaggle PEFT expects a single fused ``in_proj`` adapter.
3. **Expert fusion**: Tinker stores MoE LoRA as ``experts.w1`` / ``experts.w2``
   with a leading expert-dim; PEFT wants one adapter per expert at
   ``experts.{i}.up_proj`` / ``experts.{i}.down_proj``.
4. **Target module list**: huikang winning config uses 9 targets
   ``[q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj,
   lm_head]`` (+0.20 LB vs the 4-target baseline).

This script rewrites ``adapter_model.safetensors`` and ``adapter_config.json``
in place so the resulting directory can be zipped with the two required root
files for Kaggle submission.

Usage::

    python scripts/kg1_huikang_conversion.py \
        --tinker-dir /path/to/tinker_adapter \
        --out-dir /path/to/kaggle_adapter \
        --base-model nvidia/Nemotron-3-Nano-30B-A3B-BF16 \
        --rank 32

Dependencies: ``torch``, ``safetensors``.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants reverse-engineered from huikang cell 10 and cell 24.
# ---------------------------------------------------------------------------

HUIKANG_TARGET_MODULES: List[str] = [
    "k_proj",
    "o_proj",
    "in_proj",
    "q_proj",
    "up_proj",
    "v_proj",
    "down_proj",
    "out_proj",
    "lm_head",
]

FORCED_FUSED_RANK = 32

TINKER_PREFIX = "base_model.model.model"
KAGGLE_PREFIX = "base_model.model.backbone"


# ---------------------------------------------------------------------------
# Safetensors helpers - import torch lazily so the module is importable in
# environments without CUDA (e.g. the repo py_compile check).
# ---------------------------------------------------------------------------


def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError as exc:  # pragma: no cover - only hit in stripped CI
        raise SystemExit(
            "torch is required for huikang conversion. "
            "Install torch>=2.5 first."
        ) from exc


def _load_tensors(path: Path) -> Dict[str, "torch.Tensor"]:  # type: ignore[name-defined]
    from safetensors.torch import load_file  # type: ignore

    return load_file(str(path))


def _save_tensors(tensors: Dict[str, "torch.Tensor"], path: Path) -> None:  # type: ignore[name-defined]
    from safetensors.torch import save_file  # type: ignore

    save_file(tensors, str(path))


# ---------------------------------------------------------------------------
# SVD rank-reduce merge (huikang cell 24, verbatim math).
# ---------------------------------------------------------------------------


def _svd_merge_pair(
    lora_A_list: List["torch.Tensor"],  # type: ignore[name-defined]
    lora_B_list: List["torch.Tensor"],  # type: ignore[name-defined]
    rank: int,
) -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
    """Merge N (B, A) LoRA pairs into one rank-``rank`` pair via QR+SVD.

    Exactly reproduces huikang's ``_compress_lora_pair_to_rank``:

        Q_B, R_B = qr(B_concat)
        Q_A, R_A = qr(A_concat.T)
        core = R_B @ R_A.T
        U, S, V = svd(core, full_matrices=False)
        new_B = (Q_B @ U[:, :rank]) * S[:rank]
        new_A = V[:rank, :] @ Q_A.T
    """
    torch = _import_torch()
    # Concatenate along rank axis (column of B, row of A).
    B_block = torch.cat(lora_B_list, dim=1).float()
    A_cat = torch.cat(lora_A_list, dim=0).float()
    Q_B, R_B = torch.linalg.qr(B_block)
    Q_A, R_A = torch.linalg.qr(A_cat.T)
    core = R_B @ R_A.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    new_B = (Q_B @ U[:, :rank]) * S[:rank].unsqueeze(0)
    new_A = Vh[:rank, :] @ Q_A.T
    return new_B.contiguous(), new_A.contiguous()


# ---------------------------------------------------------------------------
# Key rewriting.
# ---------------------------------------------------------------------------


_RE_EXPERT = re.compile(r"\.experts\.w([123])\b")


@dataclass
class _ExpertBucket:
    """Collects separate (w1, w2) expert tensors before unfusing per-expert."""

    w1_A: "torch.Tensor | None" = None  # type: ignore[name-defined]
    w1_B: "torch.Tensor | None" = None  # type: ignore[name-defined]
    w2_A: "torch.Tensor | None" = None  # type: ignore[name-defined]
    w2_B: "torch.Tensor | None" = None  # type: ignore[name-defined]


def _rename_key(key: str) -> str:
    """Replace the Tinker prefix with the Kaggle backbone prefix."""
    return key.replace(TINKER_PREFIX, KAGGLE_PREFIX)


def _is_fused_proj(key: str) -> bool:
    """gate_proj and x_proj are fused into in_proj (huikang cell 24)."""
    return ".gate_proj." in key or ".x_proj." in key


def _proj_base(key: str) -> str:
    """Return the common prefix of gate_proj and x_proj so they merge."""
    return (
        key.replace(".gate_proj.", ".in_proj.")
        .replace(".x_proj.", ".in_proj.")
    )


# ---------------------------------------------------------------------------
# Main conversion.
# ---------------------------------------------------------------------------


def convert(
    tinker_dir: Path,
    out_dir: Path,
    base_model: str,
    rank: int = FORCED_FUSED_RANK,
    tensor_file: str = "adapter_model.safetensors",
) -> Path:
    """Run the full Tinker -> Kaggle conversion.

    Returns the path to the rewritten adapter directory. The output has two
    root files (``adapter_config.json``, ``adapter_model.safetensors``) ready
    to be zipped via ``zip -m submission.zip *``.
    """
    torch = _import_torch()

    tinker_dir = Path(tinker_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_tensors = tinker_dir / tensor_file
    src_config = tinker_dir / "adapter_config.json"
    if not src_tensors.exists():
        raise FileNotFoundError(src_tensors)
    if not src_config.exists():
        raise FileNotFoundError(src_config)

    tensors = _load_tensors(src_tensors)
    new_tensors: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    fused_buckets: Dict[str, Dict[str, List["torch.Tensor"]]] = {}  # type: ignore[name-defined]
    expert_buckets: Dict[str, _ExpertBucket] = {}

    for key, value in tensors.items():
        # Drop empty Tinker tensors such as ``.experts.w3`` sentinels
        # (present in some Tinker runs, always zero-size).
        if value.numel() == 0:
            continue

        renamed = _rename_key(key)
        base = renamed.rsplit(".lora_", 1)[0]

        # Route MoE experts to per-expert buckets.
        if ".experts.w1" in base or ".experts.w2" in base:
            exp_key = base.replace(".w1", "").replace(".w2", "")
            bucket = expert_buckets.setdefault(exp_key, _ExpertBucket())
            suffix = renamed.rsplit(".lora_", 1)[1]  # "A.weight" or "B.weight"
            role = suffix.split(".")[0]  # "A" | "B"
            which = "w1" if ".experts.w1" in base else "w2"
            setattr(bucket, f"{which}_{role}", value)
            continue

        # Route fused gate_proj/x_proj to in_proj buckets.
        if _is_fused_proj(base):
            merged_base = _proj_base(base)
            bucket_entry = fused_buckets.setdefault(
                merged_base, {"A": [], "B": []}
            )
            suffix = renamed.rsplit(".lora_", 1)[1]
            role = suffix.split(".")[0]
            bucket_entry[role].append(value)
            continue

        new_tensors[renamed] = value.contiguous()

    # 1. Merge fused projections via SVD down to `rank`.
    for merged_base, bucket in fused_buckets.items():
        if not (bucket["A"] and bucket["B"]):
            continue
        new_B, new_A = _svd_merge_pair(bucket["A"], bucket["B"], rank)
        new_tensors[f"{merged_base}.lora_A.weight"] = new_A
        new_tensors[f"{merged_base}.lora_B.weight"] = new_B

    # 2. Unfuse MoE experts -> per-expert up_proj / down_proj.
    for exp_base, bucket in expert_buckets.items():
        for which, proj_name in (("w1", "up_proj"), ("w2", "down_proj")):
            A = getattr(bucket, f"{which}_A")
            B = getattr(bucket, f"{which}_B")
            if A is None or B is None:
                continue
            if A.shape[0] == 1:
                A = A.expand(B.shape[0], -1, -1).contiguous()
            elif B.shape[0] == 1:
                B = B.expand(A.shape[0], -1, -1).contiguous()
            num_experts = A.shape[0]
            for i in range(num_experts):
                # Rewrite .experts.wX -> .experts.{i}.{proj_name}
                renamed = re.sub(
                    r"\.experts$",
                    f".experts.{i}.{proj_name}",
                    exp_base,
                )
                new_tensors[f"{renamed}.lora_A.weight"] = A[i].contiguous()
                new_tensors[f"{renamed}.lora_B.weight"] = B[i].contiguous()

    # 3. Write the new safetensors file.
    out_tensors = out_dir / "adapter_model.safetensors"
    _save_tensors(new_tensors, out_tensors)

    # 4. Rewrite adapter_config.json:
    #    - swap target_modules to huikang's 9-module list
    #    - set inference_mode=True, lora_dropout=0.0
    #    - update base_model_name_or_path
    cfg = json.loads(src_config.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = base_model
    cfg["target_modules"] = HUIKANG_TARGET_MODULES
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    cfg.setdefault("r", rank)
    cfg.setdefault("lora_alpha", rank)  # alpha = rank (not 2x)
    (out_dir / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    # 5. Copy any auxiliary files the Tinker directory shipped (README,
    # README.md, chat_template.jinja) so submissions stay self-documenting.
    for extra in ("README.md", "chat_template.jinja", "tokenizer_config.json"):
        src = tinker_dir / extra
        if src.exists():
            shutil.copy2(src, out_dir / extra)

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinker-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--base-model",
        default="nvidia/Nemotron-3-Nano-30B-A3B-BF16",
    )
    parser.add_argument("--rank", type=int, default=FORCED_FUSED_RANK)
    parser.add_argument(
        "--tensor-file",
        default="adapter_model.safetensors",
        help="Name of the safetensors file inside --tinker-dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = convert(
        args.tinker_dir,
        args.out_dir,
        base_model=args.base_model,
        rank=args.rank,
        tensor_file=args.tensor_file,
    )
    print(f"Converted Tinker adapter -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
