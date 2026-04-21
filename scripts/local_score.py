"""Score local pre-submit Kaggle (validation 99% rule).

Valida adapter LoRA contra train.csv hold-out antes de queimar 1/5 submits.

Usage:
    python scripts/local_score.py \\
        --adapter felipesp1983/kg1-nemotron-lora-v73-definitive \\
        --n-samples 200 \\
        --output-csv local_eval.csv

Pre-requisitos:
    pip install vllm>=0.6 pandas huggingface_hub
    Kaggle creds em ~/.kaggle/kaggle.json (download train.csv)

Retorna:
    - Score local por categoria
    - Breakdown acerto/erro
    - Gate decision: GO (score >= target) / NO-GO (regrediu)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

import math

# Metric oficial Kaggle (reverse-engineered via Agent D1, 2026-04-21):
# - Regex aceita \boxed{X} ou \boxed{X sem fechamento (EOF)
# - Ordem: boxed -> "final answer is X" -> ultimo numero -> ultima linha -> NOT_FOUND
# - verify: float + math.isclose rel_tol=1e-2 abs_tol=1e-5, fallback lower-case string
# - NAO existe branch por categoria (bit_manipulation nao tem strict [01]+ match)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")

FINAL_ANSWER_RE = re.compile(
    r"(?:[Tt]he\s+)?[Ff]inal\s+answer\s*(?:is)?\s*[:：]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Alinhado com kg1_submission_gate.py
REQUIRED_IN_TARGET = "in_proj"
FORBIDDEN_TARGETS = {"gate_proj", "x_proj"}
MAX_LORA_RANK = 32


def extract_boxed(text: str) -> str | None:
    """Extract final answer seguindo fallback EXATO do metric oficial Kaggle.

    Ordem de fallback (Agent D1 reverse-engineered):
    1. `\\boxed{X}` ou `\\boxed{X` (sem fechamento). Ultimo match nao-vazio.
    2. "The final answer is: X" variants (case-insensitive, fullwidth colon OK).
    3. Ultimo numero (regex -?\\d+(?:\\.\\d+)?).
    4. Ultima linha nao-vazia.
    5. None (equivalente a "NOT_FOUND" do metric oficial).
    """
    if not isinstance(text, str):
        return None
    # Step 1: \boxed{} com suporte a EOF sem fechamento
    matches = BOXED_RE.findall(text)
    if matches:
        # Ultimo nao-vazio (vazios apagam resposta anterior)
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        # Todos vazios: retorna ultimo stripped
        return matches[-1].strip()

    # Step 2: "final answer is: X"
    fa_matches = FINAL_ANSWER_RE.findall(text)
    if fa_matches:
        return fa_matches[-1].strip()

    # Step 3: ultimo numero
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]

    # Step 4: ultima linha nao-vazia
    for line in reversed(text.split("\n")):
        if line.strip():
            return line.strip()
    return None


def verify(answer: str, predicted: str | None) -> bool:
    """Verify predicted matches ground truth (metric OFICIAL Kaggle reproducido).

    Fonte: kaggle_research/extracted_kernel_code/huikang__*.txt L484-502.
    Funcao UNICA para TODAS 9 categorias (sem branch por categoria).
    """
    if not predicted:
        return False
    a = str(answer).strip()
    p = str(predicted).strip()
    # PRIMEIRA tentativa: parse float em ambos. Se ambos numeros, tolerancia 1%.
    # Isso aceita "01011" == "1011" (bit_manipulation sem leading-zero strict).
    try:
        a_num = float(a)
        p_num = float(p)
        return math.isclose(a_num, p_num, rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        # Fallback: string lower-case exato (case-insensitive).
        return p.lower() == a.lower()


def validate_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    """Validate adapter_config.json alinhado com kg1_submission_gate.py."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")

    cfg = json.loads(cfg_path.read_text())
    target_modules = cfg.get("target_modules", [])
    rank = cfg.get("r", cfg.get("lora_rank", 0))

    errors: list[str] = []
    if not isinstance(target_modules, list) or not target_modules:
        errors.append("target_modules invalid")
    else:
        if REQUIRED_IN_TARGET not in target_modules:
            errors.append(f"missing {REQUIRED_IN_TARGET}")
        for forbidden in FORBIDDEN_TARGETS:
            if forbidden in target_modules:
                errors.append(f"forbidden {forbidden} present")

    if rank > MAX_LORA_RANK:
        errors.append(f"rank {rank} > {MAX_LORA_RANK}")

    return {
        "valid": not errors,
        "errors": errors,
        "rank": rank,
        "target_modules": target_modules,
    }


def download_adapter_hf(repo_id: str, hf_token: str, local_dir: Path) -> Path:
    """Download adapter files from HuggingFace."""
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id}...")
    adapter_path = snapshot_download(
        repo_id=repo_id,
        token=hf_token,
        allow_patterns=["final/*", "*.json", "*.safetensors"],
        local_dir=str(local_dir),
    )
    # Find the actual adapter dir (usually {local_dir}/final)
    for candidate in [Path(adapter_path) / "final", Path(adapter_path)]:
        if (candidate / "adapter_config.json").exists():
            return candidate
    raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")


def ensure_train_csv(train_csv: Path) -> Path:
    """Download train.csv se nao existir (kaggle API)."""
    if train_csv.exists():
        return train_csv

    print("Downloading train.csv via kaggle API...")
    os.makedirs(train_csv.parent, exist_ok=True)
    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "nvidia-nemotron-model-reasoning-challenge",
            "-f", "train.csv",
            "-p", str(train_csv.parent),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")

    # Unzip
    zip_path = train_csv.parent / "train.csv.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(train_csv.parent)
        zip_path.unlink()

    assert train_csv.exists(), f"train.csv not found after download"
    return train_csv


def run_inference_vllm(
    adapter_dir: Path,
    base_model: str,
    prompts: list[str],
    max_tokens: int = 7680,
) -> list[str]:
    """Run vLLM inference with adapter."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=base_model,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        dtype="auto",
        max_model_len=8192,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_loras=1,
    )

    tokenizer = llm.get_tokenizer()
    # BOXED_INSTRUCTION alinhado com metric Kaggle oficial (Agent D1)
    BOXED_INSTRUCTION = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p + BOXED_INSTRUCTION}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # OBRIGATORIO (bug corrigido 2026-04-21)
        )
        for p in prompts
    ]

    # Kaggle metric oficial params
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        n=1,
    )
    lora_req = LoRARequest("v73", 1, str(adapter_dir))

    outputs = llm.generate(formatted, sp, lora_request=lora_req)
    return [o.outputs[0].text for o in outputs]


def score_predictions(df: pd.DataFrame, predictions: list[str]) -> dict[str, Any]:
    """Score predictions against df.answer with category breakdown."""
    df = df.copy()
    df["predicted_text"] = predictions
    df["predicted"] = df["predicted_text"].apply(extract_boxed)
    df["correct"] = df.apply(lambda r: verify(str(r.get("answer", "")), r["predicted"]), axis=1)

    overall = float(df["correct"].mean())

    per_category: dict[str, dict[str, float | int]] = {}
    if "category" in df.columns:
        for cat, grp in df.groupby("category"):
            per_category[str(cat)] = {
                "count": int(len(grp)),
                "correct": int(grp["correct"].sum()),
                "accuracy": float(grp["correct"].mean()),
            }

    return {
        "overall_score": overall,
        "n_samples": int(len(df)),
        "n_correct": int(df["correct"].sum()),
        "per_category": per_category,
        "df": df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local score pre-submit")
    parser.add_argument("--adapter", required=True, help="HF repo_id OR local dir path")
    parser.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--train-csv", default="./data/train.csv", type=Path)
    parser.add_argument("--n-samples", type=int, default=200, help="hold-out eval size")
    parser.add_argument("--seed", type=int, default=99, help="hold-out seed (NOT 42 = different from training)")
    parser.add_argument("--output-csv", default="local_eval.csv", type=Path)
    parser.add_argument("--target-score", type=float, default=0.84, help="min score to pass gate")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("WARN: HF_TOKEN env var not set. Download may fail for private repos.", file=sys.stderr)

    # 1. Adapter (HF ou local)
    if Path(args.adapter).exists():
        adapter_dir = Path(args.adapter)
        print(f"Using local adapter: {adapter_dir}")
    else:
        adapter_dir = download_adapter_hf(args.adapter, hf_token, Path("./adapters_cache"))
        print(f"Downloaded to: {adapter_dir}")

    # 2. Validate gate
    print("\n=== VALIDATE GATE ===")
    gate = validate_adapter_config(adapter_dir)
    print(f"target_modules: {gate['target_modules']}")
    print(f"rank: {gate['rank']}")
    if not gate["valid"]:
        print(f"!!! GATE FAIL: {gate['errors']}")
        sys.exit(2)
    print("[OK] Gate passed")

    # 3. Train.csv
    train_csv = ensure_train_csv(args.train_csv)

    # 4. Hold-out sample (seed=99 NOT 42 = different from kienngx training samples)
    df_full = pd.read_csv(train_csv)
    prompt_col = "prompt" if "prompt" in df_full.columns else "problem"
    answer_col = "answer" if "answer" in df_full.columns else "solution"

    df_eval = df_full.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
    print(f"\n=== EVAL on {len(df_eval)} samples (seed={args.seed}) ===")

    # 5. Inference
    predictions = run_inference_vllm(
        adapter_dir,
        args.base_model,
        df_eval[prompt_col].tolist(),
    )

    # 6. Score
    result = score_predictions(df_eval.rename(columns={answer_col: "answer"}), predictions)

    print(f"\n=== RESULTS ===")
    print(f"Overall score: {result['overall_score']:.4f}")
    print(f"Correct: {result['n_correct']}/{result['n_samples']}")
    if result["per_category"]:
        print(f"\nPer category:")
        for cat, stats in sorted(result["per_category"].items(), key=lambda x: -x[1]["accuracy"]):
            print(f"  {cat}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['count']})")

    result["df"].to_csv(args.output_csv, index=False)
    print(f"\nDetailed eval: {args.output_csv}")

    # 7. Gate decision
    print(f"\n=== GATE DECISION ===")
    if result["overall_score"] >= args.target_score:
        print(f"[GO] {result['overall_score']:.4f} >= {args.target_score}")
        print(f"Submit to Kaggle with confidence.")
        sys.exit(0)
    else:
        print(f"[NO-GO] {result['overall_score']:.4f} < {args.target_score}")
        print(f"DO NOT submit - investigate config before burning submit slot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
