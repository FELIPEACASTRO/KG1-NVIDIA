"""Data pipeline v74: multi-source + curriculum ordering + dedup.

Combina:
- kienngx baseline (1200 train.csv seed=42)
- huikang.dev rule_found (~4800 high-quality, 87.7% public)
- ritwikakancharla nemotron-math-v2-filtered-high (HIGH reasoning AoPS)
- konbu17 solver_cot_8100 (deterministic solver traces)

Output: HF dataset felipesp1983/kg1-v74-training com curriculum_order.json

Usage:
    python scripts/prepare_v74_dataset.py \\
        --output-dir data/v74 \\
        --upload-hf \\
        --target-size 10000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def deduplicate_by_prompt(df: pd.DataFrame, prompt_col: str) -> pd.DataFrame:
    """Deduplicate via hash do primeiro 500 chars do prompt."""
    df = df.copy()
    df["_hash"] = df[prompt_col].astype(str).str[:500].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )
    before = len(df)
    df = df.drop_duplicates(subset="_hash").drop(columns=["_hash"])
    after = len(df)
    print(f"  Dedup: {before} -> {after} ({before - after} dups removed)")
    return df


def validate_example(row: dict[str, Any], prompt_col: str, answer_col: str) -> bool:
    """Sanity check de um example."""
    prompt = row.get(prompt_col, "")
    answer = row.get(answer_col, "")
    if not isinstance(prompt, str) or len(prompt) < 20:
        return False
    if not isinstance(answer, str) or len(answer) < 1:
        return False
    return True


def load_kienngx_baseline(train_csv: Path, n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Kienngx baseline: train.csv sample."""
    print(f"\n=== kienngx baseline (train.csv) ===")
    df = pd.read_csv(train_csv)
    print(f"  Full: {len(df)} rows | cols: {list(df.columns)}")
    sub = df.sample(n=n, random_state=seed).reset_index(drop=True)
    sub["source"] = "kienngx_baseline"
    sub["difficulty"] = 0.5  # unknown, medium
    print(f"  Sampled: {len(sub)}")
    return sub


def load_huikang_rule_found(n_max: int = 4800, hf_token: str | None = None) -> pd.DataFrame:
    """Load huikang rule_found examples from huikang.dev + nvidia train.csv."""
    print(f"\n=== huikang rule_found (high-quality via huikang.dev) ===")
    import urllib.request

    # Download generation.jsonl
    gen_url = "https://nemotron.huikang.dev/generation.jsonl"
    prob_url = "https://nemotron.huikang.dev/problems.jsonl"

    print(f"  Downloading {prob_url}...")
    probs_by_id = {}
    with urllib.request.urlopen(prob_url, timeout=60) as r:
        for line in r.read().decode().strip().split("\n"):
            if not line: continue
            d = json.loads(line)
            probs_by_id[d["id"]] = d

    # Filter rule_found only (87.7% total = ~8333)
    rule_found_ids = [pid for pid, d in probs_by_id.items() if d.get("status") == "rule_found"]
    print(f"  rule_found available: {len(rule_found_ids)}")

    # Load difficulty from generation.jsonl (any_correct signal)
    print(f"  Downloading {gen_url}...")
    difficulty_map: dict[str, float] = {}
    with urllib.request.urlopen(gen_url, timeout=60) as r:
        for line in r.read().decode().strip().split("\n"):
            if not line: continue
            d = json.loads(line)
            pid = d["id"]
            if d.get("any_correct"):
                difficulty_map[pid] = 0.0  # easy
            elif d.get("runs"):
                avg_lp = d["runs"][0].get("avg_logprob", -10)
                difficulty_map[pid] = max(0.1, min(1.0, 1.0 + abs(avg_lp) / 10))
            else:
                difficulty_map[pid] = 1.5  # hardest

    # Join com train.csv para pegar prompt/answer reais
    import pandas as pd
    train_csv = Path("data/train.csv")
    if not train_csv.exists():
        print(f"  WARN: train.csv not at {train_csv}, skipping huikang")
        return pd.DataFrame()

    df_train = pd.read_csv(train_csv)
    df_huikang = df_train[df_train["id"].isin(set(rule_found_ids))].copy()
    df_huikang["difficulty"] = df_huikang["id"].map(lambda x: difficulty_map.get(x, 0.5))
    df_huikang["source"] = "huikang_rule_found"

    # Sort easiest first
    df_huikang = df_huikang.sort_values("difficulty").reset_index(drop=True)
    df_huikang = df_huikang.head(n_max)
    print(f"  Loaded: {len(df_huikang)}")
    return df_huikang


def load_ritwikakancharla(n_max: int = 2000, hf_token: str | None = None) -> pd.DataFrame:
    """Load ritwikakancharla nemotron-math-v2-filtered-high."""
    print(f"\n=== ritwikakancharla nemotron-math-v2-filtered-high ===")
    try:
        ds = load_dataset(
            "ritwikakancharla/nemotron-math-v2-filtered-high",
            split="train",
            streaming=True,  # lazy load
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= n_max:
                break
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"  Columns: {list(df.columns)[:10]}")
        print(f"  Loaded: {len(df)}")

        # Normalize schema to match train.csv (prompt, answer)
        if "problem" in df.columns and "prompt" not in df.columns:
            df["prompt"] = df["problem"]
        if "solution" in df.columns and "answer" not in df.columns:
            df["answer"] = df["solution"]
        elif "messages" in df.columns:
            # Extract from messages
            def extract_ans(msgs):
                try:
                    for m in reversed(msgs):
                        if m.get("role") == "assistant":
                            return m.get("content", "")
                except Exception:
                    pass
                return ""
            df["answer"] = df["messages"].apply(extract_ans)
            df["prompt"] = df["messages"].apply(
                lambda m: next((x.get("content", "") for x in m if x.get("role") == "user"), "")
            )

        df["source"] = "ritwikakancharla_math_high"
        df["difficulty"] = 0.7  # reasoning HIGH = harder
        df["category"] = "math_reasoning"
        df["id"] = df.apply(lambda r: hashlib.md5(str(r.get("prompt", "")[:100]).encode()).hexdigest()[:16], axis=1)

        # Keep only cols we need
        keep = ["id", "prompt", "answer", "category", "source", "difficulty"]
        return df[[c for c in keep if c in df.columns]]

    except Exception as e:
        print(f"  WARN: failed to load ritwikakancharla ({e}), skipping")
        return pd.DataFrame()


def load_konbu17_solver(n_max: int = 1500, hf_token: str | None = None) -> pd.DataFrame:
    """Load konbu17/nemotron-solver-cot-8100 (brand new solver CoTs)."""
    print(f"\n=== konbu17/nemotron-solver-cot-8100 ===")
    try:
        ds = load_dataset("konbu17/nemotron-solver-cot-8100", split="train")
        df = ds.to_pandas()
        print(f"  Columns: {list(df.columns)[:10]}")
        print(f"  Loaded: {len(df)}")

        # Normalize
        if "problem" in df.columns and "prompt" not in df.columns:
            df["prompt"] = df["problem"]
        if "solution" in df.columns and "answer" not in df.columns:
            df["answer"] = df["solution"]

        df["source"] = "konbu17_solver_cot"
        df["difficulty"] = 0.3  # solver verified = easier
        if "category" not in df.columns:
            df["category"] = "solver_cot"
        if "id" not in df.columns:
            df["id"] = df.apply(lambda r: hashlib.md5(str(r.get("prompt", "")[:100]).encode()).hexdigest()[:16], axis=1)

        df = df.head(n_max)

        keep = ["id", "prompt", "answer", "category", "source", "difficulty"]
        return df[[c for c in keep if c in df.columns]]
    except Exception as e:
        print(f"  WARN: failed to load konbu17 ({e}), skipping")
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v74 multi-source dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/v74"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--upload-hf", action="store_true", help="Upload to felipesp1983/kg1-v74-training")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", "")

    # Load all sources
    kienngx_df = load_kienngx_baseline(args.train_csv, n=1200, seed=42)
    huikang_df = load_huikang_rule_found(n_max=4800, hf_token=hf_token)
    ritwika_df = load_ritwikakancharla(n_max=2000, hf_token=hf_token)
    konbu_df = load_konbu17_solver(n_max=1500, hf_token=hf_token)

    # Concatenate
    all_dfs = [df for df in [kienngx_df, huikang_df, ritwika_df, konbu_df] if not df.empty]
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n=== COMBINED ===")
    print(f"  Total: {len(combined)} rows")
    print(f"  Sources: {combined['source'].value_counts().to_dict()}")

    # Deduplicate
    print(f"\n=== DEDUPLICATE ===")
    combined = deduplicate_by_prompt(combined, "prompt")

    # Validate
    print(f"\n=== VALIDATE ===")
    combined["_valid"] = combined.apply(
        lambda r: validate_example(r.to_dict(), "prompt", "answer"), axis=1
    )
    before = len(combined)
    combined = combined[combined["_valid"]].drop(columns=["_valid"])
    print(f"  Valid: {len(combined)}/{before}")

    # Cap target size
    if len(combined) > args.target_size:
        print(f"\n=== CAP to {args.target_size} ===")
        # Preserve ratio across sources via stratified sample
        combined = combined.groupby("source").apply(
            lambda g: g.sample(n=min(len(g), int(args.target_size * len(g) / len(combined))), random_state=42)
        ).reset_index(drop=True)
        print(f"  Final: {len(combined)}")

    # Curriculum ordering
    print(f"\n=== CURRICULUM (easy -> hard) ===")
    combined = combined.sort_values("difficulty").reset_index(drop=True)
    combined["curriculum_idx"] = range(len(combined))
    print(f"  Difficulty range: {combined['difficulty'].min():.2f} - {combined['difficulty'].max():.2f}")

    # Save
    csv_path = args.output_dir / "v74_curated.csv"
    parquet_path = args.output_dir / "v74_curated.parquet"
    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    # Save curriculum order
    curriculum_ids = combined.sort_values("curriculum_idx")["id"].tolist()
    (args.output_dir / "curriculum_order.json").write_text(json.dumps(curriculum_ids))

    print(f"\n=== OUTPUT ===")
    print(f"  CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Parquet: {parquet_path}")
    print(f"  Curriculum: {len(curriculum_ids)} IDs ordered")

    # Upload HF
    if args.upload_hf:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        REPO = "felipesp1983/kg1-v74-training"
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(args.output_dir), repo_id=REPO, repo_type="dataset")
        print(f"\n[OK] Uploaded to HF: https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
