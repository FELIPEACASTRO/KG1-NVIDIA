#!/usr/bin/env python3
"""Download NVIDIA's official Nemotron-RL-ReasoningGym-v1 dataset.

Source: huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1 (2026-03-10)
Categories: algebra, arithmetic, computation, cognitive reasoning

USP: released 10 days BEFORE competition opened. Community hasn't tried
pretrain warm-up on it. Novel approach.

Usage as pretraining warm-up BEFORE Tong's SFT to improve base model
reasoning signal on problem structures similar to competition data.
"""
import argparse
import os
from pathlib import Path


def download(output_dir: str, split: str = "train", limit: int = None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # Try loading with various subset names (we don't know exact config)
    candidates = [
        "nvidia/Nemotron-RL-ReasoningGym-v1",
        "nvidia/Nemotron-RL-ReasoningGym",  # fallback
    ]

    for repo in candidates:
        try:
            ds = load_dataset(repo, split=split)
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            print(f"Loaded {len(ds)} examples from {repo}")
            print(f"Columns: {ds.column_names}")
            print(f"First example: {ds[0] if len(ds) else 'empty'}")

            # Save as JSONL
            out_path = os.path.join(output_dir, f"reasoning_gym_{split}.jsonl")
            ds.to_json(out_path, lines=True, orient="records")
            print(f"Saved: {out_path}")

            # Save summary
            summary_path = os.path.join(output_dir, "summary.json")
            import json
            with open(summary_path, "w") as f:
                json.dump({
                    "repo": repo, "split": split, "count": len(ds),
                    "columns": ds.column_names,
                    "first_example": ds[0] if len(ds) > 0 else {},
                }, f, indent=2, default=str)
            return out_path
        except Exception as e:
            print(f"  Failed {repo}: {str(e)[:200]}")

    print("ERROR: no dataset variant loaded. Check HF repo name manually.")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/nemotron_reasoning_gym")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None, help="Cap rows (None=all)")
    args = p.parse_args()
    download(args.output_dir, args.split, args.limit)


if __name__ == "__main__":
    main()
