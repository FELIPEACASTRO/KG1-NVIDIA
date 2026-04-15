"""Automated Kaggle submission com validation gate.

Fluxo:
1. Download adapter from HuggingFace
2. Validate submission gate (target_modules, rank)
3. Generate submission.zip (apenas 2 files)
4. Check Kaggle submit slots disponiveis
5. Submit to competition

Usage:
    python scripts/submit_kaggle.py \\
        --hf-repo felipesp1983/kg1-nemotron-lora-v73-definitive \\
        --message "v73 kienngx replica loss 1.02" \\
        --max-daily-submits 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Alinhado com kg1_submission_gate.py
REQUIRED_TARGET = "in_proj"
FORBIDDEN_TARGETS = {"gate_proj", "x_proj"}
MAX_LORA_RANK = 32
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"


def check_kaggle_creds() -> None:
    """Verify kaggle creds configured."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print(f"!!! Kaggle creds not found at {cred_path}")
        print("Setup: export KAGGLE_USERNAME=... KAGGLE_KEY=...")
        print("   OR: save kaggle.json to ~/.kaggle/kaggle.json")
        sys.exit(1)


def count_submits_today() -> int:
    """Count submissions made today (Kaggle has 5/day limit)."""
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARN: Could not check submits: {result.stderr}", file=sys.stderr)
        return 0

    import csv
    import io
    reader = csv.DictReader(io.StringIO(result.stdout))
    today = datetime.now().date()
    count = 0
    for row in reader:
        date_str = row.get("date", "")
        try:
            submit_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            if submit_date == today:
                count += 1
        except ValueError:
            continue
    return count


def download_adapter_from_hf(repo_id: str, target_dir: Path) -> Path:
    """Download adapter from HF hub."""
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"Downloading {repo_id}...")
    local = snapshot_download(
        repo_id=repo_id,
        token=hf_token,
        allow_patterns=["final/*", "*.json", "*.safetensors"],
        local_dir=str(target_dir),
    )

    # Find adapter dir
    for candidate in [Path(local) / "final", Path(local)]:
        if (candidate / "adapter_config.json").exists():
            print(f"[OK] Adapter at: {candidate}")
            return candidate
    raise FileNotFoundError(f"adapter_config.json not in {local}")


def validate_gate(adapter_dir: Path) -> bool:
    """Validate adapter passes Kaggle submission gate."""
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())

    target_modules = cfg.get("target_modules", [])
    rank = cfg.get("r", cfg.get("lora_rank", 0))

    print(f"target_modules: {target_modules}")
    print(f"rank: {rank}")

    errors = []
    if not isinstance(target_modules, list) or not target_modules:
        errors.append("target_modules invalid")
    else:
        if REQUIRED_TARGET not in target_modules:
            errors.append(f"missing {REQUIRED_TARGET}")
        for fb in FORBIDDEN_TARGETS:
            if fb in target_modules:
                errors.append(f"forbidden {fb}")

    if rank > MAX_LORA_RANK:
        errors.append(f"rank {rank} > {MAX_LORA_RANK}")

    if errors:
        print(f"!!! GATE FAIL: {errors}")
        return False

    print("[OK] Gate passed")
    return True


def generate_submission_zip(adapter_dir: Path, output_zip: Path) -> Path:
    """Generate submission.zip (formato oficial Kaggle)."""
    REQUIRED_FILES = ["adapter_config.json", "adapter_model.safetensors"]

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in REQUIRED_FILES:
            src = adapter_dir / fname
            if not src.exists():
                raise FileNotFoundError(f"Required file missing: {src}")
            zf.write(src, arcname=fname)
            print(f"  added: {fname} ({src.stat().st_size / 1e6:.1f} MB)")

    print(f"[OK] submission.zip: {output_zip} ({output_zip.stat().st_size / 1e6:.1f} MB)")
    return output_zip


def submit_kaggle(submission_zip: Path, message: str) -> None:
    """Submit to Kaggle competition."""
    result = subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", COMPETITION,
            "-f", str(submission_zip),
            "-m", message,
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"!!! Submit failed: {result.stderr}")
        sys.exit(1)
    print("[OK] Submitted to Kaggle")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Kaggle submission")
    parser.add_argument("--hf-repo", help="HF repo_id (ou --local-dir)")
    parser.add_argument("--local-dir", type=Path, help="Local adapter dir (alt to --hf-repo)")
    parser.add_argument("--message", required=True, help="Submit message")
    parser.add_argument("--max-daily-submits", type=int, default=5, help="Kaggle 5/day limit")
    parser.add_argument("--skip-slot-check", action="store_true", help="Skip 5/day check")
    parser.add_argument("--dry-run", action="store_true", help="No real submit")
    args = parser.parse_args()

    if not args.hf_repo and not args.local_dir:
        print("Need --hf-repo OR --local-dir")
        sys.exit(1)

    # 1. Kaggle creds
    check_kaggle_creds()

    # 2. Slot check
    if not args.skip_slot_check:
        n_today = count_submits_today()
        remaining = args.max_daily_submits - n_today
        print(f"Today submits: {n_today}/{args.max_daily_submits} (remaining: {remaining})")
        if remaining <= 0:
            print("!!! No submits remaining today. Wait for 00:00 UTC reset (21:00 BRT).")
            sys.exit(1)

    # 3. Get adapter
    if args.local_dir:
        adapter_dir = args.local_dir
    else:
        adapter_dir = download_adapter_from_hf(args.hf_repo, Path("./adapters_cache"))

    # 4. Validate gate
    print("\n=== VALIDATE GATE ===")
    if not validate_gate(adapter_dir):
        print("Aborting - gate failed")
        sys.exit(2)

    # 5. Generate zip
    print("\n=== GENERATE ZIP ===")
    submission_zip = Path(f"submission_{datetime.now():%Y%m%d_%H%M%S}.zip")
    generate_submission_zip(adapter_dir, submission_zip)

    # 6. Submit
    print("\n=== SUBMIT ===")
    if args.dry_run:
        print(f"[DRY RUN] Would submit: {submission_zip}")
        print(f"  Message: {args.message}")
    else:
        submit_kaggle(submission_zip, args.message)
        print(f"\nMonitor at: https://www.kaggle.com/competitions/{COMPETITION}/submissions")


if __name__ == "__main__":
    main()
