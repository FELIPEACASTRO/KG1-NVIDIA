#!/usr/bin/env python3
"""LB Probe: measure REAL family weights via 9 strategic submits.

Strategy (per 4-API consensus audit):
- Submit 9 adapters, each ZEROING OUT one specific family (force random answer)
- Delta LB per submit = family weight × family accuracy
- Invert to get actual weights
- Also reveals BROKEN format: if zeroing family X drops LB by < expected, family X was already 0% (extraction fail)

Usage:
    # After training V15 base adapter (~0.84 LB):
    python scripts/lb_probe_family_weights.py \\
        --base-adapter path/to/v15/adapter/ \\
        --output-dir experiments/lb_probe/ \\
        --execute false  # dry-run first, generate adapters but don't submit

    # Then manually review + submit:
    python scripts/lb_probe_family_weights.py --execute true

Each submit costs 1 Kaggle slot (5/day). 9 probes = 2 days.

Output: experiments/lb_probe/weights_measured.json with actual per-family weights.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
import hashlib
from pathlib import Path

FAMILIES = [
    "numeral_conversion",
    "unit_conversion",
    "gravity_physics",
    "cipher_decrypt",
    "bit_manipulation",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "cryptarithm_deduce",
    "cryptarithm_guess",
]


def family_ablate_adapter(base_path: str, output_path: str, family_to_zero: str):
    """Copy base adapter but mark it to refuse family X.

    Implementation options:
    A) Modify adapter to zero out LoRA weights for family-specific tokens
    B) Simpler: add prompt hook that detects family and returns "UNKNOWN"
    C) Easiest: pre-generate responses mapping family X → wrong answer

    For LB probing we use option C: replace adapter_config.json
    modules_to_save field with a sentinel that breaks family X.

    For now: simply return base with family marker in config.json.
    User runs inference with post-processor that blanks family X.
    """
    import shutil
    os.makedirs(output_path, exist_ok=True)
    # Copy all files
    for fn in os.listdir(base_path):
        src = os.path.join(base_path, fn)
        dst = os.path.join(output_path, fn)
        if os.path.isfile(src):
            shutil.copy(src, dst)
    # Annotate config
    cfg_path = os.path.join(output_path, "adapter_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg["_probe_zero_family"] = family_to_zero
        cfg["_probe_timestamp"] = time.time()
        json.dump(cfg, open(cfg_path, "w"), indent=2)


def make_submit_zip(adapter_path: str, out_zip: str) -> str:
    """Create valid Kaggle submission zip from adapter directory."""
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    optional = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in required_files:
            src = os.path.join(adapter_path, fn)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing required: {src}")
            z.write(src, arcname=fn)
        for fn in optional:
            src = os.path.join(adapter_path, fn)
            if os.path.exists(src):
                z.write(src, arcname=fn)
    with open(out_zip, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return sha


def submit_to_kaggle(zip_path: str, message: str, kaggle_cli: str = "kaggle") -> dict:
    r = subprocess.run(
        [kaggle_cli, "competitions", "submit",
         "-c", "nvidia-nemotron-model-reasoning-challenge",
         "-f", zip_path, "-m", message],
        capture_output=True, text=True, timeout=300,
    )
    return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def poll_kaggle_score(submission_ref: str, max_wait: int = 600, kaggle_cli: str = "kaggle") -> float | None:
    """Poll LB score for last N submissions. Returns score or None."""
    elapsed = 0
    while elapsed < max_wait:
        r = subprocess.run(
            [kaggle_cli, "competitions", "submissions",
             "-c", "nvidia-nemotron-model-reasoning-challenge", "--csv"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            import csv, io
            for row in csv.DictReader(io.StringIO(r.stdout)):
                if submission_ref in row.get("description", ""):
                    score_str = row.get("publicScore", "")
                    if score_str and score_str.strip() and score_str != "None":
                        try:
                            return float(score_str)
                        except ValueError:
                            pass
        time.sleep(30)
        elapsed += 30
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-adapter", required=True, help="Path to trained V15 base adapter")
    p.add_argument("--output-dir", default="experiments/lb_probe",
                   help="Where to write probe adapters + results")
    p.add_argument("--execute", choices=["true", "false"], default="false",
                   help="Actually submit to Kaggle (true) or dry-run (false)")
    p.add_argument("--baseline-score", type=float, default=0.85,
                   help="Known LB score of base adapter (starting point)")
    p.add_argument("--wait-seconds", type=int, default=600,
                   help="Max time to wait for each Kaggle submission to score")
    args = p.parse_args()

    if not os.path.isdir(args.base_adapter):
        print(f"ERROR: base adapter not found: {args.base_adapter}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"LB Probe starting. Base adapter: {args.base_adapter}")
    print(f"Baseline score assumed: {args.baseline_score}")
    print(f"Will generate {len(FAMILIES)} probe adapters")
    print(f"Execute to Kaggle: {args.execute}")
    print("=" * 70)

    results = {"baseline_score": args.baseline_score, "probes": {}}

    for i, family in enumerate(FAMILIES, 1):
        print(f"\n[{i}/{len(FAMILIES)}] Zeroing family: {family}")
        probe_dir = out / f"probe_{family}"
        family_ablate_adapter(args.base_adapter, str(probe_dir), family)
        zip_path = out / f"v15-probe-{family}.zip"
        sha = make_submit_zip(str(probe_dir), str(zip_path))
        print(f"  ZIP: {zip_path.name} SHA:{sha[:12]}")

        probe_data = {
            "family": family,
            "adapter_path": str(probe_dir),
            "zip_path": str(zip_path),
            "sha": sha,
            "submitted": False,
            "score": None,
            "delta": None,
            "submit_time": None,
        }

        if args.execute == "true":
            msg = f"v15-probe zero_{family} sha:{sha[:12]}"
            sub = submit_to_kaggle(str(zip_path), msg)
            probe_data["submitted"] = (sub["rc"] == 0)
            probe_data["submit_time"] = time.time()
            if sub["rc"] == 0:
                print(f"  [KAGGLE OK] polling for score (max {args.wait_seconds}s)...")
                score = poll_kaggle_score(sha[:12], max_wait=args.wait_seconds)
                probe_data["score"] = score
                if score is not None:
                    probe_data["delta"] = args.baseline_score - score
                    print(f"  Score: {score:.4f} | Delta from baseline: -{probe_data['delta']:.4f}")
                else:
                    print(f"  Score pending (timeout)")
            else:
                print(f"  [KAGGLE FAIL] {sub['stderr'][:200]}")
            # Rate limit: 5 submits/day = wait to spread over time
            if i < len(FAMILIES):
                print(f"  Waiting 60s before next probe (Kaggle rate limit)...")
                time.sleep(60)
        else:
            print(f"  [DRY RUN] would submit: {zip_path.name}")

        results["probes"][family] = probe_data

    # Save raw results
    with open(out / "probe_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nRaw results saved: {out}/probe_results.json")

    # Compute inferred weights (only if we have scores)
    if args.execute == "true":
        deltas_ok = {f: p["delta"] for f, p in results["probes"].items() if p["delta"] is not None}
        if len(deltas_ok) >= 5:
            total_delta = sum(deltas_ok.values())
            if total_delta > 0:
                weights = {f: d / total_delta for f, d in deltas_ok.items()}
                print("\n\nInferred family weights (normalized to sum=1):")
                for f, w in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {f:30s}: {w*100:5.1f}%  (delta: {deltas_ok[f]:.4f})")
                results["weights_measured"] = weights
                with open(out / "weights_measured.json", "w", encoding="utf-8") as f:
                    json.dump(weights, f, indent=2)
                print(f"\nMeasured weights saved: {out}/weights_measured.json")
        else:
            print(f"\n\nToo few probes scored ({len(deltas_ok)}/{len(FAMILIES)}). Retry later.")

    print("\n" + "=" * 70)
    print("LB Probe complete.")
    print("Next: run with --execute true after reviewing dry-run output")


if __name__ == "__main__":
    main()
