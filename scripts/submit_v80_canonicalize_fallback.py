#!/usr/bin/env python3
"""V80 stripped canonicalize fallback submit — Melhoria 2.

Objetivo: adicionar 1 tentativa de submit com adapter DIFERENTE do V102
(correlação ~0.3) para reduzir P(all fail). Esperado: prob >=0.85 sobe
de ~95% para ~97%.

IMPORTANTE: isto NÃO aplica canonicalization no output do kernel Kaggle
(impossível — kernel controla inference). O que faz é:

1. Pega o V80 stripped adapter que já temos (SHA ab100039cb48, 105 MB)
2. Reutiliza adapter_config.json do V80 (já validado)
3. Submete via Python API Kaggle (bypass CLI 2.0.1 bug)

Racional: V80 foi treinado com dataset diferente do V102 (konbu17 recipe
com 245 steps vs huikang full huikang_v70_full 1 epoch). São adapters
estatisticamente independentes (correlação estimada ~0.3). Adicionar 1
submit de V80 stripped reduz P(all fail) de 5% → 3%.

Limitação honesta: V80 stripped já foi submetido 1x e deu score baixo
(strip perde signal dos experts). Re-submeter MESMO binário dá o mesmo
score (kernel é determinístico com temp=0). Então essa melhoria SÓ faz
sentido se você AINDA não submeteu V80 stripped OU se criar uma variante
diferente do V80 (ex: merge V80 + V70-like base).

Como usar:

    # No Colab onde V80 stripped está em /content/kg1_v80_stripped_v2/
    python scripts/submit_v80_canonicalize_fallback.py \
        --stripped-dir /content/kg1_v80_stripped_v2 \
        --message "V80 fallback submit post-V102"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path


def build_submission_zip(stripped_dir: Path, zip_path: Path) -> dict:
    """Build submission.zip ZIP_STORED from stripped dir."""
    config_path = stripped_dir / "adapter_config.json"
    weights_path = stripped_dir / "adapter_model.safetensors"

    for p in [config_path, weights_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(config_path, arcname="adapter_config.json")
        z.write(weights_path, arcname="adapter_model.safetensors")

    size_mb = zip_path.stat().st_size / 1024**2
    with open(zip_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    return {"path": str(zip_path), "size_mb": round(size_mb, 1), "sha": sha}


def validate_config(stripped_dir: Path) -> dict:
    """Validate adapter_config.json passes Kaggle gate."""
    cfg_path = stripped_dir / "adapter_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Gate constraints (from scripts/submit_kaggle.py FORBIDDEN_TARGETS)
    target_modules = cfg.get("target_modules", [])
    if isinstance(target_modules, list):
        if "gate_proj" in target_modules:
            raise ValueError("gate_proj in target_modules — FORBIDDEN pelo Kaggle gate")
        if "x_proj" in target_modules:
            raise ValueError("x_proj in target_modules — FORBIDDEN")
        if "in_proj" not in target_modules:
            raise ValueError("in_proj missing — REQUIRED pelo Kaggle gate")

    if cfg.get("r", 0) > 32:
        raise ValueError(f"r={cfg.get('r')} > 32 — exceeds max_lora_rank")

    return {
        "target_modules": target_modules,
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "base_model_name_or_path": cfg.get("base_model_name_or_path"),
    }


def submit_via_python_api(zip_path: Path, message: str, competition: str) -> dict:
    """Submit via Kaggle Python API (bypass CLI 2.0.1 bug 400)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError("kaggle not installed. pip install kaggle")

    api = KaggleApi()
    api.authenticate()

    # Check daily limit first
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    subs = api.competition_submissions(competition)
    today_count = 0
    for s in subs:
        try:
            d = s.date.date() if hasattr(s.date, "date") else datetime.strptime(str(s.date)[:10], "%Y-%m-%d").date()
            if d == today:
                today_count += 1
        except Exception:
            continue

    if today_count >= 5:
        return {
            "ok": False,
            "error": f"Daily limit reached ({today_count}/5). Reset 00:00 UTC.",
        }

    print(f"[INFO] Submits today: {today_count}/5 — proceeding")

    try:
        result = api.competition_submit(
            file_name=str(zip_path),
            message=message,
            competition=competition,
        )
        return {"ok": True, "result": str(result)[:500], "today_count": today_count + 1}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stripped-dir",
        type=Path,
        required=True,
        help="Diretório com V80 stripped (adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument(
        "--message",
        default="V80 canonicalize fallback — Melhoria 2",
        help="Mensagem do submit Kaggle",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=None,
        help="Path para submission.zip (default: ./submission.zip)",
    )
    parser.add_argument(
        "--competition",
        default="nvidia-nemotron-model-reasoning-challenge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build zip + validate mas não submete",
    )
    args = parser.parse_args()

    if args.zip_output is None:
        args.zip_output = Path("./submission.zip")

    # Validate config
    print(f"=== Validating {args.stripped_dir}/adapter_config.json ===")
    try:
        cfg_info = validate_config(args.stripped_dir)
        print(f"[OK] Config valid")
        print(f"  target_modules: {cfg_info['target_modules']}")
        print(f"  r={cfg_info['r']} alpha={cfg_info['lora_alpha']}")
        print(f"  base: {cfg_info['base_model_name_or_path']}")
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1

    # Build zip
    print(f"\n=== Building {args.zip_output} ===")
    zip_info = build_submission_zip(args.stripped_dir, args.zip_output)
    print(f"[OK] ZIP {zip_info['size_mb']} MB sha={zip_info['sha'][:12]}")

    if zip_info["size_mb"] > 500:
        print(f"[FAIL] ZIP > 500 MB ({zip_info['size_mb']}) — exceeds Kaggle limit")
        return 1

    # Submit (unless dry run)
    if args.dry_run:
        print("\n[DRY RUN] Submit skipped. ZIP built at " + str(args.zip_output))
        return 0

    print(f"\n=== Submitting to {args.competition} ===")
    msg = f"{args.message} sha={zip_info['sha'][:12]}"
    result = submit_via_python_api(args.zip_output, msg, args.competition)

    if result["ok"]:
        print(f"[OK] Submitted. Today: {result.get('today_count', '?')}/5")
        print(f"Result: {result.get('result', '')[:200]}")
        return 0
    else:
        print(f"[FAIL] {result['error']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
