#!/usr/bin/env python3
"""Run an OpenRouter model panel for the KG1 >=0.87 roadmap.

The script is intentionally cost-guarded. It can inspect the full OpenRouter
model catalog, then call a bounded panel of free/cheap/frontier models with a
single surgical prompt. It never prints API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL_PANEL = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/owl-alpha",
    "qwen/qwen3.6-vl:free",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.6-coder",
    "x-ai/grok-4.3",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
    "google/gemini-3-pro-preview",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def request_json(url: str, *, headers: dict[str, str] | None = None, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers or {})
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def model_price(model: dict[str, Any]) -> float:
    pricing = model.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 999)) + float(pricing.get("completion", 999))
    except Exception:
        return 999.0


def select_models(catalog: list[dict[str, Any]], requested: list[str], max_models: int, include_paid: bool) -> list[str]:
    by_id = {item.get("id"): item for item in catalog}
    selected: list[str] = []
    for model_id in requested:
        if model_id in by_id and model_id not in selected:
            selected.append(model_id)
    free = [
        item.get("id")
        for item in sorted(catalog, key=model_price)
        if item.get("id")
        and item.get("id") not in selected
        and (item.get("id", "").endswith(":free") or model_price(item) == 0)
    ]
    for model_id in free:
        if len(selected) >= max_models:
            return selected
        selected.append(model_id)
    if include_paid:
        for item in sorted(catalog, key=model_price):
            model_id = item.get("id")
            if not model_id or model_id in selected:
                continue
            if len(selected) >= max_models:
                break
            selected.append(model_id)
    return selected[:max_models]


def build_prompt(context: str) -> str:
    return f"""You are reviewing a Kaggle adapter-only competition pipeline.

Goal: reach public score >= 0.87 without regressing below the confirmed best baseline.

Known facts:
- Competition: NVIDIA Nemotron Model Reasoning Challenge.
- Submission artifact is only a root-level LoRA adapter zip with adapter_config.json and adapter_model.safetensors.
- Current production baseline is V194/ref 52275052: public score 0.86, user-confirmed rank 19/2613, zip SHA 49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8.
- V199B/ref 52325494 scored 0.86, not promoted because it did not beat V194.
- V198/ref 52301667 scored 0.84 and is a known regression.
- Failures seen: broad soups (V191 0.78), focal training (V174 0.41), stripped/packaging errors (0.50-0.54), V198 final lineage regression (0.84).
- Safe patterns seen: V192/V193/V194 attention-only micro lineage around 0.86, V199B baseline-gated no-regression but no gain.
- Existing train data mixture: strict V198/V195/V196/V197 data, 1875 train / 720 validation, baseline eval local around 1.123.

Additional local/web context:
{context}

Task:
Return compact JSON only, with these keys:
1. verdict: one sentence.
2. best_next_experiment: exact candidate to run next from V194, including init, data, LR, steps, trainable modules, and why it can plausibly reach 0.87.
3. reject: array of strategies to block.
4. hard_gates: array of concrete pre-submit gates that must pass.
5. anti_regression_policy: exact promotion rule.
6. probability_public_087: integer 0-100.
7. risk_notes: array of main risks.

Be ruthless. Prefer one small, testable change over broad training. Do not recommend API solvers in the Kaggle submission; they can only be used to generate or verify training data offline."""


def call_model(api_key: str, model_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/FELIPEACASTRO/KG1-NVIDIA",
        "X-Title": "KG1 V201 roadmap panel",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a terse, rigorous ML competition risk reviewer. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    started = time.time()
    try:
        data = request_json(OPENROUTER_CHAT_URL, headers=headers, payload=payload)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {
            "model": model_id,
            "ok": True,
            "elapsed_sec": round(time.time() - started, 2),
            "content": content,
            "usage": data.get("usage"),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:1200]
        return {
            "model": model_id,
            "ok": False,
            "elapsed_sec": round(time.time() - started, 2),
            "error": f"HTTP {exc.code}",
            "body": body,
        }
    except Exception as exc:
        return {
            "model": model_id,
            "ok": False,
            "elapsed_sec": round(time.time() - started, 2),
            "error": repr(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--catalog-json", type=Path)
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--include-paid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is set.")

    catalog_payload = request_json(OPENROUTER_MODELS_URL)
    catalog = catalog_payload.get("data") or []
    if args.catalog_json:
        write_json(args.catalog_json, catalog_payload)

    requested = args.model or DEFAULT_MODEL_PANEL
    models = select_models(catalog, requested, args.max_models, args.include_paid)
    prompt = build_prompt(args.context.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "catalog_model_count": len(catalog),
        "selected_models": models,
        "include_paid": args.include_paid,
        "dry_run": args.dry_run,
        "prompt": prompt,
        "results": [],
    }
    if args.dry_run:
        write_json(args.output_json, payload)
        print(json.dumps({"dry_run": True, "selected_models": models}, indent=2))
        return 0

    for model_id in models:
        print("calling", model_id, flush=True)
        payload["results"].append(call_model(api_key, model_id, prompt, args.max_tokens))
        write_json(args.output_json, payload)
        time.sleep(1.5)

    ok_count = sum(1 for item in payload["results"] if item.get("ok"))
    print(json.dumps({"results": len(payload["results"]), "ok": ok_count, "output": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
