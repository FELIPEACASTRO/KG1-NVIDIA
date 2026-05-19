#!/usr/bin/env python3
"""Build and optionally execute a post-training OpenRouter consult.

This is a CPU-only governance step. It does not train, evaluate a model, package,
or submit. Its purpose is to make every completed/failed training run leave a
complete, low-noise prompt and, when OPENROUTER_API_KEY is available, a panel of
external model responses before the next paid GPU decision.
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


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "openai/gpt-oss-120b:free",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-3-haiku",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
]
DEFAULT_MAX_CHARS = 24000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n[...TRUNCATED_BY_PROMPT_BUILDER_MIDDLE...]\n\n" + tail


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    for path in [ROOT / ".env", ROOT.parent / ".env", ROOT.parents[1] / ".env"]:
        key = parse_env_file(path).get("OPENROUTER_API_KEY", "").strip()
        if key:
            return key
    return ""


def compact_json(path: Path, max_chars: int) -> str:
    try:
        obj = read_json(path)
        text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception:
        text = read_text(path, max_chars)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...TRUNCATED_BY_PROMPT_BUILDER...]"


def collect_file_block(label: str, path: Path, max_chars: int) -> str:
    if not path.exists():
        return f"## {label}\nPath: `{path}`\nStatus: MISSING\n"
    content = compact_json(path, max_chars) if path.suffix.lower() == ".json" else read_text(path, max_chars)
    return f"## {label}\nPath: `{path}`\n\n```text\n{content}\n```\n"


def build_prompt(args: argparse.Namespace) -> str:
    blocks: list[str] = []
    blocks.append(
        "\n".join(
            [
                "# KG1 post-training crisis consult",
                "",
                "You are an external ML/MLOps/code-review panel for the KG1 NVIDIA Nemotron Model Reasoning Challenge solution.",
                "Return only actionable findings. Do not invent files, metrics, or leaderboard results.",
                "If evidence is insufficient, say exactly which local artifact or metric is missing.",
                "",
                "## Hard rules",
                "- Kaggle score is final-answer accuracy, not eval_loss alone.",
                "- False gains are forbidden: no promotion without label-free extraction, verify_answer, zero truncation, and protected-row backfire guards.",
                "- Distinguish bad decoding from adapter weights pushing the model toward wrong answers.",
                "- Loss must be cross-entropy on masked assistant/answer tokens and must stay aligned with accuracy gates.",
                "- If row_loss_weight is used in train, validation eval_loss must use the same row-weight contract.",
                "- No new paid GPU run should be recommended unless CPU/gate evidence predicts an accuracy gain and protects current correct rows.",
                "- Prefer A100-large. H200 is allowed only when A100 cannot run the stack or memory requirement is objectively proven.",
                "",
                "## Required response format",
                "1. Verdict: proceed / block / needs artifact.",
                "2. Top 5 concrete bugs or gaps, each tied to evidence in the prompt.",
                "3. Exact next experiment that is cheapest and most likely to improve weak ACC.",
                "4. Parameters to change or freeze, with values.",
                "5. Gates that must pass before another paid GPU job.",
                "6. Anything in the current plan that should be deleted because it is noise.",
                "",
            ]
        )
    )
    blocks.append(
        "\n".join(
            [
                "## Current run metadata",
                f"- run_id: `{args.run_id}`",
                f"- generated_at_utc: `{utc_now()}`",
                "- Current observed plateau: weak ACC has not exceeded the deployable baseline. V664 reached only 192/315.",
                "- Best actionable weak target remains at least 196/315 without protected-row regression, with bit >= 136 and equation >= 60.",
                "- V664 weak result: total 192/315, bit 136/160, equation 56/155, truncated 0, boxed_rate 1.0, but completions were extremely long and protected bit row 8740ed31 backfired.",
                "- V664 training moved only q_proj/v_proj LoRA tensors from V290 checkpoint-6; non-q/v tensors were unchanged.",
                "- V664 train loss decreased in 2 steps, but the generation behavior stayed long and unsafe. Loss movement alone is not acceptable evidence.",
                "",
            ]
        )
    )
    for label, path in [
        ("Roadmap", args.roadmap_md),
        ("Training or launch manifest", args.train_manifest_json),
        ("Weak/full eval summary", args.eval_summary_json),
        ("Failure analysis summary", args.failure_summary_json),
        ("Previous OpenRouter consensus", args.previous_consensus_md),
    ]:
        if path:
            blocks.append(collect_file_block(label, path, args.max_chars_per_file))
    for index, path in enumerate(args.extra_file or [], 1):
        blocks.append(collect_file_block(f"Extra artifact {index}", path, args.max_chars_per_file))
    blocks.append(
        "\n".join(
            [
                "## Final instruction",
                "Give a surgical answer that can change the next roadmap step. Do not repeat generic ML advice.",
                "Focus on concrete implementation, data, masking, decoding, LoRA contract, validation, and gate changes that can improve ACC safely.",
                "",
            ]
        )
    )
    return "\n\n".join(blocks)


def post_openrouter(model: str, prompt: str, api_key: str, timeout_s: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4096,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.kg1",
            "X-Title": "KG1 post-training consult",
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return {
            "model": model,
            "status": "ok",
            "elapsed_s": round(time.time() - started, 3),
            "response": json.loads(raw),
        }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "model": model,
            "status": "http_error",
            "http_status": exc.code,
            "elapsed_s": round(time.time() - started, 3),
            "error": raw,
        }
    except Exception as exc:
        return {
            "model": model,
            "status": "exception",
            "elapsed_s": round(time.time() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def extract_answer(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return ""
    response = result.get("response")
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return str(content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roadmap-md", type=Path, default=ROOT / "artifacts/roadmaps/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md")
    parser.add_argument("--train-manifest-json", type=Path, default=None)
    parser.add_argument("--eval-summary-json", type=Path, default=None)
    parser.add_argument("--failure-summary-json", type=Path, default=None)
    parser.add_argument("--previous-consensus-md", type=Path, default=None)
    parser.add_argument("--extra-file", type=Path, action="append", default=[])
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-chars-per-file", type=int, default=DEFAULT_MAX_CHARS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(args)
    prompt_path = args.output_dir / "KG1_POST_TRAIN_OPENROUTER_PROMPT.md"
    raw_path = args.output_dir / "openrouter_raw_results.json"
    responses_path = args.output_dir / "openrouter_responses.md"
    manifest_path = args.output_dir / "openrouter_manifest.json"
    prompt_path.write_text(prompt, encoding="utf-8", newline="\n")

    models = args.model or DEFAULT_MODELS
    api_key = load_openrouter_key()
    results: list[dict[str, Any]] = []
    if args.execute:
        if not api_key:
            results.append({"status": "skipped", "reason": "OPENROUTER_API_KEY is not set"})
        else:
            for index, model in enumerate(models, 1):
                print(f"=== OPENROUTER POST-TRAIN CALL START {index}/{len(models)} model={model} ===", flush=True)
                result = post_openrouter(model, prompt, api_key, args.timeout_s)
                print(
                    "=== OPENROUTER POST-TRAIN CALL END "
                    f"model={model} status={result.get('status')} elapsed_s={result.get('elapsed_s')} ===",
                    flush=True,
                )
                results.append(result)

    raw_path.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    response_lines = ["# KG1 Post-Train OpenRouter Responses", ""]
    for result in results:
        response_lines.extend([f"## {result.get('model', 'unknown')}", "", f"- status: `{result.get('status')}`", ""])
        answer = extract_answer(result)
        response_lines.append(answer if answer else json.dumps(result, indent=2, ensure_ascii=False)[:6000])
        response_lines.append("")
    responses_path.write_text("\n".join(response_lines), encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": "kg1_post_train_openrouter_consult_v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "execute": bool(args.execute),
        "api_key_present_without_value": bool(api_key),
        "models": models,
        "ok_count": sum(1 for item in results if item.get("status") == "ok"),
        "prompt_path": str(prompt_path),
        "raw_results_path": str(raw_path),
        "responses_path": str(responses_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
