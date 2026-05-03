from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "v198_micro_distill_colab_pack_20260503"
MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-2.5-flash",
    "qwen/qwen3.6-flash",
    "deepseek/deepseek-v4-flash",
]


def load_openrouter_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env_path = ROOT / "runs" / "score_analysis_20260426" / "todas_ias_chaves_autenticacao.normalized.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r'^OPENROUTER_API_KEY\s*=\s*"?([^"\r\n]+)"?', line.strip())
            if match:
                return match.group(1)
    raise RuntimeError("OPENROUTER_API_KEY not found")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def prompt_for() -> str:
    manifest = json.loads((ROOT / "data" / "v198" / "v198_micro_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT_DIR / "v198_training_data_gate_strict.json").read_text(encoding="utf-8"))
    sft = json.loads((OUT_DIR / "v198_sft_format_report_strict.json").read_text(encoding="utf-8"))
    v197 = json.loads((ROOT / "runs" / "v197_strict_feature_gate_lab_20260503" / "v197_strict_feature_gate_summary.json").read_text(encoding="utf-8"))
    compact = {
        "goal": "reach Kaggle KG1 public score >=0.87 without stable-family regression",
        "current_public_score_band": "0.86 public, V197 local selector 628/720=0.872222 but not directly submit-ready",
        "v198_manifest": {
            "train_rows": manifest["train_rows"],
            "family_counts": manifest["train_family_counts"],
            "source_counts": manifest["train_source_counts"],
            "role_counts": manifest["train_role_counts"],
            "v196_anti_regression_stats": manifest["v196_anti_regression_stats"],
            "v197_positive_distill_stats": manifest["v197_positive_distill_stats"],
            "notes": manifest["notes"],
        },
        "local_gates": {
            "training_data_valid": gate["valid"],
            "training_data_reasons": gate["reasons"],
            "training_data_warnings": gate["warnings"],
            "sft_clean_rate": sft["clean_rate"],
            "sft_issue_totals": sft["issue_totals"],
        },
        "v197_evidence": {
            "validation": v197["validation"],
            "stress_method_stats": v197["stress"]["method_stats"],
        },
        "v198_recipe": {
            "init_order": ["V195 final_adapter", "V195 checkpoint-110/75/55", "0.86 baseline fallback"],
            "max_steps": 45,
            "lr": "1e-5 -> 3e-6",
            "trainable_modules": "attention projections only",
            "submit_policy": "no Kaggle submit until converted adapter beats baseline locally and has no stable-family regression",
        },
    }
    return (
        "You are auditing a Kaggle KG1/Nemotron training plan. Respond only strict JSON, no prose.\n"
        "Task: decide if V198 micro-distillation is the fastest safe next GPU step, and define stop gates.\n"
        "Hard constraints: no direct training on local validation anchors, no ID hardcode, no Kaggle submit before local validation.\n"
        f"Evidence:\n{json.dumps(compact, ensure_ascii=False)}\n"
        "JSON schema: {"
        '"decision":"proceed|hold|reject",'
        '"confidence":0,'
        '"main_risk":"",'
        '"expected_best_case":"",'
        '"stop_if":"",'
        '"must_check_before_submit":[""],'
        '"suggested_change":"none|short text"'
        "}"
    )


def call_model(key: str, model: str, prompt: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/kg1",
        "X-Title": "KG1 V198 micro plan audit",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    started = time.time()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=120,
    )
    elapsed = time.time() - started
    if response.status_code >= 400:
        return {
            "model": model,
            "ok": False,
            "status_code": response.status_code,
            "elapsed_sec": round(elapsed, 3),
            "error": response.text[:1000],
        }
    data = response.json()
    raw = data["choices"][0]["message"].get("content") or ""
    try:
        parsed = extract_json(raw)
        ok = True
        error = ""
    except Exception as exc:
        parsed = {}
        ok = False
        error = f"json_parse_failed: {exc}"
    return {
        "model": model,
        "ok": ok,
        "status_code": response.status_code,
        "elapsed_sec": round(elapsed, 3),
        "usage": data.get("usage", {}),
        "content": parsed,
        "raw_content": raw,
        "error": error,
    }


def render_report(results: list[dict[str, Any]]) -> str:
    ok = [row for row in results if row.get("ok")]
    total_cost = sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in ok)
    decisions = CounterLike(row["content"].get("decision") for row in ok)
    lines = [
        "# V198 OpenRouter Micro Plan Audit",
        "",
        f"- ok_models: `{len(ok)}/{len(results)}`",
        f"- estimated_cost: `${total_cost:.6f}`",
        f"- decisions: `{dict(decisions)}`",
        "",
        "## Model verdicts",
        "",
    ]
    for row in results:
        if not row.get("ok"):
            lines.append(f"- `{row['model']}` FAILED {row.get('error', '')[:180]}")
            continue
        content = row["content"]
        lines.append(
            "- `{model}` decision=`{decision}` confidence=`{confidence}` risk=`{risk}` change=`{change}`".format(
                model=row["model"],
                decision=content.get("decision"),
                confidence=content.get("confidence"),
                risk=str(content.get("main_risk", ""))[:160],
                change=str(content.get("suggested_change", ""))[:160],
            )
        )
    return "\n".join(lines) + "\n"


def CounterLike(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    key = load_openrouter_key()
    prompt = prompt_for()
    results = [call_model(key, model, prompt) for model in MODELS]
    with (OUT_DIR / "v198_openrouter_micro_plan_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = render_report(results)
    (OUT_DIR / "V198_OPENROUTER_MICRO_PLAN_AUDIT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
