#!/usr/bin/env python3
"""V71 Stage 1 — gera CoTs para 959 prompts weak usando teachers GRATUITOS.

Teacher primario:
  nvidia/nemotron-3-super-120b-a12b:free  (via OpenRouter, 0 custo)
    - mesmo DNA arquitetural do student (Nemotron-3)
    - 120B parametros -> reasoning forte
    - :free significa 0 credits debitados, RPM ~20

Teachers fallback (se RATE_LIMIT ou erro):
  1. nvidia/nemotron-3-nano-30b-a3b:free   (OpenRouter)
  2. deepseek/deepseek-r1:free             (OpenRouter)
  3. z-ai/glm-4.5-air:free                 (OpenRouter)
  4. openai/gpt-oss-120b:free              (OpenRouter)
  5. zhipu glm-4.7-flash                   (direto - comprovado free)
  6. Qwen/Qwen3.5-397B-A17B                (HuggingFace router)
  7. deepseek-ai/DeepSeek-R1               (HuggingFace router)

Estrategia:
  - Cada prompt -> 1 CoT por teacher primario
  - Se resposta != ground truth, tenta teacher fallback ate N=3
  - Filtra apenas CoTs com answer correto (ground truth match)
  - Saida JSONL formato V70-compatible

Uso:
  # Smoke test (3 prompts)
  python scripts/v71_generate_cots_free_teachers.py --smoke

  # Full 959
  python scripts/v71_generate_cots_free_teachers.py --all

  # Categoria especifica
  python scripts/v71_generate_cots_free_teachers.py --category cryptarithm_guess
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
WEAK_JSONL = ROOT / "data" / "v71" / "weak_prompts.jsonl"
OUT_DIR = ROOT / "data" / "v71" / "cots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUT_DIR / "generation_state.json"

# API keys (read from env)
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY", "")


@dataclass
class Teacher:
    name: str
    endpoint: str
    model: str
    key: str
    extra_headers: Optional[dict] = None
    rpm_delay: float = 3.0   # conservative, avoid rate limit


TEACHERS_CHAIN = [
    Teacher(
        name="or_nemotron_3_super_120b",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        key=OR_KEY,
        extra_headers={
            "HTTP-Referer": "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge",
            "X-Title": "KG1 V71 Weak Cat Distillation",
        },
        rpm_delay=3.0,
    ),
    Teacher(
        name="or_nemotron_3_nano_30b",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        key=OR_KEY,
        extra_headers={
            "HTTP-Referer": "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge",
            "X-Title": "KG1 V71 Weak Cat Distillation",
        },
        rpm_delay=3.0,
    ),
    Teacher(
        name="or_deepseek_r1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="deepseek/deepseek-r1:free",
        key=OR_KEY,
        extra_headers={
            "HTTP-Referer": "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge",
            "X-Title": "KG1 V71 Weak Cat Distillation",
        },
        rpm_delay=5.0,
    ),
    Teacher(
        name="zhipu_glm_4_7_flash",
        endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        model="glm-4.7-flash",
        key=ZHIPU_KEY,
        rpm_delay=2.0,
    ),
    Teacher(
        name="hf_qwen3_5_397b",
        endpoint="https://router.huggingface.co/v1/chat/completions",
        model="Qwen/Qwen3.5-397B-A17B",
        key=HF_TOKEN,
        rpm_delay=4.0,
    ),
]


SYSTEM_PROMPT = (
    "You are an expert puzzle solver. For the given problem, reason step-by-step "
    "inside <think></think> tags, then give the FINAL answer in \\boxed{}. "
    "Be deterministic. Verify every hypothesis against ALL provided examples before "
    "committing. If examples are ambiguous, enumerate candidates and pick the one "
    "consistent with the largest number of examples."
)

CATEGORY_HINTS = {
    "cryptarithm_deduce": (
        "This is a cryptarithm where each letter maps to a unique digit 0-9. "
        "The operation is one of: addition, absolute difference, multiplication, "
        "concatenation (s0s1s3s4), or reverse concatenation (s3s4s0s1). "
        "Try all 5 ops; for each op, backtrack over digit assignments; keep only "
        "solutions consistent with ALL examples. Output the unique answer."
    ),
    "cryptarithm_guess": (
        "This is a cryptarithm with ambiguous operation/mapping. "
        "Examples may be insufficient to uniquely determine the rule. "
        "Enumerate most-likely candidate answers; pick the one with highest posterior "
        "given the examples (rule parsimony + frequency). "
        "If truly underdetermined, default to the most common digit pattern."
    ),
    "equation_numeric_guess": (
        "This is an equation puzzle where numeric operations are disguised. "
        "Try: simple arithmetic (+, -, *, /, %, **), digit manipulations, "
        "modular arithmetic, base conversions. "
        "Verify candidate rules on ALL examples; pick the rule consistent with most."
    ),
}


def extract_answer(content: str) -> str:
    """Extract last \\boxed{...} content."""
    if not content:
        return ""
    # Handle nested \boxed{\boxed{x}} by finding outermost balanced
    matches = list(re.finditer(r"\\boxed\{", content))
    if not matches:
        return ""
    # Take the LAST \boxed{ and extract balanced content
    last = matches[-1]
    start = last.end()
    depth = 1
    end = start
    while end < len(content) and depth > 0:
        if content[end] == "{":
            depth += 1
        elif content[end] == "}":
            depth -= 1
        end += 1
    if depth == 0:
        return content[start:end - 1].strip()
    return content[start:].strip()


def call_teacher(teacher: Teacher, prompt: str, category: str,
                 timeout: int = 120) -> dict:
    if not teacher.key:
        return {"ok": False, "err": "no_key", "teacher": teacher.name}

    hint = CATEGORY_HINTS.get(category, "")
    user_content = f"{hint}\n\n{prompt}\n\nGive the final answer in \\boxed{{}}."

    body = json.dumps({
        "model": teacher.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 4096,
        "temperature": 0.3,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {teacher.key}",
    }
    if teacher.extra_headers:
        headers.update(teacher.extra_headers)

    req = urllib.request.Request(teacher.endpoint, data=body, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        content = d.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        answer = extract_answer(content)
        return {
            "ok": True,
            "teacher": teacher.name,
            "model": teacher.model,
            "content": content,
            "answer": answer,
            "tokens": d.get("usage", {}),
            "elapsed": round(time.time() - t0, 2),
        }
    except urllib.error.HTTPError as e:
        try:
            body_err = e.read().decode()[:300]
        except Exception:
            body_err = ""
        return {"ok": False, "teacher": teacher.name, "code": e.code,
                "err": body_err, "elapsed": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "teacher": teacher.name, "err": str(e)[:300],
                "elapsed": round(time.time() - t0, 2)}


def generate_for_row(row: dict, teachers: list[Teacher]) -> dict:
    """Tenta teachers em ordem ate acertar ground truth ou esgotar chain."""
    gt = row["answer"]
    attempts = []

    for t in teachers:
        result = call_teacher(t, row["prompt"], row["category"])
        result["attempt_idx"] = len(attempts)
        attempts.append(result)
        if result["ok"]:
            if result["answer"].strip() == gt.strip():
                return {
                    "id": row["id"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "answer": gt,
                    "correct": True,
                    "winning_teacher": t.name,
                    "winning_model": t.model,
                    "content": result["content"],
                    "attempts": attempts,
                }
            time.sleep(t.rpm_delay)  # respectful between calls
        else:
            time.sleep(t.rpm_delay)

    # No teacher got it correct
    return {
        "id": row["id"],
        "category": row["category"],
        "prompt": row["prompt"],
        "answer": gt,
        "correct": False,
        "winning_teacher": None,
        "winning_model": None,
        "content": None,
        "attempts": attempts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Rodar apenas 3 prompts (smoke test)")
    ap.add_argument("--all", action="store_true", help="Rodar todas as 959 prompts")
    ap.add_argument("--category", type=str, default=None,
                    help="Rodar apenas uma categoria")
    ap.add_argument("--limit", type=int, default=None, help="Limite de prompts")
    ap.add_argument("--workers", type=int, default=2, help="Threads paralelas")
    ap.add_argument("--output", type=str, default=None,
                    help="Path de saida (default: data/v71/cots/cots_<ts>.jsonl)")
    args = ap.parse_args()

    # Validate keys
    active_teachers = [t for t in TEACHERS_CHAIN if t.key]
    if not active_teachers:
        print("[v71-gen] ERRO: nenhum teacher tem key configurada.")
        print("  Setar OPENROUTER_API_KEY ou ZHIPU_API_KEY ou HF_TOKEN.")
        sys.exit(1)
    print(f"[v71-gen] Teachers ativos ({len(active_teachers)}):")
    for t in active_teachers:
        print(f"  {t.name:30s} {t.model}")

    # Load prompts
    rows = []
    with WEAK_JSONL.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if args.category:
        rows = [r for r in rows if r["category"] == args.category]

    if args.smoke:
        rows = rows[:3]
    elif args.limit:
        rows = rows[:args.limit]
    elif not args.all:
        print("[v71-gen] Use --smoke, --all ou --limit N. Abortando.")
        sys.exit(1)

    print(f"[v71-gen] Processando {len(rows)} prompts...")

    # Output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else OUT_DIR / f"cots_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lock = Lock()
    correct_count = 0
    total_done = 0

    with out_path.open("w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(generate_for_row, row, active_teachers): row for row in rows}
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    total_done += 1
                    if rec["correct"]:
                        correct_count += 1
                    mark = "OK" if rec["correct"] else "FAIL"
                    print(f"  [{mark}] {total_done}/{len(rows)} "
                          f"{rec['category']} {rec['id']} teacher={rec['winning_teacher']}")

    pct = 100 * correct_count / max(total_done, 1)
    print(f"\n[v71-gen] FINAL: {correct_count}/{total_done} correct ({pct:.1f}%)")
    print(f"[v71-gen] Output: {out_path}")


if __name__ == "__main__":
    main()
