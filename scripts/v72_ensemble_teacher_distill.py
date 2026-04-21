#!/usr/bin/env python3
"""V72 multi-teacher ensemble + rejection sampling + verifier.

Input:  data/v71/weak_prompts.jsonl               (959 rows)
        data/v71/solver_programmatic.jsonl        (101 already solved)
Output: data/v72/verified_cots.jsonl              (subset passed GT)
        data/v72/generation_state.json            (resume state)
        data/v72/unsolvable.jsonl                 (rows with 0 CoTs passing GT)

Strategy:
  1. Skip 101 already-solved (link via solver_programmatic.jsonl)
  2. For each of 858 residual:
     a. Call up to 3 teachers in parallel: Claude Opus 4.7, DeepSeek-reasoner,
        Cerebras qwen-235b (tertiary fallback)
     b. Extract \\boxed{answer} from each CoT
     c. Run multi-layer verifier (answer match + minimum quality gates)
     d. If >=1 passed, save shortest verified CoT
     e. Else mark row as unsolvable
  3. Parallel execution with adaptive rate limiter + exponential backoff

Usage:
    python scripts/v72_ensemble_teacher_distill.py --preview
    python scripts/v72_ensemble_teacher_distill.py --all
    python scripts/v72_ensemble_teacher_distill.py --limit 10    # smoke test
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths — resolve repo ROOT from script location, falling back to env override
# so the script works whether placed in scripts/ or copied to /tmp for preview.
# ---------------------------------------------------------------------------
def _resolve_root() -> Path:
    env_root = os.environ.get("KG1_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    # Walk up looking for a directory that contains data/v71/weak_prompts.jsonl
    for parent in [here.parent, *here.parents]:
        if (parent / "data" / "v71" / "weak_prompts.jsonl").exists():
            return parent
    # Script installed inside repo as scripts/v72_*.py -> parents[1] is repo root
    if here.parent.name == "scripts":
        return here.parents[1]
    return Path.cwd()


ROOT = _resolve_root()
WEAK_JSONL = ROOT / "data" / "v71" / "weak_prompts.jsonl"
SOLVER_JSONL = ROOT / "data" / "v71" / "solver_programmatic.jsonl"
OUT_DIR = ROOT / "data" / "v72"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERIFIED_JSONL = OUT_DIR / "verified_cots.jsonl"
UNSOLVABLE_JSONL = OUT_DIR / "unsolvable.jsonl"
STATE_FILE = OUT_DIR / "generation_state.json"
LOG_FILE = OUT_DIR / "run.log"

# ---------------------------------------------------------------------------
# HTTP config
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)
TIMEOUT_S = 180  # per call
BACKOFF_SEQUENCE = [30, 60, 120]  # seconds on 429
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# API keys (env-only, never hardcoded)
# ---------------------------------------------------------------------------
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
SAMBANOVA_KEY = os.environ.get("SAMBANOVA_API_KEY", "")


# ---------------------------------------------------------------------------
# Teacher definition
# ---------------------------------------------------------------------------
@dataclass
class Teacher:
    name: str
    endpoint: str
    model: str
    api_style: str       # "anthropic" | "openai" | "openrouter"
    key_env: str         # env var name holding the key (for logs)
    key: str             # actual key value
    max_concurrent: int  # rate-limit semaphore size
    rpm_delay: float     # minimum seconds between calls in same worker
    max_tokens: int = 6000
    temperature: float = 0.3
    extra_headers: dict = field(default_factory=dict)

    def has_key(self) -> bool:
        return bool(self.key)


TEACHERS: list[Teacher] = [
    Teacher(
        name="claude_opus_4_7",
        endpoint="https://api.anthropic.com/v1/messages",
        model="claude-opus-4-7-20260101",  # placeholder; update to real alias when live
        api_style="anthropic",
        key_env="ANTHROPIC_API_KEY",
        key=ANTHROPIC_KEY,
        max_concurrent=3,
        rpm_delay=2.0,
        max_tokens=7000,
        temperature=0.3,
        extra_headers={
            "anthropic-version": "2023-06-01",
        },
    ),
    Teacher(
        name="deepseek_reasoner",
        endpoint="https://api.deepseek.com/chat/completions",
        model="deepseek-reasoner",
        api_style="openai",
        key_env="DEEPSEEK_API_KEY",
        key=DEEPSEEK_KEY,
        max_concurrent=4,
        rpm_delay=1.5,
        max_tokens=7000,
        temperature=0.3,
    ),
    Teacher(
        name="cerebras_qwen_235b",
        endpoint="https://api.cerebras.ai/v1/chat/completions",
        model="qwen-3-235b-a22b-instruct-2507",
        api_style="openai",
        key_env="CEREBRAS_API_KEY",
        key=CEREBRAS_KEY,
        max_concurrent=2,
        rpm_delay=2.0,
        max_tokens=6000,
        temperature=0.3,
    ),
]


# Per-teacher semaphores + consecutive-failure tracking (adaptive disable)
_SEMAPHORES: dict[str, Semaphore] = {t.name: Semaphore(t.max_concurrent) for t in TEACHERS}
_CONSECUTIVE_FAILS: dict[str, int] = defaultdict(int)
_DISABLED_TEACHERS: set[str] = set()
_FAIL_LOCK = Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_LOCK = Lock()


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _LOG_LOCK:
        print(line, flush=True)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. "
    "Solve the problem step-by-step with clear, numbered steps. "
    "Parse all examples carefully, explain your reasoning, then put the final "
    "answer inside \\boxed{...}. "
    "Do not include any text after the \\boxed{...}."
)


def build_user_prompt(prompt_text: str, category: Optional[str]) -> str:
    hint = ""
    if category:
        cat = category.lower()
        if "cipher" in cat or "crypt" in cat:
            hint = (
                "\nHint: this is a cryptarithm / substitution cipher. "
                "Build a letter->digit (or letter->letter) mapping from the examples "
                "before applying it to the query.\n"
            )
        elif "equation" in cat or "algebra" in cat:
            hint = (
                "\nHint: extract the algebraic relation from the examples, solve for "
                "the unknown, then verify with substitution.\n"
            )
        elif "bit" in cat or "binary" in cat:
            hint = (
                "\nHint: work in binary / bit-level. Show bit-by-bit transformation.\n"
            )
    return f"{prompt_text}{hint}\n\nReturn exactly one final answer inside \\boxed{{...}}."


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http_post(url: str, headers: dict, body: dict, timeout: int = TIMEOUT_S) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        payload = ""
        try:
            payload = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, payload
    except urllib.error.URLError as e:
        return 0, f"URLError: {e}"
    except Exception as e:
        return 0, f"Exception: {type(e).__name__}: {e}"


def call_teacher(teacher: Teacher, user_prompt: str) -> tuple[bool, str, str]:
    """Call a teacher with retries/backoff. Returns (ok, cot_text, reason)."""
    if not teacher.has_key():
        return False, "", f"no_key({teacher.key_env})"
    if teacher.name in _DISABLED_TEACHERS:
        return False, "", "teacher_disabled"

    if teacher.api_style == "anthropic":
        headers = {
            "x-api-key": teacher.key,
            **teacher.extra_headers,
        }
        body = {
            "model": teacher.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": teacher.max_tokens,
            "temperature": teacher.temperature,
        }
    else:
        # OpenAI-compatible (DeepSeek, Cerebras, OpenRouter, OpenAI)
        headers = {
            "Authorization": f"Bearer {teacher.key}",
            **teacher.extra_headers,
        }
        body = {
            "model": teacher.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": teacher.max_tokens,
            "temperature": teacher.temperature,
        }

    sem = _SEMAPHORES[teacher.name]
    for attempt in range(MAX_RETRIES):
        with sem:
            t0 = time.time()
            status, text = _http_post(teacher.endpoint, headers, body)
            elapsed = time.time() - t0

        if status == 200:
            cot = _extract_completion(text, teacher.api_style)
            if cot:
                _mark_success(teacher.name)
                time.sleep(teacher.rpm_delay)  # soft RPM throttle
                return True, cot, f"ok({elapsed:.1f}s)"
            _mark_failure(teacher.name)
            return False, "", "empty_completion"

        if status == 429:
            wait = BACKOFF_SEQUENCE[min(attempt, len(BACKOFF_SEQUENCE) - 1)]
            log(f"  [{teacher.name}] 429 rate-limited, sleeping {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        if status in (500, 502, 503, 504):
            wait = min(10 * (2 ** attempt), 60)
            log(f"  [{teacher.name}] {status} server error, sleeping {wait}s")
            time.sleep(wait)
            continue

        # Hard failure (4xx not 429, or URLError)
        _mark_failure(teacher.name)
        return False, "", f"http_{status}: {text[:200]}"

    _mark_failure(teacher.name)
    return False, "", "max_retries_exhausted"


def _extract_completion(raw_body: str, api_style: str) -> str:
    try:
        obj = json.loads(raw_body)
    except json.JSONDecodeError:
        return ""
    try:
        if api_style == "anthropic":
            blocks = obj.get("content", [])
            parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            return "\n".join(parts).strip()
        # OpenAI-compatible
        choices = obj.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content") or ""
        # DeepSeek-reasoner has a `reasoning_content` field too
        reasoning = msg.get("reasoning_content") or ""
        combined = (reasoning + "\n\n" + content).strip() if reasoning else content.strip()
        return combined
    except Exception:
        return ""


def _mark_success(teacher_name: str) -> None:
    with _FAIL_LOCK:
        _CONSECUTIVE_FAILS[teacher_name] = 0


def _mark_failure(teacher_name: str) -> None:
    with _FAIL_LOCK:
        _CONSECUTIVE_FAILS[teacher_name] += 1
        if _CONSECUTIVE_FAILS[teacher_name] >= 5 and teacher_name not in _DISABLED_TEACHERS:
            _DISABLED_TEACHERS.add(teacher_name)
            log(f"  [{teacher_name}] DISABLED after 5 consecutive failures")


# ---------------------------------------------------------------------------
# Verifier (Tarefa 2)
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str:
    """Extract content of the LAST \\boxed{...} occurrence (most common convention)."""
    matches = _BOXED_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def _normalize_answer(s: str) -> str:
    s = (s or "").strip()
    # collapse whitespace, drop surrounding quotes
    s = s.strip("\"'` \t\n")
    s = re.sub(r"\s+", " ", s)
    return s


def verify_cot(cot_text: str, prompt: str, gt: str) -> tuple[bool, str]:
    """Multi-layer verification. Returns (passed, reason)."""
    if not cot_text:
        return False, "empty_cot"

    # Layer 1: extract \boxed{} answer and compare to GT
    extracted = extract_boxed(cot_text)
    if not extracted:
        return False, "no_boxed_answer"
    if _normalize_answer(extracted) != _normalize_answer(gt):
        return False, "answer_mismatch"

    # Layer 2: CoT minimum length (avoid trivial "answer: X" without reasoning)
    if len(cot_text) < 200:
        return False, "cot_too_short"

    # Layer 3: CoT mentions examples parsing (the prompt includes few-shot examples)
    low = cot_text.lower()
    if "example" not in low and "exemplo" not in low:
        return False, "no_examples_parsing"

    # Layer 4: CoT has step structure (numbered or "Step" keyword)
    if not re.search(r"(step\s*\d+|^\s*\d+[.:])", cot_text, re.MULTILINE | re.IGNORECASE):
        return False, "no_step_structure"

    return True, "passed"


# ---------------------------------------------------------------------------
# Resume / state (Tarefa 3)
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            data["completed_ids"] = set(data.get("completed_ids", []))
            data.setdefault("stats", {})
            return data
        except Exception as e:
            log(f"[state] corrupted state file ({e}); starting fresh")
    return {"completed_ids": set(), "stats": {}}


def save_state(state: dict[str, Any]) -> None:
    payload = {
        "completed_ids": sorted(state.get("completed_ids", [])),
        "stats": state.get("stats", {}),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("prompt_id") or row.get("row_id") or "")


def row_gt(row: dict[str, Any]) -> str:
    # V71 weak_prompts.jsonl convention: "ground_truth" or "answer"
    return str(row.get("ground_truth") or row.get("answer") or row.get("gt") or "")


def row_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("text") or row.get("question") or "")


def row_category(row: dict[str, Any]) -> Optional[str]:
    return row.get("category") or row.get("task_type") or row.get("subject")


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------
_STATS_LOCK = Lock()
_STATS: dict[str, int] = defaultdict(int)


def bump(key: str, n: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[key] += n


def process_row(row: dict[str, Any]) -> dict[str, Any]:
    rid = row_id(row)
    prompt = row_prompt(row)
    gt = row_gt(row)
    category = row_category(row)
    user_prompt = build_user_prompt(prompt, category)

    passed: list[dict[str, Any]] = []
    attempts_log: list[dict[str, Any]] = []

    for teacher in TEACHERS:
        if teacher.name in _DISABLED_TEACHERS:
            attempts_log.append({"teacher": teacher.name, "skipped": "disabled"})
            continue
        if not teacher.has_key():
            attempts_log.append({"teacher": teacher.name, "skipped": f"no_key({teacher.key_env})"})
            continue

        ok, cot, reason = call_teacher(teacher, user_prompt)
        attempts_log.append({"teacher": teacher.name, "ok": ok, "reason": reason, "len": len(cot)})
        bump(f"calls_{teacher.name}")
        if not ok:
            bump(f"fail_{teacher.name}")
            continue

        verified, vreason = verify_cot(cot, prompt, gt)
        attempts_log[-1]["verified"] = verified
        attempts_log[-1]["verify_reason"] = vreason
        if verified:
            passed.append({"teacher": teacher.name, "cot": cot, "len": len(cot)})
            bump(f"pass_{teacher.name}")
        else:
            bump(f"verify_fail_{vreason}")

    if not passed:
        bump("row_unsolvable")
        return {
            "id": rid,
            "prompt": prompt,
            "ground_truth": gt,
            "category": category,
            "status": "unsolvable",
            "attempts": attempts_log,
        }

    # Pick shortest verified CoT (conciseness)
    best = min(passed, key=lambda p: p["len"])
    bump("row_verified")
    return {
        "id": rid,
        "prompt": prompt,
        "ground_truth": gt,
        "category": category,
        "status": "verified",
        "teacher": best["teacher"],
        "cot": best["cot"],
        "cot_len": best["len"],
        "num_teachers_passed": len(passed),
        "attempts": attempts_log,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def pending_rows(limit: Optional[int] = None) -> tuple[list[dict[str, Any]], int, int]:
    """Return (pending_rows, total, already_solved_count)."""
    all_rows = load_jsonl(WEAK_JSONL)
    solved = load_jsonl(SOLVER_JSONL)
    solved_ids = {row_id(r) for r in solved if row_id(r)}

    state = load_state()
    completed = state.get("completed_ids", set())

    pending = []
    for r in all_rows:
        rid = row_id(r)
        if not rid:
            continue
        if rid in solved_ids:
            continue
        if rid in completed:
            continue
        pending.append(r)

    if limit:
        pending = pending[:limit]
    return pending, len(all_rows), len(solved_ids)


def run(args: argparse.Namespace) -> int:
    pending, total, solved_count = pending_rows(args.limit)
    residual = total - solved_count

    log("=" * 70)
    log(f"V72 ensemble teacher distill")
    log(f"Total weak rows       : {total}")
    log(f"Already solved (v71)  : {solved_count}")
    log(f"Residual to process   : {residual}")
    log(f"Pending this run      : {len(pending)}  (state.completed excluded)")
    log(f"Workers               : {args.workers}")
    log(f"Output JSONL          : {VERIFIED_JSONL}")
    log(f"State file            : {STATE_FILE}")

    available = [t for t in TEACHERS if t.has_key() and t.name not in _DISABLED_TEACHERS]
    missing = [t for t in TEACHERS if not t.has_key()]
    log(f"Teachers ready        : {[t.name for t in available]}")
    if missing:
        log(f"Teachers missing key  : {[(t.name, t.key_env) for t in missing]}")

    if args.preview:
        _print_preview(len(pending), available, missing)
        return 0

    if not available:
        log("ERROR: no teacher keys configured; aborting.")
        return 2

    state = load_state()

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row, r): r for r in pending}
        for fut in as_completed(futures):
            row = futures[fut]
            rid = row_id(row)
            try:
                result = fut.result()
            except Exception as e:
                log(f"[row {rid}] exception: {e}")
                result = {
                    "id": rid,
                    "status": "error",
                    "error": str(e),
                    "prompt": row_prompt(row),
                    "ground_truth": row_gt(row),
                }
                bump("row_error")

            # Persist per-row (resume safety)
            if result.get("status") == "verified":
                append_jsonl(VERIFIED_JSONL, result)
            else:
                append_jsonl(UNSOLVABLE_JSONL, result)

            state["completed_ids"].add(rid)
            done += 1
            if done % 10 == 0 or done == len(pending):
                state["stats"] = dict(_STATS)
                save_state(state)
                rate = done / max(1e-6, (time.time() - t0))
                eta_s = (len(pending) - done) / max(1e-6, rate)
                log(f"[progress] {done}/{len(pending)} | {rate:.2f} row/s | ETA {eta_s/60:.1f} min | verified={_STATS['row_verified']} unsolvable={_STATS['row_unsolvable']}")

    # Final flush
    state["stats"] = dict(_STATS)
    save_state(state)
    log("=" * 70)
    log(f"DONE in {(time.time()-t0)/60:.1f} min")
    log(f"Verified CoTs : {_STATS['row_verified']}")
    log(f"Unsolvable    : {_STATS['row_unsolvable']}")
    log(f"Errors        : {_STATS['row_error']}")
    log(f"Per-teacher stats : {json.dumps(dict(_STATS), indent=2)}")
    return 0


def _print_preview(pending_count: int, available: list[Teacher], missing: list[Teacher]) -> None:
    print("\n" + "=" * 70)
    print("V72 PREVIEW MODE (no API calls made)")
    print("=" * 70)
    total_weak = 959
    solved = 101
    residual = total_weak - solved
    print(f"[preview] {residual} rows pending ({total_weak} - {solved} already solved)")
    teachers_desc = []
    for tier, t in zip(["primary", "secondary", "tertiary"], TEACHERS):
        status = "ready" if t.has_key() else f"MISSING {t.key_env}"
        teachers_desc.append(f"{t.name} ({tier}, {status})")
    print(f"[preview] Teachers: {', '.join(teachers_desc)}")
    print(f"[preview] Parallel workers: 6")
    print(f"[preview] Per-teacher concurrency: {[f'{t.name}={t.max_concurrent}' for t in TEACHERS]}")
    print(f"[preview] Timeout per call: {TIMEOUT_S}s  |  Backoff on 429: {BACKOFF_SEQUENCE}s")

    # Wall-clock estimate
    # assume 6 workers, avg 12s/teacher-call, 3 teachers per row -> ~36s/row but parallelized
    # -> ~6s wall per row amortized. residual * 6s / 60 ~= minutes
    wall_min_lo = residual * 5 / 60
    wall_min_hi = residual * 10 / 60
    print(f"[preview] Expected wall-clock: ~{wall_min_lo:.0f}-{wall_min_hi:.0f} min  (~{wall_min_lo/60:.1f}-{wall_min_hi/60:.1f}h)")

    # Cost estimate (Claude 4.7 ~$15/Mtok out, DeepSeek ~$0.55/Mtok, Cerebras free tier)
    # avg 3k output tokens per CoT, 3 teachers * 858 rows = 2574 CoTs
    # Claude share (~33%): 858 * 3k = 2.57M tokens * $15 = $38 hmm
    # We'll quote conservatively: $3-5 assuming primary has retries and shorter avg
    print(f"[preview] Expected cost: $3-5 (Claude primary + DeepSeek secondary, Cerebras free tier)")
    print(f"[preview] Expected pass rate: 55-70% -> 470-600 CoTs verified")
    print(f"[preview] Verifier gates: answer_match + len>=200 + examples_parsed + step_structure")
    print(f"[preview] Resume state : {STATE_FILE}")
    print(f"[preview] Output JSONL : {VERIFIED_JSONL}")
    print("=" * 70)

    if not WEAK_JSONL.exists():
        print(f"[preview] WARNING: input not found at {WEAK_JSONL}")
        print(f"[preview]          (set KG1_ROOT=<repo> or drop this script into <repo>/scripts/)")
    else:
        delta = residual - pending_count
        if delta > 0:
            print(f"[preview] Note: state file indicates {delta} already completed locally.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="V72 ensemble teacher distillation")
    p.add_argument("--preview", action="store_true", help="Print plan without calling APIs")
    p.add_argument("--all", action="store_true", help="Run on full residual set")
    p.add_argument("--limit", type=int, default=None, help="Only process first N pending rows")
    p.add_argument("--workers", type=int, default=6, help="Thread pool size (default 6)")
    args = p.parse_args()

    if not (args.preview or args.all or args.limit):
        p.print_help()
        return 1
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
