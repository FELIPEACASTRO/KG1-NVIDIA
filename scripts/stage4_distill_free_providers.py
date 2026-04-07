"""
Stage 4 Distillation Pipeline - FREE PROVIDER ROUND-ROBIN
==========================================================
Roda em PARALELO ao treino v42 H100. Gera CoTs de alta qualidade
usando providers gratuitos: NVIDIA NIM, Together AI (Llama-Nemotron),
OpenRouter (Nemotron-3-Super 120B free), Cerebras (gpt-oss/qwen3).

Formato de saida COMPATIVEL com data/distilled/distilled_*_*.jsonl
existente (mesmas chaves: id, family, prompt, answer, predicted,
correct, response, thinking, tokens, teacher, model).

Uso:
  # Geracao completa estratificada (5000 amostras)
  python scripts/stage4_distill_free_providers.py

  # Apenas equation_transform
  python scripts/stage4_distill_free_providers.py --family equation --target 1500

  # Forca um provider especifico
  python scripts/stage4_distill_free_providers.py --provider nim_official

  # Dry-run: imprime estrategia, sem chamar APIs
  python scripts/stage4_distill_free_providers.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

import pandas as pd

# ============================================================
# CONFIG: paths, keys
# ============================================================
def _find_repo_root() -> Path:
    """Walk up from this file looking for the canonical KG1 root.

    Works whether the script is invoked from the worktree under
    .claude/worktrees/<name>/scripts or from the main scripts/ folder.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "data" / "train.csv"
        if candidate.exists():
            return parent
    # Fallback for when the file is launched from a worktree without data/.
    fallback = here.parent.parent
    return fallback


BASE_DIR = Path(os.environ.get("KG1_ROOT", str(_find_repo_root())))
TRAIN_CSV = BASE_DIR / "data" / "train.csv"
CLASSIFIED_CSV = BASE_DIR / "data" / "train_classified.csv"
OUTPUT_DIR = BASE_DIR / "data" / "distilled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTPUT_DIR / "stage4_state.json"

# Keys -- NUNCA fazer commit destes valores; ler do env primeiro.
NIM_KEY = os.environ.get("NVIDIA_NIM_KEY") or os.environ.get("NVIDIA_API_KEY", "")
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ============================================================
# PROVIDER REGISTRY (Apr 2026 free-tier numbers)
# ============================================================
@dataclass
class Provider:
    name: str
    endpoint: str
    auth_header: str  # full Bearer ... string
    model: str
    rpm: int          # requests per minute (free tier)
    tpd: int          # tokens per day cap (0 = unlimited / unknown)
    max_ctx: int      # max context tokens
    max_output: int   # safe max output for CoT
    extra_headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


def build_providers() -> list[Provider]:
    providers = [
        Provider(
            name="nim_official",
            endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
            auth_header=f"Bearer {NIM_KEY}",
            model="nvidia/nemotron-cascade-2-30b-a3b",
            rpm=40,
            tpd=0,            # 1000 inference credits, no hard token cap
            max_ctx=128_000,
            max_output=4096,
            enabled=bool(NIM_KEY),
        ),
        Provider(
            name="together_llama_nemotron",
            endpoint="https://api.together.xyz/v1/chat/completions",
            auth_header=f"Bearer {TOGETHER_KEY}",
            model="nvidia/Llama-3.3-Nemotron-Super-49B-v1.5",
            rpm=20,
            tpd=0,            # uses $25 free credit, billed per-token
            max_ctx=32_768,
            max_output=4096,
            enabled=bool(TOGETHER_KEY),
        ),
        Provider(
            name="openrouter_nemotron3",
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            auth_header=f"Bearer {OPENROUTER_KEY}",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            rpm=20,
            tpd=0,            # 1000 RPD if >=$10 credit, else 50 RPD
            max_ctx=262_144,
            max_output=4096,
            extra_headers={
                "HTTP-Referer": "https://www.kaggle.com/competitions/nvidia-nemotron",
                "X-Title": "KG1 Stage 4 Distillation",
            },
            enabled=bool(OPENROUTER_KEY),
        ),
        Provider(
            name="cerebras_gptoss",
            endpoint="https://api.cerebras.ai/v1/chat/completions",
            auth_header=f"Bearer {CEREBRAS_KEY}",
            model="gpt-oss-120b",
            rpm=30,
            tpd=1_000_000,    # hard 1M tokens/day reset
            max_ctx=8_192,    # FREE tier ctx cap
            max_output=2048,
            enabled=bool(CEREBRAS_KEY),
        ),
    ]
    return [p for p in providers if p.enabled]


# ============================================================
# FAMILY-AWARE PROMPTING (mirrors distill_all_families.py logic)
# ============================================================
SYSTEM_PROMPT = (
    "You are an expert puzzle solver. For every problem, write a thorough "
    "chain-of-thought between <think> and </think> tags, then the FINAL answer "
    "in \\boxed{ ... }. Be deterministic. Never guess; verify each step against "
    "the provided examples before committing."
)

FAMILY_USER_TEMPLATES: dict[str, str] = {
    "equation": (
        "Solve this equation transformation puzzle.\n"
        "Plan: (1) detect digit substitutions; (2) define every custom operator "
        "as a Python-like function; (3) verify your operator definitions on each "
        "example; (4) apply them to the test expression and reverse digits if the "
        "examples show that pattern.\n\n{prompt}\n\nFinal answer in \\boxed{{}}."
    ),
    "bit": (
        "Solve this 8-bit transformation puzzle.\n"
        "Lay each example out as 8 bits, b7..b0. Identify the rule per bit. "
        "Test rotates, shifts, XOR, AND, OR, NOT, swap_nibbles, reverse_bits, "
        "majority and choice. Verify against ALL examples.\n\n{prompt}\n\n"
        "Final answer (8-bit binary) in \\boxed{{}}."
    ),
    "cipher": (
        "Solve this text-encryption puzzle.\n"
        "Align inputs to outputs letter by letter, compute (out-in) mod 26, "
        "look for constant, positional, or keyword cipher patterns, then decrypt "
        "the test text and explain each step.\n\n{prompt}\n\nFinal answer in \\boxed{{}}."
    ),
    "gravity": (
        "Use d = 0.5 g t^2, so g = 2d/t^2. Compute g for every example, average "
        "them, then plug t into d = 0.5 g t^2. Round to two decimals.\n\n"
        "{prompt}\n\nFinal answer in \\boxed{{}}."
    ),
    "unit": (
        "Compute the conversion factor f = output/input from each example, "
        "verify it is consistent, then apply f to the test value. Round to two "
        "decimals.\n\n{prompt}\n\nFinal answer in \\boxed{{}}."
    ),
    "numeral": (
        "Detect the target numeral system (Roman, binary, hex, base-n) from "
        "the examples, verify on each, then convert the test number.\n\n"
        "{prompt}\n\nFinal answer in \\boxed{{}}."
    ),
}


def classify_family(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p:
        return "bit"
    if "gravitational" in p or "gravity" in p:
        return "gravity"
    if "unit conversion" in p:
        return "unit"
    if "numeral system" in p:
        return "numeral"
    if "encryption" in p or "cipher" in p:
        return "cipher"
    if "equation" in p or "transformation rules" in p:
        return "equation"
    return "other"


# ============================================================
# SAMPLING STRATEGY (5000 stratified, weighted by Felipe's gaps)
# ============================================================
TARGET_PER_FAMILY = {
    "equation": 1500,   # Felipe's 12.2% gap -- HEAVY
    "bit":      1500,   # Felipe at 80% but community ~10% -- LEVERAGE
    "cipher":    800,
    "gravity":   400,
    "unit":      400,
    "numeral":   400,
}
GLOBAL_TARGET = sum(TARGET_PER_FAMILY.values())  # = 5000


def load_train() -> pd.DataFrame:
    if CLASSIFIED_CSV.exists():
        df = pd.read_csv(CLASSIFIED_CSV)
        if "family" not in df.columns:
            df["family"] = df["prompt"].apply(classify_family)
        # Normalize family naming used in distilled/*.jsonl
        family_alias = {
            "bit_manipulation": "bit",
            "text_encryption": "cipher",
            "gravity_constant": "gravity",
            "unit_conversion": "unit",
            "numeral_system": "numeral",
            "equation_transform": "equation",
        }
        df["family"] = df["family"].replace(family_alias)
    else:
        df = pd.read_csv(TRAIN_CSV)
        df["family"] = df["prompt"].apply(classify_family)
    return df


def stratified_sample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    chunks = []
    for family, target in TARGET_PER_FAMILY.items():
        pool = df[df["family"] == family]
        if pool.empty:
            print(f"[WARN] no rows for family={family}")
            continue
        idx = list(pool.index)
        rng.shuffle(idx)
        chunks.append(pool.loc[idx[:target]])
    return pd.concat(chunks, ignore_index=True)


# ============================================================
# RATE LIMIT BUDGETING
# ============================================================
class RateLimiter:
    """Per-provider token-bucket: enforces RPM and (optionally) TPD."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.lock = Lock()
        self.window: list[float] = []
        self.tokens_today = 0
        self.day_started = time.time()

    def acquire(self, est_tokens: int) -> None:
        while True:
            with self.lock:
                now = time.time()
                if now - self.day_started > 86_400:
                    self.tokens_today = 0
                    self.day_started = now
                self.window = [t for t in self.window if now - t < 60]
                if (
                    len(self.window) < self.provider.rpm
                    and (self.provider.tpd == 0 or self.tokens_today + est_tokens <= self.provider.tpd)
                ):
                    self.window.append(now)
                    self.tokens_today += est_tokens
                    return
                wait = max(0.1, 60 - (now - self.window[0]) if self.window else 1.0)
            time.sleep(wait)


# ============================================================
# HTTP CALL
# ============================================================
def call_provider(provider: Provider, family: str, prompt: str) -> dict | None:
    user_prompt = FAMILY_USER_TEMPLATES.get(family, "{prompt}").format(prompt=prompt)
    body = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": provider.max_output,
        "top_p": 1.0,
    }
    headers = {
        "Authorization": provider.auth_header,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    headers.update(provider.extra_headers)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(provider.endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(exc)
        return {"error": f"HTTP {exc.code}: {err_body[:200]}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    choice = payload.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
    usage = payload.get("usage", {}) or {}
    return {
        "text": text,
        "thinking": reasoning,
        "tokens": usage.get("total_tokens", 0),
    }


# ============================================================
# BOXED EXTRACTION + VALIDATION
# ============================================================
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def normalise(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).lower()


def verify(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    if normalise(predicted) == normalise(gold):
        return True
    try:
        return abs(float(predicted) - float(gold)) < 1e-2
    except (TypeError, ValueError):
        return False


# ============================================================
# RESUMABLE STATE
# ============================================================
class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()
        self.done: set[str] = set()
        self.stats: dict[str, dict[str, int]] = {}
        if path.exists():
            data = json.loads(path.read_text())
            self.done = set(data.get("done", []))
            self.stats = data.get("stats", {})

    def mark(self, id_: str, family: str, ok: bool) -> None:
        with self.lock:
            self.done.add(id_)
            bucket = self.stats.setdefault(family, {"ok": 0, "fail": 0})
            bucket["ok" if ok else "fail"] += 1

    def flush(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"done": sorted(self.done), "stats": self.stats}, indent=2))
            tmp.replace(self.path)


# ============================================================
# WORKER PIPELINE
# ============================================================
def worker(
    row: pd.Series,
    providers: list[Provider],
    limiters: dict[str, RateLimiter],
    out_paths: dict[str, Path],
    out_locks: dict[str, Lock],
    state: State,
) -> tuple[str, bool, str]:
    """Tries each provider in order until one returns a verified CoT."""
    family = row["family"]
    pid = row["id"]
    if pid in state.done:
        return pid, False, "skipped"

    gold = str(row["answer"])
    prompt = str(row["prompt"])
    last_error = ""

    for provider in providers:
        limiter = limiters[provider.name]
        limiter.acquire(est_tokens=2500)
        result = call_provider(provider, family, prompt)
        if not result:
            last_error = "empty"
            continue
        if "error" in result:
            last_error = result["error"]
            continue
        predicted = extract_boxed(result["text"])
        ok = verify(predicted, gold)
        record = {
            "id": pid,
            "family": family,
            "prompt": prompt,
            "answer": gold,
            "predicted": predicted,
            "correct": ok,
            "response": result["text"],
            "thinking": result["thinking"],
            "tokens": result["tokens"],
            "teacher": provider.name,
            "model": provider.model,
        }
        if ok:
            path = out_paths[family]
            with out_locks[family]:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            state.mark(pid, family, True)
            return pid, True, provider.name
        last_error = f"wrong via {provider.name}: predicted={predicted!r}"
    state.mark(pid, family, False)
    return pid, False, last_error


# ============================================================
# DRIVER
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 free-provider distillation")
    parser.add_argument("--family", choices=list(TARGET_PER_FAMILY) + ["all"], default="all")
    parser.add_argument("--target", type=int, default=0,
                        help="Override target count for the chosen family")
    parser.add_argument("--provider", default="all",
                        help="Restrict to a single provider name (see registry)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    providers = build_providers()
    if args.provider != "all":
        providers = [p for p in providers if p.name == args.provider]
    if not providers:
        sys.exit("[ERR] no providers enabled. Set NVIDIA_NIM_KEY/TOGETHER_API_KEY/...")

    print("\n=== Stage 4 free-provider distillation ===")
    print(f"providers: {[p.name for p in providers]}")
    print(f"target per family: {TARGET_PER_FAMILY}")

    df = load_train()
    if args.family != "all":
        df = df[df["family"] == args.family]
        target = args.target or TARGET_PER_FAMILY[args.family]
        rng = random.Random(args.seed)
        idx = list(df.index)
        rng.shuffle(idx)
        sample = df.loc[idx[:target]]
    else:
        sample = stratified_sample(df, seed=args.seed)

    print(f"loaded {len(df)} rows, sampled {len(sample)} for distillation")
    print(f"family breakdown: {sample['family'].value_counts().to_dict()}")

    if args.dry_run:
        print("[dry-run] no API calls. exiting.")
        return

    limiters = {p.name: RateLimiter(p) for p in providers}
    out_paths = {
        family: OUTPUT_DIR / f"distilled_{family}_stage4.jsonl"
        for family in TARGET_PER_FAMILY
    }
    out_locks = {family: Lock() for family in out_paths}
    state = State(STATE_FILE)

    rows = list(sample.to_dict("records"))
    rows = [pd.Series(r) for r in rows]
    rows = [r for r in rows if r["id"] not in state.done]
    print(f"after resume filter: {len(rows)} rows to process")

    total = len(rows)
    done_count = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(worker, row, providers, limiters, out_paths, out_locks, state)
            for row in rows
        ]
        for fut in as_completed(futures):
            pid, ok, info = fut.result()
            done_count += 1
            if done_count % 25 == 0:
                state.flush()
                rate = done_count / max(time.time() - t0, 1.0)
                print(f"[{done_count}/{total}] last={pid} ok={ok} info={info[:60]} "
                      f"rate={rate:.2f}/s stats={state.stats}")
    state.flush()

    print("\n=== final stats ===")
    print(json.dumps(state.stats, indent=2))
    print(f"outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
