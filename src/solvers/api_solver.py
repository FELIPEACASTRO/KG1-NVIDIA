#!/usr/bin/env python3
"""
API Solver — Usa Gemini/DeepSeek/OpenAI para resolver problemas
que os solvers deterministicos nao conseguem.

Cascade: Gemini Flash 2.0 (FREE) -> DeepSeek Reasoner -> GPT-4.1

Gera CoTs de alta qualidade, verifica respostas contra o ground truth.
Resume-capable: pula problemas ja resolvidos.
"""

import json, os, re, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# API KEYS
# ============================================================
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# ============================================================
# PROMPT TEMPLATE
# ============================================================

SYSTEM_PROMPT = """You are an expert puzzle solver. Solve the puzzle step by step.
Show your reasoning clearly, then put your final answer inside \\boxed{}.
Be precise: for numbers use the exact format shown in examples (e.g., 2 decimal places for floats).
For binary strings, output exactly 8 bits. For text, match case exactly."""

def make_prompt(puzzle_prompt: str) -> str:
    return f"{puzzle_prompt}\n\nSolve step by step, then put your final answer inside \\boxed{{}}."


# ============================================================
# API CALLS
# ============================================================

def call_gemini(prompt: str, timeout: int = 30) -> str:
    """Call Gemini Flash 2.0 (free tier)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": make_prompt(prompt)}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return text
    except Exception as e:
        return f"ERROR: {e}"


def call_deepseek(prompt: str, timeout: int = 60) -> str:
    """Call DeepSeek Reasoner."""
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(prompt)}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY}"
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"]
            return text
    except Exception as e:
        return f"ERROR: {e}"


def call_openai(prompt: str, timeout: int = 60) -> str:
    """Call OpenAI GPT-4.1."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(prompt)}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_KEY}"
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"]
            return text
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# EXTRACT ANSWER FROM LLM RESPONSE
# ============================================================

def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...}."""
    # Find last \boxed{...}
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "answer is: ..."
    m = re.search(r'(?:answer|result)\s*(?:is|=)\s*[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('`').strip('"').strip("'")
    return ""


# ============================================================
# VERIFY ANSWER (same as Kaggle)
# ============================================================

def verify_answer(predicted: str, expected: str) -> bool:
    """Verify using Kaggle's exact logic."""
    predicted = predicted.strip()
    expected = expected.strip()

    # Binary exact match
    if re.fullmatch(r'[01]+', expected):
        return predicted == expected

    # Float tolerance
    try:
        p_float = float(predicted)
        e_float = float(expected)
        return abs(p_float - e_float) <= max(1e-5, 1e-2 * abs(e_float))
    except (ValueError, TypeError):
        pass

    # Text case-insensitive
    return predicted.lower() == expected.lower()


# ============================================================
# SOLVE WITH CASCADE
# ============================================================

def solve_with_apis(prompt: str, expected: str = None, max_attempts: int = 3):
    """
    Try APIs in cascade until correct answer found.
    Returns (answer, cot, teacher_name, is_verified).
    """
    teachers = [
        ("gemini", call_gemini),
        ("deepseek", call_deepseek),
        ("openai", call_openai),
    ]

    for teacher_name, teacher_func in teachers:
        try:
            response = teacher_func(prompt)
            if response.startswith("ERROR:"):
                continue

            answer = extract_boxed(response)
            if not answer:
                continue

            # If we have expected answer, verify
            if expected:
                if verify_answer(answer, expected):
                    return answer, response, teacher_name, True
                # Try harder: strip quotes, whitespace
                cleaned = answer.strip('"').strip("'").strip()
                if verify_answer(cleaned, expected):
                    return cleaned, response, teacher_name, True
            else:
                # No expected answer — return first non-empty
                return answer, response, teacher_name, False

        except Exception as e:
            continue

    return None, "", "none", False


# ============================================================
# BATCH SOLVER
# ============================================================

def batch_solve(problems: list, output_path: str, workers: int = 4):
    """
    Solve a batch of problems using API cascade.

    problems: list of dicts with 'id', 'prompt', 'answer' (ground truth)
    output_path: JSON file to save results (resume-capable)
    """
    # Load existing results for resume
    results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                results[r['id']] = r

    remaining = [p for p in problems if p['id'] not in results]
    print(f"Total: {len(problems)} | Already done: {len(results)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return results

    correct = sum(1 for r in results.values() if r.get('verified'))
    total_done = len(results)

    with open(output_path, 'a') as f:
        for i, prob in enumerate(remaining):
            answer, cot, teacher, verified = solve_with_apis(
                prob['prompt'], prob.get('answer')
            )

            result = {
                'id': prob['id'],
                'answer': answer or '',
                'expected': prob.get('answer', ''),
                'teacher': teacher,
                'verified': verified,
                'cot': cot[:2000] if cot else '',  # truncate CoT for storage
            }
            f.write(json.dumps(result) + '\n')
            f.flush()

            total_done += 1
            if verified:
                correct += 1

            if (i + 1) % 10 == 0:
                pct = 100 * correct / total_done if total_done > 0 else 0
                print(f"  [{i+1}/{len(remaining)}] correct={correct}/{total_done} ({pct:.1f}%) "
                      f"last_teacher={teacher}")

            # Rate limit
            time.sleep(0.3)

    return results


# ============================================================
# MAIN — Solve unsolved problems
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.solvers.all_families_solver import MasterSolver, classify_family

    base = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(base / "data" / "train.csv")

    print("Phase 1: Running deterministic solvers...")
    solver = MasterSolver()

    solved = []
    unsolved = []

    for idx, row in df.iterrows():
        answer, cot, family, method = solver.solve(row['prompt'])
        expected = str(row['answer']).strip()

        if method == 'solver' and answer and answer.strip() == expected:
            solved.append({
                'id': row['id'],
                'prompt': row['prompt'],
                'answer': expected,
                'cot': cot,
                'family': family,
                'method': 'solver'
            })
        else:
            unsolved.append({
                'id': row['id'],
                'prompt': row['prompt'],
                'answer': expected,
                'family': family,
            })

    print(f"Deterministic: {len(solved)} solved, {len(unsolved)} need LLM")
    print()

    # Family breakdown of unsolved
    from collections import Counter
    fam_counts = Counter(p['family'] for p in unsolved)
    for fam, cnt in sorted(fam_counts.items()):
        print(f"  Unsolved {fam}: {cnt}")
    print()

    print("Phase 2: Solving with APIs (Gemini -> DeepSeek -> OpenAI)...")
    output_path = str(base / "data" / "api_solved.jsonl")
    batch_solve(unsolved, output_path)

    # Count final results
    api_results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                api_results[r['id']] = r

    api_correct = sum(1 for r in api_results.values() if r.get('verified'))
    print()
    print(f"API solved: {api_correct}/{len(api_results)} verified correct")
    print(f"TOTAL: {len(solved) + api_correct}/{len(df)} ({100*(len(solved)+api_correct)/len(df):.1f}%)")
