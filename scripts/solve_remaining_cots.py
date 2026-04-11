#!/usr/bin/env python3
"""
Resolve os 3184 problemas restantes usando APIs em cascata.
Gera CoTs de qualidade para treino SFT.

APIs: DeepSeek-chat → OpenAI GPT-4.1 → Gemini Flash 2.0
Families: cipher (1576) + equation (1555) + bit unsolved (53)
"""
import json, os, re, sys, time, urllib.request, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# API KEYS
# ============================================================
DEEPSEEK_KEY = "sk-bcb0077df2f14e7c9b9eecbf45ae44a5"
OPENAI_KEY = "sk-proj-v00XOfMi0I_YsShv8NXmLocErLAOM7Iax8CGjz0CMiAJwOyXxshPdBGE2oK4Cz8GJwjokUzPDLVT3BlbkFJvKGe6ZIjS3nIphip16siklEzgM-aj5ZzDjINzm1IATjDrPQpf9xbDqInXYPWa_J2eekeJqWHEA"
GEMINI_KEY = "AIzaSyAoPjfzI5SatMLv0HRCxVdAyB-4VW-rm3s"
GROK_KEY = "xai-oabRDoBcQ98p6Npzvfy9zzZrbd3W3ewCIbQ6i0YhnlpAtiudhGK8uHRVn5yV1lSguDZu5PZSNiOJZiHQ"

# ============================================================
# PROMPTS ESPECIALIZADOS POR FAMILIA
# ============================================================

CIPHER_PROMPT = """You are solving a substitution cipher puzzle. Each letter maps to exactly one other letter consistently.

STRATEGY:
1. Align each encrypted word with its decrypted word (same length)
2. Build a mapping table: encrypted_letter -> decrypted_letter
3. Check for conflicts (one encrypted letter mapping to two different decrypted letters = error)
4. Apply the complete mapping to decrypt the test text
5. If some letters are not in the mapping, try to infer from context or partial words

PUZZLE:
{prompt}

Show your mapping table clearly, then apply it to decrypt the test text.
Put your final decrypted text inside \\boxed{{}}.
"""

EQUATION_PROMPT = """You are solving a symbolic equation transformation puzzle. The rules involve:
- Custom operators (symbols like *, +, !, @, etc. may represent different operations)
- Possible digit/symbol substitutions
- Pattern matching from examples

STRATEGY:
1. Look at each example carefully: input -> output
2. Try to identify what each symbol/operator does
3. Check if characters are being substituted (cipher-like)
4. Check if positions matter (positional transformation)
5. Test your hypothesis against ALL examples
6. Apply to the test expression

PUZZLE:
{prompt}

Think very carefully step by step. The answer is likely a short string of symbols.
Put your final answer inside \\boxed{{}}.
"""

BIT_PROMPT = """You are solving a bit manipulation puzzle. 8-bit binary inputs are transformed to 8-bit binary outputs.

STRATEGY:
1. Each output bit is an INDEPENDENT boolean function of the input bits
2. For each output bit position (0-7, left to right):
   - Check if it's always 0 or always 1 (constant)
   - Check if it equals any input bit directly or its NOT
   - Check if it's a binary operation (AND, OR, XOR) of two input bits
3. The transformation might involve bit rotation, shifting, or position-dependent operations
4. Test your rule against ALL examples before applying to the test input

PUZZLE:
{prompt}

Show your analysis for each bit position. Put your final 8-bit binary answer inside \\boxed{{}}.
"""

# ============================================================
# API CALLERS
# ============================================================

def call_deepseek(prompt, timeout=60):
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 2000
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY}"
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]


def call_openai(prompt, timeout=60):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 2000
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_KEY}"
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]


def call_gemini(prompt, timeout=30):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result["candidates"][0]["content"]["parts"][0]["text"]


def call_grok(prompt, timeout=60):
    url = "https://api.x.ai/v1/chat/completions"
    payload = {
        "model": "grok-3-mini-fast",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 2000
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK_KEY}"
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]


# ============================================================
# EXTRACTION + VERIFICATION
# ============================================================

def extract_boxed(text):
    """Extract from \\boxed{...} with multiple patterns."""
    for pattern in [r'\\boxed\{([^}]+)\}', r'\\\\boxed\{([^}]+)\}', r'boxed\{([^}]+)\}']:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    # Fallback: last line that looks like an answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) < 50 and not line.startswith('#'):
            # Remove markdown formatting
            clean = re.sub(r'[*`]', '', line).strip()
            if clean:
                return clean
    return ""


def verify_kaggle(predicted, expected):
    """Kaggle official verify."""
    predicted = str(predicted).strip()
    expected = str(expected).strip()
    if re.fullmatch(r'[01]+', expected):
        return predicted == expected
    try:
        return abs(float(predicted) - float(expected)) <= max(1e-5, 0.01 * abs(float(expected)))
    except:
        pass
    return predicted.lower() == expected.lower()


# ============================================================
# SOLVE WITH CASCADE
# ============================================================

def solve_one(problem, family):
    """Try all APIs in cascade until one gives correct answer."""
    prompt_text = problem['prompt']
    expected = problem['answer']

    # Build specialized prompt
    if family == 'cipher':
        full_prompt = CIPHER_PROMPT.format(prompt=prompt_text)
    elif family == 'equation':
        full_prompt = EQUATION_PROMPT.format(prompt=prompt_text)
    elif family == 'bit':
        full_prompt = BIT_PROMPT.format(prompt=prompt_text)
    else:
        full_prompt = prompt_text + "\n\nSolve step by step. Put answer in \\boxed{}."

    # Cascade: try each API
    apis = [
        ("deepseek", call_deepseek),
        ("grok", call_grok),
        ("openai", call_openai),
        ("gemini", call_gemini),
    ]

    for api_name, api_func in apis:
        try:
            response = api_func(full_prompt)
            answer = extract_boxed(response)

            if answer and verify_kaggle(answer, expected):
                return {
                    'id': problem['id'],
                    'answer': answer,
                    'cot': response[:3000],
                    'teacher': api_name,
                    'verified': True,
                    'family': family
                }
        except Exception as e:
            continue
        time.sleep(0.3)

    # No API got it right — generate synthetic CoT with known answer
    synthetic_cot = f"After careful analysis of the pattern in the examples, the answer is {expected}."
    return {
        'id': problem['id'],
        'answer': expected,
        'cot': synthetic_cot,
        'teacher': 'synthetic',
        'verified': False,
        'family': family
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base / "data" / "train.csv")

    # Load existing v51 data to find what needs CoTs
    v51_path = base / "data" / "sft_v51_final.jsonl"
    existing = {}
    with open(v51_path) as f:
        for line in f:
            item = json.loads(line)
            existing[item['id']] = item

    # Find problems without CoTs
    needs_cot = []
    for item_id, item in existing.items():
        if not item.get('has_cot', False):
            row = df[df['id'] == item_id].iloc[0]
            needs_cot.append({
                'id': item_id,
                'prompt': row['prompt'],
                'answer': str(row['answer']).strip(),
                'family': item['family']
            })

    print(f"Problems needing CoTs: {len(needs_cot)}")
    from collections import Counter
    fam_counts = Counter(p['family'] for p in needs_cot)
    for fam, cnt in sorted(fam_counts.items()):
        print(f"  {fam}: {cnt}")

    # Output file (resume-capable)
    output_path = base / "data" / "api_cots.jsonl"
    done_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r['id'])
    remaining = [p for p in needs_cot if p['id'] not in done_ids]
    print(f"\nAlready done: {len(done_ids)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        sys.exit(0)

    # Process
    correct = 0
    total = 0
    with open(output_path, 'a', encoding='utf-8') as f:
        for i, prob in enumerate(remaining):
            result = solve_one(prob, prob['family'])
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

            total += 1
            if result['verified']:
                correct += 1

            if (i + 1) % 5 == 0:
                pct = 100 * correct / total if total > 0 else 0
                print(f"  [{i+1}/{len(remaining)}] verified={correct}/{total} ({pct:.0f}%) "
                      f"last={result['teacher']} fam={prob['family']}")

            time.sleep(0.5)  # rate limit

    print(f"\nFINAL: {correct}/{total} verified ({100*correct/total:.1f}%)")
