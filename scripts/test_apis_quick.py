#!/usr/bin/env python3
"""Quick test: DeepSeek cipher + OpenAI equation with improved extraction."""
import json, re, time, urllib.request, pandas as pd, sys
from pathlib import Path

DEEPSEEK_KEY = "sk-bcb0077df2f14e7c9b9eecbf45ae44a5"
OPENAI_KEY = "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"

def call_deepseek(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 1500}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def call_openai(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 1500}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_KEY}"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def extract_answer(text):
    """Extract answer from LLM response using multiple patterns."""
    # boxed patterns
    for pat in [r'\\boxed\{([^}]+)\}', r'boxed\{([^}]+)\}']:
        m = re.findall(pat, text)
        if m:
            return m[-1].strip()
    # Direct answer patterns
    for pat in [r'[Ff]inal [Aa]nswer[:\s]+["\']?([^"\'\n]+)',
                r'[Dd]ecrypted\s+text[:\s]+["\']?([^"\'\n]+)',
                r'[Rr]esult[:\s]+["\']?([^"\'\n]+)',
                r'[Oo]utput[:\s]+["\']?([^"\'\n]+)',
                r'[Aa]nswer[:\s]+["\']?([^"\'\n]+)']:
        m = re.findall(pat, text)
        if m:
            ans = m[-1].strip().rstrip(".,;:!?").strip("*` ")
            if ans and 2 <= len(ans) < 100:
                return ans
    # Last line fallback
    lines = [l.strip() for l in text.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    if lines:
        last = lines[-1].strip("*`\" ").rstrip(".,;:!?")
        if last and 2 <= len(last) < 100:
            return last
    return ""

def verify(pred, exp):
    p = str(pred).strip().lower().rstrip(".,;:!?")
    e = str(exp).strip().lower()
    return p == e

# Load data
base = Path(__file__).resolve().parent.parent
df = pd.read_csv(base / "data" / "train.csv")

# CIPHER with DeepSeek
cipher = df[df["prompt"].str.contains("encryption", case=False, na=False)].head(10)
print("=== DeepSeek CIPHER (10) ===")
c_ok = 0
for i, (idx, row) in enumerate(cipher.iterrows()):
    prompt = (
        "This is a letter substitution cipher. Each letter maps to one other letter consistently.\n"
        "Build the complete mapping from examples, then decrypt the test text.\n\n"
        + row["prompt"] + "\n\n"
        "Give ONLY the decrypted text as your final answer on the last line."
    )
    try:
        resp = call_deepseek(prompt)
        ans = extract_answer(resp)
        expected = str(row["answer"]).strip()
        match = verify(ans, expected)
        if match:
            c_ok += 1
        status = "OK" if match else "X"
        print(f"  [{i+1}] {status} exp={expected[:35]:<35s} got={str(ans)[:35]}")
    except Exception as e:
        print(f"  [{i+1}] ERR: {e}")
    time.sleep(1.5)
print(f"DeepSeek cipher: {c_ok}/10")
print()

# EQUATION with OpenAI
equation = df[df["prompt"].str.contains("transformation rules", case=False, na=False)].head(10)
print("=== OpenAI EQUATION (10) ===")
e_ok = 0
for i, (idx, row) in enumerate(equation.iterrows()):
    prompt = (
        "Solve this symbol transformation puzzle. Identify the pattern from examples.\n\n"
        + row["prompt"] + "\n\n"
        "Give ONLY the result string as your final answer on the last line."
    )
    try:
        resp = call_openai(prompt)
        ans = extract_answer(resp)
        expected = str(row["answer"]).strip()
        match = verify(ans, expected)
        if match:
            e_ok += 1
        status = "OK" if match else "X"
        print(f"  [{i+1}] {status} exp={expected[:25]:<25s} got={str(ans)[:25]}")
    except Exception as e:
        print(f"  [{i+1}] ERR: {e}")
    time.sleep(1)
print(f"OpenAI equation: {e_ok}/10")

print(f"\nTOTAL: cipher={c_ok}/10 equation={e_ok}/10")
