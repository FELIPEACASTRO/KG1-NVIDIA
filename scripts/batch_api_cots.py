#!/usr/bin/env python3
"""
Batch resolve CoTs restantes via DeepSeek + OpenAI.
Resume-capable. Salva em data/api_cots.jsonl.
"""
import json, os, re, sys, time, urllib.request, pandas as pd
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
    text = text.replace("\u2192", "->")  # fix arrow char
    for pat in [r'\\boxed\{([^}]+)\}', r'boxed\{([^}]+)\}']:
        m = re.findall(pat, text)
        if m: return m[-1].strip()
    for pat in [r'[Ff]inal [Aa]nswer[:\s]+["\']?([^"\'\n]+)',
                r'[Dd]ecrypted\s*(?:text)?[:\s]+["\']?([^"\'\n]+)',
                r'[Rr]esult[:\s]+["\']?([^"\'\n]+)',
                r'[Aa]nswer[:\s]+["\']?([^"\'\n]+)']:
        m = re.findall(pat, text)
        if m:
            ans = m[-1].strip().rstrip(".,;:!?").strip("*` ")
            if ans and 1 <= len(ans) < 100:
                return ans
    lines = [l.strip() for l in text.strip().split("\n") if l.strip() and not l.strip().startswith(("#", "Step", "Let", "The ", "So ", "Now"))]
    if lines:
        last = lines[-1].strip("*`\" ").rstrip(".,;:!?")
        if last and 1 <= len(last) < 80: return last
    return ""

def verify(pred, exp):
    pred, exp = str(pred).strip(), str(exp).strip()
    if re.fullmatch(r'[01]+', exp): return pred == exp
    try: return abs(float(pred) - float(exp)) <= max(1e-5, 0.01 * abs(float(exp)))
    except: pass
    return pred.lower().rstrip(".,;:!?") == exp.lower()

def classify(prompt):
    p = prompt.lower()
    if "bit manipulation" in p: return "bit"
    if "encryption" in p: return "cipher"
    if "gravitational" in p: return "gravity"
    if "unit conversion" in p or "measurement" in p: return "unit"
    if "numeral" in p: return "numeral"
    return "equation"

# ============================================================
# MAIN
# ============================================================
base = Path(__file__).resolve().parent.parent
df = pd.read_csv(base / "data" / "train.csv")

# Load existing v51 to find needs-CoT
v51_path = base / "data" / "sft_v51_final.jsonl"
existing = {}
with open(v51_path) as f:
    for line in f:
        item = json.loads(line)
        existing[item["id"]] = item

needs = [{"id": item["id"], "prompt": df[df["id"]==item["id"]].iloc[0]["prompt"],
          "answer": str(df[df["id"]==item["id"]].iloc[0]["answer"]).strip(),
          "family": item["family"]}
         for item in existing.values() if not item.get("has_cot")]

print(f"Need CoTs: {len(needs)}")
from collections import Counter
for f, c in Counter(n["family"] for n in needs).items():
    print(f"  {f}: {c}")

# Resume
output_path = base / "data" / "api_cots.jsonl"
done = set()
if output_path.exists():
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            done.add(r["id"])
remaining = [n for n in needs if n["id"] not in done]
print(f"Already done: {len(done)} | Remaining: {len(remaining)}")

correct = sum(1 for line in open(output_path) if json.loads(line).get("verified")) if output_path.exists() else 0
total_done = len(done)

with open(output_path, "a", encoding="utf-8") as f:
    for i, prob in enumerate(remaining):
        family = prob["family"]
        puzzle_prompt = prob["prompt"]
        expected = prob["answer"]

        # Choose API and prompt
        if family == "cipher":
            prompt = ("Substitution cipher: each letter maps to another consistently.\n"
                      "Build mapping from examples, decrypt test text.\n\n"
                      + puzzle_prompt + "\n\nFinal decrypted text:")
            try:
                resp = call_deepseek(prompt)
            except:
                try: resp = call_openai(prompt)
                except: resp = ""
        elif family == "equation":
            prompt = ("Symbol transformation puzzle. Find the pattern.\n\n"
                      + puzzle_prompt + "\n\nFinal answer:")
            try:
                resp = call_openai(prompt)
            except:
                try: resp = call_deepseek(prompt)
                except: resp = ""
        else:  # bit
            prompt = ("Bit manipulation: each output bit is independent boolean function of inputs.\n\n"
                      + puzzle_prompt + "\n\nFinal 8-bit answer:")
            try:
                resp = call_deepseek(prompt)
            except:
                resp = ""

        ans = extract_answer(resp) if resp else ""
        resp_clean = resp.replace("\u2192", "->") if resp else ""
        verified = verify(ans, expected) if ans else False

        result = {
            "id": prob["id"], "answer": ans, "expected": expected,
            "teacher": "deepseek" if family in ("cipher", "bit") else "openai",
            "verified": verified, "family": family,
            "cot": resp_clean[:2500]
        }
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

        total_done += 1
        if verified: correct += 1

        if (i + 1) % 20 == 0:
            pct = 100 * correct / total_done if total_done > 0 else 0
            print(f"  [{i+1}/{len(remaining)}] verified={correct}/{total_done} ({pct:.0f}%) fam={family}")

        time.sleep(1.0)

print(f"\nDONE: {correct}/{total_done} verified ({100*correct/total_done:.1f}%)")
