#!/usr/bin/env python3
"""Use Nemotron-Cascade-2-30B-A3B as TEACHER for CoT distillation.

Source: huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B (IMO/IOI/ICPC Gold 2025)
License: NVIDIA Open Model License (commercial use + redistribution OK)

CPMP Kaggle host (#688360 update) confirmed distillation from OPEN models is
allowed. Cascade-2 is fully open license, unlike GPT/Claude (ToS block).

Advantage over previous teachers (Gemini 0.81, DeepSeek 0.74):
- Same NemotronH architecture family as our Nano-30B
- Specialized in reasoning (Gold in math competitions)
- Competition-legal distillation target

Usage:
    python scripts/cascade2_teacher_distill.py \\
        --problems path/to/problems_needing_cots.jsonl \\
        --output path/to/cascade2_distilled_cots.jsonl \\
        --batch-size 4 \\
        --max-new-tokens 4096
"""
import argparse
import json
import os
from typing import Iterator


def generate_cot_with_cascade2(problem_text: str, model, tokenizer, max_new_tokens: int = 4096) -> str:
    """Generate a CoT+answer using Cascade-2 as teacher."""
    import torch

    prompt_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
    full_prompt = problem_text + prompt_suffix

    messages = [{"role": "user", "content": full_prompt}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=True,
        ).to(model.device)
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=max_new_tokens,
            temperature=0.0, top_p=1.0, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return response


def verify_answer(cot_response: str, expected_answer: str) -> bool:
    """Check if \\boxed{} in response matches expected answer (Kaggle scorer style)."""
    import re
    # Scorer regex: stops at first `}`
    match = re.search(r"\\boxed\{([^}]*)(?:\}|$)", cot_response)
    if not match:
        return False
    extracted = match.group(1).strip()
    return extracted.replace(" ", "").lower() == expected_answer.replace(" ", "").lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", required=True, help="JSONL with {id, question, answer, category}")
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="nvidia/Nemotron-Cascade-2-30B-A3B")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--verify", action="store_true",
                   help="Only keep CoTs where final \\boxed{} matches ground truth")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing deps: {e}")
        print("Install: pip install torch transformers bitsandbytes accelerate")
        return

    print(f"Loading Cascade-2 teacher: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load with NF4 for memory efficiency (Cascade-2 is 30B)
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, quantization_config=bnb,
    )
    model.eval()
    print("Teacher loaded.")

    # Process
    n_generated = 0
    n_verified = 0
    with open(args.problems) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.limit and n_generated >= args.limit:
                break
            problem = json.loads(line)
            qtext = problem.get("question", problem.get("prompt", ""))
            ans = str(problem.get("answer", ""))
            cat = problem.get("category", "unknown")
            pid = problem.get("id", str(n_generated))

            try:
                cot = generate_cot_with_cascade2(qtext, model, tokenizer, args.max_new_tokens)
            except Exception as e:
                print(f"  Failed {pid}: {str(e)[:100]}")
                continue

            verified = verify_answer(cot, ans)
            if args.verify and not verified:
                continue

            fout.write(json.dumps({
                "id": pid, "category": cat, "question": qtext, "answer": ans,
                "teacher_cot": cot, "verified": verified,
                "teacher_model": args.model_id,
            }) + "\n")
            n_generated += 1
            if verified:
                n_verified += 1

            if n_generated % 10 == 0:
                print(f"  {n_generated} generated | {n_verified} verified | {100*n_verified/max(1,n_generated):.1f}%")

    print(f"\nTotal generated: {n_generated}")
    print(f"Verified (matches ground truth): {n_verified} ({100*n_verified/max(1,n_generated):.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
