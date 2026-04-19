#!/usr/bin/env python3
"""Self-consistency voting inference for Kaggle Nemotron Challenge.

Claude-Opus-4-7 key insight (mega_critique): "Self-consistency voting at
inference. Nobody in your dump mentions it. Temp=0.3, n=5, majority vote on
parsed answer. Easily +2-4pp, and orthogonal to training. This is the missing
piece."

Risk: Kaggle docs say temperature=0.0, max_tokens=7680. IF Kaggle ACTUALLY
enforces temperature=0, self-consistency can't help at submit time.

BUT: this script is useful for:
1. Local pre-submit scoring to validate adapter quality
2. Test-time augmentation during development
3. If your competition container allows custom inference, drop this in

Usage (local eval):
    python scripts/self_consistency_infer.py \\
        --adapter path/to/adapter \\
        --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --prompts path/to/val_prompts.jsonl \\
        --output predictions.jsonl \\
        --n-samples 5 \\
        --temperature 0.3
"""
import argparse
import json
import os
from collections import Counter


def generate_with_sampling(model, tokenizer, prompt, n_samples=5, temperature=0.3,
                           max_new_tokens=4096, top_p=0.95):
    """Generate n_samples completions with sampling."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = []
    for _ in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        outputs.append(text)
    return outputs


def extract_boxed(text: str) -> str:
    """Kaggle scorer regex: r'\\boxed\{([^}]*)(?:\}|$)'"""
    import re
    m = re.search(r"\\boxed\{([^}]*)(?:\}|$)", text)
    return m.group(1).strip() if m else ""


def majority_vote(answers: list) -> tuple:
    """Return (most_common_answer, vote_count) with tie-break to first seen."""
    counter = Counter(a for a in answers if a)
    if not counter:
        return "", 0
    max_count = max(counter.values())
    # Tie-break: first in original order
    for a in answers:
        if counter.get(a) == max_count:
            return a, max_count
    return "", 0


def self_consistency_answer(model, tokenizer, prompt, n_samples=5, temperature=0.3,
                            format_repair_fn=None, category=None):
    """Generate n samples with sampling, extract boxed answers, majority vote."""
    outputs = generate_with_sampling(model, tokenizer, prompt,
                                      n_samples=n_samples, temperature=temperature)
    answers = []
    for text in outputs:
        if format_repair_fn:
            text = format_repair_fn(text, category)
        ans = extract_boxed(text)
        answers.append(ans)
    voted, count = majority_vote(answers)
    return voted, count, answers, outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    p.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    p.add_argument("--prompts", required=True, help="JSONL with {id, question, answer, category}")
    p.add_argument("--output", required=True)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--use-format-repair", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as e:
        print(f"Missing deps: {e}")
        return

    print(f"Loading base: {args.base_model} (NF4)")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    repair_fn = None
    if args.use_format_repair:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from format_auto_repair import repair_boxed_answer
        repair_fn = repair_boxed_answer

    PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

    correct = 0
    total = 0
    with open(args.prompts) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.limit and total >= args.limit:
                break
            prob = json.loads(line)
            qtext = prob.get("question", prob.get("prompt", ""))
            expected = str(prob.get("answer", ""))
            cat = prob.get("category", "unknown")
            pid = prob.get("id", str(total))

            # Build chat-template prompt
            messages = [{"role": "user", "content": qtext + PROMPT_SUFFIX}]
            try:
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            except TypeError:
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

            voted, count, all_ans, raw = self_consistency_answer(
                model, tokenizer, prompt_str,
                n_samples=args.n_samples, temperature=args.temperature,
                format_repair_fn=repair_fn, category=cat,
            )

            is_correct = voted.strip().lower() == expected.strip().lower()
            total += 1
            correct += int(is_correct)

            fout.write(json.dumps({
                "id": pid, "category": cat,
                "question": qtext, "expected": expected,
                "voted_answer": voted, "vote_count": count,
                "n_samples": args.n_samples,
                "all_answers": all_ans, "correct": is_correct,
            }) + "\n")

            if total % 10 == 0:
                print(f"  {total}: acc {100*correct/total:.1f}% | votes_variance = {len(set(all_ans))}")

    print(f"\nFinal: {correct}/{total} ({100*correct/max(1,total):.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
