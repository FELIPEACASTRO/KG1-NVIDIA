"""
Build v49 training data: merge solver + distilled + type-weighted sampling.

Strategy:
1. Load solver_augmented_train.jsonl (6540 rows)
2. Load ALL distilled files (equation/bit/cipher)
3. For each example, use BEST available CoT (distilled > solver)
4. Apply konbu17-style type-weighted sampling:
   - Bit: ALL available (max)
   - Equation: ALL available (max)
   - Cipher: 700
   - Unit: 700
   - Gravity: 400
   - Numeral: 300

Output: data/solver_augmented_train_v49.jsonl
"""
import json
import os
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SOLVER_JSONL = BASE_DIR / "data" / "solver_augmented_train.jsonl"
DISTILLED_DIR = BASE_DIR / "data" / "distilled"
OUTPUT = BASE_DIR / "data" / "solver_augmented_train_v49.jsonl"

# konbu17-style type-weighted sampling caps
FAMILY_CAPS = {
    "bit_manipulation": 99999,      # ALL available
    "equation_transform": 99999,    # ALL available
    "text_encryption": 700,
    "unit_conversion": 700,
    "gravity_constant": 400,
    "numeral_system": 300,
}

SEED = 123
random.seed(SEED)


def prompt_hash(p):
    return hashlib.md5(p.strip().encode()).hexdigest()


def load_distilled():
    """Load all distilled files, index by prompt hash. Keep BEST per prompt."""
    distilled = {}
    for fname in os.listdir(DISTILLED_DIR):
        if not fname.endswith(".jsonl"):
            continue
        fpath = DISTILLED_DIR / fname
        teacher = fname.split("_")[-1].replace(".jsonl", "")
        # Priority: deepseek=3, o4-mini=2, gemini=1
        priority = {"deepseek": 3, "o4-mini": 2, "gemini": 1}.get(teacher, 0)

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if not obj.get("correct"):
                        continue
                    ph = prompt_hash(obj.get("prompt", ""))
                    response = obj.get("response", "")
                    thinking = obj.get("thinking", "")
                    cot = thinking if thinking and len(thinking) > len(response) else response

                    existing = distilled.get(ph)
                    if existing is None or priority > existing["priority"] or (
                        priority == existing["priority"] and len(cot) > len(existing["cot"])
                    ):
                        distilled[ph] = {
                            "cot": cot,
                            "teacher": teacher,
                            "priority": priority,
                            "answer": obj.get("answer", ""),
                        }
                except Exception:
                    pass

    return distilled


def main():
    print("=" * 60)
    print("BUILD v49 TRAINING DATA")
    print("=" * 60)

    # 1. Load distilled
    distilled = load_distilled()
    print(f"\nDistilled unique prompts: {len(distilled)}")
    teacher_counts = Counter(d["teacher"] for d in distilled.values())
    for t, c in sorted(teacher_counts.items()):
        print(f"  {t}: {c}")

    # 2. Load solver data
    solver_rows = []
    with open(SOLVER_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                solver_rows.append(json.loads(line.strip()))
    print(f"\nSolver rows: {len(solver_rows)}")

    # 3. Merge: replace weak/acceptable with distilled
    upgraded = 0
    already_good = 0
    for row in solver_rows:
        quality = row.get("trace_quality", "")
        ph = prompt_hash(row.get("prompt", ""))

        if ph in distilled:
            d = distilled[ph]
            cot = d["cot"]
            if len(cot) < 50:
                continue

            if quality in ("weak", "acceptable") or (quality == "informative" and len(cot) > 500):
                # Replace with distilled (better quality)
                answer = row.get("answer", "")
                new_asst = f"{cot}\n\n\\boxed{{{answer}}}"

                msgs = row.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant":
                        m["content"] = new_asst

                row["trace_quality"] = "distilled"
                row["trace_strategy"] = f"distilled_{d['teacher']}"
                upgraded += 1
            else:
                already_good += 1

    print(f"\nUpgraded to distilled: {upgraded}")
    print(f"Already good (kept solver): {already_good}")

    # Quality distribution
    quality_dist = Counter(r.get("trace_quality") for r in solver_rows)
    print(f"\nQuality after merge:")
    for q, c in sorted(quality_dist.items()):
        print(f"  {q}: {c}")

    # 4. Type-weighted sampling (konbu17 style)
    by_family = defaultdict(list)
    for row in solver_rows:
        fam = row.get("family", "unknown")
        by_family[fam].append(row)

    print(f"\nPer-family before sampling:")
    for fam in sorted(by_family):
        print(f"  {fam}: {len(by_family[fam])}")

    sampled = []
    for fam, cap in FAMILY_CAPS.items():
        rows = by_family.get(fam, [])
        # Prioritize distilled > informative > acceptable > weak
        priority_order = {"distilled": 0, "informative": 1, "acceptable": 2, "weak": 3}
        rows.sort(key=lambda r: priority_order.get(r.get("trace_quality", ""), 9))

        n = min(len(rows), cap)
        sampled.extend(rows[:n])
        print(f"  {fam}: sampled {n}/{len(rows)} (cap={cap})")

    random.shuffle(sampled)
    print(f"\nFinal dataset: {len(sampled)} examples")

    # Final quality
    final_quality = Counter(r.get("trace_quality") for r in sampled)
    print(f"Final quality:")
    for q, c in sorted(final_quality.items()):
        print(f"  {q}: {c}")

    # 5. Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB, {len(sampled)} rows)")
    print("DONE!")


if __name__ == "__main__":
    main()
