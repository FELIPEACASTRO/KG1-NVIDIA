#!/usr/bin/env python3
"""V424 public-train pattern mining CPU gate.

Mines public train.csv excluding weak IDs for reusable equation_transform
patterns. This does not use weak labels to generate predictions; weak labels
are used only to audit whether mined patterns would improve the current locked
adapter baseline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_CSV = ROOT / "artifacts" / "v342_acc_first_diagnostic" / "v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = ROOT / "artifacts" / "v424_public_train_pattern_mining_gate" / "20260515T_v424_public_train_pattern_mining_gate"
HF_DATASET = "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_train_csv(path_arg: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        if path.exists():
            return path
    for candidate in (Path(r"C:\Users\davis\Downloads\competition_train.csv"), Path(r"C:\Users\davis\Downloads\train.csv")):
        if candidate.exists():
            return candidate
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename="train.csv"))


def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples: list[tuple[str, str]] = []
    for line in str(prompt).splitlines():
        if " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5 and rhs:
            examples.append((lhs, rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt))
    return examples, match.group(1).strip() if match else None


def canonicalize(examples: list[tuple[str, str]], query: str, answer: str | None = None) -> tuple[tuple, tuple | None, tuple | None, list[str]]:
    mapping: dict[str, str] = {}
    reverse: list[str] = []

    def token(ch: str) -> str:
        if ch not in mapping:
            mapping[ch] = f"v{len(mapping)}"
            reverse.append(ch)
        return mapping[ch]

    def enc(text: str) -> tuple[str, ...]:
        return tuple(token(ch) for ch in text)

    c_examples = tuple((enc(lhs), enc(rhs)) for lhs, rhs in examples)
    c_query = enc(query) if query else None
    c_answer = enc(answer) if answer is not None else None
    return c_examples, c_query, c_answer, reverse


def condition_bits(text: str) -> tuple[bool, ...]:
    if len(text) != 5:
        return ()
    return (
        text[0] == text[1],
        text[3] == text[4],
        text[:2] == text[3:5],
        text[0] == text[3],
        text[0] == text[4],
        text[1] == text[3],
        text[1] == text[4],
        len(set(text)) < 5,
        len(set(text[0:2] + text[3:5])) < 4,
        bool(set(text[:2]) & set(text[3:5])),
    )


def feature_signature(examples: list[tuple[str, str]], query: str) -> tuple:
    return (
        len(examples),
        tuple(sorted(len(rhs) for _lhs, rhs in examples)),
        len(query or ""),
        condition_bits(query or ""),
        tuple(sorted(((lhs[2] if len(lhs) == 5 else "?"), len(rhs), condition_bits(lhs)) for lhs, rhs in examples)),
    )


def source_variants(query: str) -> dict[str, str]:
    if len(query) != 5:
        return {}
    return {
        "full5": query,
        "operands4": query[:2] + query[3:5],
        "left2": query[:2],
        "right2": query[3:5],
        "op1": query[2],
    }


def direct_templates_for(query: str, answer: str) -> list[tuple[str, tuple[int, ...]]]:
    if not query or len(answer) > 6:
        return []
    out: list[tuple[str, tuple[int, ...]]] = []
    for name, source in source_variants(query).items():
        if not source or len(source) ** len(answer) > 20000:
            continue
        for positions in itertools.product(range(len(source)), repeat=len(answer)):
            pred = "".join(source[idx] for idx in positions)
            if pred == answer:
                out.append((name, positions))
    return out


def decode_canonical(c_answer: tuple[str, ...], reverse: list[str]) -> str | None:
    out: list[str] = []
    for token in c_answer:
        if not token.startswith("v"):
            return None
        idx = int(token[1:])
        if idx >= len(reverse):
            return None
        out.append(reverse[idx])
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="")
    args = parser.parse_args()

    print("=== V424 PUBLIC TRAIN PATTERN MINING START ===", flush=True)
    train_csv = resolve_train_csv(args.train_csv)
    print(f"train_csv = {train_csv}", flush=True)
    print(f"baseline_csv = {BASELINE_CSV}", flush=True)
    print(f"output_dir = {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv(BASELINE_CSV)
    weak_ids = {row["id"] for row in baseline_rows}
    train_rows = read_csv(train_csv)
    print(f"train_rows = {len(train_rows)}", flush=True)

    exact_lib: dict[tuple, set[tuple[str, ...]]] = defaultdict(set)
    template_lib: dict[tuple, Counter] = defaultdict(Counter)
    for row in train_rows:
        if row.get("id") in weak_ids:
            continue
        examples, query = parse_examples(row.get("prompt", ""))
        answer = row.get("answer", "")
        if not examples or len(query or "") != 5:
            continue
        c_examples, c_query, c_answer, _reverse = canonicalize(examples, query or "", answer)
        exact_lib[(c_examples, c_query)].add(c_answer or ())
        templates = direct_templates_for(query or "", answer)
        if len(templates) == 1:
            template_lib[feature_signature(examples, query or "")][templates[0]] += 1

    audit_rows: list[dict] = []
    accepted: list[dict] = []
    conflicts: list[dict] = []
    for row in baseline_rows:
        if row.get("type") != "equation_transform":
            continue
        examples, query = parse_examples(row.get("prompt", ""))
        if not examples or len(query or "") != 5:
            continue
        predictions: list[tuple[str, str, str]] = []
        c_examples, c_query, _c_answer, reverse = canonicalize(examples, query or "", None)
        exact_answers = exact_lib.get((c_examples, c_query), set())
        if len(exact_answers) == 1:
            pred = decode_canonical(next(iter(exact_answers)), reverse)
            if pred is not None:
                predictions.append(("exact_canonical_prompt_signature", pred, "count=1"))

        counter = template_lib.get(feature_signature(examples, query or ""), Counter())
        if counter:
            (template, count), *_ = counter.most_common(1)
            total = sum(counter.values())
            if count >= 3 and count / total >= 0.8:
                source_name, positions = template
                source = source_variants(query or "").get(source_name, "")
                if source and max(positions, default=-1) < len(source):
                    predictions.append(("feature_direct_template_library", "".join(source[idx] for idx in positions), f"count={count};total={total}"))

        unique_predictions = sorted({pred for _kind, pred, _proof in predictions})
        if len(unique_predictions) != 1:
            status = "abstain"
            pred = ""
        else:
            status = "candidate"
            pred = unique_predictions[0]
        base_correct = str(row.get("correct", "")).lower() == "true"
        cand_correct = pred == row.get("answer", "") if pred else False
        audit = {
            "id": row.get("id", ""),
            "status": status,
            "prediction": pred,
            "answer": row.get("answer", ""),
            "baseline_prediction": row.get("prediction", ""),
            "baseline_correct": str(base_correct),
            "candidate_correct": str(cand_correct),
            "prediction_sources": json.dumps(predictions, sort_keys=True),
        }
        audit_rows.append(audit)
        if status != "candidate":
            continue
        if base_correct and not cand_correct:
            conflicts.append(audit)
        elif (not base_correct) and cand_correct:
            accepted.append(audit)

    projected_total = 192 + len(accepted)
    projected_eq = 56 + len(accepted)
    manifest = {
        "schema_version": "kg1_v424_public_train_pattern_mining_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(train_csv),
        "train_rows": len(train_rows),
        "exact_signature_count": len(exact_lib),
        "feature_template_signature_count": len(template_lib),
        "candidate_count": sum(1 for row in audit_rows if row["status"] == "candidate"),
        "accepted_gain_count": len(accepted),
        "conflict_count": len(conflicts),
        "projection": {
            "correct": projected_total,
            "equation_transform_correct": projected_eq,
            "bit_manipulation_correct": 136,
            "rows": 315,
        },
        "hf_gpu_allowed": bool(accepted and not conflicts and projected_eq >= 60),
        "decision": {
            "decision": "hf_gpu_blocked_no_public_train_pattern_gain",
            "reason": f"accepted={len(accepted)}; conflicts={len(conflicts)}; projected_eq={projected_eq}",
            "next_action": "Do not train from exact/train-template signatures; mine a richer program DSL if continuing.",
        },
    }
    write_csv(OUT_DIR / "v424_public_train_pattern_audit.csv", audit_rows)
    write_csv(OUT_DIR / "v424_public_train_pattern_accepted.csv", accepted)
    write_csv(OUT_DIR / "v424_public_train_pattern_conflicts.csv", conflicts)
    (OUT_DIR / "v424_public_train_pattern_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# V424 Public Train Pattern Mining Gate

Generated: {manifest['generated_at_utc']}

| Metric | Value |
|---|---:|
| Train rows | `{manifest['train_rows']}` |
| Exact signatures | `{manifest['exact_signature_count']}` |
| Feature-template signatures | `{manifest['feature_template_signature_count']}` |
| Candidate rows | `{manifest['candidate_count']}` |
| Accepted gains | `{manifest['accepted_gain_count']}` |
| Conflicts/losses | `{manifest['conflict_count']}` |
| Projected weak total | `{projected_total}/315` |
| Projected equation_transform | `{projected_eq}/155` |
| Projected bit_manipulation | `136/160` |

Decision: `{manifest['decision']['decision']}`.
"""
    (OUT_DIR / "V424_PUBLIC_TRAIN_PATTERN_MINING_GATE.md").write_text(report, encoding="utf-8")
    print("v424_manifest =", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V424 PUBLIC TRAIN PATTERN MINING END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
