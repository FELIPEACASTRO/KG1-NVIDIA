#!/usr/bin/env python3
"""Mine public Kaggle kernels for KG1 bit/equation techniques.

The miner pulls public notebooks into a temporary directory, extracts a compact
keyword/metadata summary, writes only lightweight reports, and deletes the raw
pulled files. It does not use third-party adapters, submissions, or outputs as
training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v453_public_kernel_mining/20260515T_cpu_mining"
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"

KEYWORDS = [
    "bit_manipulation",
    "bit manipulation",
    "bitwise",
    "bitsum",
    "stride",
    "equation_transform",
    "equation_numeric",
    "symbol_transform",
    "solver",
    "verifier",
    "postprocessor",
    "reasoning-gym",
    "tong",
    "huikang",
    "lm_head",
    "gate_up_proj",
    "target_modules",
    "sft_train",
    "sft_val",
    "train_curated",
    "cryptarithm",
    "0.87",
    "0.86",
    "0.85",
]

HIGH_VALUE_MARKERS = [
    "bit_manipulation",
    "bitsum",
    "stride",
    "equation_numeric",
    "equation_transform",
    "solver",
    "verifier",
    "target_modules",
    "lm_head",
    "gate_up_proj",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout_s: int = 180) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    target = path.resolve()
    root = allowed_root.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(f"refusing to delete outside output dir: {target}")
    if target.exists():
        shutil.rmtree(target)


def list_kernels(limit: int) -> list[dict[str, str]]:
    proc = run_cmd(
        [
            "kaggle",
            "kernels",
            "list",
            "--competition",
            COMPETITION,
            "--sort-by",
            "scoreDescending",
            "--page-size",
            str(max(limit, 1)),
            "-v",
        ],
        timeout_s=180,
    )
    if proc.returncode:
        raise RuntimeError("kaggle kernels list failed:\n" + proc.stdout[-4000:])
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    return rows[:limit]


def extract_notebook_text(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.append("".join(source))
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


def extract_texts(pull_dir: Path) -> tuple[str, dict[str, Any]]:
    texts: list[str] = []
    meta: dict[str, Any] = {}
    for path in sorted(pull_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "kernel-metadata.json":
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if path.suffix.lower() in {".ipynb", ".py", ".r", ".md", ".txt", ".json"}:
            if path.stat().st_size > 4_000_000:
                continue
            if path.suffix.lower() == ".ipynb":
                texts.append(extract_notebook_text(path))
            else:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts), meta


def keyword_counts(text: str) -> dict[str, int]:
    lower = text.lower()
    return {keyword: lower.count(keyword.lower()) for keyword in KEYWORDS}


def snippet_hits(text: str, max_hits: int = 10) -> list[str]:
    lines = text.splitlines()
    hits: list[str] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        if any(marker.lower() in lower for marker in HIGH_VALUE_MARKERS):
            clean = " ".join(line.strip().split())
            if clean and clean not in hits:
                hits.append(clean[:220])
            if len(hits) >= max_hits:
                break
    return hits


def summarize_kernel(row: dict[str, str], output_dir: Path, index: int) -> dict[str, Any]:
    ref = str(row.get("ref", "")).strip()
    pull_dir = output_dir / "_pull_tmp" / f"{index:03d}_{ref.replace('/', '__')}"
    pull_dir.mkdir(parents=True, exist_ok=True)
    proc = run_cmd(["kaggle", "kernels", "pull", ref, "-p", str(pull_dir), "-m"], timeout_s=240)
    status = "ok" if proc.returncode == 0 else "pull_failed"
    text = ""
    metadata: dict[str, Any] = {}
    counts: dict[str, int] = {}
    hits: list[str] = []
    data_sources: list[str] = []
    if status == "ok":
        text, metadata = extract_texts(pull_dir)
        counts = keyword_counts(text)
        hits = snippet_hits(text)
        for item in metadata.get("dataset_sources", []) + metadata.get("model_sources", []) + metadata.get("kernel_sources", []):
            data_sources.append(json.dumps(item, sort_keys=True))
        if not data_sources and metadata.get("dataSources"):
            for item in metadata.get("dataSources", []):
                data_sources.append(json.dumps(item, sort_keys=True))
    safe_rmtree(pull_dir, output_dir)

    high_value_score = sum(counts.get(marker, 0) for marker in HIGH_VALUE_MARKERS)
    return {
        "ref": ref,
        "title": row.get("title", ""),
        "author": row.get("author", ""),
        "last_run_time": row.get("lastRunTime", ""),
        "total_votes": row.get("totalVotes", ""),
        "pull_status": status,
        "high_value_score": high_value_score,
        "keyword_counts_json": json.dumps(counts, sort_keys=True),
        "snippet_hits_json": json.dumps(hits, ensure_ascii=False),
        "data_sources_json": json.dumps(data_sources[:30], ensure_ascii=False),
        "pull_tail": proc.stdout[-1000:],
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# V453 Public Kaggle Kernel Mining",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Resultado",
        "",
        "| Item | Valor |",
        "|---|---:|",
        f"| Kernels listados | `{manifest['listed_count']}` |",
        f"| Kernels analisados | `{manifest['analyzed_count']}` |",
        f"| Pull failures | `{manifest['pull_failed_count']}` |",
        "",
        "## Top sinais tecnicos",
        "",
        "| Ref | Score | Snippets | Decisao |",
        "|---|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: int(item.get("high_value_score", 0)), reverse=True)[:15]:
        hits = json.loads(str(row.get("snippet_hits_json", "[]")))
        decision = "triage_manual" if int(row.get("high_value_score", 0)) > 0 else "baixo_sinal"
        lines.append(
            "| "
            + str(row["ref"])
            + " | `"
            + str(row["high_value_score"])
            + "` | "
            + "<br>".join(str(hit).replace("|", "\\|") for hit in hits[:3])
            + " | "
            + decision
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decisao",
            "",
            "Esta mineracao e CPU-only e nao produz artefato submit-ready. Qualquer",
            "tecnica encontrada aqui precisa virar regra local, passar por gate V452/V453",
            "e provar `total>192`, `equation>56`, `bit>=136`, `truncated=0` antes de HF.",
            "",
            "Raw notebooks baixados foram removidos apos extracao dos sinais leves.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=== V453 PUBLIC KERNEL MINING START ===", flush=True)
    print("competition =", COMPETITION, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("limit =", args.limit, flush=True)
    kernels = list_kernels(args.limit)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(kernels, start=1):
        print(f"v453_kernel_progress = {index}/{len(kernels)} ref={row.get('ref','')}", flush=True)
        rows.append(summarize_kernel(row, args.output_dir, index))

    summary_csv = args.output_dir / "v453_public_kernel_mining_summary.csv"
    manifest_json = args.output_dir / "v453_public_kernel_mining_manifest.json"
    markdown = args.output_dir / "KG1_V453_PUBLIC_KERNEL_MINING.md"
    columns = [
        "ref",
        "title",
        "author",
        "last_run_time",
        "total_votes",
        "pull_status",
        "high_value_score",
        "keyword_counts_json",
        "snippet_hits_json",
        "data_sources_json",
        "pull_tail",
    ]
    write_csv(summary_csv, rows, columns)
    status_counts = Counter(str(row["pull_status"]) for row in rows)
    manifest = {
        "schema_version": "kg1_v453_public_kernel_mining_v1",
        "generated_at_utc": utc_now(),
        "competition": COMPETITION,
        "listed_count": len(kernels),
        "analyzed_count": len(rows),
        "pull_failed_count": status_counts.get("pull_failed", 0),
        "status_counts": dict(status_counts),
        "outputs": {
            "summary_csv": str(summary_csv),
            "manifest_json": str(manifest_json),
            "markdown": str(markdown),
        },
        "raw_notebooks_deleted": True,
    }
    write_json(manifest_json, manifest)
    write_markdown(markdown, rows, manifest)
    print("status_counts =", json.dumps(dict(status_counts), sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V453 PUBLIC KERNEL MINING END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
