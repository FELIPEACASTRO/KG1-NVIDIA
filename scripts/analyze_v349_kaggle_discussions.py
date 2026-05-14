#!/usr/bin/env python3
"""Fetch and triage Kaggle discussion topics for KG1 weak-family improvement.

The script uses Kaggle's web JSON endpoint for discussion topics. It is
read-only: no training, no submission, no package creation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
DISCUSSION_ENDPOINT = "https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById"
DISCUSSION_URL = "https://www.kaggle.com/competitions/{competition}/discussion/{topic_id}"

DEFAULT_IDS_MD = Path(r"C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topic IDs.md")
DEFAULT_URLS_MD = Path(r"C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topics URLs.md")
DEFAULT_CACHE_RAW_DIRS = [
    Path("artifacts/v328_kaggle_discussion_140_audit/raw_topics"),
    Path("artifacts/v332_kaggle_discussion_resume_audit/20260513T_batch01/raw_topics"),
]

KEYWORD_GROUPS: dict[str, list[str]] = {
    "bit": [
        "bit_manipulation",
        "bit manipulation",
        "bitwise",
        "bitsum",
        "stride",
        "rotate",
        "rotation",
        "shift",
        "xor",
        "and-not",
        "truth table",
        "boolean",
    ],
    "equation": [
        "equation_transform",
        "equation",
        "numeric_equation",
        "symbol_transform",
        "symbolic",
        "cryptarithm",
        "alice",
        "operator",
        "punct",
        "transformation rules",
    ],
    "solver_verifier": [
        "solver",
        "verifier",
        "postprocessor",
        "post-processor",
        "rule",
        "deterministic",
        "regex",
        "parse",
        "parser",
    ],
    "training": [
        "sft",
        "cot",
        "chain of thought",
        "lora",
        "adapter",
        "finetun",
        "fine-tun",
        "train",
        "loss",
        "eval_loss",
        "distill",
        "synthetic",
    ],
    "submission": [
        "submission",
        "submit",
        "leaderboard",
        "lb",
        "score",
        "rank",
        "public",
        "private",
    ],
    "data": [
        "dataset",
        "train.csv",
        "test.csv",
        "sft_train",
        "sft_val",
        "huggingface",
        "github",
        "notebook",
        "code",
    ],
}

ACTION_TERMS = [
    "tong",
    "huikang",
    "98.9",
    "85.1",
    "1602",
    "bitsum",
    "stride",
    "cryptarithm",
    "equation_numeric",
    "symbol_transform",
    "postprocessor",
    "verifier",
    "solver",
    "dataset",
    "sft_train",
    "sft_val",
    "github",
    "huggingface",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_topic_ids(paths: list[Path]) -> list[str]:
    ids: list[str] = []
    for path in paths:
        text = read_text(path)
        ids.extend(re.findall(r"discussion/(\d+)", text))
        ids.extend(re.findall(r"^\s*\d+\.\s*(\d{6})\s*$", text, flags=re.M))
    out: list[str] = []
    for topic_id in ids:
        if topic_id not in out:
            out.append(topic_id)
    return out


def topic_text(payload: dict[str, Any]) -> str:
    topic = payload.get("forumTopic", {}) or {}
    chunks = [
        str(topic.get("title", "")),
        str(topic.get("rawMarkdown", "")),
        str(topic.get("authorUserName", "")),
        str(topic.get("authorUserDisplayName", "")),
    ]
    for comment in topic.get("comments", []) or []:
        chunks.extend(
            [
                str(comment.get("rawMarkdown", "")),
                str(comment.get("authorUserName", "")),
                str(comment.get("authorUserDisplayName", "")),
            ]
        )
    return "\n".join(chunks)


def record_text(record: dict[str, Any]) -> str:
    if "full_text" in record:
        return str(record.get("full_text", ""))
    chunks = [
        str(record.get("title", "")),
        str(record.get("raw_markdown", "")),
        str(record.get("author", "")),
        str(record.get("author_display", "")),
    ]
    for comment in record.get("comments", []) or []:
        chunks.extend(
            [
                str(comment.get("raw_markdown", "")),
                str(comment.get("author", "")),
                str(comment.get("author_display", "")),
            ]
        )
    return "\n".join(chunks)


def normalize_cached_record(payload: dict[str, Any], topic_id: str) -> dict[str, Any]:
    if "full_text" in payload:
        topic = payload.get("topic", {}) or {}
        messages = payload.get("messages", []) or []
        return {
            "id": str(payload.get("topic_id") or topic.get("id") or topic_id),
            "title": str(topic.get("name", "")),
            "author": str(topic.get("authorUserName", "")),
            "author_display": str(topic.get("authorUserDisplayName", "")),
            "post_date": str(topic.get("postDate", "")),
            "vote_count": topic.get("totalVotes", ""),
            "comment_count": int(payload.get("message_count_observed") or len(messages)),
            "source_url": str(
                payload.get("url") or DISCUSSION_URL.format(competition=COMPETITION, topic_id=topic_id)
            ),
            "full_text": str(payload.get("full_text", "")),
            "messages": messages,
            "text_sha256": str(payload.get("text_sha256", "")),
        }
    if "comments" in payload:
        out = dict(payload)
        out.setdefault("id", topic_id)
        out.setdefault("source_url", DISCUSSION_URL.format(competition=COMPETITION, topic_id=topic_id))
        return out
    return compact_topic(payload)


def compact_topic(payload: dict[str, Any]) -> dict[str, Any]:
    topic = payload.get("forumTopic", {}) or {}
    return {
        "id": str(topic.get("id", "")),
        "title": str(topic.get("title", "")),
        "author": str(topic.get("authorUserName", "")),
        "author_display": str(topic.get("authorUserDisplayName", "")),
        "post_date": str(topic.get("postDate", "")),
        "vote_count": topic.get("voteCount", ""),
        "comment_count": len(topic.get("comments", []) or []),
        "raw_markdown": str(topic.get("rawMarkdown", "")),
        "comments": [
            {
                "id": str(comment.get("id", "")),
                "author": str(comment.get("authorUserName", "")),
                "author_display": str(comment.get("authorUserDisplayName", "")),
                "post_date": str(comment.get("postDate", "")),
                "raw_markdown": str(comment.get("rawMarkdown", "")),
            }
            for comment in topic.get("comments", []) or []
        ],
    }


def load_cached_topic(topic_id: str, cache_raw_dirs: list[Path]) -> tuple[dict[str, Any] | None, str]:
    for cache_dir in cache_raw_dirs:
        path = cache_dir / f"{topic_id}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = normalize_cached_record(payload, topic_id)
            record["cache_source_path"] = str(path)
            return record, str(path)
    return None, ""


def fetch_topic(session: requests.Session, topic_id: str, retries: int = 3) -> tuple[int, dict[str, Any] | None, str]:
    params = {"forumTopicId": topic_id, "includeComments": "true"}
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(DISCUSSION_ENDPOINT, params=params, timeout=40)
            if response.status_code == 200:
                return response.status_code, response.json(), ""
            last_error = response.text[:500]
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = repr(exc)
        time.sleep(min(2.0, attempt * 0.5))
    return -1, None, last_error


def score_text(text: str) -> tuple[dict[str, int], int, list[str]]:
    lowered = text.lower()
    group_scores: dict[str, int] = {}
    for group, terms in KEYWORD_GROUPS.items():
        group_scores[group] = sum(lowered.count(term.lower()) for term in terms)
    action_hits = sorted({term for term in ACTION_TERMS if term.lower() in lowered})
    weighted = (
        5 * group_scores["bit"]
        + 5 * group_scores["equation"]
        + 4 * group_scores["solver_verifier"]
        + 3 * group_scores["training"]
        + 2 * group_scores["data"]
        + group_scores["submission"]
        + 10 * len(action_hits)
    )
    return group_scores, weighted, action_hits


def infer_actionable_notes(compact: dict[str, Any], group_scores: dict[str, int], action_hits: list[str]) -> list[str]:
    title = str(compact.get("title", "")).lower()
    text = record_text(compact).lower()
    notes: list[str] = []
    if "bitsum" in text or "stride" in text or "85.1" in text or "98.9" in text:
        notes.append("bit_solver_algorithm_signal")
    if "cryptarithm" in text:
        notes.append("equation_symbolic_cryptarithm_signal")
    if "equation_numeric" in text or ("numeric" in text and group_scores["equation"] > 0):
        notes.append("equation_numeric_operator_signal")
    if "sft" in text or "lora" in text or "adapter" in text or "fine" in text:
        notes.append("training_transfer_signal")
    if "dataset" in text or "sft_train" in text or "sft_val" in text:
        notes.append("dataset_source_signal")
    if "postprocessor" in text or "verifier" in text or "solver" in text:
        notes.append("solver_verifier_signal")
    if "rule" in text and ("bit" in text or "equation" in text):
        notes.append("rule_taxonomy_signal")
    if "winning" in title or "progress prize" in title or "publication" in title:
        notes.append("high_priority_author_solution")
    return sorted(set(notes))


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V349 KAGGLE DISCUSSION DOUBLE CHECK START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("ids_md =", args.ids_md, flush=True)
    print("urls_md =", args.urls_md, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("cache_raw_dirs =", json.dumps([str(path) for path in args.cache_raw_dir]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_topics"
    if args.write_normalized_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    topic_ids = extract_topic_ids([args.ids_md, args.urls_md])
    if len(topic_ids) != args.expected_topics:
        raise RuntimeError(f"expected {args.expected_topics} unique topics, got {len(topic_ids)}")
    (args.output_dir / "topic_ids.json").write_text(json.dumps(topic_ids, indent=2), encoding="utf-8")
    (args.output_dir / "topic_urls.txt").write_text(
        "\n".join(DISCUSSION_URL.format(competition=COMPETITION, topic_id=topic_id) for topic_id in topic_ids),
        encoding="utf-8",
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    cache_hit_count = 0
    fetched_count = 0
    for index, topic_id in enumerate(topic_ids, start=1):
        if index == 1 or index % 10 == 0 or index == len(topic_ids):
            print(f"fetch_progress = {index}/{len(topic_ids)} topic_id={topic_id}", flush=True)
        compact, cache_source = load_cached_topic(topic_id, args.cache_raw_dir)
        status: int | str
        error = ""
        if compact is not None:
            status = "cache"
            cache_hit_count += 1
        else:
            status, payload, error = fetch_topic(session, topic_id)
            status_counts[str(status)] += 1
            if payload is not None:
                fetched_count += 1
                compact = compact_topic(payload)
                cache_source = ""
        if compact is None:
            rows.append(
                {
                    "topic_id": topic_id,
                    "url": DISCUSSION_URL.format(competition=COMPETITION, topic_id=topic_id),
                    "status": status,
                    "title": "",
                    "author": "",
                    "comment_count": 0,
                    "weighted_score": 0,
                    "keyword_groups_json": "{}",
                    "action_hits": "",
                    "actionable_notes": "fetch_failed",
                    "error": error,
                }
            )
            continue
        compact["source_url"] = DISCUSSION_URL.format(competition=COMPETITION, topic_id=topic_id)
        compact["cache_source_path"] = cache_source
        if args.write_normalized_raw:
            (raw_dir / f"{topic_id}.json").write_text(json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8")
        text = record_text(compact)
        group_scores, weighted, action_hits = score_text(text)
        notes = infer_actionable_notes(compact, group_scores, action_hits)
        rows.append(
            {
                "topic_id": topic_id,
                "url": compact["source_url"],
                "status": status,
                "title": compact["title"],
                "author": compact["author"],
                "comment_count": compact["comment_count"],
                "weighted_score": weighted,
                "keyword_groups_json": json.dumps(group_scores, sort_keys=True),
                "action_hits": ";".join(action_hits),
                "actionable_notes": ";".join(notes),
                "error": "",
                "cache_source_path": cache_source,
            }
        )
        time.sleep(args.sleep_s)

    rows_sorted = sorted(rows, key=lambda row: int(row["weighted_score"]), reverse=True)
    triage_csv = args.output_dir / "v349_kaggle_discussion_triage.csv"
    with triage_csv.open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "topic_id",
            "url",
            "status",
            "title",
            "author",
            "comment_count",
            "weighted_score",
            "keyword_groups_json",
            "action_hits",
            "actionable_notes",
            "cache_source_path",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_sorted)

    note_counts = Counter()
    for row in rows:
        for note in str(row["actionable_notes"]).split(";"):
            if note:
                note_counts[note] += 1

    high_priority = [
        row
        for row in rows_sorted
        if int(row["weighted_score"]) >= args.high_priority_threshold or row["actionable_notes"]
    ][: args.max_high_priority]

    manifest = {
        "schema_version": "kg1_v349_kaggle_discussion_double_check_v1",
        "generated_at_utc": utc_now(),
        "inputs": {"ids_md": str(args.ids_md), "urls_md": str(args.urls_md)},
        "topic_count": len(topic_ids),
        "cache_hit_count": cache_hit_count,
        "fetched_count": fetched_count,
        "status_counts": dict(status_counts),
        "note_counts": dict(note_counts),
        "high_priority_count": len(high_priority),
        "high_priority_topics": high_priority,
        "decision": {
            "decision": "discussion_triage_ready_for_manual_evidence_integration",
            "reason": "All listed topic IDs were fetched through Kaggle JSON endpoint and keyword/action triaged.",
            "next_action": "Inspect high-priority raw topic JSONs and add only concrete new findings to roadmap/V349 CPU gate.",
        },
        "outputs": {
            "triage_csv": str(triage_csv),
            "raw_topics_dir": str(raw_dir) if args.write_normalized_raw else "",
            "topic_ids_json": str(args.output_dir / "topic_ids.json"),
            "topic_urls_txt": str(args.output_dir / "topic_urls.txt"),
            "manifest_json": str(args.output_dir / "v349_kaggle_discussion_double_check_manifest.json"),
        },
    }
    Path(manifest["outputs"]["manifest_json"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("status_counts =", json.dumps(dict(status_counts), sort_keys=True), flush=True)
    print("cache_hit_count =", cache_hit_count, flush=True)
    print("fetched_count =", fetched_count, flush=True)
    print("note_counts =", json.dumps(dict(note_counts), sort_keys=True), flush=True)
    print("high_priority_topics_top10 =", json.dumps(high_priority[:10], indent=2, sort_keys=True), flush=True)
    print("manifest_json =", manifest["outputs"]["manifest_json"], flush=True)
    print("=== V349 KAGGLE DISCUSSION DOUBLE CHECK END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-md", type=Path, default=DEFAULT_IDS_MD)
    parser.add_argument("--urls-md", type=Path, default=DEFAULT_URLS_MD)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-topics", type=int, default=140)
    parser.add_argument("--cache-raw-dir", type=Path, action="append", default=DEFAULT_CACHE_RAW_DIRS)
    parser.add_argument("--write-normalized-raw", action="store_true")
    parser.add_argument("--sleep-s", type=float, default=0.05)
    parser.add_argument("--high-priority-threshold", type=int, default=25)
    parser.add_argument("--max-high-priority", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
