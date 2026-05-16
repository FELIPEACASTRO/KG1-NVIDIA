#!/usr/bin/env python3
"""Fetch and triage Kaggle discussion topics for KG1 actionable evidence.

This script is read-only. It does not train, evaluate adapters, package, submit,
or use weak/full labels as training data. It fetches discussion topic JSON via
Kaggle's read endpoint and extracts posts/URLs likely relevant to the current
bit_manipulation and equation_transform plateau.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = Path(r"C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topic IDs.md")
DEFAULT_URLS = Path(r"C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topics URLs.md")
DEFAULT_OUT = ROOT / "artifacts" / "v512_kaggle_discussions_audit"
ENDPOINT = "https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById"
TOPIC_URL_PREFIX = "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/"


KEYWORD_GROUPS: dict[str, list[str]] = {
    "bit_solver": [
        "bit_manipulation",
        "bit manipulation",
        "bitwise",
        "bitsum",
        "bit sum",
        "stride",
        "rotation",
        "rot(",
        "shift",
        "shl",
        "shr",
        "xor",
        "and-not",
        "or-not",
        "fullbyte",
        "tong hui kang",
        "huikang",
    ],
    "equation_solver": [
        "equation_transform",
        "transformation/equation",
        "symbol_transform",
        "symbol transform",
        "numeric_equation",
        "equation_numeric",
        "cryptarithm",
        "equation",
        "deduce",
        "operator",
        "dsl",
        "verifier",
        "parser",
        "postprocessor",
    ],
    "adapter_training": [
        "adapter",
        "lora",
        "qlora",
        "peft",
        "sft",
        "cot",
        "chain of thought",
        "trace",
        "distill",
        "synthetic",
        "teacher",
        "student",
        "nemotron",
    ],
    "concrete_artifact": [
        "github.com",
        "huggingface.co",
        "kaggle.com/code",
        "dataset",
        "notebook",
        "solver.py",
        ".ipynb",
        ".jsonl",
        ".csv",
        "release",
        "repo",
        "code",
    ],
}

URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
TOPIC_ID_RE = re.compile(r"\b\d{6}\b")


@dataclass
class PostHit:
    topic_id: int
    topic_name: str
    topic_url: str
    post_id: int | str
    post_date: str
    author: str
    score: int
    groups: dict[str, list[str]]
    urls: list[str]
    excerpt: str


def read_ids(paths: list[Path]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for path in paths:
        if not path.exists():
            print(f"warning_missing_topic_file = {path}", flush=True)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw in TOPIC_ID_RE.findall(text):
            value = int(raw)
            if value not in seen:
                seen.add(value)
                ids.append(value)
    return ids


def fetch_topic(session: requests.Session, topic_id: int, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        response = session.get(
            ENDPOINT,
            params={"forumTopicId": str(topic_id), "includeComments": "true"},
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0 KG1 discussion audit"},
            timeout=60,
        )
        if response.status_code == 429:
            wait_s = 10 * attempt
            print(f"rate_limit topic_id={topic_id} attempt={attempt}/{retries} sleep_s={wait_s}", flush=True)
            time.sleep(wait_s)
            last_error = requests.HTTPError(f"429 Too Many Requests for topic_id={topic_id}")
            continue
        response.raise_for_status()
        return response.json()
    assert last_error is not None
    raise last_error


def compact_text(value: str, limit: int = 900) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def author_name(message: dict[str, Any], fallback: str = "") -> str:
    author = message.get("author") or {}
    if isinstance(author, dict):
        return str(author.get("userName") or author.get("displayName") or fallback or "")
    return fallback


def iter_posts(topic: dict[str, Any]) -> list[dict[str, Any]]:
    ft = topic.get("forumTopic") or {}
    posts: list[dict[str, Any]] = []
    first = ft.get("firstMessage")
    if isinstance(first, dict):
        first = dict(first)
        first["_kg1_post_kind"] = "firstMessage"
        posts.append(first)
    for comment in ft.get("comments") or []:
        if isinstance(comment, dict):
            comment = dict(comment)
            comment["_kg1_post_kind"] = "comment"
            posts.append(comment)
    return posts


def score_post(text: str) -> tuple[int, dict[str, list[str]]]:
    lower = text.lower()
    groups: dict[str, list[str]] = {}
    score = 0
    for group, keywords in KEYWORD_GROUPS.items():
        matched = sorted({kw for kw in keywords if kw in lower})
        if not matched:
            continue
        groups[group] = matched
        if group in {"bit_solver", "equation_solver"}:
            score += 5 * len(matched)
        elif group == "concrete_artifact":
            score += 4 * len(matched)
        else:
            score += 2 * len(matched)
    return score, groups


def build_hit(topic: dict[str, Any], post: dict[str, Any]) -> PostHit | None:
    ft = topic.get("forumTopic") or {}
    topic_id = int(ft.get("id") or 0)
    topic_name = str(ft.get("name") or "").strip()
    topic_url = TOPIC_URL_PREFIX + str(topic_id)
    raw = str(post.get("rawMarkdown") or post.get("content") or "")
    if not raw:
        return None
    score, groups = score_post(topic_name + "\n" + raw)
    urls = sorted(set(URL_RE.findall(raw)))
    if score <= 0 and not urls:
        return None
    author = author_name(post, str(ft.get("authorUserName") or ""))
    return PostHit(
        topic_id=topic_id,
        topic_name=topic_name,
        topic_url=topic_url,
        post_id=post.get("id") or post.get("_kg1_post_kind") or "",
        post_date=str(post.get("postDate") or ft.get("postDate") or ""),
        author=author,
        score=score + (6 if urls else 0),
        groups=groups,
        urls=urls,
        excerpt=compact_text(raw),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(manifest: dict[str, Any], hits: list[PostHit], path: Path) -> None:
    lines = [
        "# V512 Kaggle Discussion Audit",
        "",
        f"Generated UTC: {manifest['generated_at_utc']}",
        "",
        "## Scope",
        "",
        f"- Topic IDs requested: `{manifest['topic_count_requested']}`",
        f"- Topics fetched: `{manifest['topic_count_fetched']}`",
        f"- Posts scanned: `{manifest['post_count_scanned']}`",
        f"- Relevant post hits: `{manifest['hit_count']}`",
        "",
        "## Highest-Signal Hits",
        "",
    ]
    for hit in hits[:40]:
        groups = ", ".join(f"{name}: {', '.join(words[:8])}" for name, words in hit.groups.items())
        url_text = ", ".join(hit.urls[:5])
        lines.extend(
            [
                f"### {hit.topic_id} - {hit.topic_name}",
                "",
                f"- URL: {hit.topic_url}",
                f"- Post: `{hit.post_id}` by `{hit.author}` at `{hit.post_date}`",
                f"- Score: `{hit.score}`",
                f"- Groups: {groups or 'url-only'}",
                f"- URLs: {url_text or 'none'}",
                "",
                "Excerpt:",
                "",
                "```text",
                hit.excerpt,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--urls-file", type=Path, default=DEFAULT_URLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sleep-s", type=float, default=0.15)
    parser.add_argument("--refresh", action="store_true", help="Refetch topics even when raw JSON already exists.")
    args = parser.parse_args()

    print("=== V512 KAGGLE DISCUSSION AUDIT START ===", flush=True)
    topic_ids = read_ids([args.ids_file, args.urls_file])
    if not topic_ids:
        raise RuntimeError("No topic IDs found.")
    print("topic_count_requested =", len(topic_ids), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    raw_dir = args.output_dir / "raw_topics"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    topics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, topic_id in enumerate(topic_ids, start=1):
        print(f"fetch_topic_progress = {idx}/{len(topic_ids)} topic_id={topic_id}", flush=True)
        raw_path = raw_dir / f"{topic_id}.json"
        try:
            if raw_path.exists() and not args.refresh:
                topic = json.loads(raw_path.read_text(encoding="utf-8"))
                topics.append(topic)
                print(f"fetch_topic_cached topic_id={topic_id}", flush=True)
                continue
            topic = fetch_topic(session, topic_id)
            topics.append(topic)
            write_json(raw_path, topic)
        except Exception as exc:
            failures.append({"topic_id": topic_id, "error": repr(exc)})
            print(f"fetch_topic_failure topic_id={topic_id} error={exc!r}", flush=True)
        time.sleep(args.sleep_s)

    hits: list[PostHit] = []
    post_count = 0
    for topic in topics:
        for post in iter_posts(topic):
            post_count += 1
            hit = build_hit(topic, post)
            if hit is not None:
                hits.append(hit)
    hits.sort(key=lambda item: (-item.score, item.topic_id, str(item.post_id)))

    manifest = {
        "schema_version": "kg1_v512_kaggle_discussion_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT,
        "topic_count_requested": len(topic_ids),
        "topic_count_fetched": len(topics),
        "failure_count": len(failures),
        "failures": failures,
        "post_count_scanned": post_count,
        "hit_count": len(hits),
        "keyword_groups": KEYWORD_GROUPS,
        "top_hits_json": str(args.output_dir / "v512_kaggle_discussion_top_hits.json"),
        "summary_md": str(args.output_dir / "V512_KAGGLE_DISCUSSION_AUDIT_SUMMARY.md"),
    }
    serial_hits = [hit.__dict__ for hit in hits]
    write_json(args.output_dir / "v512_kaggle_discussion_manifest.json", manifest)
    write_json(args.output_dir / "v512_kaggle_discussion_top_hits.json", serial_hits)
    render_markdown(manifest, hits, args.output_dir / "V512_KAGGLE_DISCUSSION_AUDIT_SUMMARY.md")
    print("topic_count_fetched =", len(topics), flush=True)
    print("post_count_scanned =", post_count, flush=True)
    print("hit_count =", len(hits), flush=True)
    print("manifest_path =", args.output_dir / "v512_kaggle_discussion_manifest.json", flush=True)
    print("summary_path =", args.output_dir / "V512_KAGGLE_DISCUSSION_AUDIT_SUMMARY.md", flush=True)
    print("=== V512 KAGGLE DISCUSSION AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
