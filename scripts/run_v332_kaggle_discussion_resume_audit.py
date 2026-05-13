#!/usr/bin/env python3
"""Resume Kaggle discussion fetch/audit for KG1 weak-family work.

This script intentionally fetches in small batches. Kaggle discussion endpoints
rate-limit aggressively; the goal is to build an evidence cache incrementally
while avoiding repeated 429 failures.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kagglesdk.discussions.services.discussions_api_service import (
    ApiGetTopicRequest,
    ApiListCommentsRequest,
    DiscussionApiClient,
)
from kagglesdk.kaggle_http_client import KaggleHttpClient


KEYWORDS = [
    "bit",
    "bitwise",
    "bit_manipulation",
    "bitsum",
    "stride",
    "equation",
    "equation_transform",
    "numeric",
    "symbol",
    "symbolic",
    "cryptarithm",
    "operator",
    "solver",
    "verifier",
    "postprocessor",
    "cot",
    "chain of thought",
    "synthetic",
    "loss",
    "eval",
    "accuracy",
    "adapter",
    "lora",
    "token",
    "prompt",
    "train",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def html_to_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.I)
    value = re.sub(r"</p\s*>", "\n", value, flags=re.I)
    value = re.sub(r"</li\s*>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s+", "\n", value)
    return value.strip()


def parse_topic_ids(paths: list[Path]) -> list[int]:
    ids: list[int] = []
    for path in paths:
        if not path.exists():
            continue
        text = read_text(path)
        ids.extend(int(x) for x in re.findall(r"(?:discussion/|^|\D)(\d{6})(?=\D|$)", text))
    return sorted(set(ids))


def flatten_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def visit(comment: dict[str, Any], depth: int) -> None:
        item = dict(comment)
        replies = item.pop("replies", []) or []
        item["depth"] = depth
        flat.append(item)
        for reply in replies:
            visit(reply, depth + 1)

    for comment in comments:
        visit(comment, 0)
    return flat


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Unsupported response type: {type(response)!r}")


def fetch_topic(client: DiscussionApiClient, topic_id: int, page_size: int) -> dict[str, Any]:
    topic_req = ApiGetTopicRequest()
    topic_req.id = topic_id
    topic_resp = response_to_dict(client.get_topic(topic_req))
    topic = topic_resp.get("topic") or {}

    comments: list[dict[str, Any]] = []
    page_token = ""
    while True:
        comments_req = ApiListCommentsRequest()
        comments_req.topic_id = topic_id
        comments_req.page_size = page_size
        comments_req.page_token = page_token
        comments_resp = response_to_dict(client.list_comments(comments_req))
        comments.extend(comments_resp.get("comments") or [])
        next_token = (
            comments_resp.get("nextPageToken")
            or comments_resp.get("next_page_token")
            or comments_resp.get("pageToken")
            or ""
        )
        if not next_token or next_token == page_token:
            break
        page_token = next_token

    messages: list[dict[str, Any]] = []
    first = {
        "id": topic.get("firstMessageId") or topic.get("id"),
        "authorName": topic.get("authorName") or topic.get("authorUserDisplayName") or "",
        "postDate": topic.get("postDate") or "",
        "content": topic.get("content") or "",
        "depth": 0,
        "first": True,
    }
    messages.append(first)
    for comment in flatten_comments(comments):
        comment = dict(comment)
        comment["first"] = False
        messages.append(comment)

    full_parts = []
    for msg in messages:
        text = msg.get("rawMarkdown") or html_to_text(str(msg.get("content") or ""))
        author = msg.get("authorName") or msg.get("authorUserName") or ""
        full_parts.append(
            f"[message_id={msg.get('id', '')} author={author} first={bool(msg.get('first'))}]\n{text}"
        )
    full_text = "\n\n".join(full_parts).strip()
    return {
        "topic_id": topic_id,
        "url": f"https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/{topic_id}",
        "topic": topic,
        "messages": messages,
        "message_count_observed": len(messages),
        "full_text": full_text,
        "text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def keyword_hits(text: str) -> dict[str, int]:
    lower = text.lower()
    return {kw: lower.count(kw) for kw in KEYWORDS if lower.count(kw)}


def load_cached_records(raw_dirs: list[Path]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        for path in raw_dir.glob("*.json"):
            try:
                topic_id = int(path.stem)
                records[topic_id] = json.loads(read_text(path))
            except Exception:
                continue
    return records


def build_summary(records: dict[int, dict[str, Any]], expected_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topic_id in sorted(records):
        record = records[topic_id]
        text = str(record.get("full_text") or "")
        hits = keyword_hits(text)
        relevance = sum(hits.values())
        title = ""
        topic = record.get("topic") or {}
        if isinstance(topic, dict):
            title = str(topic.get("title") or topic.get("name") or "")
        rows.append(
            {
                "topic_id": topic_id,
                "expected_input": topic_id in set(expected_ids),
                "title": title,
                "url": record.get("url")
                or f"https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/{topic_id}",
                "message_count_observed": record.get("message_count_observed"),
                "text_sha256": record.get("text_sha256"),
                "relevance_score": relevance,
                "keyword_hits": hits,
                "preview": re.sub(r"\s+", " ", text)[:500],
            }
        )
    return sorted(rows, key=lambda row: (-int(row["relevance_score"]), int(row["topic_id"])))


def write_markdown(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# KG1 V332 Kaggle Discussion Resume Audit",
        "",
        "## Scope",
        "",
        f"- Generated at UTC: `{manifest['generated_at_utc']}`",
        f"- Expected topic IDs: `{manifest['expected_topic_ids']}`",
        f"- Cached/fetched topic records: `{manifest['records_total_after']}`",
        f"- Newly fetched this run: `{manifest['records_fetched_this_run']}`",
        f"- Missing after this run: `{manifest['missing_after_count']}`",
        f"- Errors this run: `{manifest['errors_this_run']}`",
        "",
        "## Highest-Relevance Cached Topics",
        "",
    ]
    for row in rows[:30]:
        lines.extend(
            [
                f"### {row['topic_id']} - {row['title'] or '(no title)'}",
                "",
                f"- URL: {row['url']}",
                f"- Messages: `{row['message_count_observed']}`",
                f"- Relevance score: `{row['relevance_score']}`",
                f"- Keyword hits: `{json.dumps(row['keyword_hits'], sort_keys=True, ensure_ascii=False)}`",
                f"- Preview: {row['preview']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls-path", type=Path, required=True)
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--previous-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--sleep-s", type=float, default=3.0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--priority-ids",
        default="",
        help="Comma-separated topic IDs to fetch first when they are still missing.",
    )
    args = parser.parse_args()

    print("=== V332 KAGGLE DISCUSSION RESUME AUDIT START ===", flush=True)
    print(f"urls_path = {args.urls_path} exists = {args.urls_path.exists()}", flush=True)
    print(f"ids_path = {args.ids_path} exists = {args.ids_path.exists()}", flush=True)
    print(f"previous_raw_dir = {args.previous_raw_dir} exists = {args.previous_raw_dir.exists()}", flush=True)
    print(f"output_dir = {args.output_dir}", flush=True)
    print(f"max_new = {args.max_new}", flush=True)
    print(f"sleep_s = {args.sleep_s}", flush=True)

    expected_ids = parse_topic_ids([args.urls_path, args.ids_path])
    if not expected_ids:
        raise RuntimeError("No topic IDs found in input files")

    raw_dir = args.output_dir / "raw_topics"
    cached = load_cached_records([args.previous_raw_dir, raw_dir])
    missing_before = [topic_id for topic_id in expected_ids if topic_id not in cached]
    print(f"expected_topic_ids = {len(expected_ids)}", flush=True)
    print(f"cached_before = {len(cached)}", flush=True)
    print(f"missing_before = {len(missing_before)}", flush=True)

    client = DiscussionApiClient(KaggleHttpClient())
    fetched = 0
    errors: list[dict[str, Any]] = []
    priority_ids = []
    for raw_id in args.priority_ids.split(","):
        raw_id = raw_id.strip()
        if raw_id:
            priority_ids.append(int(raw_id))
    fetch_order = []
    for topic_id in priority_ids + missing_before:
        if topic_id in missing_before and topic_id not in fetch_order:
            fetch_order.append(topic_id)

    for topic_id in fetch_order[: max(0, args.max_new)]:
        print(f"fetch_topic_start topic_id={topic_id}", flush=True)
        try:
            record = fetch_topic(client, topic_id, args.page_size)
            write_json(raw_dir / f"{topic_id}.json", record)
            cached[topic_id] = record
            fetched += 1
            print(
                f"fetch_topic_ok topic_id={topic_id} messages={record['message_count_observed']} sha256={record['text_sha256']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - audit keeps evidence of API failures.
            error = {"topic_id": topic_id, "type": type(exc).__name__, "error": str(exc)}
            errors.append(error)
            print(f"fetch_topic_error {json.dumps(error, sort_keys=True, ensure_ascii=False)}", flush=True)
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                print("rate_limit_detected; stopping this batch", flush=True)
                break
        time.sleep(max(0.0, args.sleep_s))

    cached = load_cached_records([args.previous_raw_dir, raw_dir])
    missing_after = [topic_id for topic_id in expected_ids if topic_id not in cached]
    rows = build_summary(cached, expected_ids)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_urls_path": str(args.urls_path),
        "input_ids_path": str(args.ids_path),
        "previous_raw_dir": str(args.previous_raw_dir),
        "raw_dir": str(raw_dir),
        "expected_topic_ids": len(expected_ids),
        "records_total_after": len(cached),
        "records_fetched_this_run": fetched,
        "missing_before_count": len(missing_before),
        "missing_after_count": len(missing_after),
        "missing_after_first50": missing_after[:50],
        "errors_this_run": len(errors),
        "errors": errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "v332_kaggle_discussion_resume_manifest.json", manifest)
    write_json(args.output_dir / "v332_kaggle_discussion_relevance_summary.json", rows)
    write_markdown(args.output_dir / "KG1_V332_KAGGLE_DISCUSSION_RESUME_AUDIT.md", manifest, rows)
    print("manifest = " + json.dumps(manifest, sort_keys=True, ensure_ascii=False), flush=True)
    print("=== V332 KAGGLE DISCUSSION RESUME AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
