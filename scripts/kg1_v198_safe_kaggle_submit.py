#!/usr/bin/env python3
"""Safely submit the gated V198 adapter ZIP to Kaggle.

This script is intended for Colab after the V198 final double-check has passed.
It validates the double-check JSON, verifies the ZIP SHA256, prepares Kaggle
credentials from standard Colab/Drive locations, and submits through the Python
Kaggle API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_kaggle_credentials(kaggle_json: Path | None) -> dict[str, Any]:
    home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
    candidates = []
    if kaggle_json:
        candidates.append(kaggle_json)
    candidates.extend([
        home_kaggle,
        Path("/content/drive/MyDrive/kaggle.json"),
        Path("/content/kaggle.json"),
        Path.cwd() / "kaggle.json",
    ])

    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "kaggle.json not found. Put it at /content/drive/MyDrive/kaggle.json, "
            "or upload it to /content/kaggle.json, or pass --kaggle-json."
        )

    home_kaggle.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != home_kaggle.resolve():
        shutil.copy(source, home_kaggle)
    home_kaggle.chmod(stat.S_IRUSR | stat.S_IWUSR)
    creds = load_json(home_kaggle)
    if not creds.get("username") or not creds.get("key"):
        raise ValueError(f"Invalid Kaggle credentials file: {home_kaggle}")
    return {"path": str(home_kaggle), "source": str(source), "username": creds.get("username")}


def validate_doublecheck(candidate_zip: Path, doublecheck_json: Path, expected_sha256: str | None) -> dict[str, Any]:
    if not candidate_zip.exists():
        raise FileNotFoundError(candidate_zip)
    if not doublecheck_json.exists():
        raise FileNotFoundError(doublecheck_json)

    report = load_json(doublecheck_json)
    decision = report.get("decision") or {}
    reasons = decision.get("reasons") or []
    if not decision.get("submit_ready") or reasons:
        raise RuntimeError(f"Doublecheck did not approve submission: submit_ready={decision.get('submit_ready')} reasons={reasons}")

    actual_sha = sha256_file(candidate_zip)
    expected = expected_sha256 or str(report.get("zip", {}).get("sha256") or "")
    if expected and actual_sha != expected:
        raise RuntimeError(f"Candidate ZIP SHA mismatch: actual={actual_sha} expected={expected}")

    report_zip = Path(str(report.get("candidate_zip") or ""))
    if report_zip.name and report_zip.name != candidate_zip.name:
        raise RuntimeError(f"Doublecheck candidate mismatch: report={report_zip} submit={candidate_zip}")

    return {
        "candidate_zip": str(candidate_zip),
        "zip_sha256": actual_sha,
        "doublecheck_json": str(doublecheck_json),
        "doublecheck_generated_at": report.get("generated_at"),
    }


def parse_score(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def exception_details(exc: BaseException) -> dict[str, Any]:
    details: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    for attr in ("status", "reason", "body", "headers"):
        if hasattr(exc, attr):
            value = getattr(exc, attr)
            if value is not None:
                details[attr] = str(value)
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        details["cause"] = {
            "type": type(cause).__name__,
            "message": str(cause),
        }
        for attr in ("status", "reason", "body", "headers"):
            if hasattr(cause, attr):
                value = getattr(cause, attr)
                if value is not None:
                    details["cause"][attr] = str(value)
    return details


def summarize_latest_submissions(api: Any, limit: int = 5) -> list[dict[str, Any]]:
    try:
        submissions = api.competition_submissions(COMPETITION)
    except Exception as exc:
        return [{"error": exception_details(exc)}]
    rows = []
    for sub in submissions[:limit]:
        rows.append({
            "ref": getattr(sub, "ref", None),
            "date": getattr(sub, "date", None),
            "description": getattr(sub, "description", None),
            "status": getattr(sub, "status", None),
            "public_score": parse_score(getattr(sub, "publicScore", None)),
            "private_score": parse_score(getattr(sub, "privateScore", None)),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-zip", type=Path, required=True)
    parser.add_argument("--doublecheck-json", type=Path, required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--kaggle-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-raise-on-submit-error",
        action="store_true",
        help="Write diagnostics and exit 0 when Kaggle rejects the submission. Useful in notebooks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_zip = args.candidate_zip.resolve()
    doublecheck_json = args.doublecheck_json.resolve()
    validation = validate_doublecheck(candidate_zip, doublecheck_json, args.expected_sha256)
    creds = ensure_kaggle_credentials(args.kaggle_json)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError("kaggle package is required. Run: pip install -q --upgrade kaggle") from exc

    api = KaggleApi()
    api.authenticate()
    before = summarize_latest_submissions(api, limit=5)
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "competition": COMPETITION,
        "dry_run": args.dry_run,
        "validation": validation,
        "credentials": creds,
        "message": args.message,
        "submissions_before": before,
    }

    print("=== KG1 V198 SAFE KAGGLE SUBMIT ===")
    print(f"candidate_zip: {candidate_zip}")
    print(f"zip_sha256: {validation['zip_sha256']}")
    print(f"kaggle_user: {creds['username']}")
    print(f"dry_run: {args.dry_run}")

    if args.dry_run:
        print("DRY RUN: not submitting.")
        result["submitted"] = False
    else:
        try:
            response = api.competition_submit(str(candidate_zip), args.message, COMPETITION)
        except Exception as exc:
            details = exception_details(exc)
            result["submitted"] = False
            result["submit_error"] = details
            result["submissions_after"] = summarize_latest_submissions(api, limit=8)
            write_json(args.output_json, result)
            print("submit_failed: True")
            print(f"submit_error_type: {details.get('type')}")
            print(f"submit_error_message: {details.get('message')}")
            if details.get("status"):
                print(f"submit_error_status: {details.get('status')}")
            if details.get("reason"):
                print(f"submit_error_reason: {details.get('reason')}")
            if details.get("body"):
                print("submit_error_body:")
                print(str(details["body"])[:4000])
            print(f"report: {args.output_json}")
            if args.no_raise_on_submit_error:
                return 0
            raise
        else:
            result["submitted"] = True
            result["submit_response"] = str(response)
            print(f"submit_response: {response}")

    if args.poll_seconds > 0:
        print(f"Waiting {args.poll_seconds}s before polling submissions...")
        time.sleep(args.poll_seconds)
    result["submissions_after"] = summarize_latest_submissions(api, limit=8)
    write_json(args.output_json, result)
    print(f"report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
