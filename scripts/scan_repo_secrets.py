#!/usr/bin/env python3
"""Cheap repository secret scan for active KG1 release code.

This is not a replacement for provider-side secret scanning. It blocks the
patterns that previously appeared in active scripts and keeps CI from accepting
new hardcoded API/Kaggle credentials.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT / ".github",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "notebooks" / "KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb",
]
SK_PREFIX = "s" + "k-"
PATTERNS = {
    "openai_or_deepseek_key": re.compile(rf"{re.escape(SK_PREFIX)}(?:proj-)?[A-Za-z0-9_-]{{20,}}"),
    "gemini_key": re.compile(r"AI" + r"za[A-Za-z0-9_-]{20,}"),
    "xai_key": re.compile(r"xai-[A-Za-z0-9_-]{20,}"),
    "literal_kaggle_key": re.compile(r"KAGGLE_KEY\s*=\s*['\"][^'\"]{10,}['\"]"),
}
EXTENSIONS = {".py", ".ipynb", ".yml", ".yaml"}
SKIP_NAMES = {"scan_repo_secrets.py"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    return sorted({path for path in files if path.name not in SKIP_NAMES})


def main() -> int:
    findings: list[str] = []
    for path in iter_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{rel}:{line}: {name}")
    if findings:
        print("secret_scan_failed=true")
        for finding in findings:
            print(finding)
        return 1
    print("secret_scan_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
