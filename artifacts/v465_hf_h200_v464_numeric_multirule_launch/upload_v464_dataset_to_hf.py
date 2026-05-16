#!/usr/bin/env python3
"""Archived KG1 uploader for quarantined V464 dataset; fail-closed."""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "Archived KG1 launcher: quarantined V464 dataset upload; fail-closed. "
        "Do not upload or reuse V464 without rebuilding a clean version."
    )


if __name__ == "__main__":
    raise SystemExit(main())
