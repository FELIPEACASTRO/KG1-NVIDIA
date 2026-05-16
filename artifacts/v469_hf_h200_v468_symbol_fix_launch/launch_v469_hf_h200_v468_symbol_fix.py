#!/usr/bin/env python3
"""Archived KG1 launcher for quarantined V469/V468 symbol-fix route; fail-closed."""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "Archived KG1 launcher: quarantined V469/V468 symbol-fix route; fail-closed. "
        "V468 still contains full-reference contamination."
    )


if __name__ == "__main__":
    raise SystemExit(main())
