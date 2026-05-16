#!/usr/bin/env python3
"""Archived KG1 weak-eval launcher for quarantined V469 adapter; fail-closed."""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "Archived KG1 launcher: quarantined V469 adapter from V468 data; fail-closed. "
        "Do not spend HF GPU evaluating this adapter."
    )


if __name__ == "__main__":
    raise SystemExit(main())
