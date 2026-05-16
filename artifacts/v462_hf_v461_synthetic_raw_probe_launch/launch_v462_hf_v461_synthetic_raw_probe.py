#!/usr/bin/env python3
"""Archived KG1 launcher for quarantined V462/V461 synthetic route; fail-closed.

V472 found a full-reference exact prompt/answer seed in the V461 prompt pack.
The raw-output probe is no longer allowed because it can regenerate evidence
from contaminated prompts.
"""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "Archived KG1 launcher: quarantined V462/V461 synthetic route; fail-closed. "
        "Build a new prompt pack without full-reference overlap before probing."
    )


if __name__ == "__main__":
    raise SystemExit(main())
