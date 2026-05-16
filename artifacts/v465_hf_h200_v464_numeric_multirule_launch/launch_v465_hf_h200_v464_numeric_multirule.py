#!/usr/bin/env python3
"""Archived KG1 launcher for quarantined V465/V464 numeric route; fail-closed."""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "Archived KG1 launcher: quarantined V465/V464 numeric multirule route; fail-closed. "
        "V464 contains rejected-candidate contradictions and must not be trained."
    )


if __name__ == "__main__":
    raise SystemExit(main())
