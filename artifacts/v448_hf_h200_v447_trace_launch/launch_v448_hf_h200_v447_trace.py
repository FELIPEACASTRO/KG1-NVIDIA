#!/usr/bin/env python3
"""Archived KG1 launcher for quarantined V448/V447 trace route; fail-closed.

V472 found that the V447 trace dataset contains ``hypothesis_formed`` traces
with contradictory boxed answers. Do not relaunch this H200 train job. Rebuild
a new clean dataset under a new version and pass V286/reference gates first.
"""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "Archived KG1 launcher: quarantined V448/V447 trace route; fail-closed. "
        "Use a new clean dataset/version after V286/reference gates pass."
    )


if __name__ == "__main__":
    raise SystemExit(main())
