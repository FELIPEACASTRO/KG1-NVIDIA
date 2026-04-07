"""Fix Cell 3 EXCLUDE_OUT_PROJ block insertion using regex (whitespace-tolerant)."""
from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/KG1_v42_NONUPLE.ipynb")


def fix_cell3(source: str) -> str:
    """Replace simple `target = 'all-linear'` line with full EXCLUDE_OUT_PROJ block.

    Uses regex to match across whitespace variations.
    """
    new_block = '''    # NONUPLE: choose target modules
    if EXCLUDE_OUT_PROJ:
        # NVIDIA NeMo official YAML: exclude out_proj (Mamba uses custom kernels - LoRA broken)
        target = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "in_proj",                                 # Mamba in_proj only (no out_proj)
            "up_proj", "down_proj", "gate_proj",       # FFN
        ]
        print(f"  NONUPLE target_modules (exclude out_proj): {target}")
    else:
        target = "all-linear"  # PROVEN v30/v41 baseline
        print(f"  Standard target: all-linear (v30/v41 PROVEN)")
'''

    # Match: optional comments, blank lines, target = "all-linear" assignment
    pattern = re.compile(
        r"    # Determine target modules:.*?target = \"all-linear\".*?\n",
        re.DOTALL,
    )
    if not pattern.search(source):
        print("  WARNING: pattern not found, trying fallback")
        # Fallback: just match target = "all-linear" line
        pattern = re.compile(r'    target = "all-linear".*?\n')

    matches = pattern.findall(source)
    print(f"  Found {len(matches)} matches")

    new_source, n = pattern.subn(new_block, source, count=1)
    print(f"  Substituted {n} occurrence(s)")

    return new_source


def main():
    with NOTEBOOK_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    cell3 = nb["cells"][3]
    cell3_src = "".join(cell3["source"])

    new_src = fix_cell3(cell3_src)

    if "if EXCLUDE_OUT_PROJ:" not in new_src:
        print("FAILED: substitution did not apply")
        return 1

    cell3["source"] = new_src.splitlines(keepends=True)

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("SUCCESS: Cell 3 patched with EXCLUDE_OUT_PROJ block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
