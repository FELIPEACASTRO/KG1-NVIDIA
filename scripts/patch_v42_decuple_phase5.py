"""Add Phase 5 (DECUPLE livctr hybrid) to v42 NONUPLE notebook.

Phase 5 incorporates hyperparameters from livctr/nvidia-nemotron-kaggle (LB 0.74 confirmed):
- learning_rate = 1e-4 (exp41)
- max_length = 1536 (exp41)
- n_epochs = 1 (exp41)
- grad_accum = 16 (matches kbsooo v1 effective batch size)
- use_cot = True (solver-augmented CoTs from solver_augmented_train.jsonl)
- 4 submit_steps (respects Kaggle 5/day limit, leaves 1 slot for averaged)

All Phase 4 NONUPLE flags stay on (freeze_moe_router, checkpoint_averaging, smoke test, etc).

Run from project root:
    python scripts/patch_v42_decuple_phase5.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/KG1_v42_NONUPLE.ipynb")

PHASE_5_BLOCK = '''    5: {  # v43 DECUPLE: v42 Phase 4 + livctr hyperparams (LB 0.74 confirmed) + solver CoT data
        "name": "v43-decuple-livctr",
        "output_repo": "felipesp1983/kg1-nemotron-lora-v43-decuple",
        # livctr exp41 hyperparams (LB 0.74 CONFIRMED, github.com/livctr/nvidia-nemotron-kaggle):
        "n_examples": 5000,     # total, distributed per family (not flat)
        "n_epochs": 1,          # livctr exp41: 1 epoch only
        "learning_rate": 1e-4,  # livctr exp41 (vs v41/v42 Phase 4: 5e-5)
        "max_length": 1536,     # livctr exp41 (vs v41: 1024)
        "grad_accum": 16,       # effective bs=16 (kbsooo v1 matches)
        "warmup_ratio": 0.05,
        "use_thinking": True,   # solver CoTs have <think> tags
        "use_cot": True,        # enable CoT path (solver-generated)
        "warm_start_repo": None,  # fresh LoRA (adapter chaining optional in separate run)
        # NONUPLE + DECUPLE additions:
        "submit_steps": [100, 200, 300, 400],  # 4 submits (leaves 1 Kaggle daily slot for averaged)
        "eq_oversample": 2.0,
        "freeze_moe_router": True,     # NONUPLE: Aman Atar - prevent router collapse
        "exclude_out_proj": False,     # keep baseline (test True in follow-up)
        "checkpoint_averaging": True,  # NONUPLE: AIMO-2 averaging (+1-2 pts free)
        "skip_pretrain_smoke": False,  # NONUPLE: enforce smoke test
        "skip_prescore_gate": False,   # NONUPLE: enforce pre-score gate
        "prescore_min_threshold": 0.60,
        # DECUPLE NEW:
        "use_solver_augmented_data": True,  # load data/solver_augmented_train.jsonl in Cell 2
        "solver_augmented_ratio": 0.70,     # 70% solver CoTs + 30% original train.csv
        "source_recipe": "livctr_exp41_hybrid",
    },
}'''


def main() -> int:
    if not NOTEBOOK_PATH.exists():
        print(f"ERROR: {NOTEBOOK_PATH} not found")
        return 1

    with NOTEBOOK_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    cell1 = nb["cells"][1]
    cell1_src = "".join(cell1["source"])

    # 1. Update PHASE param to include 5
    cell1_src = cell1_src.replace(
        'PHASE = 1  #@param [1, 2, 3, 4] {type:"integer"}',
        'PHASE = 1  #@param [1, 2, 3, 4, 5] {type:"integer"}',
    )

    # 2. Add Phase 5 right after Phase 4's closing "    },"
    # Use literal string replacement (more robust than regex with comments)
    phase_4_end = '        "prescore_min_threshold": 0.60,  # gate: only train if smoke prescore >= 0.60\n    },\n}'
    phase_4_plus_5 = (
        '        "prescore_min_threshold": 0.60,  # gate: only train if smoke prescore >= 0.60\n'
        '    },\n'
        + PHASE_5_BLOCK  # this already ends with "    },\n}"
    )

    if phase_4_end not in cell1_src:
        print("ERROR: Phase 4 end marker not found in Cell 1 source")
        return 2

    new_src = cell1_src.replace(phase_4_end, phase_4_plus_5, 1)

    # 3. Add new DECUPLE flags exposure (USE_SOLVER_AUG + SOLVER_AUG_RATIO + SOURCE_RECIPE)
    decuple_flags = '''
# ============================================================
# DECUPLE FLAGS (Phase 5 only)
# ============================================================
USE_SOLVER_AUGMENTED_DATA = CFG.get("use_solver_augmented_data", False)
SOLVER_AUGMENTED_RATIO = CFG.get("solver_augmented_ratio", 0.0)
SOURCE_RECIPE = CFG.get("source_recipe", "v41_proven")

if PHASE == 5:
    print(f"\\n{'='*60}")
    print(f"  DECUPLE FLAGS (Phase 5) — livctr LB 0.74 hybrid")
    print(f"  source_recipe: {SOURCE_RECIPE}")
    print(f"  use_solver_augmented_data: {USE_SOLVER_AUGMENTED_DATA}")
    print(f"  solver_augmented_ratio: {SOLVER_AUGMENTED_RATIO}")
    print(f"  learning_rate: {CFG['learning_rate']} (livctr exp41: 1e-4)")
    print(f"  max_length: {CFG['max_length']} (livctr exp41: 1536)")
    print(f"  n_epochs: {CFG['n_epochs']} (livctr exp41: 1)")
    print(f"  grad_accum: {CFG['grad_accum']} (effective bs={CFG['grad_accum']})")
    print(f"{'='*60}")
'''

    # Insert DECUPLE flags after NONUPLE flags block (before "✅ CELL 1 COMPLETE")
    # The NONUPLE block ends with the "if PHASE == 4:" check
    nonuple_end_marker = 'print(f"{\'=\'*60}")'
    # Find the NONUPLE flags print block and insert DECUPLE after it
    if 'DECUPLE FLAGS' not in new_src:
        # Insert before the final "✅ CELL 1 COMPLETE" print
        insert_before = 'print("\\n✅ CELL 1 COMPLETE")'
        new_src = new_src.replace(
            insert_before,
            decuple_flags + '\n' + insert_before,
        )

    cell1["source"] = new_src.splitlines(keepends=True)

    # 4. Patch Cell 2 (data loading) to handle USE_SOLVER_AUGMENTED_DATA
    cell2 = nb["cells"][2]
    cell2_src = "".join(cell2["source"])

    # Add solver-augmented loading at the beginning of Cell 2 data loading
    solver_aug_loader = '''
# ============================================================
# DECUPLE: LOAD SOLVER-AUGMENTED DATA (Phase 5 only)
# ============================================================
solver_aug_records = []
if USE_SOLVER_AUGMENTED_DATA:
    print("\\n=== DECUPLE: Loading solver-augmented data ===")
    # Try local path first, then HF repo fallback
    solver_aug_path = None
    for candidate in [
        "data/solver_augmented_train.jsonl",
        "/tmp/kg1_data/data/solver_augmented_train.jsonl",
        "/content/kg1-nvidia/data/solver_augmented_train.jsonl",
    ]:
        if os.path.exists(candidate):
            solver_aug_path = candidate
            break

    if solver_aug_path is None:
        # Try to download from HF (Felipe needs to have uploaded it)
        try:
            hf_hub_download(
                repo_id=DATA_REPO,
                filename="solver_augmented_train.jsonl",
                local_dir="/tmp/kg1_data/",
                repo_type="dataset",
            )
            solver_aug_path = "/tmp/kg1_data/solver_augmented_train.jsonl"
        except Exception as e:
            print(f"  ⚠️  Could not load solver_augmented_train.jsonl: {e}")
            print(f"  ⚠️  Falling back to original training data only")

    if solver_aug_path:
        with open(solver_aug_path, encoding="utf-8") as f:
            for line in f:
                solver_aug_records.append(json.loads(line))
        print(f"  ✓ Loaded {len(solver_aug_records)} solver-augmented records")
        # Stats per family
        from collections import Counter as _Counter
        fam_counts = _Counter(r["family"] for r in solver_aug_records)
        for fam, cnt in sorted(fam_counts.items()):
            print(f"    {fam}: {cnt}")

'''

    # Insert right after "# ============================================================\n# DOWNLOAD DATA"
    download_marker = 'print("=== Downloading training data ===")'
    if 'DECUPLE: LOAD SOLVER-AUGMENTED DATA' not in cell2_src:
        cell2_src = cell2_src.replace(
            download_marker,
            download_marker + '\n' + solver_aug_loader.rstrip() + '\n',
        )

    # 5. Modify the sampling loop to use solver_aug_records when USE_SOLVER_AUGMENTED_DATA=True
    # Find the "for _, row in sampled.iterrows():" block and add solver_aug injection before it
    mixer_code = '''
# ============================================================
# DECUPLE: MIX SOLVER-AUGMENTED WITH ORIGINAL (Phase 5)
# ============================================================
if USE_SOLVER_AUGMENTED_DATA and solver_aug_records:
    print(f"\\n=== DECUPLE: Mixing {SOLVER_AUGMENTED_RATIO*100:.0f}% solver-CoT + {(1-SOLVER_AUGMENTED_RATIO)*100:.0f}% original ===")
    # Compute how many from each source
    n_total_target = CFG["n_examples"]
    n_solver = int(n_total_target * SOLVER_AUGMENTED_RATIO)
    n_original = n_total_target - n_solver
    print(f"  target: {n_total_target} = {n_solver} solver + {n_original} original")

    # Shuffle and take n_solver from solver_aug_records
    random.seed(42)
    random.shuffle(solver_aug_records)
    solver_take = solver_aug_records[:n_solver]

    # Convert solver records to the same "examples" format as original
    solver_examples = []
    solver_ids = []
    for rec in solver_take:
        solver_examples.append({"messages": rec["messages"]})
        solver_ids.append(rec["id"])
    print(f"  ✓ Collected {len(solver_examples)} solver-CoT examples")

    # Prepend solver examples to the existing examples list
    # (they will be shuffled together in the next step)
    examples = solver_examples + examples[:n_original]
    example_ids = solver_ids + example_ids[:n_original]
    print(f"  ✓ Combined dataset: {len(examples)} examples")

'''

    # Insert mixer before the shuffle block
    shuffle_marker = '# Shuffle and truncate (IDENTICAL TO v30'
    if 'DECUPLE: MIX SOLVER-AUGMENTED' not in cell2_src:
        cell2_src = cell2_src.replace(
            shuffle_marker,
            mixer_code.rstrip() + '\n\n' + shuffle_marker,
        )

    cell2["source"] = cell2_src.splitlines(keepends=True)

    # Save
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"SUCCESS: Phase 5 DECUPLE added to {NOTEBOOK_PATH}")
    print(f"Cells: {len(nb['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
