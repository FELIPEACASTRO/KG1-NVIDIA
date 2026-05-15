# V453 Public Kaggle Kernel Mining

Generated: 2026-05-15T21:19:50.035941+00:00

## Resultado

| Item | Valor |
|---|---:|
| Kernels listados | `30` |
| Kernels analisados | `30` |
| Pull failures | `1` |

## Top sinais tecnicos

| Ref | Score | Snippets | Decisao |
|---|---:|---|---|
| suryamilenial/end-to-end-finetuning-for-lb-0-83-6e2fa5 | `25` | TARGET_MODULES = [<br>"lm_head",<br>target_modules=TARGET_MODULES, | triage_manual |
| huikang/end-to-end-finetuning-for-lb-0-85 | `25` | TARGET_MODULES = [<br>"lm_head",<br>target_modules=TARGET_MODULES, | triage_manual |
| matthewblakeward/steinifrank | `13` | ('bit_manipulation', [r'bit', r'xor', r'and\b', r'or\b', r'rotate', r'mask', r'binary', r'hexadecimal', r'0x']),<br>('equation_numeric', [r'equation', r'function', r'f\(', r'solve for', r'deduce', r'guess', r'sequence', r'pattern']),<br>return 'equation_numeric' | triage_manual |
| huikang/tinker-submission-notebook | `10` | trained_adapter_config["target_modules"] = [<br>"lm_head",<br># # Skip lm_head (not in reference adapter) | triage_manual |
| kienngx/nvidia-nemotron-trained-models-submission | `8` | * target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]<br>* target_modules="all-linear"<br>* target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"] | triage_manual |
| drchenb/nvidia-nemotron-trained-models-submission | `8` | * target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]<br>* target_modules="all-linear"<br>* target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"] | triage_manual |
| rauffauzanrambe/lora-nvidia-nemotron-models-with-pytorch | `8` | * target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]<br>* target_modules="all-linear"<br>* target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"] | triage_manual |
| markjcooper/thk-public-fork-2026-05-14-v14-tinker-adapter | `8` | * target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]<br>* target_modules="all-linear"<br>* target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"] | triage_manual |
| teasue05/tinker-submission-notebook | `7` | trained_adapter_config["target_modules"] = [<br>"lm_head",<br># # Skip lm_head (not in reference adapter) | triage_manual |
| afr1ste/nemotron-0-86-tinker-adapter-guide | `6` | \| Sluitel broad adapter with best `lm_head` fill \| 0.62 \| Structural compatibility alone is not enough \|<br>\| Error1249x broad adapter with best `lm_head` fill \| 0.59 \| Another valid but task-misaligned broad adapter \|<br>For this baseline, the important signs are a broad LoRA target set, rank 32, alpha 32, and root-level adapter files. The target modules include both attention projections and expert/feed-forward projections, plus `lm_hea | triage_manual |
| rn8205/adapter-validation-notebook | `6` | CRYPTARITHM_SOLVER_PREFIX = """<br>if category in ["cryptarithm_guess", "equation_numeric_guess"]:<br>return "bit_manipulation" | triage_manual |
| dgxchen/training-with-unsloth-to-achieve-0-85-lb | `5` | In this version, training on **"lm_head"** in target_modules has been **removed**. At the same time, the microbatch size has been reset to **1**. In my experiments, increasing the microbatch size to 2 reduced training ti<br>target_modules = [<br>target_modules=target_modules, | triage_manual |
| huikang/adapter-validation-notebook | `3` | return "bit_manipulation"<br>return "equation_numeric_deduce"<br>return "equation_numeric_guess" | triage_manual |
| asalhi/tinker-adapter-to-ready-to-submit-adapter | `2` | target_modules,<br>A._add_peft_weight(peft_target_key, merged_lora_A, merged_lora_B, peft_weights, target_modules) | triage_manual |
| mirzayasirabdullah07/nvidia-nemotron-model-notebook | `2` | cfg["target_modules"] = [<br>"up_proj", "v_proj", "down_proj", "out_proj", "lm_head", | triage_manual |

## Decisao

Esta mineracao e CPU-only e nao produz artefato submit-ready. Qualquer
tecnica encontrada aqui precisa virar regra local, passar por gate V452/V453
e provar `total>192`, `equation>56`, `bit>=136`, `truncated=0` antes de HF.

Raw notebooks baixados foram removidos apos extracao dos sinais leves.
