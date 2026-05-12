# mistralai/mistral-nemo

role: assistant
sha256: 4063645a6daf8fd1923a25d8bbf6120354df6a61966a3599cda8873f703c20b7
chars: 4165

msg_tmp_ttfm4fswi4e
message
completed
assistant
output_text
**Distillation Plan**

**1. Root Cause Hypotheses**

- **V313 (SFT verifier synthetic)**:
  - Insufficient data diversity for equation transformation.
  - Lack of targeted loss weighting for equation gains.
  - Possible overfitting to synthetic data.

- **V315 (preference on V312)**:
  - Inadequate fine-tuning of LoRA modules, leading to minimal improvement.
  - Insufficient exposure to equation transformation examples during training.
  - Loss of bit manipulation accuracy due to aggressive optimization.

**2. Equation Transformation Plan**

- **Trace Style**: Use detailed, step-by-step traces for equation transformation gains. Include intermediate steps and guardrails to prevent loss of existing accuracy.
- **Data Mix**:
  - Increase the ratio of equation transformation examples in the dataset to 60% (vs. 40% bit manipulation).
  - Ensure a balanced mix of positive and negative examples for each gain rule.
- **Loss Weighting**:
  - Assign higher weights to equation transformation examples (e.g., 1.5x) to emphasize their importance.
  - Implement gain-specific loss weights based on the difficulty and importance of each rule.
- **Guardrails**:
  - Implement a confidence threshold for equation transformation predictions to prevent incorrect changes.
  - Use a conservative promotion gate that prioritizes equation transformation gains without sacrificing bit manipulation accuracy.

**3. Bit Manipulation Plan**

- **Trace Style**: Maintain detailed, step-by-step traces for bit manipulation examples, focusing on fullbyte_safe_ternary and fullbyte_binary operations.
- **Data Mix**:
  - Maintain a 40% ratio of bit manipulation examples in the dataset.
  - Ensure a balanced mix of fullbyte_safe_ternary and fullbyte_binary examples.
- **Loss Weighting**:
  - Assign equal loss weights to bit manipulation examples to maintain existing accuracy.
  - Implement gain-specific loss weights for fullbyte_safe_ternary and fullbyte_binary rules.
- **Guardrails**:
  - Monitor bit manipulation accuracy closely during training and validation.
  - Implement a stop rule if bit manipulation accuracy drops below the current baseline.

**4. Dataset Blueprint**

- **Train**: 700 rows (60% equation transformation, 40% bit manipulation)
  - Equation transformation: 420 rows (60% positive, 40% negative)
  - Bit manipulation: 280 rows (50% fullbyte_safe_ternary, 50% fullbyte_binary)
- **Validation**: 247 rows (60% equation transformation, 40% bit manipulation)
  - Equation transformation: 148 rows (60% positive, 40% negative)
  - Bit manipulation: 99 rows (50% fullbyte_safe_ternary, 50% fullbyte_binary)

**5. Training Recipe**

- **LoRA Modules**: Fine-tune LoRA modules for equation transformation and bit manipulation tasks separately. Use a shared LoRA module for common layers.
- **LR/Steps/Epochs**:
  - Learning rate: 1e-4 (conservative) to 5e-4 (aggressive)
  - Steps: 10,000 (conservative) to 20,000 (aggressive)
  - Epochs: 10 (conservative) to 20 (aggressive)
- **Conservative vs. Aggressive Branches**:
  - Conservative: Lower learning rate, fewer steps, and fewer epochs.
  - Aggressive: Higher learning rate, more steps, and more epochs.

**6. Validation Gate**

- **Metrics**:
  - Overall accuracy
  - Equation transformation accuracy (weak)
  - Bit manipulation accuracy (weak)
  - Truncation rate
- **Stop/Promote Rules**:
  - Stop if weak equation transformation < 60 or weak bit manipulation < 136.
  - Promote if weak equation transformation ≥ 60, weak bit manipulation ≥ 136, overall ≥ 193, and truncation is acceptable.

**7. Risks**

- **Failure Modes**:
  - Overfitting to equation transformation examples, leading to bit manipulation regression.
  - Insufficient exposure to bit manipulation examples, causing accuracy loss.
  - Loss of existing accuracy due to aggressive optimization or inadequate guardrails.

**8. Do Not Do**

- **Waste HF Budget**:
  - Do not use excessive compute resources for fine-tuning LoRA modules.
  - Avoid unnecessary full model training or evaluation steps.
  - Do not use expensive data augmentation techniques without clear benefits.
