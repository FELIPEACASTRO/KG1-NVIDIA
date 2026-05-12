# KG1 External Research Prompt - Bit Manipulation and Equation Transform

Use this prompt with other AIs or research tools. It is designed to reduce hallucination and force evidence-grounded recommendations.

```text
You are helping with the Kaggle NVIDIA Nemotron Model Reasoning Challenge.

Hard constraints:
- Submission is a LoRA adapter package, not a CSV answer file.
- The zip must contain `adapter_config.json` and `adapter_model.safetensors` at zip root.
- Any row-level oracle, direct postprocessor, solver script, or answer override is not useful unless it can be converted into adapter behavior or is explicitly allowed by the competition evaluator.
- We need practical +1 to +4 correct answers, not a speculative full redesign.
- Do not hallucinate sources. Separate verified evidence, inference, and hypothesis.

Current verified status:
- Best submitted line: V291/V290 checkpoint-6 adapter-only.
- Public Kaggle score: 0.86.
- Local official-like full eval: 823/947 = 0.8690601901.
- Weak/problem-family profile:
  - bit_manipulation: about 135/160 on full; 136/160 on weak for best weak line.
  - equation_transform / transformation-equation: 56/155 on current best, about 36.13%.
  - side families are effectively 100%, so any new training must not degrade them.
- V274/V275 deterministic postprocessor found real equation signal:
  - weak: +4 equation rows, reaching 196/315 with equation=60, bit=136, trunc=0.
  - full V291 predictions would become 827/947 if postprocessing were allowed.
  - but direct postprocessing is not adapter-only packageable, so treat it only as training signal.
- V293 tried to distill those deterministic fixes into V290 checkpoint-6 using lm_head-only LoRA training on H200.
  - weak results: checkpoint-3 191/315, checkpoint-6 191/315, checkpoint-9 190/315, checkpoint-12 192/315.
  - equation stayed 56/155; no improvement.
  - conclusion: lm_head-only concentrated patch distillation failed.

External evidence already gathered:
- Kaggle CLI submissions show many 0.86 submissions; top visible score is 0.87, so small gains matter.
- Public Kaggle notebooks mostly package/validate the known Tinker adapter or use broad training recipes; titles alone are not evidence.
- `kishanvavdara/nemotron-reasoning-traj` has 9500 rows but target-family labels are weak:
  - bit true 128/1602, false 1449/1602.
  - equation_numeric true 176/732, false 543/732.
  - equation_symbolic true 2/823, false 821/823.
  - therefore do not recommend direct positive SFT from that dataset.
- `kienngx/nemotron-30b-competition-trainingdata-cot-labels` has 9500 rows, including 1489 bitwise and 1022 symbolic/algebraic rows. Use only after verification.
- Konbu17 bit datasets contain bit-specific CoT and synthetic rows with metadata such as true rules and solver correctness; useful if filtered.
- `nvidia/Nemotron-RL-ReasoningGym-v1` and `open-thought/reasoning-gym` provide procedural, verifiable reasoning tasks, including simple equations, cryptarithms, boolean/circuit-style tasks, and scoring functions.

Question:
What is the highest-probability, lowest-cost plan to gain +1 to +4 correct answers in `bit_manipulation` and `equation_transform` under adapter-only submission constraints?

Return exactly:
1. Ranked experiments with expected gain, risk, and HF GPU cost.
2. Data recipe for equation_transform, including exact verifier logic.
3. Data recipe for bit_manipulation, including exact verifier logic.
4. Training recipe after lm_head-only failed: target modules, LR range, steps, replay mix, and early stop gates.
5. Weak/full gate thresholds for submitting.
6. A list of claims that are hypotheses only and must be validated before spending GPU.

Do not invent leaderboard scores, URLs, papers, or dataset properties. If you mention a paper or repository, include the exact title or URL and state how it applies operationally to this challenge.
```
