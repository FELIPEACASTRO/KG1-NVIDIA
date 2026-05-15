# KG1 V446 URL Double Check - OpenRouter Chat Fri May 15 2026 (1)

- Source file: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026 (1).json`
- Source SHA256: `e571b110fed9179b2c247ff6a9c16493a60c3f6770c3a1bbf5c1fc8b9584d54d`
- Unique URLs extracted: `101`
- Challenge/content URLs: `61`
- Provider metadata/status/TOS URLs: `40`
- Method: bounded HTTP GET with `Range: bytes=0-4095`, Kaggle CLI pages/files where available, Hugging Face Hub API metadata, and raw GitHub source inspection. No large datasets/PDFs were downloaded.

## Decision

No new submit-safe ACC gain was found in the URL audit. The audit does change the implementation order: stop GPU training until CPU target-alignment is proven, and prioritize adapting public Tong Hui Kang reasoner/corpus techniques into our CPU gates.

## Actionable Findings

1. `tonghuikang/nemotron` is the strongest concrete source. It contains public reasoners for `bit_manipulation` and `equation_numeric`, corpus masking, training metrics, and exact workflow. The useful action is not to copy a submit, but to adapt the algorithmic trace generation into our CPU target-alignment gate.

2. `reasoners/equation_numeric.py` provides a concrete operation inventory for equation-style numeric transforms: concatenation, reverse concatenation, add/sub/multiply, abs difference, +/-1 variants, division/modulo, digit-wise mod/add/sub/multiply, determinant, reversed operands, reversed results, prefix/suffix handling. This is the exact seed for DSL v2.

3. `reasoners/bit_manipulation.py` confirms the bit route should remain bit-pair/bitsum/stride oriented. Our bit target is preservation at `>=136/160`, not broad relearning.

4. `andy279/nemotron-reasoning-challenge` and `andy279/nemotron-reasoning-challenge-raw-traces` document SFT/teacher traces, including solver-guided transformation and bit traces. Access to payload files is gated/unauthorized in this environment, but local copies already supplied by the user should be treated as trace sources only after strict provenance/anti-leakage gates.

5. `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` is an Apache-2.0 public train/test mirror with 9.5k train rows. It is useful for hash/provenance checks, not as a new signal beyond official data.

6. NVIDIA/vLLM/NeMo/Megatron Bridge pages reinforce the official scaffold gate: max LoRA rank 32, vLLM official loading path, model compatibility, and package validation.

## Rejections / Non-Actions

- Kaggle discussion URLs in the export returned only a generic Kaggle shell through HTTP, and Kaggle CLI 2.0.2 exposes competition pages/files/submissions but no discussion-body endpoint. No new discussion content was extracted in this pass.

- `passagereptile455/*` LoRA repos have only `.gitattributes` by HF API; no usable adapter weights.

- `GaryNENE/nemotron-nano-8b-reasoning-lora` targets an 8B base with rank 128; it is not submit-compatible for this challenge.

- Generic NVIDIA math/ReasoningGym/logical-puzzles datasets are P2 fixtures or trace-style references only. They do not justify GPU spend unless a CPU target-alignment gate shows family-specific lift.

- OpenRouter/provider TOS/status/API/favicons do not affect ACC and are excluded from roadmap actions.

## Required Roadmap Change

Add V446D: URL/source audit confirms the next implementation must be CPU-only `Tong-source target-alignment`: extract operation inventory and trace format from public source, generate certified equation/bit candidate traces on allowed data, run anti-leakage/provenance gates, then GPU only if `equation>56`, `bit>=136`, `truncated=0` is plausible before checkpoint-1.

## URL Audit Matrix

| # | Class | HTTP | Domain | Action | Note | URL |
|---:|---|---:|---|---|---|---|
| 1 | provider/meta | ERR | `dashscope-intl.aliyuncs.com` | ignore | HTTPError: HTTP Error 404: Not Found | https://dashscope-intl.aliyuncs.com/compatible-mode/v1 |
| 2 | provider/meta | 200 | `www.alibabacloud.com` | ignore | Alibaba Cloud International Product Terms of Service Overview - Legal - Alibaba Cloud | https://www.alibabacloud.com/help/en/legal/latest/alibaba-cloud-international-website-product-terms-of-service-v-3-8-0 |
| 3 | provider/meta | 200 | `www.alibabacloud.com` | ignore | How we handle your personal data and your rights - Alibaba Cloud International Website Pri | https://www.alibabacloud.com/help/en/legal/latest/alibaba-cloud-international-website-privacy-policy |
| 4 | provider/meta | 200 | `status.alibabacloud.com` | ignore |  | https://status.alibabacloud.com/ |
| 5 | provider/meta | 206 | `t0.gstatic.com` | ignore |  | https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256 |
| 6 | provider/meta | 200 | `cloud.google.com` | ignore | Google Cloud Platform Terms Of Service | https://cloud.google.com/terms/ |
| 7 | provider/meta | 200 | `cloud.google.com` | ignore | Google Cloud Privacy Notice | https://cloud.google.com/terms/cloud-privacy-notice |
| 8 | provider/meta | 206 | `status.cloud.google.com` | ignore | Google Cloud Service Health | https://status.cloud.google.com/products/sdXM79fz1FS6ekNpu37K/history |
| 9 | provider/meta | ERR | `generativelanguage.googleapis.com` | ignore | HTTPError: HTTP Error 404: Not Found | https://generativelanguage.googleapis.com/v1beta |
| 10 | provider/meta | ERR | `api.deepseek.com` | ignore | HTTPError: HTTP Error 401: Unauthorized | https://api.deepseek.com/beta |
| 11 | provider/meta | ERR | `chat.deepseek.com` | ignore | HTTPError: HTTP Error 403: Forbidden | https://chat.deepseek.com/downloads/DeepSeek%20Terms%20of%20Use.html |
| 12 | provider/meta | ERR | `chat.deepseek.com` | ignore | HTTPError: HTTP Error 403: Forbidden | https://chat.deepseek.com/downloads/DeepSeek%20Privacy%20Policy.html |
| 13 | provider/meta | ERR | `status.deepseek.com` | ignore | URLError: <urlopen error _ssl.c:1018: The handshake operation timed out> | https://status.deepseek.com/ |
| 14 | provider/meta | ERR | `openrouter-foundry-east-resource.cognitiveservices.azure.com` | ignore | HTTPError: HTTP Error 404: Resource Not Found | https://openrouter-foundry-east-resource.cognitiveservices.azure.com/openai |
| 15 | provider/meta | ERR | `www.microsoft.com` | ignore | HTTPError: HTTP Error 403: Forbidden | https://www.microsoft.com/en-us/legal/terms-of-use?oneroute=true |
| 16 | provider/meta | ERR | `www.microsoft.com` | ignore | HTTPError: HTTP Error 403: Forbidden | https://www.microsoft.com/en-us/privacy/privacystatement |
| 17 | provider/meta | 200 | `status.azure.com` | ignore | Azure status | https://status.azure.com/ |
| 18 | provider/meta | ERR | `api.deepinfra.com` | ignore | HTTPError: HTTP Error 404: Not Found | https://api.deepinfra.com/v1/openai |
| 19 | provider/meta | 206 | `deepinfra.com` | ignore | DeepInfra Terms of Service | https://deepinfra.com/terms |
| 20 | provider/meta | 206 | `deepinfra.com` | ignore | DeepInfra Privacy Policy | https://deepinfra.com/privacy |
| 21 | provider/meta | 200 | `status.deepinfra.com` | ignore | Deep Infra status | https://status.deepinfra.com/ |
| 22 | provider/meta | ERR | `api.x.ai` | ignore | HTTPError: HTTP Error 404: Not Found | https://api.x.ai/v1 |
| 23 | provider/meta | ERR | `x.ai` | ignore | HTTPError: HTTP Error 403: Forbidden | https://x.ai/legal/terms-of-service-enterprise |
| 24 | provider/meta | ERR | `x.ai` | ignore | HTTPError: HTTP Error 403: Forbidden | https://x.ai/legal/privacy-policy |
| 25 | provider/meta | ERR | `status.x.ai` | ignore | HTTPError: HTTP Error 403: Forbidden | https://status.x.ai/ |
| 26 | provider/meta | 206 | `t0.gstatic.com` | ignore |  | https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256 |
| 27 | provider/meta | ERR | `api.intelligence.io.solutions` | ignore | HTTPError: HTTP Error 404: Not Found | https://api.intelligence.io.solutions/api/openrouter/v1 |
| 28 | provider/meta | 206 | `t0.gstatic.com` | ignore |  | https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://io.net/&size=256 |
| 29 | content | 200 | `openrouter.ai` | ignore |  | https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning |
| 30 | provider/meta | ERR | `integrate.api.nvidia.com` | ignore | HTTPError: HTTP Error 404: Not Found | https://integrate.api.nvidia.com/v1 |
| 31 | provider/meta | 206 | `assets.ngc.nvidia.com` | ignore | pdf not downloaded in audit | https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf |
| 32 | provider/meta | 200 | `www.nvidia.com` | ignore |  | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/ |
| 33 | provider/meta | 206 | `t0.gstatic.com` | ignore |  | https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.nvidia.com/en-us/&size=256 |
| 34 | provider/meta | ERR | `chat.ionstream.ai` | ignore | HTTPError: HTTP Error 404: Not Found | https://chat.ionstream.ai/v1 |
| 35 | provider/meta | 200 | `ionstream.ai` | ignore | Terms and Conditions - ionstream | https://ionstream.ai/terms-and-conditions/ |
| 36 | provider/meta | 200 | `ionstream.ai` | ignore | Privacy Policy - ionstream | https://ionstream.ai/privacy-policy/ |
| 37 | provider/meta | 206 | `t0.gstatic.com` | ignore |  | https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://huggingface.co/&size=256 |
| 38 | provider/meta | ERR | `api.openai.com` | ignore | HTTPError: HTTP Error 404: Not Found | https://api.openai.com/v1 |
| 39 | provider/meta | ERR | `openai.com` | ignore | HTTPError: HTTP Error 403: Forbidden | https://openai.com/policies/row-terms-of-use/ |
| 40 | provider/meta | ERR | `openai.com` | ignore | HTTPError: HTTP Error 403: Forbidden | https://openai.com/policies/privacy-policy/ |
| 41 | provider/meta | 200 | `status.openai.com` | ignore |  | https://status.openai.com/ |
| 42 | content | 200 | `www.kaggle.com` | rules/eval/data source; adapter-only constraints | NVIDIA Nemotron Model Reasoning Challenge / Kaggle | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules |
| 43 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/Ayman-Sabek/NVIDIA_Kaggle_Nemotron |
| 44 | content | 200 | `www.kaggle.com` | rules/eval/data source; adapter-only constraints | NVIDIA Nemotron Model Reasoning Challenge / Kaggle | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge |
| 45 | content | 200 | `huggingface.co` | reject for submit: wrong 8B base/rank128; maybe infra reference only | GaryNENE/nemotron-nano-8b-reasoning-lora · Hugging Face | https://huggingface.co/GaryNENE/nemotron-nano-8b-reasoning-lora |
| 46 | content | 200 | `huggingface.co` | infra reference only; no direct ACC signal | Naribow/nemotron-training-runtime · Datasets at Hugging Face | https://huggingface.co/datasets/Naribow/nemotron-training-runtime |
| 47 | content | 200 | `huggingface.co` | mirror of public train/test; use only for provenance/hash cross-check |  | https://huggingface.co/datasets/jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge |
| 48 | content | 200 | `www.kaggle.com` | rules/eval/data source; adapter-only constraints | NVIDIA Nemotron Model Reasoning Challenge / Kaggle | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data |
| 49 | content | 200 | `www.kaggle.com` | blocked: shell only; no new content extracted | shell only; no discussion body via HTTP/CLI | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/682355 |
| 50 | content | 200 | `www.kaggle.com` | blocked: shell only; no new content extracted | shell only; no discussion body via HTTP/CLI | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 |
| 51 | content | 200 | `luma.com` | ignore | NVIDIA Nemotron Model Reasoning Challenge · Luma | https://luma.com/5ugsphtp |
| 52 | content | 200 | `www.kaggle.com` | scaffold/gate reference: official loader/model compatibility | NVIDIA Nemotron Reasoning Challenge Dataset / Kaggle | https://www.kaggle.com/datasets/nctuan/nvidia-nemotron-reasoning-challenge |
| 53 | content | 200 | `www.kaggle.com` | blocked: shell only; no new content extracted | shell only; no discussion body via HTTP/CLI | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685462 |
| 54 | content | 200 | `huggingface.co` | scaffold/gate reference: official loader/model compatibility | Naribow/nvidia-nemotron-progress-prize · Datasets at Hugging Face | https://huggingface.co/datasets/Naribow/nvidia-nemotron-progress-prize |
| 55 | content | 200 | `blogs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | NVIDIA Releases Open Synthetic Data Generation Pipeline for Training Large Language Models | https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training |
| 56 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Synthetic Data Generation — NVIDIA NeMo Framework User Guide | https://docs.nvidia.com/nemo-framework/user-guide/24.12/datacuration/syntheticdata.html |
| 57 | content | 200 | `huggingface.co` | P2 fixtures/trace style only; no GPU without target-alignment gate |  | https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3 |
| 58 | content | 200 | `huggingface.co` | P2 fixtures/trace style only; no GPU without target-alignment gate |  | https://huggingface.co/datasets/nvidia/Nemotron-Math-v2 |
| 59 | content | 206 | `d6108366.hf-mirror.com` | scaffold/gate reference: official loader/model compatibility | nvidia/Nemotron-CC-Math-v1 · Datasets at Hugging Face | https://d6108366.hf-mirror.com/datasets/nvidia/Nemotron-CC-Math-v1 |
| 60 | content | 200 | `huggingface.co` | P2 fixtures/trace style only; no GPU without target-alignment gate |  | https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning |
| 61 | content | 206 | `arxiv.org` | ignore | pdf not downloaded in audit | https://arxiv.org/pdf/2512.15489 |
| 62 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/NVIDIA/workbench-example-nemotron-finetune |
| 63 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/llm/nemotron3.md |
| 64 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/NVIDIA-NeMo/NeMo/blob/v2.5.3/tutorials/nlp/lora.ipynb |
| 65 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/NVIDIA-NeMo/NeMo/issues/14856 |
| 66 | content | 200 | `www.kaggle.com` | rules/eval/data source; adapter-only constraints | NVIDIA Nemotron Model Reasoning Challenge / Kaggle | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview |
| 67 | content | 200 | `www.kaggle.com` | rules/eval/data source; adapter-only constraints | NVIDIA Nemotron Model Reasoning Challenge / Kaggle | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models |
| 68 | content | 200 | `huggingface.co` | scaffold/gate reference: official loader/model compatibility | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 · Hugging Face | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| 69 | content | 200 | `docs.api.nvidia.com` | scaffold/gate reference: official loader/model compatibility | nvidia / nemotron-3-nano-30b-a3b | https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-30b-a3b |
| 70 | content | 200 | `huggingface.co` | scaffold/gate reference: official loader/model compatibility | README.md · nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 at refs/pr/32 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/refs%2Fpr%2F32/README.md |
| 71 | content | 206 | `research.nvidia.com` | scaffold/gate reference: official loader/model compatibility |  | https://research.nvidia.com/labs/nemotron/Nemotron-3/ |
| 72 | content | 200 | `research.nvidia.com` | scaffold/gate reference: official loader/model compatibility | pdf not downloaded in audit | https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf |
| 73 | content | 206 | `github.com` | ignore |  | https://github.com/Jerry2003826/nivida |
| 74 | content | 200 | `huggingface.co` | scaffold/gate reference: official loader/model compatibility |  | https://huggingface.co/datasets/felipesp1983/nemotron-cryptarithm-cot-813 |
| 75 | content | 200 | `huggingface.co` | scaffold/gate reference: official loader/model compatibility |  | https://huggingface.co/datasets/alex-gapch/nemotron-cryptarithm-narrative |
| 76 | content | 200 | `huggingface.co` | scaffold/gate reference: official loader/model compatibility |  | https://huggingface.co/datasets/felipesp1983/nemotron-cryptarithm-synth-4425 |
| 77 | content | 200 | `huggingface.co` | P2 fixtures/trace style only; no GPU without target-alignment gate |  | https://huggingface.co/datasets/codelion/logical-puzzles-cot |
| 78 | content | 206 | `github.com` | strong: inspect reasoner/corpus code; adapt algorithms CPU-first |  | https://github.com/tonghuikang/nemotron |
| 79 | content | 200 | `huggingface.co` | strong if access/data available: trace source; anti-leakage gate required | andy279/nemotron-reasoning-challenge-raw-traces · Datasets at Hugging Face | https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces |
| 80 | content | 206 | `github.com` | ignore |  | https://github.com/tonghuikang?tab=achievements |
| 81 | content | 206 | `github.com` | ignore |  | https://github.com/tonghuikang |
| 82 | content | 206 | `arxiv.org` | ignore | Nemotron-cc-math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset | https://arxiv.org/html/2508.15096v1 |
| 83 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/NVIDIA-NeMo/Nemotron |
| 84 | content | 200 | `openreview.net` | ignore |  | https://openreview.net/forum?id=rhPnkTKfMy |
| 85 | content | 200 | `aifactory.space` | ignore | NVIDIA Nemotron 모델 추론 챌린지 | https://aifactory.space/task/9273/overview |
| 86 | content | 200 | `huggingface.co` | P2 fixtures/trace style only; no GPU without target-alignment gate |  | https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1 |
| 87 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/vllm-project/vllm/pull/30802 |
| 88 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Nemotron 3 Nano — Megatron Bridge | https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/llm/nemotron3.html |
| 89 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Nemotron 3 Nano — Megatron Bridge | https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/models/llm/nemotron3.html |
| 90 | content | 200 | `huggingface.co` | reject: no adapter weights in repo API | passagereptile455/nemotron-reasoning-lora-v10-kaggle-1epoch · Hugging Face | https://huggingface.co/passagereptile455/nemotron-reasoning-lora-v10-kaggle-1epoch |
| 91 | content | 200 | `huggingface.co` | reject: no adapter weights in repo API | passagereptile455/nemotron-reasoning-lora-v11-kaggle-alpha64-1epoch · Hugging Face | https://huggingface.co/passagereptile455/nemotron-reasoning-lora-v11-kaggle-alpha64-1epoch |
| 92 | content | 200 | `recipes.vllm.ai` | scaffold/gate reference: official loader/model compatibility |  | https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| 93 | content | 200 | `docs.vllm.ai` | scaffold/gate reference: official loader/model compatibility | NVIDIA Nemotron-3-Nano-30B-A3B User Guide - vLLM Recipes | https://docs.vllm.ai/projects/recipes/en/stable/NVIDIA/Nemotron-3-Nano-30B-A3B.html |
| 94 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Llama Nemotron Models — NVIDIA NeMo Platform Documentation | https://docs.nvidia.com/nemo/microservices/26.3.0/customizer/models/llama-nemotron.html |
| 95 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Remove Reasoning Traces — NeMo Evaluator SDK | https://docs.nvidia.com/nemo/evaluator/latest/tutorials/how-to/reasoning.html |
| 96 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Evaluation of Reasoning Models — NeMo Evaluator SDK | https://docs.nvidia.com/nemo/evaluator/nightly/evaluation/run-evals/reasoning.html |
| 97 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Nemotron 3 Nano — Megatron Bridge | https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html |
| 98 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | Parameter Efficient Fine-Tuning (PEFT) — NVIDIA NeMo Framework User Guide | https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/nemotron/peft.html |
| 99 | content | 206 | `github.com` | scaffold/gate reference: official loader/model compatibility |  | https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nano-v3/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py |
| 100 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | bridge.recipes.nemotronh.nemotron_3_nano — Megatron Bridge | https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/apidocs/bridge/bridge.recipes.nemotronh.nemotron_3_nano.html |
| 101 | content | 200 | `docs.nvidia.com` | scaffold/gate reference: official loader/model compatibility | LoRA Fine-Tuning & Deployment of Nemotron 3 Nano for Text2SQL — Nemotron | https://docs.nvidia.com/nemotron/latest/use-case-examples/sql-lora-finetuning-and-deployment/README.html |
