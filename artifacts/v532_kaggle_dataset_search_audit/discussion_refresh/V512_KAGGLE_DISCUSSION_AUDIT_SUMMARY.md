# V512 Kaggle Discussion Audit

Generated UTC: 2026-05-17T01:43:54.841993+00:00

## Scope

- Topic IDs requested: `140`
- Topics fetched: `58`
- Posts scanned: `357`
- Relevant post hits: `238`

## Highest-Signal Hits

### 690307 - Strategy to solve 85% of bit manipulation

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- Post: `3439770` by `Tong Hui Kang` at `2026-04-11T07:43:56.657Z`
- Score: `119`
- Groups: bit_solver: and-not, bit manipulation, bit_manipulation, bitsum, huikang, or-not, rot(, rotation, equation_solver: equation, numeric_equation, operator, symbol_transform, adapter_training: chain of thought, cot, nemotron, sft, synthetic, trace, concrete_artifact: code, kaggle.com/code, notebook, repo
- URLs: https://nemotron.huikang.dev/corpus.html?category=bit_manipulation, https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2, https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/prizes

Excerpt:

```text
This is part of my [publication](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/prizes) for the Open Progress Prize. I read the 0.73 scoring [notebook](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2) from @llkh0a / Kh0a. The approach described in Kh0a's notebook is actually very similar to mine - Use code to write synthetic CoT traces - Train SFT on the synthetic CoT traces - Make the submission Kh0a reports the following validation score. ``` Per-category: bit_manipulation: 35/160 = 21.88% gravity_physics: 160/160 = 100.00% numeral_system: 158/158 = 100.00% numeric_equation: 51/73 = 69.86% symbol_transform: 0/82 = 0.00% text_decryption: 145/158 = 91.77% unit_conversion: 159/159 = 100.00% Overall: 708/950 = 74.53% Weighted CV score: 74.76% ``` Kh0a's algorithm solves only 35/160 of bit manipulation problems. I have an al...
```

### 689915 - [Open Progress Prize Publication] SFT to maximize minimum logprob

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- Post: `3438961` by `Tong Hui Kang` at `2026-04-10T03:02:11.813Z`
- Score: `98`
- Groups: bit_solver: bit manipulation, bit_manipulation, huikang, rot(, shl, shr, xor, equation_solver: cryptarithm, deduce, equation, equation_numeric, operator, adapter_training: adapter, chain of thought, distill, lora, nemotron, sft, synthetic, trace, concrete_artifact: code, github.com, kaggle.com/code, notebook
- URLs: https://en.wikipedia.org/wiki/Verbal_arithmetic, https://github.com/tonghuikang/nemotron, https://github.com/tonghuikang/nemotron/blob/master/reasoners/cipher.py, https://github.com/tonghuikang/nemotron/blob/master/reasoners/cryptarithm.py, https://github.com/tonghuikang/nemotron/blob/master/reasoners/equation_numeric.py

Excerpt:

```text
I would like to thank the competition hosts and Kaggle for organizing this competition. I did manage to find something interesting to bet on, and I am happy to see my gamble paying off. You might have made some predictions that I have asked for. These are the answers. - The score I was aiming for - 0.877 - How many tokens are used to train - [27,850,703](https://nemotron.huikang.dev/metrics.html?logpath=04-08-16-14) tokens for the winning solution, 598,958,637 in total - How much money I have spent - $212.48 in Tinker credits, approximately $60 in Modal credits, $10 for Kaggle / Colab subscription. - What do you think is the secret - bit manipulation, you only need SFT, deterministic chain-of-thought design, use of min logprob as objective, use of Tinker for training # Quick links - Original [notebook](https://www.kaggle.com/code/huikang/tinker-submission-notebook) - Validation [noteb...
```

### 697491 - Why a "Better" Dataset Scored Worse: Lessons on Logprobs, Gradient Saturation, and SFT Bugs

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697491
- Post: `3453960` by `Taha` at `2026-05-06T09:21:05.257Z`
- Score: `57`
- Groups: bit_solver: bit_manipulation, equation_solver: cryptarithm, deduce, equation, equation_numeric, adapter_training: chain of thought, cot, nemotron, sft, synthetic, concrete_artifact: code, dataset, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/tahaalam2009/end-to-end-finetuning-for-lb-0-82-csv-custom, https://www.kaggle.com/code/tahaalam2009/nemotron-batched-logprob-filter-train, https://www.kaggle.com/code/tahaalam2009/nemotron-sft-final-0-83-lb, https://www.kaggle.com/datasets/tahaalam2009/nemotron-0-90, https://www.kaggle.com/datasets/tahaalam2009/nemotron-logprob

Excerpt:

```text
Hey everyone, Over the last week, I went down a massive rabbit hole trying to improve the synthetic Chain of Thought (CoT) generation for the hard categories in this competition (`cryptarithm_deduce`, `cryptarithm_guess`, `equation_numeric_guess`). I managed to write a much better deterministic algorithm to solve these, pushing my synthetic dataset accuracy from the baseline **87.7% to 95.8%**. I assumed this would guarantee a leaderboard boost. Instead, my first few runs crashed to 0.73, and even my stabilized runs hovered around 0.82–0.84, failing to beat the 0.85 baseline. I wanted to share exactly why a "better" dataset doesn't automatically equal a better LB score, and the three massive traps I fell into (and how to fix them). #### The Data Comparison Here is the baseline (Tong's) generation vs. my custom generation. Notice the massive jump in the hard categories: **Original Base...
```

### 694556 - symbol_transformation class problem can have multiple valid candidate answer

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694556
- Post: `3454832` by `Murugesan Narayanaswamy` at `2026-05-08T01:06:23.643Z`
- Score: `51`
- Groups: bit_solver: huikang, equation_solver: cryptarithm, deduce, equation, equation_numeric, symbol transform, symbol_transform, adapter_training: cot, concrete_artifact: .csv, dataset
- URLs: https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10218904%2F5edb96b48f890bfe1c4c7e14e7173c86%2FPicture1.jpg?generation=1778202058227263&alt=media

Excerpt:

```text
Did you notice that there are a lot of duplicate prompts in symbol transformation category? Only 54 prompts are unique in cryptarithm_deduce category and overall only 10% of around 1000 prompts are unique! I spent time and gpu hours on creating new accurate CoT and training for crypatrithm_deduce and equation_numeric, but there was no change in LB score, then I realized that I am training for some 21 prompts, 54 prompts etc!! ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10218904%2F5edb96b48f890bfe1c4c7e14e7173c86%2FPicture1.jpg?generation=1778202058227263&alt=media) Note: this analysis is on @huikang's dataset titled 'problem_ids_matched.csv'
```

### 694556 - symbol_transformation class problem can have multiple valid candidate answer

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694556
- Post: `3454909` by `NguyenThanhNhan` at `2026-05-08T07:00:36.230Z`
- Score: `46`
- Groups: bit_solver: huikang, shift, xor, equation_solver: dsl, operator, symbol_transform, adapter_training: lora, nemotron, sft, trace, concrete_artifact: .csv, notebook
- URLs: none

Excerpt:

```text
@toolazyhhh123 Of the 1555 symbol_transformation puzzles in train.csv, the public DSL (pick / shift / xor) doesn't fit on 878 of them (~57%): * 661 the public reference solver (Huikang's notebook) skips entirely. * 217 more it traces but tags individual operators as unknown and emits a fallback guess. I ran two independent solvers over a subset of 561 puzzles for training, and kept 53 puzzles for validation: 1. z3-bounded search restricted to the public DSL - pick a character from a fixed input position, Caesar-shift, xor-with-constant, xor across two positions. For each output character of each example, the search walks through which DSL operation could produce that character, then labels the position as either derivable (some same-operator example fixes the operation) or a guess (no same-operator example constrains it). One step-by-step trace per puzzle becomes one training row. 2....
```

### 694710 - How to Cut Nemotron Training from 11 Hours to 5h 40m (And Fix the "Loss Illusion")

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694710
- Post: `3448732` by `Taha` at `2026-04-26T13:56:00.393Z`
- Score: `45`
- Groups: bit_solver: huikang, shift, equation_solver: cryptarithm, equation, equation_numeric, adapter_training: adapter, cot, lora, nemotron, sft, synthetic, concrete_artifact: dataset, repo
- URLs: none

Excerpt:

```text
Hey everyone, If you are burning through your Kaggle/Modal GPU quotas watching standard HuggingFace `SFTTrainer` crawl for 10–11 hours, I want to share a pipeline shift that cut my LoRA training time down to **5 hours and 40 minutes** while vastly improving convergence. I initially struggled with standard SFT runs taking 11 hours and throwing artificially high loss rates (~2.5 to 3.0), which led to catastrophic forgetting of easy categories when injecting custom synthetic data. By adapting the custom training loop shared by Tong and others, I achieved a massive performance jump. *Note: I originally misunderstood the exact tensor-level mechanics of why this loop is so much better, but thanks to some direct corrections from Tong, here is the actual mathematical reality of why this pipeline works:* ### 1. The VRAM Unlock: Cut Cross-Entropy Standard `SFTTrainer` materializes a massive log...
```

### 683172 - Kaggle CLI — Develop Locally and Run on RTX Pro 6000 GPU

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683172
- Post: `3424706` by `Keanan` at `2026-03-19T22:58:24.990Z`
- Score: `42`
- Groups: adapter_training: adapter, lora, nemotron, sft, concrete_artifact: .ipynb, code, dataset, github.com, kaggle.com/code, notebook, repo
- URLs: https://download.pytorch.org/whl/nightly/cu128, https://github.com/Kaggle/kaggle-api, https://github.com/Kaggle/kaggle-api.git, https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md, https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md

Excerpt:

```text
I have also added a notebook version for easier navigation (table of contents): [https://www.kaggle.com/code/citerne/from-local-dev-rtx-6000-kaggle-cli-guide] > Practical guide for the **NVIDIA Nemotron Model Reasoning Challenge**. > Hard-won lessons on setting up a CLI → Kaggle GPU workflow, pitfalls to avoid, and everything you need to get started. --- ## Why This Workflow? The Kaggle web interface is great for exploration, but once you want to: - Iterate quickly on training code - Use a proper IDE with autocomplete and dev tools (e.g. Claude Code, Copilot) - Version and push from a local Git repo ...the **develop locally → push with CLI** workflow becomes essential. --- ## 1. Install the Kaggle CLI (from GitHub) The PyPI version (`kaggle==1.7.x`) does **not** support the `--accelerator` flag, which is required to target the RTX Pro 6000 GPU. You need to install from GitHub. > 📦 **R...
```

### 693260 - 90.7% Synthetic CoT Accuracy -> LB Drop: A Warning on Data Generation & Thanks to Donald

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/693260
- Post: `3445737` by `Taha` at `2026-04-20T12:47:14.677Z`
- Score: `37`
- Groups: bit_solver: bit manipulation, bit_manipulation, equation_solver: cryptarithm, adapter_training: adapter, cot, lora, nemotron, synthetic, trace, concrete_artifact: dataset
- URLs: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461

Excerpt:

```text
First, a massive shoutout to Donald Galliano III for his incredible [100% Solve Rate / Reverse Engineering post](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461). His insights completely changed how I was approaching the dataset. I wanted to share my experience of implementing his methodology to build a custom synthetic Chain-of-Thought (CoT) dataset, how I hit **98.9% on Bit Manipulation**, and the catastrophic mistake I made that actually caused my LB score to drop—plus how I'm fixing it. ### The Win: Solving Bit Manipulation (98.9%) Following Donald's advice, I realized the baseline models cap out at ~85% on `bit_manipulation` because LLMs hallucinate when forced to do parallel array math (e.g., `1100 AND 1010`), and the resulting CoTs often hit the 7680 token limit. I wrote a custom Python generator to enforce **Bit-Serial Computatio...
```

### 681745 - How to Get Started + Nemotron Model Reasoning Challenge Resources

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681745
- Post: `3422112` by `Jamil C Semaan` at `2026-03-17T00:08:37.323Z`
- Score: `34`
- Groups: adapter_training: nemotron, synthetic, concrete_artifact: code, dataset, github.com, huggingface.co, notebook, repo
- URLs: https://brev.nvidia.com/launchable/deploy?launchableID=env-32kC34ErT9wsqTcJyaKMxBEuhr2, https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/, https://developer.nvidia.com/nemotron, https://developer.nvidia.com/topics/ai/how-to-build-agentic-ai-rag, https://developer.nvidia.com/topics/ai/how-to-build-an-ai-agent

Excerpt:

```text
## Information for First-Timers New to NVIDIA Nemotron open model family and its NeMo open libraries? If you’re just starting to explore these models, tools, or docs, feel free to start a new thread in this channel, or drop any first-timer questions here and folks can help you get unblocked. If you’re new to Nemotron and NeMo, skim the official Nemotron model pages, NeMo docs, and the resources listed below, then come back with specific questions about setup, fine-tuning, RL, or evaluation. Happy reasoning with Nemotron, and good luck climbing the leaderboard! ### 1. Core Nemotron model family (models & baselines) - [Nemotron overview (model family, capabilities, benchmarks)](https://developer.nvidia.com/nemotron) - [Nemotron GitHub](https://github.com/NVIDIA-NeMo/Nemotron)(main repo and examples) - Nemotron 3 Nano[ technical blog](https://developer.nvidia.com/blog/inside-nvidia-nemot...
```

### 690307 - Strategy to solve 85% of bit manipulation

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- Post: `3445372` by `Taha` at `2026-04-19T16:38:53.767Z`
- Score: `30`
- Groups: bit_solver: bit manipulation, bit_manipulation, equation_solver: cryptarithm, deduce, equation, equation_numeric
- URLs: none

Excerpt:

```text
| Category | Found | Total | Accuracy | Avg ms | |--------------------------|------:|------:|---------:|-------:| | bit_manipulation | 1584 | 1602 | 98.9% | 7.7 | | cipher | 1576 | 1576 | 100.0% | 0.0 | | cryptarithm_deduce | 98 | 659 | 14.9% | 41.1 | | cryptarithm_guess | 14 | 164 | 8.5% | 39.8 | | equation_numeric_deduce | 553 | 596 | 92.8% | 0.9 | | equation_numeric_guess | 21 | 136 | 15.4% | 0.9 | | gravity | 1597 | 1597 | 100.0% | 0.0 | | numeral | 1576 | 1576 | 100.0% | 0.0 | | unit_conversion | 1594 | 1594 | 100.0% | 0.0 | |--------------------------|-------|-------|----------|--------| | **TOTAL** | 8613 | 9500 | 90.7% | 33.0 | Guess what?
```

### 694556 - symbol_transformation class problem can have multiple valid candidate answer

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694556
- Post: `3448421` by `toolazyhhh123` at `2026-04-25T13:09:00.063Z`
- Score: `30`
- Groups: equation_solver: dsl, equation, operator, symbol_transform, adapter_training: sft, concrete_artifact: code, dataset
- URLs: none

Excerpt:

```text
# `symbol_transformation` would benefit from a stated rule class: finite examples cannot identify arbitrary operations First off — thank you to the organizers for putting this benchmark together. Rule-induction puzzles are a great testbed, and I've enjoyed working on this category. I'd like to share a concern in the spirit of making the task even stronger, and I'd love to hear the team's thoughts. Quick terminology note: I don't mean to imply that `symbol_transformation` is an official competition label unless the organizers use that term elsewhere. I'm using it as a shorthand for the prompt family identifiable by wording like "a secret set of transformation rules is applied to equations." In other words, this is the equation/symbol rule-induction family, not a claim about a published taxonomy column. ## TL;DR The `symbol_transformation` tasks ask models to infer hidden operations fro...
```

### 684192 - [Dataset Hallucination?] How did you resolve these problems by human?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684192
- Post: `3431860` by `Ashutosh Kumar` at `2026-03-30T19:15:15.820Z`
- Score: `29`
- Groups: bit_solver: bit manipulation, bit_manipulation, rotation, equation_solver: equation, equation_transform, concrete_artifact: dataset
- URLs: none

Excerpt:

```text
Exactly, for bit_manipulation and equation_transformation, despite so many rotations and handlings model hallucinates and its not fit for arithmetic operations Bit manipulation (46% per-bit, 100% rotation): Rotation: 9/9 = 100% — model recognizes rotation patterns perfectly Per-bit: 42/91 = 46% — model writes correct format but derives WRONG boolean rules Failed predictions heavily use complex ops (xnor: 80, or_not: 42, and_not: 33) Correct predictions dominated by simple ops (COPY: 290, AND: 90) The model can't actually compute boolean operations — this is a 3B model capability limit Equation transformation (54% numeric, 4% symbolic): 28/100 equations have numeric answers, 72/100 have symbolic Numeric: 15/28 correct (54%) — model gets arithmetic wrong in 13 cases Symbolic: 3/72 correct (4%) — arbitrary character substitutions are fundamentally impossible Common failure: model claims...
```

### 684212 - Visualize the problems and completions from the base model

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212
- Post: `3427425` by `Tong Hui Kang` at `2026-03-24T04:07:04.903Z`
- Score: `29`
- Groups: equation_solver: equation, adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook, repo
- URLs: https://modal.com/pricing, https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook, https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo, https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684192

Excerpt:

```text
FAQ - Is this the base model? It is the base model. - Why are the partially solved problems in the middle? Most of the questions are ran only once, except some at the start and some in the middle. I ran the inference script in alphabetical and reverse alphabetical order and went for a nap. When I woke up, each script is slightly more than half done and I terminated it. - Where is the prompt from? It is from the official [metric](https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook) notebook. - How much money did it cost? On Modal I ran on an RTX PRO 6000 which is [$3.03](https://modal.com/pricing) per hour. I think I ran this for five hours. The throughput was 2.5k tokens per second. Comments - You see that the solve rate is almost 50%, which aligns with the demo submission from the [organizer](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo). -...
```

### 693251 - [Discussion] Concerns about copied notebooks and misleading submissions in the Notebooks section , Heavy Plagiarism

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/693251
- Post: `3445714` by `Taha` at `2026-04-20T11:54:50.843Z`
- Score: `29`
- Groups: bit_solver: huikang, adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook, repo
- URLs: https://www.kaggle.com/atahalam, https://www.kaggle.com/code/huikang/tinker-submission-notebook, https://www.kaggle.com/code/itahiro/nvidia-nemotron-trained-models-submission, https://www.kaggle.com/code/kienngx/nvidia-nemotron-trained-models-submission, https://www.kaggle.com/code/koushikrudra/0-86-just-train-see-u-in-leaderboard

Excerpt:

```text
I’d like to bring attention to a pattern I’ve been noticing recently in the Notebooks section that may be harmful to the integrity of the platform. After [This Guy](https://www.kaggle.com/atahalam) everyone has started to do the same following his path ; even worse and they are not getting banded or removed!! There are several notebooks being published with high claimed scores, but upon closer inspection: Minimal or no original work: Some notebooks appear to be direct copies of existing public notebooks, with little to no modification. Output reuse: In certain cases, the outputs or submission files seem to be reused rather than generated by the code in the notebook itself. Misleading titles: Titles sometimes claim scores or improvements that are not reproducible from the provided code. Limited transparency: Comments are occasionally disabled, which makes it harder for others to questi...
```

### 697491 - Why a "Better" Dataset Scored Worse: Lessons on Logprobs, Gradient Saturation, and SFT Bugs

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697491
- Post: `3456281` by `MAJ0RT0M` at `2026-05-11T16:30:11.057Z`
- Score: `28`
- Groups: equation_solver: cryptarithm, deduce, equation, equation_numeric, adapter_training: cot, sft, concrete_artifact: dataset
- URLs: none

Excerpt:

```text
@tahaalam2009 - theres something very fishy about your results cryptarithm_guess 164 85.4% equation_numeric_deduce 596 90.6% equation_numeric_guess 136 92.6% how can the guess categories achieve parity w/ the non-guess categories? even if you SMT solved these problems (which definitely is not a COT compatible algorithm) I dont think these results are possible Are you sure you aren't passing the golden answer into your solver?
```

### 689915 - [Open Progress Prize Publication] SFT to maximize minimum logprob

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- Post: `3440973` by `einherjer` at `2026-04-13T10:21:13.323Z`
- Score: `27`
- Groups: bit_solver: huikang, adapter_training: nemotron, sft, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/metric/nvidia-nemotron-metric

Excerpt:

```text
>The temperature in the leaderboard evaluation metric is 0.0. @huikang the [competition metric notebook](https://www.kaggle.com/code/metric/nvidia-nemotron-metric) shows temperature as 1.0. Not 0.0. Is that just a typo? ``` def score( ... temperature: float = 1.0, ... ) -> float: ```
```

### 690307 - Strategy to solve 85% of bit manipulation

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- Post: `3440047` by `Giovanny Rodríguez` at `2026-04-11T20:18:55.943Z`
- Score: `27`
- Groups: bit_solver: bit manipulation, bitwise, equation_solver: symbol_transform, concrete_artifact: .csv, dataset, solver.py
- URLs: none

Excerpt:

```text
It worked—thanks (It went up almost 20%.):``` (.venv) dreuxx@dreuxx-HP-ZBook-Fury-15-6-inch-G8-Mobile-Workstation-PC:~/Documents/data$ python3 solver.py "train(7).csv" Verifying 9500 samples from train(7).csv bitwise : 918/ 1602 ( 57.3%) [unsolvable: 0] cipher : 1576/ 1576 (100.0%) [unsolvable: 0] physics : 1597/ 1597 (100.0%) [unsolvable: 0] symbol_transform : 168/ 1555 ( 10.8%) [unsolvable: 0] symbolic : 1576/ 1576 (100.0%) [unsolvable: 0] unit_conversion : 1594/ 1594 (100.0%) [unsolvable: 0] Total verified: 7429/9500 (78.2%) Generating curated dataset... Saved 7429 solved rows to train_curated.csv. (.venv) dreuxx@dreuxx-HP-ZBook-Fury-15-6-inch-G8-Mobile-Workstation-PC:~/Documents/data$ python3 solver.py "train(7).csv" Verifying 9500 samples from train(7).csv bitwise : 1157/ 1602 ( 72.2%) [unsolvable: 0] cipher : 1576/ 1576 (100.0%) [unsolvable: 0] physics : 1597/ 1597 (100.0%) [uns...
```

### 690161 - Why GRPO is Painfully Slow on Nemotron (and the Fix)

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161
- Post: `3439466` by `Komil Parmar` at `2026-04-10T18:51:33.823Z`
- Score: `26`
- Groups: adapter_training: nemotron, sft, student, teacher, concrete_artifact: code, dataset, repo
- URLs: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686615

Excerpt:

```text
If you've tried GRPO fine-tuning on Nemotron-3-Nano-30B and noticed generation crawling at ~2 tokens/sec, you're not alone. I spent a while debugging this, so here's a full breakdown of what's going on, why it happens, and how to fix it. ## SFT vs GRPO: Why One Works Fine and the Other Doesn't ### SFT is "Open-Book" Think of SFT (Supervised Fine-Tuning) like a student studying with the answer key right next to them. The model (student) sees the full conversation, i.e. both the question AND the correct answer, all at once. It just does one big forward pass across the entire sequence, compares its predictions to the right answer, and adjusts. Like if you are simply reading the question and the answer and say 'Ah! that's how to solve it. Okay, got it. Let's move on'. The key thing: **SFT never actually generates text.** The student never try to answer the question on their own. It hence...
```

### 681714 - How to get started + Competition's Official Discord

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681714
- Post: `3422190` by `KaizaburoChubachi` at `2026-03-17T05:47:20.160Z`
- Score: `25`
- Groups: bit_solver: shl, adapter_training: adapter, lora, nemotron, concrete_artifact: code, kaggle.com/code
- URLs: https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo

Excerpt:

```text
Hi @ashleyoldacre! On the Overview page, it says: >Submitting >You must submit a LoRA adapter of rank at most 32 for the NVIDIA Nemotron-3-Nano-30B model packaged into a submission.zip file. You may consider adapting the [NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo) to produce your submission. But the link https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo gives a 404 error. Could you look into it?
```

### 692950 - Is ~0.86 a current ceiling for most approaches?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/692950
- Post: `3444925` by `Taha` at `2026-04-18T17:38:39.807Z`
- Score: `25`
- Groups: bit_solver: tong hui kang, adapter_training: cot, distill, lora, nemotron, synthetic, concrete_artifact: notebook
- URLs: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915

Excerpt:

```text
I’ve been experimenting quite a bit with the Nemotron reasoning setup over the past few days, and I’m starting to feel like many of us might be hitting a similar ceiling around ~0.86. From what I can tell, a lot of current approaches seem to fall into two buckets: * building directly on top of strong public notebooks (which already reach ~0.86), or * trying to reproduce / adapt methods like [ Tong Hui Kang’s](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915) pipeline I’ve personally tried a mix of things on top of that: * different LoRA variants (including DoRA) * some knowledge distillation setups * hybrid synthetic + distilled data * prompt / formatting tweaks But so far, nothing has meaningfully pushed me past that same range. Right now I even have a long run going (~14 hours in), trying a slightly different data pipeline, but I’m not...
```

### 681745 - How to Get Started + Nemotron Model Reasoning Challenge Resources

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681745
- Post: `3428335` by `Ashutosh Kumar` at `2026-03-25T09:42:48.897Z`
- Score: `24`
- Groups: bit_solver: bit manipulation, bit_manipulation, equation_solver: equation, equation_transform, adapter_training: nemotron, trace
- URLs: none

Excerpt:

```text
Hi Jamil, There appears to be some trainining data bug which I identified lately, as the model was being trained on buggy data and gave wrong answer. Below are the areas critical: - Bit manipulation: 50.5% of training traces are WRONG (764/1513 mismatches between computed Result and answer) - Equation transformation: 49% have I/O length mismatches making char_map fundamentally broken, 44% have unknown chars Bit Manipulation: 1988 total training examples 1513 have a Result line 764 have MISMATCHES between computed Result and \boxed{} answer 50.5% mismatch rate - HALF the training data teaches WRONG reasoning! 344 examples have f() unknowns (where we couldn't determine the rule) Equation Transformation: 3157 total training examples 1555 (49%) have input/output lengths that differ - char_map approach is fundamentally wrong for these Another 1382 (44%) have unknown '?' characters when app...
```

### 687961 - Training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961
- Post: `3435428` by `Tong Hui Kang` at `2026-04-04T06:52:51.353Z`
- Score: `24`
- Groups: adapter_training: adapter, lora, nemotron, concrete_artifact: github.com, notebook, repo
- URLs: https://arxiv.org/abs/2211.15841, https://arxiv.org/abs/2312.00752, https://github.com/huggingface/transformers/pull/44390/changes, https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md, https://jax-ml.github.io/scaling-book/transformers/

Excerpt:

```text
I want to understand the theoretical limitations when training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences. I have not proven that any of the configurations listed here works in practice. I am making my own training implementation, and I want to understand whether my inefficiencies are avoidable with better implementation. Please help me check if I have missed any theoretical limits, thanks! This table calculates how much memory is needed to train Nemotron-3-Nano-30B-A3B-BF16 with different microbatch sizes (μ). Larger microbatch sizes can improve hardware utilization and speed up training, but only if they fit in memory [1]. | Component | Formula | μ=1 | μ=4 | μ=16 | μ=64 | |---|---|---|---|---|---| | Base model weights (BF16) | W × 2 | 63.6 GB | 63.6 GB | 63.6 GB | 63.6 GB | | LoRA adapter weights (FP32) | P × 4 | 3.5 GB | 3.5 GB | 3.5 GB | 3.5 GB | | LoR...
```

### 683172 - Kaggle CLI — Develop Locally and Run on RTX Pro 6000 GPU

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683172
- Post: `3452634` by `Tong Hui Kang` at `2026-05-04T02:12:51.293Z`
- Score: `23`
- Groups: bit_solver: huikang, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85/log

Excerpt:

```text
Thanks for the post! Is there a way to read notebook run [logs](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85/log)?
```

### 686069 - Per-Category Error Analysis After SFT (0.63 LB) — Where the Real Bottlenecks Are

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686069
- Post: `3432754` by `James Day` at `2026-03-31T22:44:25.753Z`
- Score: `23`
- Groups: bit_solver: bit manipulation, bit_manipulation, equation_solver: symbol_transform, adapter_training: distill, sft, concrete_artifact: dataset
- URLs: none

Excerpt:

```text
Your SFT results are quite a bit different from mine... I'm having more trouble with the bit manipulation puzzles and less trouble with the cipher & gravity ones. CV results from my best model (0.68 LB) are included below. These were measured with 1K questions from outside the training dataset. | Category | Accuracy with 8K token limit | Accuracy with 16K token limit | | ---------------- | ----------------------------: | -----------------------------: | | bit_manipulation | 9.9% | 18.8% | | numeral_system | 100.0% | 100.0% | | physics_gravity | 98.8% | 98.8% | | symbol_transform | 17.3% | 20.7% | | text_cipher | 75.5% | 76.1% | | unit_conversion | 100.0% | 100.0% | | overall | 67.1% | 69.3% | --- I suspect my bit manipulation failures largely stem from distilling models which yap too much. Qwen3.5 27B with thinking "disabled" and no special instructions can solve 49% of those problems...
```

### 684271 - ModuleNotFoundError: No module named 'cutlass'

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684271
- Post: `3428879` by `newduck` at `2026-03-26T03:00:04.287Z`
- Score: `22`
- Groups: adapter_training: nemotron, sft, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/newduck/nvidia-nemotron-soft-balanced-sampling-sft/comments

Excerpt:

```text
I had the same "No module named 'cutlass'" error and fixed it in my notebook. Not sure if this applies to your case, but it might help. Please see the comments in the notebook below. https://www.kaggle.com/code/newduck/nvidia-nemotron-soft-balanced-sampling-sft/comments
```

### 687798 - Rescore After Metric Update

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798
- Post: `3436215` by `Yurnero` at `2026-04-05T14:19:43.110Z`
- Score: `22`
- Groups: adapter_training: lora, nemotron, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13977249%2F2558cd409c1e98a690dfd190659dad08%2F1.jpg?generation=1775398654086016&alt=media, https://www.kaggle.com/code/metric/nvidia-nemotron-metric

Excerpt:

```text
@ryanholbrook In the overview tab we can find this. ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13977249%2F2558cd409c1e98a690dfd190659dad08%2F1.jpg?generation=1775398654086016&alt=media) but in the last version (14th) of the [NVIDIA Nemotron Metric notebook](https://www.kaggle.com/code/metric/nvidia-nemotron-metric) I see this ``` def score( solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, max_lora_rank: int = 32, max_tokens: int = 3584, top_p: float = 1.0, temperature: float = 1.0, max_num_seqs: int = 128, gpu_memory_utilization: float = 0.85, max_model_len: int = 4096, debug: bool = False, ) ``` Is it a placeholder and actual scoring parameters are correct in the overview tab. I'm specifically interested in `max_tokens` and `max_model_length`
```

### 688360 - [update] Read CPMP's reply. [original] Do not distill models that do not allow distillation (e.g. gemini, gpt5)

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688360
- Post: `3436206` by `c-number` at `2026-04-05T13:55:19.090Z`
- Score: `22`
- Groups: adapter_training: cot, distill, nemotron, trace, concrete_artifact: dataset, notebook
- URLs: https://www.kaggle.com/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels

Excerpt:

```text
~~The dataset that many top-scoring public notebooks rely on includes traces that were generated by Gemini-2.0-flash. In my understanding, using such data will result in the revocation of your eligibility to receive prizes and, in the worst case, could lead to your account being deleted from the LB.~~ https://www.kaggle.com/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels It seems it's okay.
```

### 684212 - Visualize the problems and completions from the base model

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212
- Post: `3427405` by `Tong Hui Kang` at `2026-03-24T03:34:27.017Z`
- Score: `21`
- Groups: bit_solver: huikang, adapter_training: nemotron, concrete_artifact: code, github.com
- URLs: https://github.com/tonghuikang/nemotron, https://nemotron.huikang.dev/base, https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1680925%2F2ec0a9bf244160a624ce832546c8454f%2FScreenshot%202026-03-23%20at%2020.52.27.png?generation=1774324365015607&alt=media, https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1680925%2Fc52362d70d95e25bcd2c90b32cb66e4a%2FScreenshot%202026-03-23%20at%2020.43.10.png?generation=1774323809172370&alt=media, https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528

Excerpt:

```text
I ran the base model (yes, the base model) over all the 9500 problems at least once. These are the results. https://nemotron.huikang.dev/base ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1680925%2Fc52362d70d95e25bcd2c90b32cb66e4a%2FScreenshot%202026-03-23%20at%2020.43.10.png?generation=1774323809172370&alt=media) You can see the model generation as well. ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1680925%2F2ec0a9bf244160a624ce832546c8454f%2FScreenshot%202026-03-23%20at%2020.52.27.png?generation=1774324365015607&alt=media) The code is [here](https://github.com/tonghuikang/nemotron). This is derived from my work in [AIMO 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528) and [ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2025...
```

### 696059 - What if the answer contains square brackets?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/696059
- Post: `3453927` by `Ogurtsov` at `2026-05-06T07:37:11.657Z`
- Score: `21`
- Groups: equation_solver: equation, adapter_training: nemotron, concrete_artifact: dataset, repo
- URLs: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697333

Excerpt:

```text
@ryanholbrook If we have any chance to get fixed train dataset, please also inspect such cases (task **00d8b3db**): ``` In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 34/44 = 1 41/32 = 9 34|25 = 69 87\64 = 8853 Now, determine the result for: 69/52 ``` It assumes answer `17/` but where does the symbol `/` come from? Another example (task **0c8a8a16**): answer `{17` contains redundant `{`. One more example is reported here https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697333
```

### 681968 - Unable to load Nemotron-3-Nano-30B-A3B-BF16 due to mamba_ssm dependency (No Internet Environment)

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681968
- Post: `3422901` by `Kh0a` at `2026-03-18T02:57:56.117Z`
- Score: `20`
- Groups: adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/ryanholbrook/nvidia-utility-script

Excerpt:

```text
You can add this utility script to your notebook https://www.kaggle.com/code/ryanholbrook/nvidia-utility-script
```

### 684192 - [Dataset Hallucination?] How did you resolve these problems by human?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684192
- Post: `3427334` by `Dennis` at `2026-03-24T01:13:54.920Z`
- Score: `20`
- Groups: equation_solver: deduce, equation, concrete_artifact: dataset
- URLs: https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F145164%2Fcb7ae8b9240d5418b30f437e8e7f292d%2Fnvda.png?generation=1774324860344503&alt=media

Excerpt:

```text
**eeae398e** ```python In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 63]67 = 4 18]81 = 9 72-22 = 95 64]48 = 16 65]15 = 5 Now, determine the result for: 65/58 ``` When we look at the above question, slash has never appeared in the example. How can we deduce the solution? We know that ] is max(A, B) % min(A, B). No example is given for / **e7cf0394** ```python In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 88\87 = 7656 30]47 = 3047 52*15 = *37 Now, determine the result for: 97]83 ``` If the symbol works across the "Alice Wonderland", here 30] 47 should not use concat. It seems that the training dataset has hallucination? One more thing, **7993452d** ```python In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few e...
```

### 684432 - Equation Symbolic has anyone figured out the pattern?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684432
- Post: `3428029` by `Шерхан Масакбаев` at `2026-03-24T20:27:05.143Z`
- Score: `20`
- Groups: bit_solver: huikang, xor, equation_solver: equation, operator
- URLs: none

Excerpt:

```text
I've been analyzing the problem types and wanted to share some findings on what seems to be the hardest category. Based on @huikang 's visualization, the base model solves only 2 out of 823 Equation Symbolic problems (0.2%). For comparison, Numeral is at 96%, Unit Conversion at 75%, Gravity at 59%. I did some deeper analysis of the structure: Format: Each problem has equations like `AB_CD = result` where _ is an operator character and A,B,C,D are ASCII characters (printable range). What makes it so hard: On average only 1.6 examples per operator per problem - very few data points to infer a rule Variable output length - the same operator within the same problem can produce outputs of length 1, 2, 3, or 4. This rules out simple per-position mappings 137 out of 813 problems have a query operator that doesn't appear in any example at all - you need to infer the rule with zero examples Th...
```

### 684654 - Fixed: Any fix for trl installation?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684654
- Post: `3431231` by `Yann` at `2026-03-29T17:44:01.933Z`
- Score: `20`
- Groups: adapter_training: nemotron, peft, sft, concrete_artifact: code, dataset
- URLs: https://www.kaggle.com/datasets/mayukh18/nemotron-packages

Excerpt:

```text
I'm using https://www.kaggle.com/datasets/mayukh18/nemotron-packages now. The following code should be in top cell: ``` import os, sys os.environ["WANDB_DISABLED"] = "true" os.environ["WANDB_MODE"] = "disabled" !pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages \ --ignore-installed \ unsloth trl peft transformers datasets accelerate bitsandbytes !pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl !pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl from trl import SFTTrainer ``` You may need to uninstall wandb juste before with: `!pip uninstall -y wandb`
```

### 685920 - Something wrong -- My notebook of 0.80+ now scores 0.77

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685920
- Post: `3431234` by `bliao` at `2026-03-29T17:53:49.500Z`
- Score: `20`
- Groups: adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/metric/nvidia-nemotron-metric, https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683853

Excerpt:

```text
The verify function (https://www.kaggle.com/code/metric/nvidia-nemotron-metric) is changed. See discussion at https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683853
```

### 687798 - Rescore After Metric Update

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798
- Post: `3435110` by `Ryan Holbrook` at `2026-04-03T17:30:58.430Z`
- Score: `20`
- Groups: adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/metric/nvidia-nemotron-metric

Excerpt:

```text
Hi everyone, Recently, there was an update to the [evaluation metric](https://www.kaggle.com/code/metric/nvidia-nemotron-metric) that fixed a bug that was causing binary answers to match as floats instead of exactly as strings. This update led to a drop of about 0.3-0.4 points on new submissions. (The update is in the `verify()` function of the linked notebook.) So that the leaderboard will only reflect submissions using the new metric, I will be initiating a rescore on Monday. This rescore will only evaluate those submissions that both are still present on the Public Leaderboard and that were made on or before March 28th, that is, prior to the metric update. This will affect a little less than 2/3 of the current LB. Other submissions made on or before March 28th, I will invalidate. Submissions made after that time will be preserved. Why are we not rescoring every submission? Unfortun...
```

### 688360 - [update] Read CPMP's reply. [original] Do not distill models that do not allow distillation (e.g. gemini, gpt5)

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688360
- Post: `3438798` by `Halla Yang` at `2026-04-09T18:13:36.227Z`
- Score: `20`
- Groups: adapter_training: distill, synthetic, trace, concrete_artifact: code, dataset
- URLs: https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles

Excerpt:

```text
In the 2025 edition of ARC-AGI-2, it looked as though NVARC (NVIDIA's 1st place solution) found it permissible to use Claude to generate some limited types of synthetic data (see https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles ). The creator of that dataset wrote: >License concerns: In case you need to filter the records created using Claude model, you have to filter all the records that contain ["summary1", "summary5", "summary6"] or ["mix"]. That text suggests that they were aware that Claude usage could lead to license concerns. Maybe there are some nuances about how synthetic data from Claude can be used, e.g. perhaps it's okay to generate text summaries that then generate code, but it's not okay to generate reasoning traces, though I would be interested in more certainty on where that boundary is located.
```

### 688461 - Answers To Everything Data: Read Me! 100% Solve Rate

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461
- Post: `3436396` by `Donald Galliano III` at `2026-04-05T21:55:54.277Z`
- Score: `20`
- Groups: bit_solver: xor, equation_solver: operator, adapter_training: trace, concrete_artifact: code, dataset
- URLs: none

Excerpt:

```text
I reverse engineered **100% of the dataset.** It's all solvable. Below I'm going to show exactly how. Since my compute isn't good enough to actually run this (Kaggle GPU environment is still broken for me), I'm bowing out of the comp. My only goal was the midpoint prize given my time constraints, and that's clearly off the table, so I'm opening up the playbook for anyone who can run it. I'll break this into sections by category type, with my own think tracings included. One full think tracing per category will be posted in the comments below. If you have questions, please ask, because I obviously can't document every edge case that deviates from the exact pattern but is still solvable under the same framework. I've put **200+ hours into this**. If anyone wants to show appreciation, a like helps others see it. I asked for teammates about a week ago and nobody reached out, which is unfo...
```

### 690689 - [Fake Notebook Alert] Watch out for fake laptops that copy and upload other people's submission.

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690689
- Post: `3440558` by `EISLab_hwlee` at `2026-04-12T16:01:29.320Z`
- Score: `20`
- Groups: adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/atahalam/nvidianemotron-0-81-new-making-0-85-public-soon

Excerpt:

```text
I would like to bring the community's attention to a deceptive practice currently happening in the notebooks section to farm upvotes. [nvidianemotron 0.81 new making 0.85 public soon](https://www.kaggle.com/code/atahalam/nvidianemotron-0-81-new-making-0-85-public-soon) **Why this notebook is fake and deceptive:** 1. **Output Copying (Plagiarism):** The code inside this notebook is completely fake and does not generate the score it claims through actual modeling. Instead, it simply copies the `submission.zip` or outputs from other hard-working Kagglers' top-performing public notebooks directly into `/kaggle/working/`. 2. **Clickbait Titles:** The author intentionally uses misleading titles like *"making 0.85 public soon"* to deceive users and farm upvotes. Furthermore, they have a history of writing fabricated scores in the titles that are completely different from what the actual code...
```

### 691641 - corrupted or puzzel (numeric equations)

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691641
- Post: `3443275` by `Mark Cooper` at `2026-04-16T10:17:00.643Z`
- Score: `20`
- Groups: bit_solver: xor, equation_solver: equation, numeric_equation, operator
- URLs: none

Excerpt:

```text
We ran exhaustive GPU brute-force sweeps across operator primitives (±, ×, ÷, mod, concat, reverse-concat, absolute diff, XOR, logical ops, and compositions up to depth 3) against every example. ~15-16% of the numeric_equation category genuinely has no formula that fits all examples AND the query — they're information-theoretically underdetermined when the query operator doesn't appear in the examples. A few things we found useful beyond "assume absolute difference": 1. **Output length is the strongest signal.** If all example outputs are 1-2 digits, it's probably `a - b` or `a + b` not multiplication. If 3-4 digits, multiplication or concatenation. Pick the guess operator that matches the expected output length from the query operands. 2. **Reversal patterns cluster.** Puzzles where example operators produce `ab -> ba` style patterns (reverse result) tend to keep the reversal behavio...
```

### 694975 - GRPO is must for this competition

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694975
- Post: `3452331` by `Taha` at `2026-05-03T05:40:59.893Z`
- Score: `20`
- Groups: adapter_training: nemotron, concrete_artifact: code, kaggle.com/code, notebook
- URLs: https://www.kaggle.com/code/banwait13/nemotron-on-steroids

Excerpt:

```text
See this amazing GPRO notebook yet it yields same results [HERE](https://www.kaggle.com/code/banwait13/nemotron-on-steroids)
```
