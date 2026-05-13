# KG1 V332 Kaggle Discussion Resume Audit

## Scope

- Generated at UTC: `2026-05-13T17:58:35.833414+00:00`
- Expected topic IDs: `11`
- Cached/fetched topic records: `113`
- Newly fetched this run: `7`
- Missing after this run: `0`
- Errors this run: `0`

## Highest-Relevance Cached Topics

### 688461 - Answers To Everything Data: Read Me! 100% Solve Rate

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461
- Messages: `50`
- Relevance score: `285`
- Keyword hits: `{"accuracy": 25, "bit": 123, "bitwise": 3, "chain of thought": 1, "equation": 12, "eval": 10, "numeric": 2, "operator": 23, "prompt": 10, "solver": 1, "symbol": 47, "synthetic": 1, "token": 2, "train": 25}`
- Preview: [message_id=688461 author=Donald Galliano III first=True] I reverse engineered 100% of the dataset. It's all solvable. Below I'm going to show exactly how. Since my compute isn't good enough to actually run this (Kaggle GPU environment is still broken for me), I'm bowing out of the comp. My only goal was the midpoint prize given my time constraints, and that's clearly off the table, so I'm opening up the playbook for anyone who can run it. I'll break this into sections by category type, with my 

### 689915 - [Open Progress Prize Publication] SFT to maximize minimum logprob

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- Messages: `53`
- Relevance score: `235`
- Keyword hits: `{"accuracy": 1, "adapter": 15, "bit": 15, "bit_manipulation": 2, "chain of thought": 11, "cot": 4, "cryptarithm": 16, "equation": 7, "eval": 3, "lora": 8, "loss": 18, "numeric": 4, "operator": 11, "solver": 1, "symbol": 4, "symbolic": 1, "synthetic": 4, "token": 35, "train": 75}`
- Preview: [message_id=689915 author=Tong Hui Kang first=True] I would like to thank the competition hosts and Kaggle for organizing this competition. I did manage to find something interesting to bet on, and I am happy to see my gamble paying off. You might have made some predictions that I have asked for. These are the answers. The score I was aiming for - 0.877 How many tokens are used to train - 27,850,703 tokens for the winning solution, 598,958,637 in total How much money I have spent - $212.48 in Ti

### 690307 - Strategy to solve 85% of bit manipulation

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- Messages: `12`
- Relevance score: `101`
- Keyword hits: `{"accuracy": 1, "bit": 29, "bit_manipulation": 3, "bitsum": 7, "bitwise": 2, "chain of thought": 3, "cot": 4, "cryptarithm": 2, "equation": 3, "numeric": 3, "operator": 4, "solver": 2, "stride": 5, "symbol": 5, "symbolic": 2, "synthetic": 2, "token": 16, "train": 8}`
- Preview: [message_id=690307 author=Tong Hui Kang first=True] This is part of my publication for the Open Progress Prize. I read the 0.73 scoring notebook from @llkh0a / Kh0a. The approach described in Kh0a's notebook is actually very similar to mine Use code to write synthetic CoT traces Train SFT on the synthetic CoT traces Make the submission Kh0a reports the following validation score. Per-category: bit_manipulation: 35/160 = 21.88% gravity_physics: 160/160 = 100.00% numeral_system: 158/158 = 100.00% 

### 694556 - symbol_transformation class problem can have multiple valid candidate answer

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694556
- Messages: `15`
- Relevance score: `101`
- Keyword hits: `{"accuracy": 1, "bit": 6, "cot": 5, "cryptarithm": 2, "equation": 3, "eval": 4, "lora": 1, "numeric": 1, "operator": 21, "prompt": 10, "solver": 8, "symbol": 11, "synthetic": 1, "train": 27}`
- Preview: [message_id=694556 author=toolazyhhh123 first=True] symbol_transformation would benefit from a stated rule class: finite examples cannot identify arbitrary operations First off — thank you to the organizers for putting this benchmark together. Rule-induction puzzles are a great testbed, and I've enjoyed working on this category. I'd like to share a concern in the spirit of making the task even stronger, and I'd love to hear the team's thoughts. Quick terminology note: I don't mean to imply that 

### 689792 - Is anyone able to get inference/generation speeds >2 tokens/sec?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689792
- Messages: `17`
- Relevance score: `92`
- Keyword hits: `{"accuracy": 6, "adapter": 3, "bit": 6, "bitwise": 1, "equation": 5, "eval": 15, "lora": 17, "numeric": 1, "prompt": 16, "token": 16, "train": 6}`
- Preview: [message_id=689792 author=MAJ0RT0M first=True] I'm sure we are all familiar w/ this error by now: NemotronH requires an initialized NemotronHHybridDynamicCache to return a cache. None was provided, so no cache will be returned. It makes generating reasonable length (2000 tokens+) infeasible - generation speed is ~2toks/sec - so this would take 15m GPRO becomes impossible - local evaluation also impossible. Even sanity checks to see if the model is producing reasonable output and learning thinkin

### 693260 - 90.7% Synthetic CoT Accuracy -> LB Drop: A Warning on Data Generation & Thanks to Donald

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/693260
- Messages: `21`
- Relevance score: `84`
- Keyword hits: `{"accuracy": 4, "adapter": 2, "bit": 28, "bit_manipulation": 2, "bitwise": 1, "cot": 13, "cryptarithm": 1, "lora": 5, "solver": 5, "synthetic": 2, "token": 12, "train": 9}`
- Preview: [message_id=693260 author=Taha first=True] First, a massive shoutout to Donald Galliano III for his incredible 100% Solve Rate / Reverse Engineering post . His insights completely changed how I was approaching the dataset. I wanted to share my experience of implementing his methodology to build a custom synthetic Chain-of-Thought (CoT) dataset, how I hit 98.9% on Bit Manipulation , and the catastrophic mistake I made that actually caused my LB score to drop—plus how I'm fixing it. The Win: Solvi

### 685886 - sharing high quality synthetic data generation prompt 

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685886
- Messages: `4`
- Relevance score: `77`
- Keyword hits: `{"bit": 59, "bitwise": 9, "chain of thought": 1, "prompt": 2, "symbol": 1, "symbolic": 1, "synthetic": 1, "train": 3}`
- Preview: [message_id=685886 author=lucian kucera first=True] Use this prompt to generate high quality bit data reasoning trace, when Iam done with generating traces I will share dataset. ```""" SYSTEM ROLE: You are a deterministic logic-trace engine. Your goal is to generate high-fidelity Supervised Fine-Tuning (SFT) data that explicitly demonstrates the search and verification process of symbolic logic. ATOMIC EXECUTION RULES: Bitwise Delta Analysis : Before proposing any hypothesis, you must compare In

### 697491 - Why a "Better" Dataset Scored Worse: Lessons on Logprobs, Gradient Saturation, and SFT Bugs

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697491
- Messages: `13`
- Relevance score: `74`
- Keyword hits: `{"accuracy": 6, "bit": 4, "bit_manipulation": 2, "chain of thought": 1, "cot": 5, "cryptarithm": 13, "equation": 10, "eval": 2, "loss": 3, "numeric": 10, "solver": 7, "synthetic": 3, "token": 4, "train": 4}`
- Preview: [message_id=697491 author=Taha first=True] Hey everyone, Over the last week, I went down a massive rabbit hole trying to improve the synthetic Chain of Thought (CoT) generation for the hard categories in this competition ( cryptarithm_deduce , cryptarithm_guess , equation_numeric_guess ). I managed to write a much better deterministic algorithm to solve these, pushing my synthetic dataset accuracy from the baseline 87.7% to 95.8% . I assumed this would guarantee a leaderboard boost. Instead, my 

### 690756 - 2 interpretations of the bit manipulation problem

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690756
- Messages: `2`
- Relevance score: `55`
- Keyword hits: `{"accuracy": 1, "bit": 26, "bit_manipulation": 1, "bitsum": 2, "cot": 2, "operator": 3, "solver": 5, "stride": 4, "token": 5, "train": 6}`
- Preview: [message_id=690756 author=Darren Amadeus Martin first=True] After seeing fhe discussions and analyzing the bit manipulation problems for many times, there can be 2 ways to tackle the bit problem FUNCTION ON FULL BITS This is the first way which I think kinda aligns with the first LB winner. The approach is first we take an unary function which is rotate, shift, and their not variations and call it U. U is applied not to a single bit but to the whole number. So NOT(11100111) would simply be (0001

### 684192 - [Dataset Hallucination?] How did you resolve these problems by human?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684192
- Messages: `20`
- Relevance score: `46`
- Keyword hits: `{"bit": 5, "bit_manipulation": 1, "cot": 1, "equation": 14, "equation_transform": 1, "numeric": 3, "operator": 6, "prompt": 2, "symbol": 6, "symbolic": 3, "token": 1, "train": 3}`
- Preview: [message_id=684192 author=Dennis first=True] eeae398e In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 63]67 = 4 18]81 = 9 72-22 = 95 64]48 = 16 65]15 = 5 Now, determine the result for: 65/58 When we look at the above question, slash has never appeared in the example. How can we deduce the solution? We know that ] is max(A, B) % min(A, B). No example is given for / e7cf0394 In Alice's Wonderland, a secret set of transformation rules i

### 691380 - Nemotron ATLAS: Architecture-Targeting LoRA with Augmented Solvers

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691380
- Messages: `5`
- Relevance score: `44`
- Keyword hits: `{"adapter": 1, "cot": 1, "equation": 4, "equation_transform": 3, "lora": 6, "loss": 7, "operator": 1, "prompt": 2, "solver": 8, "token": 2, "train": 9}`
- Preview: [message_id=691380 author=Shehab Anwer first=True] Hi Kaggle community & NVIDIA team 👋 I'd like to share ATLAS , my end-to-end pipeline for the NVIDIA Nemotron Reasoning Challenge. NOTEBOOK LINK ATLAS stands for Architecture-Targeting LoRA with Augmented Solvers . It combines high-quality programmatic reasoning traces with efficient LoRA targeting tailored to Nemotron’s hybrid Mamba-2 + MoE + Attention architecture. Key Techniques Solver-Augmented Training (SAT) Programmatic solvers generated ve

### 698293 - 97.2% Gold-Conditioned Symbolic Solver Exposing Digit Mappings and Operators

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698293
- Messages: `4`
- Relevance score: `43`
- Keyword hits: `{"cot": 1, "equation": 5, "lora": 1, "numeric": 1, "operator": 3, "prompt": 2, "solver": 7, "symbol": 11, "symbolic": 10, "train": 2}`
- Preview: [message_id=698293 author=lkevincc first=True] I have been using this gold-conditioned symbolic solver to study the rule structure of the equation_symbolic category. To be clear, this is not an inference-time competition solution. The solver uses the known target answer as a constraint, so it should be viewed as a research oracle rather than something directly usable in a Kaggle submission. What it shows is that for many examples, there exists a latent symbolic rule that can explain the puzzle. 

### 686069 - Per-Category Error Analysis After SFT (0.63 LB) — Where the Real Bottlenecks Are

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686069
- Messages: `7`
- Relevance score: `39`
- Keyword hits: `{"accuracy": 5, "bit": 7, "bit_manipulation": 1, "cot": 4, "lora": 1, "numeric": 2, "prompt": 2, "symbol": 4, "synthetic": 1, "token": 5, "train": 7}`
- Preview: [message_id=686069 author=EnDream first=True] I ran error analysis on 300 stratified samples (50 per category) after SFT training (1200 samples, LoRA rank 32, 2 epochs). Here are the per-category accuracy numbers: Category Accuracy Error Pattern numeral 100% — unit_conv 100% — bit_ops 30% Model guesses plausible but wrong bit patterns gravity 12% Numerical errors of 10-25%, model doesn't compute g correctly cipher 0% Outputs random plausible words, no actual decryption symbol 6% Completely wrong

### 687961 - Training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961
- Messages: `10`
- Relevance score: `36`
- Keyword hits: `{"adapter": 3, "lora": 7, "loss": 2, "symbol": 1, "token": 4, "train": 19}`
- Preview: [message_id=687961 author=Tong Hui Kang first=True] I want to understand the theoretical limitations when training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences. I have not proven that any of the configurations listed here works in practice. I am making my own training implementation, and I want to understand whether my inefficiencies are avoidable with better implementation. Please help me check if I have missed any theoretical limits, thanks! This table calculates how

### 682355 - Clarification needed: Experimenting with prompting strategies vs. strict sequence length constraints?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/682355
- Messages: `4`
- Relevance score: `35`
- Keyword hits: `{"adapter": 5, "eval": 8, "lora": 1, "prompt": 12, "solver": 1, "token": 3, "train": 5}`
- Preview: [message_id=682355 author=Kamal Raj Kanakarajan first=True] Hi, I have a question regarding how we can experiment with prompting strategies given the fixed evaluation pipeline and some strict constraints on sequence length. Model Sequence Length Constraints Based on the submission demo and competition overview, we are working with very specific length limits: Prompt Limit: The input prompt length is strictly capped at <512 tokens. Total Max Length: The model's max_length is fixed at 8192. Reason

### 686444 - It seems the KV cache is not enabled during RL training

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686444
- Messages: `5`
- Relevance score: `35`
- Keyword hits: `{"adapter": 10, "bit": 1, "eval": 1, "lora": 7, "prompt": 4, "token": 7, "train": 5}`
- Preview: [message_id=686444 author=WillTLing first=True] I encountered the following issue: transformers_modules.nemotron_model.modeling_nemotron_h|WARNING] NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was provided, so no cache will be returned When I was conducting GRPO training, the training speed was very slow. I suspect this might be related to the issue mentioned above. I noticed that a similar problem was also mentioned in this discussion , but I'm not sur

### 690891 - There are still many missing pieces of the puzzle: equation and cryptarithm.

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690891
- Messages: `3`
- Relevance score: `34`
- Keyword hits: `{"accuracy": 1, "bit": 4, "bit_manipulation": 2, "bitsum": 1, "cot": 5, "cryptarithm": 6, "equation": 2, "numeric": 2, "operator": 4, "solver": 3, "token": 3, "train": 1}`
- Preview: [message_id=690891 author=Zejun_ first=True] Thank huikang for providing such a powerful new starting point for the latter part of this competition. Congratulations! If you run the longer notebook and look at the report results, you will find that the accuracy rates for the three categories of problems, cryptarithm_deduce , cryptarithm_guess , and equation_numeric_guess , are all very low. These will be the main directions for everyone's efforts after this incredibly powerful baseline. To solve 

### 692092 - Please help me, my predict code as slow as 1 token/second

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/692092
- Messages: `3`
- Relevance score: `32`
- Keyword hits: `{"adapter": 1, "bit": 1, "eval": 1, "lora": 2, "prompt": 2, "token": 21, "train": 4}`
- Preview: [message_id=692092 author=zkhdGuoFeng first=True] blow is my predict code, it's too slow that predicting 1token/second. My server is 4 cards of L20(48G GPU): PyTorch: 2.6.0+cu118 CUDA available: True CUDA version: 11.8 GPU count: 4 GPU name: NVIDIA L20 nvidia-smi Thu Apr 16 16:28:16 2026 +-----------------------------------------------------------------------------------------+ | NVIDIA-SMI 580.126.20 Driver Version: 580.126.20 CUDA Version: 13.0 | +-----------------------------------------+----

### 687798 - Rescore After Metric Update

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798
- Messages: `20`
- Relevance score: `29`
- Keyword hits: `{"accuracy": 1, "bit": 9, "equation": 2, "eval": 2, "lora": 2, "symbol": 4, "symbolic": 2, "token": 2, "train": 5}`
- Preview: [message_id=687798 author=Ryan Holbrook first=True] Hi everyone, Recently, there was an update to the evaluation metric that fixed a bug that was causing binary answers to match as floats instead of exactly as strings. This update led to a drop of about 0.3-0.4 points on new submissions. (The update is in the verify() function of the linked notebook.) So that the leaderboard will only reflect submissions using the new metric, I will be initiating a rescore on Monday. This rescore will only evalu

### 692950 - Is ~0.86 a current ceiling for most approaches? 

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/692950
- Messages: `8`
- Relevance score: `28`
- Keyword hits: `{"bit": 1, "cot": 6, "cryptarithm": 3, "lora": 2, "prompt": 1, "synthetic": 5, "token": 5, "train": 5}`
- Preview: [message_id=692950 author=Taha first=True] I’ve been experimenting quite a bit with the Nemotron reasoning setup over the past few days, and I’m starting to feel like many of us might be hitting a similar ceiling around ~0.86. From what I can tell, a lot of current approaches seem to fall into two buckets: building directly on top of strong public notebooks (which already reach ~0.86), or trying to reproduce / adapt methods like Tong Hui Kang’s pipeline I’ve personally tried a mix of things on t

### 688120 - Calling experienced fine-tuners: share LoRA / PEFT resources for newcomers in this challenge

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688120
- Messages: `2`
- Relevance score: `26`
- Keyword hits: `{"adapter": 2, "bit": 2, "chain of thought": 1, "cot": 2, "equation": 4, "eval": 4, "lora": 5, "train": 6}`
- Preview: [message_id=688120 author=Sahil Patil first=True] Hi everyone, A lot of participants here are new to LoRA/PEFT and training adapters. This comp’s setup-Nemotron base + LoRA zip submission + exact-match puzzle eval-is a concrete goal, but the learning curve is still steep without a good reading order and a few battle-tested patterns. If you’ve fine-tuned LLMs before (especially LoRA on HF stacks, Kaggle GPUs, or similar reasoning/supervised setups), it would help a lot if you could pay it forward

### 683172 - Kaggle CLI — Develop Locally and Run on RTX Pro 6000 GPU

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683172
- Messages: `14`
- Relevance score: `25`
- Keyword hits: `{"adapter": 1, "eval": 5, "lora": 1, "token": 2, "train": 16}`
- Preview: [message_id=683172 author=Keanan first=True] I have also added a notebook version for easier navigation (table of contents): [ https://www.kaggle.com/code/citerne/from-local-dev-rtx-6000-kaggle-cli-guide ] Practical guide for the NVIDIA Nemotron Model Reasoning Challenge . Hard-won lessons on setting up a CLI → Kaggle GPU workflow, pitfalls to avoid, and everything you need to get started. Why This Workflow? The Kaggle web interface is great for exploration, but once you want to: Iterate quickly

### 688360 - [update] Read CPMP's reply. [original] Do not distill models that do not allow distillation (e.g. gemini, gpt5)

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688360
- Messages: `28`
- Relevance score: `25`
- Keyword hits: `{"adapter": 4, "bit": 2, "cot": 1, "lora": 3, "synthetic": 2, "train": 13}`
- Preview: [message_id=688360 author=c-number first=True] https://www.kaggle.com/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels It seems it's okay. [message_id=3439425 author=CPMP first=False] @cnumber Thanks for raising this, it is a recurring issue in Kaggle competitions. Not kaggle fault, rather the complexity of IP laws and regulations. Disclaimer: We cannot provide legal advice. Contact someone trained on US IP laws if you need advice. With the above disclaimer, here is our thought.

### 692879 -  Is DoRA allowed does it actually improve LB scores?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/692879
- Messages: `8`
- Relevance score: `24`
- Keyword hits: `{"adapter": 4, "bit": 1, "cryptarithm": 2, "eval": 2, "lora": 8, "train": 7}`
- Preview: [message_id=692879 author=Taha first=True] Hey everyone, I’m currently prepping a final training run and wanted to get the community's thoughts on using DoRA (Weight-Decomposed Low-Rank Adaptation) instead of standard LoRA for this specific reasoning challenge. I’ve managed to hit a solid baseline (around 0.84 - 0.85) using standard SFT with a carefully balanced data mix and a cosine scheduler. However, I'm looking for that last 1-2% push to handle the harder algorithmic categories (like cryptar

### 686865 - Has anyone successfully run NVIDIA Nemotron using DeepSpeed ZeRO-3?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686865
- Messages: `4`
- Relevance score: `22`
- Keyword hits: `{"bit": 1, "loss": 1, "token": 1, "train": 19}`
- Preview: [message_id=686865 author=HaoKwok first=True] Hi everyone, I am seeking advice on a persistent issue when transitioning from DeepSpeed ZeRO-2 to ZeRO-3 for fine-tuning Nemotron-30B on a local setup with 2x NVIDIA A100 (80GB). Current Context & Motivation Previously, I was using ZeRO-2, which works but is pushed to the absolute physical limit. With per_device_train_batch_size=2 and max_seq_len=3072, my VRAM usage sits at 79.5GB / 80GB. To gain more headroom for larger batches or longer sequences,

### 681793 - Are problem types the same for train and test?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681793
- Messages: `7`
- Relevance score: `21`
- Keyword hits: `{"adapter": 2, "bit": 2, "equation": 1, "eval": 2, "lora": 4, "prompt": 5, "train": 5}`
- Preview: [message_id=681793 author=Devin Anzelmo first=True] There appear to be six different problem types in the training set: numbers are secretly converted into a different numeral system the gravitational constant has been secretly changed a secret set of transformation rules is applied to equations secret encryption rules are used on text a secret bit manipulation rule transforms 8-bit binary numbers a secret unit conversion is applied to measurements Are these the same set of problem types that ap

### 682167 - What is the minimum VRAM for training?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/682167
- Messages: `4`
- Relevance score: `21`
- Keyword hits: `{"adapter": 1, "bit": 2, "lora": 9, "train": 9}`
- Preview: [message_id=682167 author=c-number first=True] I tried training with unsloth using my RTX 5090, but failed due to OOM. Has anyone been able to train locally? [message_id=3422854 author=lucian kucera first=False] Bruh u have RTX PRO 6000 on kaggle, there is no point in using 5090. Kaggle GPU has 3 times the vram. Model is in BF16 and has 30B params so simple math u need 60GB vram to load model. Than ofc when u train lora u add additional params + activations. Optimizer has copy of each trainable 

### 689840 - Is RLVR worth it? or should I work on SFT only?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689840
- Messages: `6`
- Relevance score: `18`
- Keyword hits: `{"accuracy": 2, "bit": 2, "bitwise": 1, "cot": 2, "equation": 3, "eval": 2, "numeric": 1, "symbol": 2, "symbolic": 1, "token": 1, "train": 1}`
- Preview: [message_id=689840 author=m4nocha first=True] here are my results with SFT ======================================================= ✅ OVERALL ACCURACY : 81.34% (693/852) ------------------------------------------------------- > Bit Manipulation : 9.23% (12/130) > Equation Transformation : 52.11% (37/71) > Gravitational Constant : 100.00% (148/148) > Number Base Conversion : 100.00% (162/162) > Text Encryption : 95.86% (162/169) > Unit Conversion : 100.00% (172/172) -------------------------------

### 689877 - Hallucination in equation problems?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689877
- Messages: `3`
- Relevance score: `16`
- Keyword hits: `{"equation": 3, "eval": 1, "operator": 8, "prompt": 3, "symbol": 1}`
- Preview: [message_id=689877 author=Darren Amadeus Martin first=True] I have noticed that on some of the equations problem, around 20% of the prompt asks for equations that did not even have any examples at all. For example is id 260f20c1 where the problem is In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 84[69 = 153 13+97 = 1260 46+47 = 2161 52[80 = 132 Now, determine the result for: 22\65 Answer: 43 Since the sign '\' did not appear in the 

### 696059 - What if the answer contains square brackets?

- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/696059
- Messages: `13`
- Relevance score: `16`
- Keyword hits: `{"adapter": 2, "cryptarithm": 2, "equation": 4, "eval": 1, "lora": 2, "symbol": 2, "train": 3}`
- Preview: [message_id=696059 author=Birtley Doru (DB) first=True] fa5dfa46,"In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: $< &@ = &$@\ }->@ = -!$ }>+@< = !}} \@ @@ = &<>$ Now, determine the result for: }^-`}",-^} What if the answer contains square brackets? Can putting a box over a character correctly identify whether it's right or wrong? [message_id=3453927 author=Ogurtsov first=False] @ryanholbrook If we have any chance to get fixed train
