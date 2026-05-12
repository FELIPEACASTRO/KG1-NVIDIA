# Tong Hui Kang - NVIDIA Nemotron Discussion Sweep

Source: Kaggle authenticated DiscussionApiService.

Queries: `huikang, Tong Hui Kang, NVIDIA Nemotron huikang, NVIDIA Nemotron Tong Hui Kang, Open Progress Prize huikang, Strategy to solve bit manipulation huikang, nemotron.huikang.dev`

Topic refs collected: `29`
Author-authored items in target competition: `24`

## Query Stats

- {"next_left": false, "pages": 1, "query": "huikang", "seen": 20, "total": 99}
- {"next_left": false, "pages": 1, "query": "Tong Hui Kang", "seen": 20, "total": 67}
- {"next_left": false, "pages": 1, "query": "NVIDIA Nemotron huikang", "seen": 14, "total": 14}
- {"next_left": false, "pages": 1, "query": "NVIDIA Nemotron Tong Hui Kang", "seen": 6, "total": 6}
- {"next_left": false, "pages": 1, "query": "Open Progress Prize huikang", "seen": 10, "total": 10}
- {"next_left": false, "pages": 1, "query": "Strategy to solve bit manipulation huikang", "seen": 1, "total": 1}
- {"next_left": false, "pages": 1, "query": "nemotron.huikang.dev", "seen": 2, "total": 2}

## Author Items

### topic - Visualize the problems and completions from the base model (2026-03-24T03:34:27.018Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212

I ran the base model (yes, the base model) over all the 9500 problems at least once. These are the results.

https://nemotron.huikang.dev/base

You can see the model generation as well.

The code is here. This is derived from my work in AIMO 3 and ARC-AGI-2.

### reply - Visualize the problems and completions from the base model (2026-03-27T03:46:30.330Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212#3429662

Thanks! I uploaded my copy as well https://www.kaggle.com/datasets/huikang/nemotron-base-model-generation

### reply - Visualize the problems and completions from the base model (2026-03-24T04:45:10.827Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212#3427444

If I only consider the latest run for each of the 9500 problems, apparently there are 48,217,898 tokens generated.

At 2.5k tokens per second it is 5.35 hours, which is about right.

### comment - Visualize the problems and completions from the base model (2026-03-24T04:07:04.903Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212#3427425

FAQ

Is this the base model? It is the base model.

Why are the partially solved problems in the middle? Most of the questions are ran only once, except some at the start and some in the middle. I ran the inference script in alphabetical and reverse alphabetical order and went for a nap. When I woke up, each script is slightly more than half done and I terminated it.

Where is the prompt from? It is from the official metric notebook.

How much money did it cost? On Modal I ran on an RTX PRO 6000 which is $3.03 per hour. I think I ran this for five hours. The throughput was 2.5k tokens per second.

Comments

You see that the solve rate is almost 50%, which aligns with the demo submission from the organizer.

For many entries of equation numeric, and almost all entries in equation symbolic, I could not figure the pattern. This is reported in other threads.

### reply - Visualize the problems and completions from the base model (2026-03-24T03:44:41.777Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212#3427409

Oh, that is extracted from the problem. If you see, the problems categorizes nicely into 6 (or 7) formats.

### reply - Visualize the problems and completions from the base model (2026-03-24T10:06:16.957Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212#3427625

lol I somewhat gave up on AIMO 3, I want to prove that I can finetune models here, then I go back to AIMO 3

### topic - Training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences (2026-04-04T06:52:51.352Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961

I want to understand the theoretical limitations when training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences.

I have not proven that any of the configurations listed here works in practice. I am making my own training implementation, and I want to understand whether my inefficiencies are avoidable with better implementation. Please help me check if I have missed any theoretical limits, thanks!

This table calculates how much memory is needed to train Nemotron-3-Nano-30B-A3B-BF16 with different microbatch sizes (μ). Larger microbatch sizes can improve hardware utilization and speed up training, but only if they fit in memory [1].

Component
Formula
μ=1
μ=4
μ=16
μ=64

Base model weights (BF16)
W × 2
63.6 GB
63.6 GB
63.6 GB
63.6 GB

LoRA adapter weights (FP32)
P × 4
3.5 GB
3.5 GB
3.5 GB
3.5 GB

LoRA gradients (FP32)
P × 4
3.5 GB
3.5 GB
3.5 GB
3.5 GB

Optimizer m + v (FP32)
P × 8
7.1 GB
7.1 GB
7.1 GB
7.1 GB

CUDA context & buffers
~3 GB
3.0 GB
3.0 GB
3.0 GB
3.0 GB

Checkpointed layer inputs
L × μ × S × H × 2
2.3 GB
9.2 GB
36.6 GB
146.6 GB

Peak intra-layer intermediates
μ × S × D × 2
331 MB
1.3 GB
5.3 GB
21.2 GB

Backward intra-layer gradient
μ × S × H × 2
44...

### reply - Training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences (2026-04-04T13:17:52.897Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961#3435611

There is no point training with a length of 16384 since the limit is 8192.

If you limit to 8192, you should be able to train with 
μ=2 I guess?

### reply - Training Nemotron-3-Nano-30B-A3B-BF16 with rank 32 LoRA on length 8192 sequences (2026-04-22T02:57:36.123Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961#3446735

Good question. The Github PR I cited has strings like 
_supports_flash_attn_2 = True.

When I run the training script on Kaggle, it has logs with 
FA2 = False

==((====))== Unsloth 2026.3.17: Fast Nemotron_H patching. Transformers: 4.57.6.
 \\ /| NVIDIA RTX PRO 6000 Blackwell Server Edition. Num GPUs = 1. Max memory: 94.971 GB. Platform: Linux.
O^O/ \_/ \ Torch: 2.10.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.6.0
\ / Bfloat16 = TRUE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-" Free license: http://github.com/unslothai/unsloth

In summary, I am not sure.

### topic - Midpoint Cut-off Date and the Open Progress Prize (2026-04-06T01:12:02.640Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688482

Congratulations to @yiyangzheng for scoring 0.84, whose team also won the AIMO3 Longest Leader Prize.

For the Open Progress Prize of this competition

Awarded to the team with the highest leaderboard score as of the Midpoint Cut-off Date: April 9, 2026.

How exactly is April 9, 2026 defined - is it the end of the day? Also, is the defined by the submission time as well?

Edit: "The deadline is 11:59pm UTC unless otherwise noted", see comment

I also presume that winning the Open Progress Prize requires sharing the submissions by April 16, 2026, as this section of the rules apply to the prize.

To be eligible for any prize, teams must publish a public Kaggle notebook and solution write-up documenting the methods, datasets, and techniques used to produce the submission.

### comment - Midpoint Cut-off Date and the Open Progress Prize (2026-04-07T13:30:26.493Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688482#3437302

The eve of April 9 is approaching, @ryanholbrook for clarification

### topic - How many examples are there in the public leaderboard? (2026-04-08T02:54:47.312Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689257

I think first ranked below "Alice's Wonderland" with 0.82, and then I ranked above them with 0.82.

If there are 250 examples in the public leaderboard, the scores would look like this

205 / 250 -> 0.820 -> 0.82

206 / 250 -> 0.824 -> 0.82

207 / 250 -> 0.828 -> 0.82

The training set is 9500, it felt like 10000 were generated and 250 goes to the public leaderboard and 250 goes to the private leaderboard.

### reply - How many examples are there in the public leaderboard? (2026-04-08T07:41:28.517Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689257#3437846

9500 puzzles took me 5 hours on the H100, I am very surprised if the public leaderboard could score 50000 puzzles in one hour.

I guess 50000 puzzles per hour is still possible if your completion is just 20 tokens long.

### topic - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-10T03:02:11.814Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915

I would like to thank the competition hosts and Kaggle for organizing this competition.
I did manage to find something interesting to bet on, and I am happy to see my gamble paying off.

You might have made some predictions that I have asked for. These are the answers.

The score I was aiming for - 0.877

How many tokens are used to train - 27,850,703 tokens for the winning solution, 598,958,637 in total

How much money I have spent - $212.48 in Tinker credits, approximately $60 in Modal credits, $10 for Kaggle / Colab subscription.

What do you think is the secret - bit manipulation, you only need SFT, deterministic chain-of-thought design, use of min logprob as objective, use of Tinker for training

Quick links

Original notebook

Validation notebook

Training metrics and logs for winning submission

Github containing the relevant code

What was I betting on

This is what I am betting on

Nemotron can act as a simple computer after LoRA training.

These are the assumptions in my bet

I can craft the chain-of-thought better than my competitors. I think this turned out to be true. The main differentiator was the bit manipulation problem where I managed to figure out a chain of thou...

### comment - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-23T20:47:18.063Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3447754

I have received the DGX Spark!

### comment - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-12T20:01:17.150Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3440666

I have published my work, all the best for the remainder of the competition!

Please reach out here if I am missing anything!

### reply - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-15T05:40:37.580Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3442215

Thanks for trying out my code and data!

I am curious how did you even get to submit the Tinker adapter even for the 0.74 submission. It was really not easy for me to figure out.

The idea of doing SVD was from Claude Code, I wonder if you did something similar.

My view is that 0.87 is achievable without the breakthrough in cryptarithm.

My bit manipulation chain of thought design is actually not that easy for LLMs to figure out the pattern. It seems that it is still quite hard to get the LLM to decide whether to produce a whitespace (it means there is a match in the correct position) or produce a 
y (there is a match, but at the wrong position).

### reply - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-12T20:00:04.270Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3440665

Thanks for the kind words!

### reply - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-13T17:04:37.017Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3441211

The defaults are defaults and were being overridden

### reply - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-10T04:15:48.293Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3438988

Thanks!

It seems that you only made 4 submissions on April 9 the final day. I was waiting for your 43rd submission before I submit mine, but I was too tired and gave up at 6:30am, made my final submissions, and went to sleep.

### reply - [Open Progress Prize Publication] SFT to maximize minimum logprob (2026-04-15T02:28:34.563Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915#3442135

Just for the winning submission - 27,850,703 x $0.40 = $11.14

### topic - Strategy to solve 85% of bit manipulation (2026-04-11T07:43:56.656Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307

This is part of my publication for the Open Progress Prize.

I read the 0.73 scoring notebook from @llkh0a / Kh0a.

The approach described in Kh0a's notebook is actually very similar to mine

Use code to write synthetic CoT traces

Train SFT on the synthetic CoT traces

Make the submission

Kh0a reports the following validation score.

Per-category:
 bit_manipulation: 35/160 = 21.88%
 gravity_physics: 160/160 = 100.00%
 numeral_system: 158/158 = 100.00%
 numeric_equation: 51/73 = 69.86%
 symbol_transform: 0/82 = 0.00%
 text_decryption: 145/158 = 91.77%
 unit_conversion: 159/159 = 100.00%
Overall: 708/950 = 74.53%
Weighted CV score: 74.76%

Kh0a's algorithm solves only 35/160 of bit manipulation problems.

I have an algorithm that solves 1364 of 1602 bit manipulation problems (85.1%).

85.1% of 160 is around 136. The additional 136 - 35 = 101 correct solutions will bring the overall score from 708/950 to 809/950 which is approximately 85%, which is the same as my winning submission score.

If Kh0a was actually able to perfectly train the model to generate exactly the chain of thought, Kh0a would have won the progress prize.

I describe my algorithm for bit manipulation here in a sep...

### reply - Strategy to solve 85% of bit manipulation (2026-04-11T12:26:11.480Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307#3439888

Yes, I will need to release all the CoT for my winning submissions.

### comment - How to Cut Nemotron Training from 11 Hours to 5h 40m (And Fix the "Loss Illusion") (2026-04-26T16:00:16.953Z)

URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694710#3448796

Thanks for listing three of the changes that I have made.

However, none of the the changes here makes a material difference to training time directly.

Point 1 on cut cross entropy do make a difference to memory, otherwise the sequences would not fit in memory. It does not materially change how many floats are being computed. The mechanism on how this affects training time is that I could fit more sequences in a GPU, which makes training faster.

Point 2 on loss masking is simply necessary to ensure you are training the model on the correct things. If you ask the model to memorize the question the model is less effective at memorizing the solution approach. This does not materially change how many floats are being computed either.

Point 3 on weight tying does not affect training time much either. Most of the forward and backward passes involves the base Nemotron model and that it not being reduced. If implemented correctly weight tying reduces memory usage of the adapter, but I did not implement it correctly. In my implementation, each expert still has a copy of weights, they are just synchronized before weight updates. I have not talked to the team behind Tinker, I am still curi...

## Actionable KG1 Signals

- Prioritize the bit-manipulation per-bit relation/stride algorithm from discussion `690307` as V296 CPU audit and teacher-signal extraction.
- Do not assume direct solver-code submission is packageable; use it for verifier, synthetic CoT/data, and no-loss weak/full auditing.
- The reply stating all CoTs would be released points to checking public HF/Kaggle releases before further GPU spend.
