# V260B V259 Eq-Focus Weak Eval Summary

- Job: https://huggingface.co/jobs/felipesp1983/6a0145edaff1cd33e8f333a0
- Upload commit: https://huggingface.co/felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/commit/496d31f4284ec45b278e561aee4543005767a661
- Contract: V221 reproduced, H200, thinking enabled, max_tokens=7680, max_model_len=8192, max_num_seqs=64. Shared row contract `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`, weak CSV SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.

| Candidate | Total | Equation | Bit | Trunc | Decision |
|---|---:|---:|---:|---:|---|
| `v259_checkpoint_4_v221_contract` | 192 | 56 | 136 | 0 | best but weak gate failed |
| `v259_checkpoint_8_v221_contract` | 191 | 56 | 135 | 1 | worse than best |
| `v259_final_v221_contract` | 190 | 56 | 134 | 1 | worse than best |

Changed rows vs V258 checkpoint-4 predictions: `7`. Net score is unchanged at `192/315`: V260B lost one previously correct bit row (`4ef88f92`) and gained one bit row (`59bee375`); four changed equation predictions remain incorrect.

Conclusion: the equation-focused V259 smoke did not improve equation beyond 56/155; do not run a longer blind continuation from this recipe without new data/solver evidence.
