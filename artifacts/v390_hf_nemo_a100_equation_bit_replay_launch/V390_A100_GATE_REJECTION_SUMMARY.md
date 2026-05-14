# V390 A100 Gate Rejection Summary

Data: 2026-05-14

Job: https://huggingface.co/jobs/felipesp1983/6a0627f4e48bea4538b9d56e

| Item | Valor |
|---|---|
| Status | `ERROR` por preflight gate |
| Hardware | `NVIDIA A100-SXM4-80GB` |
| Torch/CUDA | `torch=2.9.0a0+50eac811a6.nv25.09`, `cuda=13.0` |
| Decisao | bloquear A100 com CUDA 13 |
| Motivo | jobs HF anteriores expuseram incompatibilidade CUDA 13/A100; o gate deve cortar antes de treino caro |
| Proxima acao | relancar como V391 em H200, mantendo dataset/gates V390 |

Conclusao: V390 nao treinou. O bloqueio foi correto por FinOps e por validacao de runtime.
