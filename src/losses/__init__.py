"""Custom loss functions for KG1 training.

- max_min_logprob: Tong winner (0.877) loss — penalizes worst-predicted token per sample.
  Empirically validated (Agent V5, 2026-04-21) — converges without NaN.
  Requires: grad_clip=1.0, warmup CE 100-500 steps, LR 0.5x vs CE baseline.
"""
from .max_min_logprob import max_min_logprob_loss, MaxMinLogProbLoss  # noqa: F401
