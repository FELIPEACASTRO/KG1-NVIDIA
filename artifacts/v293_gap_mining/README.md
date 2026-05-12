# V293 Gap Mining

This directory records the postprocessor audit over V291 full predictions.

Key result:

- V291 adapter-only baseline: `823/947`, `equation_transform=56/155`, `bit_manipulation=135/160`.
- Applying the previously verified V274 rules to V291 predictions would produce `827/947`, `equation_transform=60/155`, `bit_manipulation=135/160`, with four zero-loss equation gains on the evaluated rows.

The rules are not directly packageable in the Kaggle adapter-only submission path, so V293 focuses on distilling those patterns into a small LoRA continuation.
