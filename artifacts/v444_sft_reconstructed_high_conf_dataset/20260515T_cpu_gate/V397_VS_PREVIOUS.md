# V397 vs Previous

| Metric | Previous active state | V397 candidate | Decision |
|---|---:|---:|---|
| Best adapter-only weak | `192/315` | not trained | preserve as baseline |
| equation_transform weak | `56/155` | not trained | tokenization gate first |
| bit_manipulation weak | `136/160` | not trained | tokenization gate first |
| Train rows | n/a | `1848` | new transfer corpus |
| Validation rows | n/a | `172` | new transfer corpus |
| Weak row overlap | must be `0` | `0` | pass |
| Train/val prompt overlap | must be `0` | `0` | pass |

V397 is not a submit candidate. It is a CPU-gated dataset candidate built from local reconstructed public-train traces.
