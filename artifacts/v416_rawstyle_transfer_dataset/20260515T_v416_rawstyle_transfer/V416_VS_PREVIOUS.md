# V416 Rawstyle Transfer Dataset

| Item | V410 | V416 |
|---|---:|---:|
| Train rows | `2320` | `2320` |
| Val rows | `580` | `580` |
| Weak/full rows used for train | `0` | `0` |
| Completion style | `Rule / Final answer` | `raw-output-style boxed suffix` |
| Prior transfer result | V413 failed: `190/315`, eq `56`, bit `134` | not launched yet |

V416 exists only to test the V370/V415 finding that solver teachers are not transferring into the adapter. It is not a submit artifact.

HF/Kaggle GPU remains blocked until tokenization and debug gates pass.