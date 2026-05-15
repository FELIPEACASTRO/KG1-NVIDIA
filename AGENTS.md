# KG1 Agent Instructions

## Notebook Generation

- Every generated Colab/Jupyter notebook must include explicit progress logs in each operational cell.
- Long-running cells must print start/end markers, key input paths, output paths, command lines, return codes, and success/failure summaries.
- Script cells must make it possible to verify execution progress without waiting for the final manifest only.
- If a script is long-running and currently silent, update either the script or the notebook wrapper to emit progress checkpoints before using it.
- Every created or changed Colab notebook must be accompanied by the exact Colab execution URL in the response.
- If the notebook exists only locally and has not been pushed to GitHub yet, explicitly say that the URL will work only after the notebook is pushed to the referenced branch.

## Notebook Release Gate

- Every created or changed `.ipynb` must pass `python scripts/notebook_release_gate.py <notebook path>` before it is delivered, committed, or pushed.
- If multiple notebooks were created or changed, pass all of them to the gate in the same command.
- The gate must validate the notebook structure, Python syntax in code cells, explicit `# CELL:` headers, START/END progress markers, Colab URL, long-running command logging, hard Kaggle-submit lock, data hash/path checks, V194 adapter checks, runtime/dependency checks, train completion checks, and weak/full eval gates.
- Training/evaluation notebooks must include executable checks for H100/CUDA capacity, `causal-conv1d`, `mamba_ssm`, vLLM install ordering, exact dataset hashes, exact Drive adapter path, adapter tensor/config compatibility, tokenization/truncation/offset-mask contracts, final adapter completeness, and quality gates before full eval/submission.
- The CI workflow `.github/workflows/notebook-release-gate.yml` enforces this gate for every changed notebook on push and pull request.
- Historical notebooks are not retroactively required to pass until they are edited; once edited, they must satisfy the current gate.

## Static Safety Gate

- Every created or changed script, HF/Kaggle job launcher, workflow, or notebook must pass `python scripts/kg1_static_safety_gate.py <changed paths>` before it is delivered, committed, pushed, or executed.
- This gate blocks active jobs/notebooks from using archived mixed V435E preference data, `format_negative_*` preference rows, or `ALLOW_FORMAT_NEGATIVES` unless the file is an explicit CPU diagnostic gate/builder.
- Any preference-training job must use hard-negative-only preference data by default and must fail closed if a dataset contains format-only negatives.
- The CI workflow `.github/workflows/notebook-release-gate.yml` runs this gate over changed files on push and pull request.

## HF Job Log Monitoring

- While actively analyzing a running HF/Kaggle/Colab job, check job status and logs every ~30 seconds unless a command is blocked, a human explicitly pauses monitoring, or the job reaches a terminal state.
- Each monitoring update should identify the current stage, any new progress markers, and whether there is a failure, regression, or next automated action.
- H200 is authorized for KG1 jobs up to 1 hour per execution. If a run needs more than 1 hour, stop and ask for explicit human authorization before continuing.
