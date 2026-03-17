"""Convert baseline_submission.py into a Kaggle notebook (.ipynb)"""
import json

with open("C:/tmp/kaggle-nemotron/baseline_submission.py", "r") as f:
    content = f.read()

cells = []
parts = content.split("# ============================================================\n# CELL")
for part in parts:
    if not part.strip():
        continue
    lines = part.strip().split('\n')
    code_lines = []
    for i, line in enumerate(lines):
        if i == 0 and (': ' in line or line.startswith('=')):
            continue
        if line.strip() == '# ============================================================':
            continue
        code_lines.append(line)
    code = '\n'.join(code_lines).strip()
    if code:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": code.split('\n')
        })

for cell in cells:
    lines = cell["source"]
    cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

notebook = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [{"sourceId": 125197, "databundleVersionId": 12345, "sourceType": "competition", "isSourceIdPinned": True}],
            "dockerImageVersionId": 30000,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

with open("C:/tmp/kaggle-nemotron/baseline_submission.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Created baseline notebook with {len(cells)} cells")
