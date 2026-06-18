"""Convert nemotron_training.py into a Kaggle notebook (.ipynb)"""
import json

# Read the training script
with open("C:/tmp/kaggle-nemotron/nemotron_training.py", "r") as f:
    content = f.read()

# Split by CELL markers
cells = []
parts = content.split("# ============================================================\n# CELL")
for part in parts:
    if not part.strip():
        continue
    # Remove the cell header line
    lines = part.strip().split('\n')
    # Skip the header line (e.g., "1: Setup and imports") and separator
    code_lines = []
    skip_next = False
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
            "source": code.split('\n')  # Will rejoin with newlines
        })

# Fix source format: each line needs \n except the last
for cell in cells:
    lines = cell["source"]
    cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {
                    "sourceId": 125197,
                    "databundleVersionId": 12345,
                    "sourceType": "competition",
                    "isSourceIdPinned": True
                }
            ],
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

with open("C:/tmp/kaggle-nemotron/nemotron_training.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Created notebook with {len(cells)} cells")
