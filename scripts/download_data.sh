#!/usr/bin/env bash
# One-time bootstrap: download the raw Lending Club dataset from Kaggle into data/raw/.
# After `dvc add`, the DVC remote (not Kaggle) is the source of truth —
# collaborators and CI run `dvc pull`, never this script.
set -euo pipefail

DATASET="ethon0426/lending-club-20072020q1"
OUT_DIR="data/raw"

mkdir -p "$OUT_DIR"
uv run kaggle datasets download "$DATASET" -p "$OUT_DIR" --quiet
