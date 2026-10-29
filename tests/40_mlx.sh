#!/bin/bash
# MLX: Nemotron-Nano decode with stock attention vs the GQA kernel, plus the 128K spec-gate run. ~40 min.
# Needs mlx-lm in $MLX_PY and the model under models/. Other models: MLX_NEMO=models/mlx-qwen30b-4bit.
. "$(dirname "$0")/common.sh"; need "$MLX_NEMO"
bench/run.sh "$MLX_NEMO" --mode shared --suffix 1024 --ctx ${CTX:-4096 131072} --users ${USERS:-1 4 16} --gen 256 \
  --variants stock:0,gqa:0,gqa:4 --tag test-mlx --out results/results_mlx_matrix.jsonl
bench/run.sh "$MLX_NEMO" --ctx 131072 --users 1 --gen 2048 --trials ${TRIALS:-3} --gqa-kernel --tag gate --out results/results_gate.jsonl
python3 bench/report_tables.py results/results_mlx_matrix.jsonl
