#!/bin/zsh
# Gemma 4 26B-A4B on MLX: same single-user, concurrency-matrix and gate runs as Nemotron/Qwen.
# Stock attention only: gqa_decode_attn supports D=128, Gemma uses D=256 (sliding) / 512 (global).
cd "$(dirname "$0")/.."
M=models/mlx-gemma26b-4bit; P=(--prefill-step 512)
bench/run.sh $M --ctx 4096 32768 131072 --users 1 --gen 512 $P --tag single-gemma26b --out results/results_mlx.jsonl
for L in 4096 32768 131072; do
  bench/run.sh $M --mode shared --suffix 1024 --ctx $L --users 1 2 4 8 16 32 --gen 256 $P --variants stock:0,stock:8,stock:4 --tag matrix-gemma26b --out results/results_mlx_matrix.jsonl
done
bench/run.sh $M --ctx 131072 --users 1 --gen 2048 --trials 5 $P --tag gate --out results/results_gate.jsonl
echo ALL_DONE
