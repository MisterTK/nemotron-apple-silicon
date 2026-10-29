#!/bin/zsh
cd "$(dirname "$0")/.."
M=models/mlx-qwen30b-4bit; O=results/results_mlx_matrix.jsonl; T=matrix-qwen30b
# done: bench/run.sh $M --mode shared --suffix 1024 --ctx 4096 --users 1 2 4 8 16 32 --gen 256 --variants gqa:0 --tag $T --out $O
bench/run.sh $M --mode shared --suffix 1024 --ctx 32768 --users 1 2 4 8 16 32 --gen 256 --variants stock:0,gqa:0,stock:8,gqa:8,gqa:4 --tag $T --out $O
bench/run.sh $M --ctx 131072 --users 1 --gen 256 --gqa-kernel --tag $T --out $O
bench/run.sh $M --ctx 131072 --users 1 --gen 256 --gqa-kernel --kv-bits 8 --tag $T --out $O
echo ALL_DONE
