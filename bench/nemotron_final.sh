#!/bin/zsh
cd "$(dirname "$0")/.."
M=models/mlx-nemotron30b-4bit
PY=(env PYTHON_GIL=0 ${MLX_PY:-python3} -I)
$PY bench/validate.py $M --test agreement --ctx 32768 --gen 128 --out results/results_validate.jsonl 2>&1 | grep -v Warning
$PY bench/validate.py $M --test needle --ctx 131072 --depths 0.1 0.5 0.9 --variants gqa:0,gqa:8,gqa:4,stock:8 --out results/results_validate.jsonl 2>&1 | grep -v Warning
for v in "" "--gqa-kernel" "--gqa-kernel --kv-bits 4"; do
  bench/run.sh $M --ctx 131072 --users 1 --gen 2048 --trials 5 ${=v} --tag gate --out results/results_gate.jsonl
done
echo ALL_DONE
