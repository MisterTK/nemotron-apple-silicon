#!/bin/zsh
# Re-run Gemma 32K quantized points lost to the OOM abort, then quality checks.
cd "$(dirname "$0")/.."
M=models/mlx-gemma26b-4bit; P=(--prefill-step 512)
bench/run.sh $M --mode shared --suffix 1024 --ctx 32768 --users 1 2 4 8 16 32 --gen 256 $P --variants stock:4 --tag matrix-gemma26b --out results/results_mlx_matrix.jsonl
bench/run.sh $M --mode shared --suffix 1024 --ctx 32768 --users 32 --gen 256 $P --variants stock:8 --tag matrix-gemma26b --out results/results_mlx_matrix.jsonl
bench/run.sh $M --mode shared --suffix 1024 --ctx 131072 --users 16 32 --gen 256 $P --variants stock:4 --tag matrix-gemma26b --out results/results_mlx_matrix.jsonl
PY=(env PYTHON_GIL=0 ${MLX_PY:-python3} -I)
$PY bench/validate.py $M --test agreement --ctx 32768 --gen 128 --variants stock:8,stock:4 --out results/results_validate.jsonl 2>&1 | grep -v Warning
$PY bench/validate.py $M --test needle --ctx 131072 --depths 0.1 0.5 0.9 --variants stock:8,stock:4 --out results/results_validate.jsonl 2>&1 | grep -v Warning
echo FOLLOWUP_DONE
