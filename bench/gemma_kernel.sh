#!/bin/zsh
# Gemma 4 26B-A4B with the wide-D GQA kernel (global layers, D=512), plus stock points lost to OOM
# aborts, quality checks, kernel gate runs. Then the llama.cpp concurrency sweep.
cd "$(dirname "$0")/.."
M=models/mlx-gemma26b-4bit; P=(--prefill-step 512); T=matrix-gemma26b; O=results/results_mlx_matrix.jsonl
bench/run.sh $M --ctx 4096 32768 131072 --users 1 --gen 512 $P --gqa-kernel --tag single-gemma26b --out results/results_mlx.jsonl
bench/run.sh $M --mode shared --suffix 1024 --ctx 4096 --users 1 2 4 8 16 32 --gen 256 $P --variants gqa:0,gqa:8,gqa:4 --tag $T --out $O
bench/run.sh $M --mode shared --suffix 1024 --ctx 32768 --users 1 2 4 8 16 32 --gen 256 $P --variants gqa:0,gqa:8,gqa:4,stock:4 --tag $T --out $O
bench/run.sh $M --mode shared --suffix 1024 --ctx 32768 --users 32 --gen 256 $P --variants stock:8 --tag $T --out $O
bench/run.sh $M --mode shared --suffix 1024 --ctx 131072 --users 1 2 4 8 16 32 --gen 256 $P --variants gqa:0,gqa:8,gqa:4 --tag $T --out $O
bench/run.sh $M --mode shared --suffix 1024 --ctx 131072 --users 16 32 --gen 256 $P --variants stock:4 --tag $T --out $O
PY=(env PYTHON_GIL=0 ${MLX_PY:-python3} -I)
$PY bench/validate.py $M --test agreement --ctx 32768 --gen 128 --out results/results_validate.jsonl 2>&1 | grep -v Warning
$PY bench/validate.py $M --test agreement --ctx 32768 --gen 128 --reference ref32 --variants stock:0,gqa:0,gqa:8,stock:8 --out results/results_validate.jsonl 2>&1 | grep -v Warning
$PY bench/validate.py $M --test needle --ctx 131072 --depths 0.1 0.5 0.9 --variants gqa:0,gqa:8,gqa:4,stock:8 --out results/results_validate.jsonl 2>&1 | grep -v Warning
for v in "--gqa-kernel" "--gqa-kernel --kv-bits 4"; do
  bench/run.sh $M --ctx 131072 --users 1 --gen 2048 --trials 5 $P ${=v} --tag gate --out results/results_gate.jsonl
done
echo MLX_DONE
bash bench/llamacpp_concurrency.sh
