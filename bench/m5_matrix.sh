#!/bin/zsh
# M5 Pro (24 GB) MLX matrix: one process per (model, ctx, users) point so an out-of-memory point is recorded and skipped;
# for each ctx, stop raising users after the first failure.
cd "$(dirname "$0")/.."
O=results/results_m5_mlx_matrix.jsonl; L=logs/m5_matrix.log
for m in nemotron30b qwen30b gemma26b; do
  for ctx in 4096 32768 131072; do
    for u in 1 2 4 8 16 32; do
      echo "== $m ctx=$ctx users=$u $(date +%T)" | tee -a $L
      bench/run.sh models/mlx-$m-4bit --mode shared --suffix 1024 --ctx $ctx --users $u --gen 256 \
        --variants stock:0,gqa:0 --tag m5-matrix-$m --out $O 2>&1 | tee -a $L
      if [[ ${pipestatus[1]} -ne 0 ]] || tail -5 $L | grep -qiE "error|Traceback|killed|out of memory"; then
        echo "FAILED $m ctx=$ctx users=$u" | tee -a $L; break
      fi
    done
  done
done
echo ALL_DONE | tee -a $L
