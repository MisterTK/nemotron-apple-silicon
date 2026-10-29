#!/bin/bash
# Optimized Bolt (native NVFP4 + SSM/routing kernels, scales folded): depth sweep and concurrency sweep.
cd "$(dirname "$0")/.."
M=models/bolt/Bolt-74B-A9B-24GB-folded.gguf
B=llama.cpp-bolt/build/bin
for kv in f16 q8_0; do
  $B/llama-bench -m $M -fa 1 -ngl 99 -p 2048 -n 128 -d 0,32768,131072 -r 2 -ctk $kv -ctv $kv -o jsonl >> results/results_llamabench_bolt_opt.jsonl 2>> logs/bolt_bench.err
done
echo BENCH_DONE
BENCH=$B/llama-batched-bench python3 sweep.py $M --depths 4096,32768 --users 1,2,4,8,16,32 --ntg 64 --max-tokens 2200000 --tag bolt-opt-f16 --out results/results_llamacpp_concurrency.jsonl
BENCH=$B/llama-batched-bench python3 sweep.py $M --depths 131072 --users 1,4,8,16 --ntg 64 --max-tokens 4300000 --extra "-ctk q8_0 -ctv q8_0" --tag bolt-opt-q8 --out results/results_llamacpp_concurrency.jsonl
echo ALL_DONE
