#!/bin/bash
# Bolt-74B-A9B on the second optimization pass (scan/conv state written straight into the recurrent cache)
cd "$(dirname "$0")/.."
M=models/bolt/Bolt-74B-A9B-24GB-folded.gguf
R=models/bolt/Bolt-74B-A9B-24GB-folded-rf16.gguf   # + f16 router (bench/gguf_retype.py --src F32 --dst F16 --match 'ffn_gate_inp\.weight$')
B=llama.cpp-bolt/build/bin
# 1. single-user depth sweep
for kv in f16 q8_0; do
  $B/llama-bench -m $M -fa 1 -ngl 99 -p 2048 -n 128 -d 0,32768,131072 -r 2 -ctk $kv -ctv $kv -o jsonl >> results/results_llamabench_bolt_opt2.jsonl 2>> logs/bolt_bench.err
done
echo BENCH_DONE
# 2. concurrency sweep (patched batched-bench, shared prompt)
BENCH=$B/llama-batched-bench python3 sweep.py $M --depths 4096,32768 --users 1,2,4,8,16,32 --ntg 64 --max-tokens 2200000 --tag bolt-opt2-f16 --out results/results_llamacpp_concurrency.jsonl
BENCH=$B/llama-batched-bench python3 sweep.py $M --depths 131072 --users 1,4,8,16,32 --ntg 64 --max-tokens 4300000 --extra "-ctk q8_0 -ctv q8_0" --tag bolt-opt2-q8 --out results/results_llamacpp_concurrency.jsonl
echo SWEEP_DONE
# 3. f16 router (run last: the two 22 GB files cannot both stay in the page cache)
for i in 1 2; do $B/llama-bench -m $R -fa 1 -ngl 99 -p 2048 -n 128 -r 3 -o jsonl >> results/results_llamabench_bolt_opt2_rf16.jsonl 2>> logs/bolt_bench.err; done
echo ALL_DONE
