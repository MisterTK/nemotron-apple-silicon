#!/bin/bash
# llama.cpp concurrency sweep (patched llama-batched-bench, shared prompt, f16 KV) for all three 30B-class models.
# Points whose total context cannot fit the ~40 GB Metal budget are skipped via --max-tokens.
cd "$(dirname "$0")/.."
export BENCH=llama.cpp/build/bin/llama-batched-bench
OUT=results/results_llamacpp_concurrency.jsonl
U=1,2,4,8,16,32
python3 sweep.py models/gemma-4-26B-A4B-it-Q4_0.gguf --depths 4096,32768,131072 --users $U --ntg 64 --max-tokens 1100000 --tag gemma-f16 --out $OUT
python3 sweep.py models/gemma-4-26B-A4B-it-Q4_0.gguf --depths 131072 --users $U --ntg 64 --max-tokens 2200000 --extra "-ctk q8_0 -ctv q8_0" --tag gemma-q8 --out $OUT
python3 sweep.py models/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf --depths 4096,32768,131072 --users $U --ntg 64 --max-tokens 2200000 --tag nemotron-f16 --out $OUT
python3 sweep.py models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --depths 4096,32768 --users $U --ntg 64 --max-tokens 200000 --tag qwen-f16 --out $OUT
echo ALL_DONE
