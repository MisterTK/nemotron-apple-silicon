#!/bin/bash
# Aggregate decode vs concurrent sequences (patched llama-batched-bench, shared prompt). ~20 min per model.
# usage: tests/20_concurrency.sh [model.gguf] [bin dir]
. "$(dirname "$0")/common.sh"
M=${1:-$BOLT}; B=${2:-$FORK}; need "$M" "$B/llama-batched-bench"
TAG=${TAG:-$(basename "$M" .gguf)}
OUT=results/results_concurrency.jsonl
BENCH=$B/llama-batched-bench python3 sweep.py "$M" --depths "${DEPTHS:-4096,32768}" --users "${USERS:-1,2,4,8,16,32}" --ntg 64 --max-tokens "${MAX_TOKENS:-2200000}" ${KV:+--extra "-ctk $KV -ctv $KV"} --tag "$TAG" --out $OUT
echo "-> $OUT"
