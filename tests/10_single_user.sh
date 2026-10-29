#!/bin/bash
# Single-user decode and prefill vs context depth (llama-bench). ~15 min per model with the default depths.
# usage: tests/10_single_user.sh [model.gguf] [bin dir]   e.g. tests/10_single_user.sh $GEMMA $MASTER
. "$(dirname "$0")/common.sh"
M=${1:-$BOLT}; B=${2:-$FORK}; need "$M" "$B/llama-bench"
DEPTHS=${DEPTHS:-0,32768,131072}; KV=${KV:-f16}
OUT=results/results_single_user.jsonl
$B/llama-bench -m "$M" -fa 1 -ngl 99 -p 2048 -n 128 -d "$DEPTHS" -r 2 -ctk $KV -ctv $KV -o jsonl 2>>logs/single_user.err | tee -a $OUT | python3 -c '
import sys, json
for l in sys.stdin:
    r = json.loads(l); print(f'"'"'{r["model_filename"].split("/")[-1]:45s} depth {r["n_depth"]:>7} {r["n_prompt"] and "pp" or "tg"} {r["avg_ts"]:7.1f} tok/s'"'"')'
echo "-> $OUT"
