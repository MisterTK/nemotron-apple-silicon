#!/bin/bash
# Deployment views: 1-user context scaling, agents per user, capacity. Probes the largest sequence count that fits.
# Default grid is the short version (~1 h for all three models). Full grid: GRID=full. Raise the wired limit first.
. "$(dirname "$0")/common.sh"
if [ "${GRID:-short}" = full ]; then
  G="8192:1,4,8,16,32,64,128;32768:1,4,8,16,32,64;131072:1,2,4,8,16,32;260000:1,2,4,8"
else
  G="8192:1,8,32;32768:1,8;131072:1,4"
fi
for spec in "$BOLT $FORK bolt" "$GEMMA $MASTER gemma" "$QWEN $MASTER qwen"; do
  set -- $spec; need "$1" "$2/llama-batched-bench"
  python3 bench/scenarios.py "$1" --bin "$2/llama-batched-bench" --tag "$3" --kv "${KV:-f16}" --grid "$G" --out results/results_scenarios.jsonl
done
echo "-> results/results_scenarios.jsonl"
