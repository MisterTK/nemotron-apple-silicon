#!/bin/bash
# Full context x users matrix (bench/scenarios.py), optimized then stock builds.
cd "$(dirname "$0")/.."
S="python3 bench/scenarios.py"
BOLT=models/bolt/Bolt-74B-A9B-24GB-folded.gguf
BOLT_SHIP=models/bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
GEMMA=models/gemma-4-26B-A4B-it-Q4_0.gguf
QWEN=models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
FORK=llama.cpp-bolt/build/bin/llama-batched-bench
SPLITK=llama.cpp-upstream-splitk/build/bin/llama-batched-bench
MASTER=llama.cpp-upstream-base/build/bin/llama-batched-bench
G_F16="8192:1,4,8,16,32,64,128;32768:1,4,8,16,32,64;131072:1,2,4,8,16,32;260000:1,2,4,8"
G_Q8="131072:1,4,8,16,32,64;260000:1,2,4,8,16"
G_STOCK="8192:1,8,16,32,64;32768:1,16,32;131072:1,4"
for spec in "$BOLT $FORK bolt-opt" "$GEMMA $SPLITK gemma-opt" "$QWEN $SPLITK qwen-opt"; do
  set -- $spec
  $S $1 --bin $2 --tag $3 --kv f16 --grid "$G_F16"
  $S $1 --bin $2 --tag $3 --kv q8_0 --grid "$G_Q8"
done
$S $GEMMA --bin $MASTER --tag gemma-stock --kv f16 --grid "$G_STOCK"
$S $QWEN --bin $MASTER --tag qwen-stock --kv f16 --grid "$G_STOCK"
$S $BOLT_SHIP --bin $MASTER --tag bolt-stock --kv f16 --grid "8192:1,8,32;32768:1;131072:1"
echo ALL_DONE
