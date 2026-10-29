#!/bin/bash
# Nemotron-3.5-Bolt-74B-A9B-24GB (official IQ2_XXS GGUF) on llama.cpp: same tests as the 30B models.
cd "$(dirname "$0")/.."
# Metal has no NVFP4 matmul, so the NVFP4 Mamba projections (and drafter weights) are re-encoded
# as Q8_0 by bench/gguf_retype.py; everything else is byte-identical to NVIDIA's file.
M=models/bolt/Bolt-74B-A9B-24GB-mambaQ8.gguf
D=models/bolt-lilicorr/LiLiCorr-Q8.gguf
# 620d2225c + 704366d76 (Metal MoE kernel for 9 experts/token) + the batched-bench patch
B=llama.cpp-bolt/build/bin
# 1. single-user depth sweep
for kv in f16 q8_0; do
  $B/llama-bench -m $M -fa 1 -ngl 99 -p 2048 -n 128 -d 0,32768,131072 -r 2 -ctk $kv -ctv $kv -o jsonl >> results/results_llamabench_bolt.jsonl 2>> logs/bolt_bench.err
done
echo BENCH_DONE
# 2. concurrency sweep (patched batched-bench, shared prompt)
BENCH=$B/llama-batched-bench python3 sweep.py $M --depths 4096,32768,131072 --users 1,2,4,8,16,32 --ntg 64 --max-tokens 2200000 --tag bolt-f16 --out results/results_llamacpp_concurrency.jsonl
BENCH=$B/llama-batched-bench python3 sweep.py $M --depths 131072 --users 8,16,32 --ntg 64 --max-tokens 4300000 --extra "-ctk q8_0 -ctv q8_0" --tag bolt-q8 --out results/results_llamacpp_concurrency.jsonl
echo SWEEP_DONE
# 3. serving: NVIDIA's recipe (DFlash drafter, q8_0 KV, 1 slot) vs no speculation
serve() {  # $1 = tag, rest = extra server args
  tag=$1; shift
  $B/llama-server -m $M --port 8090 --host 127.0.0.1 -ngl 999 --batch-size 2048 --ubatch-size 512 --ctx-size 65536 \
    --parallel 1 --flash-attn on --jinja --cache-type-k q8_0 --cache-type-v q8_0 "$@" > logs/bolt_server_$tag.log 2>&1 &
  pid=$!
  until curl -sf 127.0.0.1:8090/health >/dev/null; do sleep 2; kill -0 $pid 2>/dev/null || { echo "server $tag died"; return; }; done
  for pw in 2000 22000; do for u in 1 2 4; do
    python3 loadtest.py --url http://127.0.0.1:8090 --users $u --prompt-words $pw --gen 256 --tag $tag --out results/results_bolt_serving.jsonl
  done; done
  kill $pid; wait $pid 2>/dev/null
}
serve bolt-dflash6-q8kv-par1 --spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-model $D --spec-draft-ngl 999
serve bolt-nospec-q8kv-par1
echo ALL_DONE
