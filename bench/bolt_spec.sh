#!/bin/bash
# Bolt + DFlash (LiLiCorr) draft-length sweep on the optimized llama.cpp-bolt build (native NVFP4 kernels).
cd "$(dirname "$0")/.."
M=${M:-models/bolt/Bolt-74B-A9B-24GB-folded.gguf}
D=${D:-models/bolt-lilicorr/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf}
B=llama.cpp-bolt/build/bin
OUT=${OUT:-results/results_bolt_serving.jsonl}
serve() {  # $1 = tag, rest = extra server args
  tag=$1; shift
  $B/llama-server -m $M --port 8090 --host 127.0.0.1 -ngl 999 --batch-size 2048 --ubatch-size 512 --ctx-size 65536 \
    --parallel 1 --flash-attn on --jinja --cache-type-k q8_0 --cache-type-v q8_0 "$@" > logs/bolt_server_$tag.log 2>&1 &
  pid=$!
  until curl -sf 127.0.0.1:8090/health >/dev/null; do sleep 2; kill -0 $pid 2>/dev/null || { echo "server $tag died"; return; }; done
  for pw in ${PW:-2000}; do for u in ${USERS:-1}; do
    python3 loadtest.py --url http://127.0.0.1:8090 --users $u --prompt-words $pw --gen 256 --tag $tag --out $OUT
  done; done
  kill $pid; wait $pid 2>/dev/null
}
serve opt-nospec-q8kv-par1
for n in ${NMAX:-2 3 4 5 6 8}; do
  serve opt-dflash$n-q8kv-par1 --spec-type draft-dflash --spec-draft-n-max $n --spec-draft-model $D --spec-draft-ngl 999
done
echo SPEC_DONE
