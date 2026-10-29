#!/bin/bash
# llama-server load test for Bolt: no speculation vs DFlash (n_max 8, p_min 0.5). ~10 min.
. "$(dirname "$0")/common.sh"; need "$BOLT" "$BOLT_DRAFT" "$FORK/llama-server"
OUT=results/results_serving.jsonl
serve() {
  tag=$1; shift
  $FORK/llama-server -m "$BOLT" --port 8090 --host 127.0.0.1 -ngl 999 --batch-size 2048 --ubatch-size 512 --ctx-size 65536 \
    --parallel 1 --flash-attn on --jinja --cache-type-k q8_0 --cache-type-v q8_0 "$@" > logs/server_$tag.log 2>&1 &
  pid=$!
  until curl -sf 127.0.0.1:8090/health >/dev/null; do sleep 2; kill -0 $pid 2>/dev/null || { echo "server $tag died, see logs/server_$tag.log"; return; }; done
  for u in ${USERS:-1 4}; do python3 loadtest.py --url http://127.0.0.1:8090 --users $u --prompt-words ${PROMPT_WORDS:-2000} --gen 256 --tag $tag --out $OUT; done
  kill $pid; wait $pid 2>/dev/null
}
serve nospec
serve dflash8 --spec-type draft-dflash --spec-draft-n-max 8 --spec-draft-p-min 0.5 --spec-draft-model "$BOLT_DRAFT" --spec-draft-ngl 999
echo "-> $OUT"
