#!/bin/sh
# The 1-slot 128K scenario set against the app's server, run from anywhere:
# short prompts (loadtest.py, 2,000 and 22,000 words), natural-text prompts of 8K, 32K and 100K tokens
# (long_prompt.py), and the growing agent session to ~126K tokens (session.py).
#
# usage: webai/scenarios_128k.sh TAG OUT.jsonl [SESSION_TURNS]   [WEBAI_URL=http://127.0.0.1:11436]
set -eu
cd "$(dirname "$0")/.."
TAG=${1:?usage: scenarios_128k.sh TAG OUT.jsonl [SESSION_TURNS]}; OUT=${2:?}; TURNS=${3:-28}
URL=${WEBAI_URL:-http://127.0.0.1:11436}

curl -sf "$URL/health" >/dev/null || { echo "no server at $URL" >&2; exit 1; }
curl -s "$URL/props" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("server:", d["model_path"].split("/")[-1], "slots", d["total_slots"], "n_ctx", d["default_generation_settings"]["n_ctx"])'

for w in 2000 22000; do
    python3 loadtest.py --url "$URL" --users 1 --prompt-words $w --gen 256 --tag "$TAG" --out "$OUT"
done
for n in 8000 32000 100000; do
    python3 webai/long_prompt.py $n "$TAG-natural" "$OUT"
done
python3 webai/session.py "$TURNS" "$TAG-session" "$OUT"
