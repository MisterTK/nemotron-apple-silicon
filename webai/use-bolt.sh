#!/bin/sh
# Run NVIDIA's Bolt-74B GGUF inside the webAI desktop app on a patched llama-server.
#
# webAI (com.apogee.shell) spawns the Homebrew llama-server (llama.cpp 0.7.0, commit 620d2225c)
# and resolves models as "<hf-repo>:<quant>" into ~/Library/Caches/webai/gguf-cache-v1.
# This script
#   1. registers an already-downloaded Bolt GGUF in that cache (hard link + origin.json),
#   2. points webAI at a patched llama-server through the APOGEE_LLAMA_SERVER environment variable,
#   3. sets the app's selected model to Bolt, and
#   4. relaunches webAI.
# Observed on webAI 2.5.2 nightly (2026-10-29). Undo with `webai/use-bolt.sh --undo`.
#
# usage: webai/use-bolt.sh [--undo] [path/to/llama-server] [path/to/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf]
set -eu

SERVER=${2:-$(dirname "$0")/../llama.cpp-bolt/build/bin/llama-server}
SRC=${3:-$(dirname "$0")/../models/bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf}
REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
FILE=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
MODEL_ID="$REPO:24GB"            # the quant label must be a suffix of the file name; "24GB" is.
GEMMA_ID="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0"
CACHE="$HOME/Library/Caches/webai/gguf-cache-v1"
LS=$(ls "$HOME"/Library/WebKit/com.apogee.shell/WebsiteData/Default/*/*/LocalStorage/localstorage.sqlite3 2>/dev/null | head -1)

hash16() { printf '%s' "https://huggingface.co/$REPO/resolve/main/$1" | shasum -a 256 | cut -c1-16; }
DST="$CACHE/$(printf '%s' "$REPO" | tr / _)__${FILE}__$(hash16 "$FILE").gguf"

set_pref() {   # set_pref <model-id>; the app must not be running
    [ -n "$LS" ] || { echo "webAI localStorage not found; start the app once first" >&2; exit 1; }
    sqlite3 "$LS" "PRAGMA wal_checkpoint(TRUNCATE);" >/dev/null
    python3 - "$LS" "$1" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); v = sys.argv[2].encode('utf-16-le')
for k in ('oasis_model_base_pref_v1', 'oasis_model_dropdown_v1'):
    db.execute('update ItemTable set value=? where key=?', (v, k))
db.commit()
PY
}

quit_app() {
    osascript -e 'quit app "webAI"' >/dev/null 2>&1 || true
    i=0; while pgrep -x webAI >/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i+1)); done
    pkill -f 'llama-server --model' 2>/dev/null || true
}

if [ "${1:-}" = "--undo" ]; then
    quit_app
    launchctl unsetenv APOGEE_LLAMA_SERVER
    set_pref "$GEMMA_ID"
    rm -f "$DST" "$DST.origin.json" "$DST.last_used"
    open -a webAI
    echo "restored: Homebrew llama-server, $GEMMA_ID, Bolt removed from the cache"
    exit 0
fi

[ -x "$SERVER" ] || { echo "llama-server not found at $SERVER (build it with patches/apply.sh fork)" >&2; exit 1; }
[ -f "$SRC" ] || { echo "Bolt GGUF not found at $SRC" >&2; exit 1; }

mkdir -p "$CACHE"
[ -e "$DST" ] || ln "$SRC" "$DST" 2>/dev/null || cp "$SRC" "$DST"
printf '{\n  "repo": "%s",\n  "file_name": "%s"\n}\n' "$REPO" "$FILE" > "$DST.origin.json"
touch "$DST.last_used"

quit_app
SERVER_ABS=$(cd "$(dirname "$SERVER")" && pwd)/$(basename "$SERVER")
launchctl setenv APOGEE_LLAMA_SERVER "$SERVER_ABS"   # per login session; re-run after a reboot
set_pref "$MODEL_ID"
open -a webAI
echo "webAI relaunched with $MODEL_ID on $SERVER_ABS"
echo "check: ps -o command= -p \$(pgrep -f 'llama-server --model')"
