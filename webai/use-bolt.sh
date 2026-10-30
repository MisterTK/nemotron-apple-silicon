#!/bin/sh
# Run NVIDIA's Bolt-74B GGUF inside the webAI desktop app on a patched llama-server.
#
# webAI (com.apogee.shell) spawns the Homebrew llama-server (llama.cpp 0.7.0, commit 620d2225c)
# and resolves models as "<hf-repo>:<quant>" into ~/Library/Caches/webai/gguf-cache-v1.
# This script
#   1. registers an already-downloaded Bolt GGUF in that cache (hard link + origin.json),
#   2. points webAI at webai/llama-server-wrapper.sh through the APOGEE_LLAMA_SERVER environment
#      variable; the wrapper execs the patched llama-server with the context and slot count
#      below instead of webAI's fixed 30720 tokens over 3 slots,
#   3. sets the app's selected model to Bolt, and
#   4. relaunches webAI.
# Observed on webAI 2.5.2 nightly (2026-10-29). Undo with `webai/use-bolt.sh --undo`.
#
# Any other GGUF works the same way: --model PATH --repo OWNER/REPO [--quant LABEL] [--draft MTP.gguf],
# for example ggml-org's Nemotron-3.5-Lightning-30B-A3B Q4_0 with its MTP draft.
#
# usage: webai/use-bolt.sh [--undo] [--ctx TOKENS_PER_SLOT] [--slots N] [--server PATH]
#                          [--model PATH] [--repo OWNER/REPO] [--quant LABEL] [--draft PATH]
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SERVER=$HERE/../llama.cpp-bolt/build/bin/llama-server
SRC=$HERE/../models/bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
QUANT=           # webAI matches "<repo>:<quant>" against the file name suffix; default: the token after the last "-"
DRAFT=           # MTP draft GGUF, passed to the server as --spec-type draft-mtp --model-draft
CTX=131072       # per slot; q8_0 KV costs 0.5 GB per 128K slot on Bolt, the recurrent state 165 MB
SLOTS=3          # webAI's own count; its parallel features expect 3
UNDO=
while [ $# -gt 0 ]; do
    case $1 in
        --undo) UNDO=1 ;;
        --ctx) CTX=$2; shift ;;
        --slots) SLOTS=$2; shift ;;
        --server) SERVER=$2; shift ;;
        --model) SRC=$2; shift ;;
        --repo) REPO=$2; shift ;;
        --quant) QUANT=$2; shift ;;
        --draft) DRAFT=$2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
FILE=$(basename "$SRC")
[ -n "$QUANT" ] || QUANT=$(printf '%s' "${FILE%.gguf}" | sed 's/.*-//')
MODEL_ID="$REPO:$QUANT"
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

if [ -n "$UNDO" ]; then
    quit_app
    for v in APOGEE_LLAMA_SERVER WEBAI_LLAMA_SERVER WEBAI_CTX_PER_SLOT WEBAI_SLOTS WEBAI_EXTRA_ARGS; do launchctl unsetenv $v; done
    set_pref "$GEMMA_ID"
    rm -f "$DST" "$DST.origin.json" "$DST.last_used"
    open -a webAI
    echo "restored: Homebrew llama-server, $GEMMA_ID, $FILE removed from the cache"
    exit 0
fi

[ -x "$SERVER" ] || { echo "llama-server not found at $SERVER (build it with patches/apply.sh fork)" >&2; exit 1; }
[ -f "$SRC" ] || { echo "GGUF not found at $SRC" >&2; exit 1; }
[ -z "$DRAFT" ] || [ -f "$DRAFT" ] || { echo "draft GGUF not found at $DRAFT" >&2; exit 1; }

mkdir -p "$CACHE"
[ -e "$DST" ] || ln "$SRC" "$DST" 2>/dev/null || cp "$SRC" "$DST"
printf '{\n  "repo": "%s",\n  "file_name": "%s"\n}\n' "$REPO" "$FILE" > "$DST.origin.json"
touch "$DST.last_used"

quit_app
SERVER_ABS=$(cd "$(dirname "$SERVER")" && pwd)/$(basename "$SERVER")
# launchctl setenv is per login session; re-run this script after a reboot
launchctl setenv APOGEE_LLAMA_SERVER "$HERE/llama-server-wrapper.sh"
launchctl setenv WEBAI_LLAMA_SERVER "$SERVER_ABS"
launchctl setenv WEBAI_CTX_PER_SLOT "$CTX"
launchctl setenv WEBAI_SLOTS "$SLOTS"
if [ -n "$DRAFT" ]; then
    launchctl setenv WEBAI_EXTRA_ARGS "--spec-type draft-mtp --model-draft $(cd "$(dirname "$DRAFT")" && pwd)/$(basename "$DRAFT")"
else
    launchctl unsetenv WEBAI_EXTRA_ARGS
fi
set_pref "$MODEL_ID"
open -a webAI
echo "webAI relaunched with $MODEL_ID on $SERVER_ABS, $SLOTS slots x $CTX tokens"
echo "check: ps -o command= -p \$(pgrep -f 'llama-server --model')"
