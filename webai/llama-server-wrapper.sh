#!/bin/sh
# Stand-in for llama-server that webAI launches through APOGEE_LLAMA_SERVER.
# webAI fixes --ctx-size and --parallel from its own device-tier table (Bolt gets
# 30720 tokens over 3 slots, 10240 per request, and its load-model API refuses
# anything larger). This script rewrites those two arguments and execs the real
# server. Everything else, including --version, passes through unchanged.
#
#   WEBAI_LLAMA_SERVER   real binary (default: ../llama.cpp-bolt/build/bin/llama-server)
#   WEBAI_CTX_PER_SLOT   context per slot in tokens (default: keep webAI's total)
#   WEBAI_SLOTS          number of slots (default: keep webAI's value)
#
# webai/use-bolt.sh sets these with `launchctl setenv`, which is what the app reads.
here=$(cd "$(dirname "$0")" && pwd)
real=${WEBAI_LLAMA_SERVER:-$here/../llama.cpp-bolt/build/bin/llama-server}

args=""; ctx=""; par=""; prev=""
for a in "$@"; do
    case $prev in
        --ctx-size) ctx=$a; prev=""; continue ;;
        --parallel) par=$a; prev=""; continue ;;
    esac
    case $a in
        --ctx-size|--parallel) prev=$a; continue ;;
    esac
    args="$args '$(printf '%s' "$a" | sed "s/'/'\\\\''/g")'"
done

slots=${WEBAI_SLOTS:-$par}
if [ -n "$WEBAI_CTX_PER_SLOT" ] && [ -n "$slots" ]; then
    total=$((WEBAI_CTX_PER_SLOT * slots))
else
    total=$ctx
fi
[ -n "$total" ] && args="$args --ctx-size $total"
[ -n "$slots" ] && args="$args --parallel $slots"
eval exec "'$real'" $args
