#!/usr/bin/env bash
# Check out llama.cpp at a pinned commit, apply one patch set, build.
#
#   patches/apply.sh fork      [dir]   # 620d2225c + fork-llama.cpp-bolt-metal-opt.patch   (fastest Bolt; fork-only types)
#   patches/apply.sh upstream  [dir]   # 418f905a4 + the four upstream branches (clean master plus NVFP4, SSM fusions,
#                                      #   sigmoid routing fusion, batched-bench -pps fix)
#   patches/apply.sh master    [dir]   # 418f905a4 + batched-bench fix only (baseline for A/B runs)
#
# dir defaults to llama.cpp-<set>. Set LLAMA_REPO to a local clone to speed up the clone.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
set_name=${1:?usage: apply.sh fork|upstream|master [dir]}
dir=${2:-llama.cpp-$set_name}

case $set_name in
  fork)     commit=620d2225c; patches=(fork-llama.cpp-bolt-metal-opt.patch) ;;
  upstream) commit=418f905a4; patches=(upstream-batched-bench-pps-recurrent.patch upstream-metal-topk-moe-sigmoid.patch
                                       upstream-metal-nvfp4.patch upstream-metal-ssm-fusions.patch) ;;
  master)   commit=418f905a4; patches=(upstream-batched-bench-pps-recurrent.patch) ;;
  *) echo "unknown set: $set_name" >&2; exit 1 ;;
esac

if [ ! -d "$dir/.git" ]; then
  git clone ${LLAMA_REPO:+--reference "$LLAMA_REPO"} https://github.com/ggml-org/llama.cpp "$dir"
fi
git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
git -C "$dir" checkout -q --detach "$commit"
git -C "$dir" reset -q --hard
for p in "${patches[@]}"; do
  echo "applying $p"
  git -C "$dir" apply --3way "$here/$p"
done

sdk=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
cmake -S "$dir" -B "$dir/build" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_OSX_SYSROOT:+-DCMAKE_OSX_SYSROOT="$CMAKE_OSX_SYSROOT"} \
  $([ -d "$sdk" ] && [ -z "${CMAKE_OSX_SYSROOT:-}" ] && echo "-DCMAKE_OSX_SYSROOT=$sdk")
cmake --build "$dir/build" -j --target llama-bench llama-batched-bench llama-server llama-perplexity llama-cli test-backend-ops
echo "done: $dir/build/bin"
