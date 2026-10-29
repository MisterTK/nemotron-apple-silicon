#!/bin/bash
# Check builds, models and Python, and measure GPU bandwidth. Takes about a minute.
. "$(dirname "$0")/common.sh"
echo "== host"; sysctl -n machdep.cpu.brand_string hw.memsize; sw_vers -productVersion
echo "== wired limit (MB, 0 = default budget)"; sysctl -n iogpu.wired_limit_mb
echo "== builds"
for b in $FORK $MASTER llama.cpp/build/bin llama.cpp-upstream-base/build/bin; do
  printf '%-40s ' "$b"; ls "$b" 2>/dev/null | grep -E '^(llama-bench|llama-batched-bench|llama-server|llama-perplexity|test-backend-ops)$' | tr '\n' ' '; echo
done
echo "== models"
for m in $BOLT $BOLT_DRAFT $GEMMA $QWEN $NEMO $MLX_NEMO; do printf '%-70s %s\n' "$m" "$([ -e "$m" ] && du -shL "$m" | cut -f1 || echo MISSING)"; done
echo "== python"
"$MLX_PY" -c 'import mlx.core as mx, mlx_lm; print("mlx", mx.__version__, "mlx-lm", mlx_lm.__version__)' 2>&1 | tail -1
echo "== GPU bandwidth"; PYTHON_GIL=0 "$MLX_PY" -I bench/bw.py
