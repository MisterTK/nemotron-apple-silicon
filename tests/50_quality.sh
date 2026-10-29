#!/bin/bash
# Correctness: fork kernels vs CPU, Bolt perplexity, MLX kernel logit agreement and 128K needle retrieval. ~45 min.
. "$(dirname "$0")/common.sh"
echo "== test-backend-ops (fork, Metal vs CPU)"
for op in MUL_MAT MUL_MAT_ID SSM_SCAN SSM_CONV SSM_SCAN_CACHE_FUSION SSM_CONV_CAT_FUSION TOPK_MOE GET_ROWS; do
  printf '%-24s ' $op; $FORK/test-backend-ops test -b MTL0 -o $op 2>&1 | grep -E '^\s*[0-9]+/[0-9]+ tests passed' | tail -1
done
echo "== Bolt perplexity (16 x 2048 tokens of stdlib source; reference 1.999 +- 0.027)"
python3 - <<'PY'
import glob, os
lib = os.path.dirname(os.__file__); out = open("logs/ppl_fixture.txt", "w"); n = 0
for f in sorted(glob.glob(os.path.join(lib, "*.py"))):
    try: t = open(f, encoding="utf-8").read()
    except Exception: continue
    out.write(t); n += len(t)
    if n > 600_000: break
PY
need "$BOLT"; $FORK/llama-perplexity -m "$BOLT" -f logs/ppl_fixture.txt -c 2048 --chunks 16 -fa 1 -ngl 99 2>&1 | grep -E 'Final estimate'
if [ -e "$MLX_NEMO" ]; then
  echo "== MLX kernel vs stock vs fp32 attention (32K, 128 teacher-forced tokens)"
  PYTHON_GIL=0 "$MLX_PY" -I bench/validate.py "$MLX_NEMO" --test agreement --ctx 32768 --gen 128 --reference ref32 --variants stock:0,gqa:0,gqa:8 --out results/results_validate.jsonl 2>&1 | grep -v Warning
  echo "== needle at 131K"
  PYTHON_GIL=0 "$MLX_PY" -I bench/validate.py "$MLX_NEMO" --test needle --ctx 131072 --variants gqa:0,gqa:4 --out results/results_validate.jsonl 2>&1 | grep -v Warning
fi
