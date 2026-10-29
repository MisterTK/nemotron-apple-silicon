#!/bin/bash
# usage: bench/run.sh <mlx_bench args...>   (any cwd); prints compact lines.
# MLX_PY: python with mlx-lm installed (default: python3). PYTHON_GIL=0 is only needed for a free-threaded build.
cd "$(dirname "$0")/.."
PY=${MLX_PY:-python3}; [ "$PY" = python3 ] && ! python3 -c "import mlx" 2>/dev/null && [ -x "$(uv tool dir 2>/dev/null)/mlx-lm/bin/python" ] && PY="$(uv tool dir)/mlx-lm/bin/python"
PYTHON_GIL=0 "$PY" -I bench/mlx_bench.py "$@" 2>&1 | python3 bench/fmt.py
