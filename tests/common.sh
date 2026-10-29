# Shared defaults for tests/*.sh. Override any variable in the environment.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results logs
FORK=${FORK:-llama.cpp-bolt/build/bin}                 # Bolt fork (NVFP4, SSM fusions, split-K)
MASTER=${MASTER:-llama.cpp-upstream-splitk/build/bin}  # master 2e0d86beb + split-K
BOLT=${BOLT:-models/bolt/Bolt-74B-A9B-24GB-folded.gguf}
BOLT_DRAFT=${BOLT_DRAFT:-models/bolt-lilicorr/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf}
GEMMA=${GEMMA:-models/gemma-4-26B-A4B-it-Q4_0.gguf}
QWEN=${QWEN:-models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}
NEMO=${NEMO:-models/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf}
MLX_NEMO=${MLX_NEMO:-models/mlx-nemotron30b-4bit}
# Python with mlx-lm: $MLX_PY, else python3, else the uv tool environment "mlx-lm".
if [ -z "$MLX_PY" ]; then
  if python3 -c 'import mlx' 2>/dev/null; then MLX_PY=python3
  elif [ -x "$(uv tool dir 2>/dev/null)/mlx-lm/bin/python" ]; then MLX_PY="$(uv tool dir)/mlx-lm/bin/python"
  else MLX_PY=python3; fi
fi
export MLX_PY
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }; done; }
