# nemotron-apple-silicon

Decode benchmarks and Metal kernel work for hybrid Mamba-2 MoE models (Nemotron-3-Nano-30B, Nemotron-3.5-Bolt-74B) against attention MoE models (Qwen3-30B-A3B, Gemma 4 26B-A4B) on Apple Silicon, with MLX and llama.cpp.

**Read first:** [REPORT.md](REPORT.md) has the results, the optimizations, the recommendations and the open items.

## Layout

| path | contents |
|---|---|
| `REPORT.md` | findings, numbers, recommendations |
| `tests/` | the test set; `tests/run_all.sh` runs everything, each script also runs alone |
| `bench/` | harness: MLX benchmark and GQA decode-attention kernel, llama.cpp sweeps, GGUF re-encoders, validation |
| `results/` | every measurement as JSON lines, one record per point; `bench/report_tables.py` renders the MLX matrices |
| `logs/` | raw benchmark and server logs from the recorded runs |
| `patches/` | llama.cpp diffs: `fork-*` the full Bolt fork diff, `upstream-*` five upstream-ready branches each with a notes file, `tool-*` the batched-bench prefill-reuse patch and the op-skip profiler, `experiment-*` measured changes not kept; `UPSTREAM-INDEX.md` indexes them |
| `pr/`, `llama.cpp*` | git worktrees (not included in a copy of this repo; see Rebuild) |
| `models/` | model files (not included; see Models) |
| `ref/` | MLX's stock SDPA source, kept for comparison |
| `sweep.py`, `loadtest.py` | llama-batched-bench driver and llama-server load test |
| `webai/` | `use-bolt.sh` runs Bolt inside the webAI desktop app on the patched server |

## Using the patches

To run Bolt-74B (or any Mamba-2 hybrid) faster on Apple Silicon without reading further:

```sh
patches/apply.sh upstream        # llama.cpp master 418f905a4 + the four upstream-ready patches -> llama.cpp-upstream/build/bin
patches/apply.sh fork            # llama.cpp 620d2225c + the full Bolt fork (fastest, uses fork-only formats) -> llama.cpp-fork/build/bin
```

Both sets add Metal NVFP4 support, so NVIDIA's `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` runs on the GPU as is. Decode
on an M3 Max, one user, that file: stock master 9.7 tok/s, upstream set 43.9 (measured 2026-10-29 on 418f905a4), fork 44.8.
The fork reaches 46.9 with the experts re-encoded as IQ2_XXS_U3 (`bench/gguf_iq2u3.py`, +10 GB, fork-only format). Folding
the NVFP4 scales into the norm weights (`bench/gguf_fold_scales.py`) is optional; it was worth 0.7 tok/s on the fork and
nothing on the upstream set. The patches are described in `patches/UPSTREAM-INDEX.md`; each upstream patch has a notes file
with tests and numbers.

## Bolt in webAI

The webAI desktop app runs its chat on the Homebrew `llama-server` (llama.cpp 620d2225c, the fork's base commit) and
resolves models as `<hugging-face-repo>:<quant>` into its own GGUF cache. `webai/use-bolt.sh` registers the downloaded
Bolt file in that cache, points the app at `webai/llama-server-wrapper.sh` through the `APOGEE_LLAMA_SERVER`
environment variable, sets Bolt as the selected model and relaunches the app. The wrapper execs the fork build with
the context and slot count the script chooses, because webAI itself gives Bolt 30,720 tokens over 3 slots (10,240 per
request) and its load-model API refuses more; a 22,000-word prompt is rejected at that size. Defaults are 3 slots of
131,072 tokens (`--ctx`, `--slots` change them). Bolt's cache is cheap: 8 attention layers, 4.2 KB per token at q8_0,
so a 128K slot costs 0.5 GB plus 165 MB of recurrent state.

Measured through the app's server, 2,000-word prompts, 256 tokens, q8_0 KV, 3 x 128K:

| users | decode per user | aggregate | first token |
|---|---|---|---|
| 1 | 43.9 | 24 | 4.5 s |
| 2 | 29.2 | 58 | 9.0 s |
| 3 | 25.4 | 75 | 13.8 s |

The same as the standalone server (llama-bench 44.8). A 22,000-word prompt prefills at 448 tok/s and decodes at 40.8.
Prefill is serialized across slots, so a long prompt on one slot stalls the others while it runs.
`webai/use-bolt.sh --undo` puts Gemma and the Homebrew server back. The environment variables are per login session,
so re-run the script after a reboot. The wrapper appends to `WEBAI_SERVER_LOG` if set; webAI discards the server's
stderr. Twice after a launch the server answered every request with "Compute error" until relaunched; the log is there
to catch the cause next time.

Any GGUF goes the same way with `--model`, `--repo`, optional `--quant` and `--draft`. Nemotron-3.5-Lightning-30B-A3B
(ggml-org Q4_0, 17.6 GB, with its 1.1 GB MTP draft) in webAI, 3 x 128K, q8_0 KV, `results/results_webai_lightning.jsonl`,
with Gemma 4 26B-A4B as webAI ships it (Q4_0, MTP, `results/results_webai_serving.jsonl`):

| prompt | Lightning decode | Lightning first token | Gemma 4 decode | Gemma first token |
|---|---|---|---|---|
| 2,000 words, 1 user | 102 | 1.6 s | 97 | 1.6 s |
| 2,000 words, 2 / 3 users each | 50 / 37 | 3.2 / 5.2 s | | |
| 22,000 words | 91 | 20 s | 65 | 23 s |
| 8K tokens, natural text | 137 | 7.7 s | | |
| 32K tokens, natural text | 104 | 34 s | | |
| 100K tokens, natural text | 69 | 150 s | | |

Lightning's draft is accepted 68-77% of the time on natural text and 43-48% on the repeated-word load-test prompts,
which is why the natural-text decode is higher. Without the draft it decodes at 94 (2K) and 85 (22K words). Its cache
is 3.2 KB per token at q8_0 (6 attention layers), 0.4 GB per 128K slot; native context is 1M tokens.

## Setup

Requirements: macOS with Xcode, cmake, Python 3 with `numpy`, and for the MLX tests a Python with `mlx-lm` (tested with mlx 0.32.1, mlx-lm 0.31.3). Set `MLX_PY` to that interpreter if it is not `python3`.

### Rebuild llama.cpp

`patches/apply.sh <set> [dir]` does the checkout, patch and build for the `fork`, `upstream` and `master` sets. The recorded runs used four builds, each a plain llama.cpp checkout with a patch applied:

| directory | base commit | patch |
|---|---|---|
| `llama.cpp` | 620d2225c | `patches/upstream-batched-bench-pps-recurrent.patch` |
| `llama.cpp-bolt` | 620d2225c | `patches/fork-llama.cpp-bolt-metal-opt.patch` (the Bolt fork, includes the batched-bench fixes) |
| `llama.cpp-upstream-base` | 2e0d86beb | `patches/tool-batched-bench-pps-reuse.patch` (includes the pps fix) |
| `llama.cpp-upstream-splitk` | 2e0d86beb | the above plus `patches/upstream-metal-mul-mm-split-k.patch` |

```sh
git clone https://github.com/ggml-org/llama.cpp llama.cpp-bolt && cd llama.cpp-bolt
git checkout 620d2225c && git apply ../patches/fork-llama.cpp-bolt-metal-opt.patch
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
cmake --build build -j --target llama-bench llama-batched-bench llama-server llama-perplexity test-backend-ops
```

The explicit SDK path is needed on macOS 27 when the Command Line Tools SDK fails to link.

### Models

| file | source |
|---|---|
| `models/bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF |
| `models/bolt/Bolt-74B-A9B-24GB-folded.gguf` | `python3 bench/gguf_fold_scales.py <above> <this>` (NVFP4 scales folded into norms; optional, the recorded fork runs used it) |
| `models/bolt-lilicorr/...LiLiCorr.gguf` | NVIDIA's DFlash drafter for Bolt |
| `models/gemma-4-26B-A4B-it-Q4_0.gguf` | ggml-org/gemma-4-26B-A4B-it-GGUF |
| `models/lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, `mtp-...-Q4_0.gguf` | ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF |
| `models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` | unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF |
| `models/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf` | unsloth/Nemotron-3-Nano-30B-A3B-GGUF |
| `models/mlx-nemotron30b-4bit`, `mlx-qwen30b-4bit`, `mlx-gemma26b-4bit` | mlx-community 4-bit conversions |

MLX runs need a token fixture per model: `python3 bench/build_fixture.py models/mlx-<name>-4bit` (9M tokens of the stdlib, deterministic).

### GPU memory

The default Metal budget is 40 GB on a 48 GB machine. The capacity runs used `sudo sysctl iogpu.wired_limit_mb=44032` (resets on reboot).

## Running tests

```sh
tests/00_env.sh            # builds, models, bandwidth (1 min)
tests/10_single_user.sh    # decode vs context depth, one model (15 min)
tests/20_concurrency.sh    # aggregate decode vs sequences (20 min)
tests/30_scenarios.sh      # 1-user / agents / capacity views, three models (1 h; GRID=full for the report's grid)
tests/40_mlx.sh            # MLX stock vs GQA kernel, 128K gate (40 min)
tests/50_quality.sh        # kernel correctness, perplexity, logit agreement, needle (45 min)
tests/60_serving.sh        # llama-server, no-spec vs DFlash (10 min)
tests/run_all.sh
```

Every script takes its model and binary from `tests/common.sh`, overridable through the environment, for example `tests/10_single_user.sh models/gemma-4-26B-A4B-it-Q4_0.gguf llama.cpp-upstream-splitk/build/bin`. Results append to `results/`; existing files hold the recorded runs.

Measurement notes: close screen sharing and other GPU users (10-15% noise), interleave A/B runs, and expect two 22 GB GGUFs not to stay page-cached at once.

## Longer sweeps

The scripts in `bench/*.sh` are the exact runs behind the report (MLX matrices, Gemma and Bolt suites, DFlash sweep, the full scenario grid). They take hours each and are kept as the record.
