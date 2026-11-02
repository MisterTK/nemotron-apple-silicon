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
nothing on the upstream set. Homebrew llama.cpp 0.8.0 (build 11960, 8642541b2, 2026-10-30) loads both models unpatched:
Bolt decodes at 9.0 tok/s (no Metal NVFP4 kernels), Lightning with its MTP draft at 98 (2K prompt) and 86 (32K), against
102 and 104 on the fork. The patches are described in `patches/UPSTREAM-INDEX.md`; each upstream patch has a notes file
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

A long agent session, 2,000-word tool result per turn plus a 128-token reply, history growing to 126K tokens, same
3 x 128K setup, `results/results_webai_session.jsonl`. llama-server reuses the cached prefix, so each turn processes only
its new ~4,500 tokens; the cost of the history shows up as slower prefill and decode at depth, not as re-reading it.
Per turn: time to first token / decode tok/s / whole turn:

| history | Lightning | Bolt | Gemma 4 (fork, MTP) |
|---|---|---|---|
| 4K (turn 1) | 4.2 s / 118 / 5.4 s | 11.1 s / 43 / 14.1 s | 4.2 s / 108 / 5.4 s |
| 31K (turn 7) | 5.4 s / 96 / 6.7 s | 12.5 s / 38 / 15.8 s | 7.3 s / 64 / 9.3 s |
| 58K (turn 13) | 7.8 s / 80 / 9.4 s | 14.5 s / 37 / 18.0 s | 12.0 s / 43 / 14.9 s |
| 85K (turn 19) | 8.4 s / 75 / 10.1 s | 15.3 s / 35 / 18.9 s | server died at 86K |
| 126K (turn 28) | 12.3 s / 54 / 14.6 s | 19.8 s / 32 / 23.8 s | |

Turn 29 is rejected by all of them: the history exceeds the 128K slot, and `--context-shift` does not apply to prompts
larger than the slot. Gemma's server was killed mid-prefill with 100 MB of system memory free; its cache at 128K is
about six times the Mamba models' and the machine was under memory pressure from other applications.

Lightning at 3 x 256K, `results/results_webai_lightning_256k.jsonl`. The server took 23.3 GB at load and peaked at
27.8 GB; free memory touched 0.1 GB during the 250K prefill without heavy compression, and the server log shows no
errors over the whole run. Single natural-text prompts:

| prompt | prefill tok/s | first token | decode |
|---|---|---|---|
| 150K tokens | 535 | 281 s | 49 |
| 250K tokens | 371 | 674 s | 44 |

The same growing session continued to 258K tokens of history, still processing only the new ~4,500 tokens per turn:

| history | first token | decode | whole turn |
|---|---|---|---|
| 4K (turn 1) | 3.8 s | 129 | 4.9 s |
| 58K (turn 13) | 7.1 s | 76 | 8.8 s |
| 131K (turn 29) | 12.0 s | 59 | 14.2 s |
| 203K (turn 45) | 19.2 s | 43 | 22.2 s |
| 258K (turn 57) | 23.6 s | 37 | 27.2 s |

Turn 58 (262,199 tokens) is rejected, 55 tokens over the slot. Three 100K prompts sent at once finished after 152, 306
and 458 s; the first two decoded at 1-2 tok/s while the remaining prefills ran, the last at 48 alone. llama-server
schedules one batch per step across all slots: a slot that is prefilling fills the batch (`--batch-size`, 2048 here),
so other prefills wait and a decoding slot gets one token per 2048-token chunk. A smaller batch size gives decoding
slots more turns at the cost of prefill speed; there is no priority setting. Multiple slots pay off for overlapping
short chats, not for simultaneous long prompts.

The two probes are `webai/long_prompt.py TOKENS TAG OUT.jsonl` (one natural-text prompt built from the repo's markdown,
256-token answer) and `webai/session.py TURNS TAG OUT.jsonl` (the growing session, about 4,500 tokens per turn). Run
them from the repo root against the app's server on port 11436; both append one JSON record per request.

To repeat the Lightning test on another Mac: build `llama.cpp-bolt` as below, download the two Lightning files into
`models/lightning/`, then

```sh
webai/use-bolt.sh --model models/lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
  --repo ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF \
  --draft models/lightning/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf --ctx 131072 --slots 1
curl -s localhost:11436/props | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["total_slots"], d["default_generation_settings"]["n_ctx"])'
python3 webai/long_prompt.py 8000 m5-lightning-1x128k results/results_m5.jsonl
python3 webai/long_prompt.py 100000 m5-lightning-1x128k results/results_m5.jsonl
python3 webai/session.py 28 m5-lightning-1x128k-session results/results_m5.jsonl
```

On a 24 GB machine the 17.6 GB weights plus the 1.1 GB draft are close to the GPU working-set limit; if the load
fails or decode falls far below the figures above, drop `--draft` first (decode was 94 instead of 102 without it here),
then raise the wired limit (`sudo sysctl iogpu.wired_limit_mb=20480`). The cache itself is not the problem: 0.4 GB at 128K.

The same Lightning sweep on oMLX (jundot/omlx 0de3b0ba, mlx 0.32.4, mlx-lm 0.32.3.dev, `mlx-community` 4-bit
checkpoint, bf16 KV, no MTP head in that checkpoint), one engine, 4 concurrent
requests, `results/results_omlx_lightning.jsonl`. llama.cpp figures are the fork in webAI with q8_0 KV and the MTP draft.
Single natural-text prompts, prefill tok/s / first token / decode:

| prompt | oMLX | llama.cpp fork |
|---|---|---|
| 2K | 1170 / 1.6 s / 128 | 460 / 1.6 s / 102 |
| 8K | 1357 / 6.1 s / 124 | 1083 / 7.7 s / 137 |
| 32K | 1190 / 27 s / 110 | 937 / 34 s / 104 |
| 100K | 876 / 114 s / 84 | 668 / 150 s / 69 |
| 150K | 747 / 201 s / 73 | 535 / 281 s / 49 |
| 250K | 565 / 443 s / 58 | 371 / 674 s / 44 |

Growing session, first token / decode per turn:

| history | oMLX | llama.cpp fork |
|---|---|---|
| 4K | 4.2 s / 124 | 3.8 s / 129 |
| 60K | 5.6 s / 98 | 7.1 s / 76 |
| 135K | 9.3 s / 76 | 12.0 s / 59 |
| 210K | 13.3 s / 63 | 19.2 s / 43 |
| 257K | 15.8 s / 57 | 23.6 s / 37 |

Both reject the turn that passes 262,144 tokens, the model's configured context. Three 100K prompts at once on oMLX:
first tokens at 112, 339 and 343 s, decode 28 / 37 / 44 tok/s; the second and third prefilled together after the first,
and decode was shared rather than starved. Free memory touched 0.1 GB during the 250K prefill (bf16 attention scores),
no errors. Without its draft the fork decodes at 94 (2K) and 85 (22K words), so oMLX's lead at depth is the MLX attention
path, not speculation; with a converted checkpoint oMLX can run Nemotron-H MTP as well. oMLX's native Metal kernels
made no difference for this model (2K 1393 / 127, 32K 1199 / 110, 100K 885 / 84 with them built); they target other
architectures.

Lightning's draft is accepted 68-77% of the time on natural text and 43-48% on the repeated-word load-test prompts,
which is why the natural-text decode is higher. Without the draft it decodes at 94 (2K) and 85 (22K words). Its cache
is 3.2 KB per token at q8_0 (6 attention layers), 0.4 GB per 128K slot; native context is 1M tokens.

## Bolt on oMLX

Bolt runs on oMLX from txgsync/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-MLX: NVIDIA's 24 GB checkpoint unchanged plus
Metal kernels for its IQ2_XXS and NVFP4 tensors, loaded with Trust Remote Code. The `-LiLiCorr-MLX` variant adds
NVIDIA's DFlash drafter (7 draft tokens, enabled with the model's DFlash setting after the first load). Same oMLX build
as the Lightning run, `results/results_omlx_bolt.jsonl` and `results/results_omlx_bolt_lilicorr.jsonl`. Memory after
load 23.2 GB, 24.1 GB with the drafter.

Single natural-text prompts, prefill tok/s / first token / decode:

| prompt | oMLX | oMLX + DFlash | llama.cpp fork |
|---|---|---|---|
| 2K | 544 / 3.5 s / 50 | 540 / 3.5 s / 59 | 460 / 4.5 s / 44 |
| 8K | 535 / 16 s / 49 | | |
| 32K | 482 / 66 s / 46 | 498 / 64 s / 62 | 448 / 65 s / 41 |
| 100K | 427 / 235 s / 43 | 427 / 235 s / 48 | |
| 150K | 380 / 396 s / 40 | | |
| 250K | 312 / 803 s / 36 | | |

Growing session, first token / decode per turn:

| history | oMLX | llama.cpp fork |
|---|---|---|
| 4K | 13.2 s / 48 | 11.1 s / 43 |
| 55K | 11.8 s / 45 | 14.5 s / 37 (58K) |
| 105K | 15.0 s / 42 | 19.8 s / 32 (126K) |
| 155K | 16.4 s / 39 | |
| 205K | 18.7 s / 37 | |
| 255K | 21.0 s / 35 | |

Short answers with the drafter, 300 tokens, temperature 0: code 101-104 tok/s (accepted 85-96%), a markdown table 85,
prose 51-55 (65-71%); the natural-text probes above accept 74-80%. Prose is not slower with the drafter here.

Three 100K prompts at once: first tokens at 236, 711 and 723 s, decode 11.5 / 14.8 / 16.8 tok/s. The fast kernels are
single-sequence; batched requests take the generic mlx-vlm path. For more than one user at long context the llama.cpp
fork is the better server for Bolt. Free memory went to 0.1 GB during the first long prefill and stayed there; no errors.

oMLX 0de3b0ba fails to attach the drafter (`__eq__(): incompatible function arguments ... mlx.core.Dtype, NoneType`):
`match_target_dtype` in `omlx/speculative/dflash/__init__.py` compares the target embedding's dtype, which is None for
Bolt's custom embedding, with `mx.float16`. Guarding the None (`target_dtype is None or target_dtype != mx.float16`)
fixes it.

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
