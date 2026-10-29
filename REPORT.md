# Hybrid Mamba-2 MoE models vs attention MoE models on Apple Silicon

Decode-throughput study of NVIDIA Nemotron-3-Nano-30B-A3B, Nemotron-3.5-Bolt-74B-A9B, Qwen3-30B-A3B-2507 and Gemma 4 26B-A4B on a MacBook Pro M3 Max (48 GB, 40-core GPU, 369 GB/s measured). Measured 2026-10-26 to 2026-10-28. All numbers are measured on that machine unless marked *estimate*.

Sections: [Summary](#summary) · [Setup](#setup) · [Results](#results) · [Optimizations](#optimizations) · [Recommendations](#recommendations) · [Open items](#open-items)

## Summary

1. **Bytes per token decide long-context decode speed.** Nemotron-Nano is a hybrid of 23 Mamba-2, 23 MoE and 6 attention layers. Its KV cache is 6 KB/token against 96 KB/token for Qwen3-30B. Bolt-74B has 8 attention layers out of 88 and ~8 KB/token. Decode on Apple Silicon is bandwidth-bound, so these models lose little speed as context grows while Qwen and Gemma lose most of theirs.
2. **Single user, matched 4-bit MLX:** Nemotron-Nano 108 vs Qwen 75 tok/s at 4K, 69 vs 15 at 128K. On llama.cpp Q4_K_M Qwen leads at short context (101 vs 79) and Nemotron leads from 32K (67 vs 41) to 128K (50 vs 14).
3. **Concurrency:** at 128K Qwen fits one user in the 40 GB Metal budget. Nemotron-Nano ran 32 users (214 tok/s aggregate with 4-bit KV). Bolt fits 16 users at 128K with f16 KV and 26 with q8_0.
4. **MLX attention kernel:** a split-K, GQA-head-reuse decode attention kernel (`bench/gqa_decode_attn.py`) raises Nemotron-Nano 128K decode from 68 to 82 tok/s at 1 user and from 112 to 184 aggregate at 16 users. Gemma gains 33-48% at 128K. Quality is unchanged (closer to fp32 attention than stock MLX).
5. **Bolt on llama.cpp:** NVIDIA's file ran at 9.8 tok/s because Metal lacked NVFP4 kernels. Native Metal NVFP4 kernels, fused SSM scan/conv, fused sigmoid routing and split-K mat-mat take it to 44.8 tok/s at 1 user and 154 aggregate at 32 users, perplexity unchanged. DFlash speculation reaches 82 tok/s on code and 52 on prose.
6. **Spec gate (100 tok/s, 1 user, 128K) not met on the M3 Max:** best 82.3 tok/s. M3 Ultra *estimate* ~130 tok/s.

## Setup

| item | value |
|---|---|
| machine | MacBook Pro M3 Max, 40-core GPU, 48 GB; macOS 27.0.1, Xcode 26.2 |
| GPU bandwidth | 369 GB/s read, 257 GB/s copy (`bench/bw.py`) |
| GPU memory budget | 40.2 GB default; raised to 44 GB with `sudo sysctl iogpu.wired_limit_mb=44032` for the capacity runs |
| MLX | mlx 0.32.1, mlx-lm 0.31.3, free-threaded Python 3.14 |
| llama.cpp | 620d2225c (the commit webAI 2.5.2 ships) and master 2e0d86beb; built from source with Metal |
| models (MLX 4-bit) | `mlx-community` Nemotron-3-Nano-30B-A3B, Qwen3-30B-A3B-Instruct-2507, gemma-4-26B-A4B-it |
| models (GGUF) | unsloth Nemotron-Nano Q4_K_M (24.6 GB, experts Q5_0/Q8_0), unsloth Qwen3-30B Q4_K_M (18.6 GB), ggml-org Gemma 4 Q4_0, NVIDIA Bolt-74B-A9B-24GB (IQ2_XXS experts, NVFP4 Mamba projections) |
| fixture | 9M tokens of CPython 3.14 stdlib source, tokenized per model (`bench/build_fixture.py`) |

### Architecture and bytes per decode step

| | Nemotron-Nano-30B | Bolt-74B | Gemma 4 26B | Qwen3-30B |
|---|---|---|---|---|
| layers | 23 Mamba-2 + 23 MoE + 6 attn | 40 Mamba-2 + 40 LatentMoE + 8 attn | 25 sliding + 5 global attn, MoE + dense FFN | 48 attn + MoE |
| KV per token | 6 KB | 8 KB | 20 KB (global layers) | 96 KB |
| fixed state per user | 48 MB | 165 MB | ~210 MB sliding window | none |
| MoE | 128 experts, top-6 + shared | 240-496 experts, top-9 + shared | 128 experts, top-8 | 128 experts, top-8 |
| weights read per token (B=1) | ~2.0 GB (4-bit) | ~4.8 GB | ~2.55 GB | ~1.9 GB (4-bit) |

Unsloth's Nemotron "Q4_K_M" keeps experts at Q5_0/Q8_0 (2.57 GB/token vs 1.92 for Qwen), so the MLX 4-bit pair is the matched comparison.

## Results

### Nemotron-Nano vs Qwen3-30B, single user

MLX 4-bit, stock mlx-lm, 512 generated tokens:

| context | Nemotron tok/s | Qwen tok/s | Nemotron prefill | Qwen prefill |
|---|---|---|---|---|
| 4K | 108.6 | 75.1 | 3.0 s | 3.1 s |
| 32K | 95.2 | 38.7 | 28 s | 55 s |
| 128K | 68.5 | 15.4 | 168 s | 660 s |

llama.cpp Q4_K_M, `llama-bench -fa 1`:

| depth | Nemotron tg | Qwen tg | Nemotron pp | Qwen pp |
|---|---|---|---|---|
| 0 | 79.2 | 101.4 | 1237 | 1286 |
| 32K | 67.5 | 41.2 | 792 | 241 |
| 128K | 50.5 | 14.2 | 402 | not run |

### Nemotron-Nano vs Qwen3-30B, concurrency (MLX, aggregate tok/s, best variant)

Shared-prefix mode, 256 tokens per user. `mem` = exceeds the Metal budget.

| point | Nemotron | Qwen | ratio |
|---|---|---|---|
| 4K, 1 user | 108 | 75 | 1.4x |
| 4K, 32 users | 453 | 298 | 1.5x |
| 32K, 1 user | 98 | 44 | 2.2x |
| 32K, 16 users | 293 | 99 | 3.0x |
| 32K, 32 users | 362 | mem | |
| 128K, 1 user | 83 | 21 | 3.9x |
| 128K, 16 users | 188 | mem | |
| 128K, 32 users | 214 | mem | |

Full matrices (per variant, per-user tok/s, p95 gap): `python3 bench/report_tables.py results/results_mlx_matrix.jsonl`.

### MLX decode attention kernel

Stock MLX attention on Nemotron's 16-query-heads-per-KV-head shape ran at ~140 GB/s (38% of bandwidth): each query head gets its own simdgroup and the kernel is ALU-bound. The new kernel loads each 32-key K/V tile once into threadgroup memory, reuses it for all G query heads with `simdgroup_matrix` MMA, splits keys across threadgroups so one user fills the GPU, and dequantizes 8/4-bit KV on load. It supports G = 8 or 16 and D = 128, with a wide-D variant for Gemma (D = 256/512).

Per layer call, Nemotron shape, 128K:

| | 1 user | 16 users |
|---|---|---|
| stock bf16 | 0.90 ms, 147 GB/s | 15.8 ms, 136 GB/s |
| stock 8-bit KV | 1.32 ms | 39.4 ms |
| kernel bf16 | 0.55 ms, 245 GB/s | 7.2 ms, 297 GB/s |
| kernel 4-bit KV | 0.50 ms | 6.9 ms |

Spec gate, 1 user, 128K prompt, 5 trials of 2,048 tokens (`results/results_gate.jsonl`):

| config | median tok/s | p95 gap |
|---|---|---|
| stock bf16 | 68.2 | 14.8 ms |
| kernel bf16 | 81.7 | 12.3 ms |
| kernel 4-bit KV | 82.3 | 12.3 ms |

Quality (`bench/validate.py`, 32K, 128 teacher-forced tokens): kernel bf16 agrees with exact fp32 attention at 97.7% top-1 / 0.0034 KL, stock bf16 at 96.1% / 0.0051. 8-bit KV costs about the bf16 noise floor; 4-bit triples the KL. Needle retrieval at 131K: 15/15 for every variant.

Stock mlx-lm quantized KV is slower than bf16 at long context (58 vs 69 tok/s at 128K). With the kernel, 8-bit gives 77 and 4-bit 83.

### Gemma 4 26B-A4B

webAI 2.5.2 runs this model on llama.cpp 620d2225c with q8_0 KV, one slot and MTP speculation. Sliding-window attention keeps 25 of 30 layers context-free; the 5 global layers still grow KV at 20 KB/token.

llama.cpp, `llama-bench -fa 1`:

| depth | tg f16 KV | tg q8_0 KV | pp f16 |
|---|---|---|---|
| 0 | 91.4 | 80.1 | 1349 |
| 32K | 64.9 | 49.5 | 568 |
| 128K | 34.1 | 28.9 | 209 |

q8_0 KV costs 12-24% of decode on Metal: dequantizing in the attention kernel costs more than the bandwidth it saves. MLX with the wide-D kernel: 128K decode 36 -> 47.5 tok/s (bf16), 52.8 with 4-bit KV; 128K x 8 users 31 -> 91 aggregate. Gemma fits 4 users at 128K bf16, 8 with quantized KV. Quality matches stock (needle 15/15).

### Bolt-74B-A9B on llama.cpp

Two gaps in llama.cpp 620d2225c: no Metal MoE kernel for 9 experts per token (segfault, fixed upstream in 704366d76), and no Metal NVFP4 kernels, so 80 Mamba projections ran on the CPU (9.8 tok/s). Re-encoding those tensors to Q8_0 (`bench/gguf_retype.py`) gave 27 tok/s with perplexity unchanged (1.9987 vs 1.9986). Decode drops only 20% from 0 to 128K.

Serving with NVIDIA's DFlash drafter on that build: 49 tok/s at 2,000-word prompts, 37 at 22,000 words, vs 26 and 24 without speculation.

## Optimizations

All llama.cpp changes live in `llama.cpp-bolt/` (620d2225c + 704366d76 + PR #31075), diff in `patches/fork-llama.cpp-bolt-metal-opt.patch`. Single user, `llama-bench -fa 1`, depth 0:

| step | tg tok/s |
|---|---|
| Q8_0 re-encode of NVFP4 on stock kernels | 26.8 |
| native Metal NVFP4 kernels (mat-vec, MMA, mat-mat, MoE, get_rows), NVIDIA's file unmodified | 31.4 |
| + single-pass SSM scan and full-threadgroup SSM conv at decode | 33.9 |
| + PR #31075 (full threadgroups, LUT-free FP4 decode) | 38.2 |
| + NVFP4 scales folded into RMSNorm weights (`bench/gguf_fold_scales.py`) | 38.9 |
| + fused sigmoid+bias top-k routing, 9-expert reduce fusion | 40.9 |
| + SSM scan and conv write state straight into the recurrent cache (no CPY) | **44.8** |
| + IQ2_XXS_U3 expert re-encode (fork-only type, +10 GB) | 46.9 |

Per-op in-graph cost was measured by skipping each op type (`patches/tool-metal-op-skip-profiling.snippet.cpp`); timestamp profiling misleads because kernels overlap. The 40 x 4 MB state copies per step were the largest item not at a hardware limit. Removing them matters most under load: the copies scale with users while weights are shared.

Depth and concurrency, folded NVFP4 file (`results/results_llamabench_bolt_opt2.jsonl`, `results/results_llamacpp_concurrency.jsonl`):

| | stock kernels | optimized |
|---|---|---|
| 1 user, depth 0 / 32K / 128K (f16 KV) | 23.2 / 20.5 / 18.7 | 44.8 / 40.1 / 29.9 |
| aggregate at 4K, 8 / 16 / 32 users | 53 / 58 / 58 | 122 / 134 / 136 |
| aggregate at 4K, 24 / 32 users, + split-K mat-mat | | 141 / 154 |
| aggregate at 128K q8_0, 8 / 16 users | 32 / 37 | 55 / 57 |
| DFlash, code / prose (n_max 8, p_min 0.5) | | 82 / 52 |

Split-K in the Metal mat-mat kernel (fourth pass) also helps stock models on master: Qwen3-8B +13%/+9% and Qwen3-30B-A3B +5%/+6% at 24/32 users.

At 32 users Bolt is compute-bound: routed experts ~55 ms, NVFP4 mat-mat ~40 ms, shared experts ~25 ms of a ~176 ms step. It uses 31-43% of bandwidth under load, where Gemma and Qwen use 55-79%.

Quality checks for every pass: perplexity 1.9987-1.9990 (16 x 2048 tokens, within the ±0.027 error), greedy output identical to the CPU backend, DFlash output identical to no-spec, `test-backend-ops` passes including new SSM_SCAN cache-fusion and SSM_CONV fusion cases. The fusion tests exposed one bug (scan fusion silently skipped when the batch did not start at cache cell 0) and one race (fusion without the host-side row-id check fails 5/5 at 16 sequences); both are fixed.

Measured and not kept: 12 IQ2_XXS mat-vec variants (ALU and lookups in balance, ~650 G weights/s), grouped expert mat-vec over `map0` tiles (slower below 64 tokens), f16 recurrent state with stochastic rounding (`patches/experiment-ssm-f16-state.patch`, applies on the fork; no end-to-end gain because the scan overlaps other ops), encoder thread and command-buffer changes (no effect).

### Upstream candidates

Five branches on master 2e0d86beb, each with a patch and a notes file in `patches/`, indexed in `patches/UPSTREAM-INDEX.md`: batched-bench fix for recurrent models, Metal NVFP4, sigmoid top-k MoE fusion, the Mamba-2 decode fusions (scan and conv, one branch), mat-mat split-K.

### Three deployment views (llama.cpp, 2026-10-28)

Bolt on the fork, Gemma and Qwen on master + split-K. Capacity = largest number of sequences at that context that run without a Metal out-of-memory error at the 44 GB wired limit. `bench/scenarios_all.sh`, `results/results_scenarios.jsonl`.

1 user, decode tok/s and cold prefill time:

| context | Bolt | Gemma 4 | Qwen3-30B | Bolt prefill | Gemma prefill | Qwen prefill |
|---|---|---|---|---|---|---|
| 8K | 43.5 | 79.3 | 75.4 | 21 s | 7 s | 9 s |
| 32K | 40.6 | 63.9 | 42.8 | 81 s | 39 s | 75 s |
| 128K | 32.6 | 38.9 | 14.9 | 7.0 min | 6.0 min | 16.4 min |
| 256K | 25.5 | 21.7 | fails | 19.7 min | 21.0 min | |

Agents for one user, aggregate tok/s (per agent):

| agents x context | Bolt | Gemma 4 | Qwen3-30B |
|---|---|---|---|
| 8 x 32K | 102 (12.8) | 127 (15.8) | max 7: 59 (8.4) |
| 4 x 128K | 54 (13.5) | 30 (7.6) | max 2 (q8_0): 14 (7.2) |
| 8 x 128K | 64 (8.0) | 40 (5.0), q8_0 | does not fit |
| 8 x 256K | 40 (5.0), q8_0 | does not fit | does not fit |

Capacity (sequences that fit, f16 / q8_0 KV) and aggregate at that load:

| context | Bolt | Gemma 4 | Qwen3-30B |
|---|---|---|---|
| 8K | 96: 157 tok/s | 56: 318 | 32: 155 |
| 32K | 48: 117 | 26: 127 | 8: 59 |
| 128K | 16 / 26: 66 / 62 | 8 / 14: 39 / 40 | 2 / 3: 15 / 14 |
| 256K | 9 / 15: 33 / 41 | 4 / 7: 22 / 20 | 1 / 1 |

Stock master with NVIDIA's Bolt file: 12.4 tok/s at 1 user, ~25 aggregate under load. The fork is 3.5x at 1 user and 4-6x under load.

## Recommendations

Assumptions: an interactive conversation needs 20+ tok/s and each agent or user about 10 tok/s. Answer quality was not measured. Bolt is 74B total / 9B active against 26-30B / 3-4B active for Gemma and Qwen, so it should answer better.

| use case | model | context | measured |
|---|---|---|---|
| single conversation | Bolt | 128K (256K if needed) | 32.6 tok/s at 128K, 25.5 at 256K |
| single user, several agents | Bolt | 4 x 128K or 8 x 32K | 13.5 / 12.8 tok/s per agent |
| many users, short chats | Gemma 4 | 8K-32K | 24-32 users at >= 9 tok/s at 8K; ~10 at 32K |
| many users, long documents | Bolt | 32K-128K | ~10 users at 32K, 4-6 at 128K, >= 8 tok/s each |

- Below ~100K context Gemma is faster than Bolt (79 vs 44 tok/s at 8K). Choose it when speed matters more than answer quality.
- Qwen3-30B is not suited beyond 32K: 15 tok/s at 128K and at most 2-3 users.
- Prefill dominates long prompts (Bolt reads 128K in 7 minutes). The application must cache prompts between turns.
- Bolt needs the fork (or the upstream branches) and the folded NVFP4 GGUF. On stock llama.cpp it runs at 12 tok/s.
- Use f16 KV on Bolt and Gemma below 128K. Use q8_0 KV on Bolt at 128K+ for 60-70% more users at ~5% speed cost. q8_0 KV costs Gemma 12-24% decode speed.
- Serve Bolt with `--spec-type draft-dflash --spec-draft-n-max 8 --spec-draft-p-min 0.5`: 1.9x on code, 1.3x on prose, output identical to no-spec.
- Leave capacity headroom. Running at the last sequence that fits costs 30% speed (Bolt 256K: 23 tok/s at 8 vs 33 at 4).
- Raise the GPU wired limit to 44 GB for the long-context multi-user points.
- On MLX, install the GQA kernel for Nemotron-Nano, Qwen3-MoE and Gemma 4 at long context. Set `GQA_MIN_KEYS=8192` so short contexts stay on the stock path.
- webAI's bundled llama.cpp (620d2225c) cannot load Bolt until it moves past 704366d76.

## Open items

- **Spec gate.** 82 tok/s at 128K on the M3 Max vs a 100 tok/s target. M3 Ultra *estimate* ~130 tok/s (2.2x bandwidth on the bandwidth-bound ~9.5 ms of a 12.2 ms step; ~2.7 ms dispatch overhead does not scale). The harness runs unchanged on the Ultra.
- **MLX dispatch overhead** at batch 1 is ~2.7 ms/step (34% at short context). Fusing router + top-k + gather and the Mamba conv/scan chain is the largest single-user lever left. `mx.compile` on a MoE block gave ~8%.
- **Bolt multi-user** is compute-bound in the IQ2_XXS expert mat-vec and NVFP4 mat-mat. Kernel tuning is exhausted; the remaining lever is the expert format.
- **Bolt single user**: fusing D-skip and gate into the scan (~2%), residual add into the mat-vec store.
- **Quality** beyond perplexity and needle retrieval (multi-document reasoning, tool output) was not evaluated.
- **M5 Pro (24 GB)**: 280 GB/s, 23.6 TFLOPS f16 (1.8x the M3 Max on large matmuls, 0.75x bandwidth). Nemotron-Nano 4K decode 95 tok/s. The MLX matrix ran out of memory at 2 users; 24 GB leaves no room for long contexts with 30B models. `bench/m5_matrix.sh` reruns it one process per point.

### Findings worth keeping

- `llama-batched-bench -pps` fails on recurrent models ("inconsistent sequence positions"); fixed in `patches/upstream-batched-bench-pps-recurrent.patch`. Prefill reuse (`BB_REUSE_PP=1`, `patches/tool-batched-bench-pps-reuse.patch`) matched fresh prefill within 0.3% on Bolt but is invalid on stock master (0.37 tok/s slow path).
- mlx-lm quantized-KV prefill materializes the full score matrix (8.6 GB at 128K); the harness prefills in bf16 and quantizes after. mlx-lm also computes full-vocab logits for every prefill position; the harness prefills through the backbone only. Its BatchGenerator cannot batch QuantizedKVCache.
- MoE does not batch for free: each added user touches more experts, so weight bytes per step grow until most experts are read. 4 users at 4K cost 1.8x the step time of 1 user.
- Benchmark noise: screen sharing costs 10-15% bandwidth; two 22 GB GGUFs cannot both stay page-cached; another process on the GPU adds up to 10% run-to-run variance. Interleave A/B runs.
