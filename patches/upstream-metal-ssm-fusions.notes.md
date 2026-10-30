# metal: Mamba-2 decode step fusions (SSM_SCAN rows kernel + scan/cache fusion, CONCAT/conv/cache fusion)

Branch: `metal-ssm-fusions` in worktree `pr/metal-ssm-fusions`, on origin/master 418f905a4 (rebased 2026-10-29 from 2e0d86beb, applied cleanly), staged and uncommitted.
Patch: `patches/upstream-metal-ssm-fusions.patch` (see UPSTREAM-INDEX.md).
Source: the Bolt fork (`llama.cpp-bolt`), second and third passes in REPORT.md. Merged 2026-10-29 from two
branches (`metal-ssm-scan-cache`, `metal-ssm-conv-cat`) that conflicted textually and only pay off together.

## What changes

Scan side:
- New `kernel_ssm_scan_f32_rows` (ssm.metal): one simdgroup per head_dim row. The row's state stays in registers
  (d_state/32 floats per lane, d_state <= 256) across all tokens of the ubatch; no threadgroup memory, no barriers.
  Writes the K-1 rollback snapshots like `kernel_ssm_scan_impl`.
- Dispatch: whenever the SSD MMA path is not taken and `d_state % 32 == 0 && d_state <= 256` (decode, speculative
  verify, prefill of models with head_dim != 64).
- New fusion `GGML_METAL_FUSION_SSM_SCAN_CACHE` (SSM_SCAN + CPY, plus a 4-op variant with the empty GET_ROWS + CPY that
  `build_rs` inserts): the kernel writes the final state and snapshots straight into the recurrent cache view the CPY
  targets, and the CPY is skipped. `nb_s_slot` kargs field gives the snapshot slot stride (output tail or cache).
- Host-side safety check for n_seqs > 1: the scan reads state rows `ids[i]` from the cache it writes. The ids are graph
  inputs already uploaded at encode time; if any sequence would read a row another sequence writes (cell swaps, shared
  zero state), the fusion is skipped. Only for shared (host-visible) Metal buffers.
- `ggml_metal_fusion_check_memory_ranges` exempts exactly that src0/dst overlap for this fusion.

Conv side:
- New fusion `GGML_METAL_FUSION_SSM_CONV_CAT` for the decode/verify conv step mamba-base builds:
  CONCAT(conv state, x^T) + m CPYs of the rolled window into the conv cache (m = 1..10 snapshot slots) + SSM_CONV +
  ADD(bias) + UNARY(silu). `kernel_ssm_conv_cat_f32` keeps the window (d_conv-1 state values + n_t <= 16 tokens) in
  registers per channel, writes the silu output and the new state and snapshots straight to the cache. The
  concatenated tensor is never materialized. Hooked in `ggml_metal_op_concat`.
- Existing SSM_CONV kernels: one channel per thread in 256-thread threadgroups instead of a 1-thread threadgroup per
  channel; optional fused bias via function constant `FC_SSM_CONV + 3`.
- Existing SSM_CONV_SILU fusion also accepts SSM_CONV + ADD(bias) + UNARY.

Tests:
- `test_ssm_scan_cache_fusion` gains `ids_mode` (state read from the cache itself: own cell, own cell in slot 1,
  reversed cells that alias another sequence's write) and 18 new cases (n_seqs 1/4/16, decode and K=4 verify).
- New `test_ssm_conv_cat_fusion`, 8 cases: decode, 4-token verify with 4 slots, 8 tokens / 3 slots, n_ch = 300;
  n_seqs 1 and 4. It reads the cache back so the CPYs are checked.

## Why
Master's decode scan runs one threadgroup of d_state threads per row with barriers, then a separate CPY moves the full
state into the cache. The conv step runs 4-5 small kernels per layer and launches one 1-thread threadgroup per
channel (11 GB/s at the decode shape). Bolt has 40 Mamba layers with a 4 MiB f32 state per user per layer; the state
copies scale with the number of sequences while weights are shared, which is why the gain is largest under load.

## Tests (M3 Max, merged branch, 2026-10-29)
- `test-backend-ops -b MTL0`: SSM_SCAN 15/15, SSM_CONV 60/60, SSM_SCAN_CACHE_FUSION 26/26, SSM_CONV_CAT_FUSION 8/8,
  CONCAT 226/226, CPY 199/199.
- On the separate branches: the scan fusion fired in 21/26 SSM_SCAN_CACHE_FUSION cases (the 5 others are the
  aliasing cases the host check refuses, and the 300-token SSD case); the conv fusion in 8/8. In the models:
  162 / 108 fused scan / conv steps per short decode run on Nano-9B, 115 / 92 on Nano-30B.
- With the host-side ids check disabled, the aliasing cases fail 1 in 6 at 4 sequences and 5/5 at 16, so the check
  catches a real race.
- Perplexity, Nemotron-Nano-9B-v2 Q4_K_M, 16 x 512, `-ub 8` (so the rows kernel runs): 3.7133 master and branch. Identical per chunk on 418f905a4.
- Both fused kernels use the same arithmetic order as the kernels they replace; output is bit-identical.

## Kernel time, M3 Max, `test-backend-ops perf`, best of 2, us

| case | master | branch |
|---|---|---|
| SSM_SCAN d_state 128, head_dim 64, 48 heads, 1 token | 27.0 | 14.8 |
| SSM_SCAN d_state 128, head_dim 80, 128 heads, 1 token (Nano-9B) | 81.0 | 36.0 |
| SSM_SCAN same, 512 tokens (no MMA) | 6728 | 3965 |
| SSM_SCAN d_state 128, head_dim 64, 512 tokens (MMA path, untouched) | 1713 | 1711 |
| SSM_CONV [4, 3328] (decode) | 10.0 | 2.8 |
| SSM_CONV [515, 3328] / [937, 8192] (prefill, batched kernel) | 68.4 / 237.9 | 69.5 / 237.7 |

## End to end (tok/s, best of 2 interleaved rounds)

Each part alone on Nemotron-Nano 9B-v2 / 30B-A3B was within noise of master (<= 1.3%). The merged branch on
Nano-30B-A3B Q4_K_M, `llama-bench -fa 1 -n 128 -r 2`, three interleaved rounds (2026-10-29): master 74.2 / 76.5 / 76.8,
branch 81.9 / 86.7 / 87.1 tok/s (+13%). Fusions disabled (`GGML_METAL_FUSION_DISABLE=1`) on the branch: 65.6.
After the rebase onto 418f905a4: master 75.1 / 79.3 / 79.2, branch 89.9 / 89.5 / 89.5 (+13%); fusions disabled 79.6.

Bolt-74B-A9B (NVIDIA 24 GB GGUF) needs NVFP4 on Metal, so: master + `metal-nvfp4` (base) vs + this branch:

| | base | scan part alone | conv part alone | both (this branch) |
|---|---|---|---|---|
| tg128 | 31.7 | 35.7 (+13%) | 33.1 (+5%) | 38.1 (+20%) |
| aggregate, 4 / 8 / 16 seqs | 47.6 / 57.0 / 60.9 | 50.8 / 62.0 / 66.6 | 50.6 / 60.5 / 64.6 | 85.9 / 120.4 / 144.0 |

Each part alone gives +5-13% at 1 seq and +6-9% at 16; together 1.8x / 2.1x / 2.4x at 4 / 8 / 16 seqs. Not
investigated why the combination exceeds the sum. Unverified guess: with only one fused, the remaining unfused CPY
into the recurrent cache still serializes the multi-sequence step. This is why the two are one branch.

## Overlap with other work
- Open #25533 (`kernel_ssm_scan_group_f32`, no review): same register-resident-state idea, but n_seq_tokens == 1
  only, d_state 128/256 only, no rollback snapshots, no cache fusion, `exp(dt*A)` once per token (scalar A only).
- Merged #26647 (SSD MMA prefill): untouched; the rows kernel only runs where MMA is not used.
- Merged #28948 added SSM_CONV + silu fusion on Metal; this extends it with the bias and adds the concat/cache fusion.
- Master already has `test_ssm_scan_cache_fusion` (CUDA has that fusion); this adds the Metal side.

## Open points a reviewer may ask about
- The host-side ids check reads graph input data during encode and relies on the ids buffer being shared memory.
  With private buffers the scan fusion is not used for n_seqs > 1.
- Rows kernel for long prefills with head_dim != 64: faster than master in the perf test, but a serial loop over
  tokens; not tuned for very long ubatches.
- Ten conv fusion-table entries (`ops_ssm_conv_cat_1..10`), one per number of snapshot CPYs, because the matcher
  matches fixed op sequences. Longest is 14 nodes, under GGML_METAL_FUSION_MAX (16).
- The conv check walks the view chain of the state and x inputs so neither aliases the cache being written.
- d_conv must be 2..4 and n_t <= 16 for the conv fusion.
- Not measured on other Apple GPUs.
- CONTRIBUTING.md requires an AI-use disclosure in the PR.

## Walkthrough (hunk by hunk)
1. `ggml-metal-impl.h`: `nb_s_slot` at the end of `ggml_metal_kargs_ssm_scan` (slot stride: n_seqs*nb03 unfused,
   cache view nb[2] fused). New `ggml_metal_kargs_ssm_conv_cat`: n_t, n_ch, n_slots, strides of state, x, conv
   weights, dst and cache (row stride per sequence, byte offset per snapshot slot).
2. `kernels/ssm.metal`:
   - `kernel_ssm_scan_f32_rows`: grid (ceil(d_inner/8), n_head, n_seqs), 8 simdgroups per threadgroup. Each lane
     loads its d_state/32 state values and A once. Per token: softplus of dt (same formula and 20.0 cutoff as the
     existing kernel), `s = s*exp(dt*A) + B*x*dt`, partial dot with C, `simd_sum`, lane 0 writes y. Snapshot slot
     `n_t-1-t` is written when 0 < slot < K. Final state written once to `dst_s` (buffer 9): the op's own output
     unfused, or the cache.
   - `FC_ssm_conv_bias` and a `bias` buffer on the four existing conv kernels; non-batched kernels use
     `ir = tgpig.x*ntg.x + tpitg.x` with an `ir >= ne01` guard.
   - `kernel_ssm_conv_cat_f32`: loads d_conv-1 state values and n_t new values into `w[]` and the weights into
     `cw[]`; per token the same dot as the existing kernel (float4 when nc == 4), bias, silu, store; then for each
     slot s writes `w[n_t-s .. n_t-s+ncs-1]` to cache slot s (slot 0 = final state).
3. `ggml-metal-device.cpp/.h`: `ggml_metal_library_get_pipeline_ssm_scan_rows`; `use_bias` on the two conv pipeline
   getters; `ggml_metal_library_get_pipeline_ssm_conv_cat(nc)`.
4. `ggml-metal-ops.cpp`:
   - `ggml_metal_op_ssm_scan`: sets `nb_s_slot`; new branch before the existing `!use_mma` one: if
     `d_state % 32 == 0 && d_state <= 256`, optionally take the fusion (buffer 9 at the cache view, `s_off = 0`, slot
     stride from the view), dispatch the rows kernel, return 1 or 2 nodes consumed.
   - `ggml_metal_op_ssm_conv_cat`: fills kargs from the fused nodes (cache slot stride = view offset of CPY 1 minus
     CPY 0), binds 6 buffers, dispatches ceil(n_ch/256) x n_seqs threadgroups of 256. `ggml_metal_op_concat` tries it
     first. `ggml_metal_op_ssm_conv`: `use_bias` when the SSM_CONV_SILU fusion matched 3 nodes; 256-thread dispatch.
5. `ggml-metal-fusion.h`: ids `GGML_METAL_FUSION_SSM_SCAN_CACHE` and `GGML_METAL_FUSION_SSM_CONV_CAT`.
6. `ggml-metal-fusion.cpp`:
   - `ggml_metal_fusion_check_ssm_scan_cache`: CPY source must be the contiguous view of the scan output right after
     y with D*n_seqs*min(n_tokens,K) elements; destination an f32 [D, n_seqs, n_written] view with row stride D*4. In
     FULL mode the cache must be allocated and, for n_seqs > 1, the ids aliasing check runs. The 4-op variant accepts
     `build_rs`'s GET_ROWS + CPY only when empty (decode).
   - `ggml_metal_fusion_check_ssm_conv_silu`: optional middle ADD with a contiguous f32 bias of conv->ne[0].
   - `ggml_metal_fusion_check_ssm_conv_cat`: shapes and types of the concat inputs and weights; ADD/UNARY wiring; use
     counts; no output flags; CPY j source = view of the concat at offset (n_t-j) with the state's shape; CPY
     destinations = views of one cache tensor at off0 + j*slot; neither input aliases the cache.
   - `ggml_metal_fusion_check_memory_ranges`: skips the overlap check only for the scan fusion, node 0, src 0, when
     src and dst are views of the same cache tensor.
   - Table: two scan entries (2-op, 4-op), ten conv-cat entries, the 3-op SSM_CONV + ADD + UNARY entry before the
     2-op one so it wins.
7. `tests/test-backend-ops.cpp`: `ids_mode` on `test_ssm_scan_cache_fusion` (state input is a reshape of the cache
   itself, cache randomly initialised, ids set per mode); `test_ssm_conv_cat_fusion` builds the same subgraph as
   mamba-base and reads the cache back.
