# metal: split-K in kernel_mul_mm

Branch: `metal-mul-mm-split-k` in worktree `llama.cpp-upstream-splitk`, on origin/master 2e0d86beb (moved from 8ca4060b7; applied without changes, no Metal commits in between), uncommitted.
Patch: `patches/upstream-metal-mul-mm-split-k.patch` (see UPSTREAM-INDEX.md).

## What changes
- Non-tensor `kernel_mul_mm`: takes a K range `[k0, k0+nkc)`; grid z = split * (ne12*ne13) + batch; slice z of the output is the partial result.
- New `kernel_mul_mm_sk_reduce`: sums the nsk slices into dst.
- Scratch is `nsk * ggml_nbytes(op)` after dst, reserved in `ggml_backend_metal_buffer_type_get_alloc_size` (same pattern as MUL_MAT_ID extras).
- Split only when tiles (64x32) < 512 and ne11 > 8; slices are multiples of 256 (block boundary for every type), >= 512, at most 16.
- Tensor-API kernel (has_tensor): unchanged, nsk = 1.

## Why
At ne11 = 17..64 a narrow src0 gives too few tiles: ne01 = 368 -> 6 threadgroups, 1024 -> 16, 4096 -> 64 (M3 Max has 40 cores).
ne11 <= 16 goes to the few-row MMA path; raising its cap to 32/64 was slower and erratic (e.g. NVFP4 18560x4096 n=24: 522 -> 3056 us).

## Tests (M3 Max, master + patch)
- On 2e0d86beb: `test-backend-ops test -b MTL0 -o MUL_MAT`: 2490/2490; `-o MUL_MAT_ID`: 1031/1031 (same counts as on 8ca4060b7).
- On 8ca4060b7: `test-backend-ops -o MUL_MAT`: 2490/2490; `-o MUL_MAT_ID`: 1031/1031. ~239 existing MUL_MAT cases take the split path, incl. K = 2880 (partial last slice).
- Perplexity (Bolt-74B, fork build, 16 x 2048, `-ub 32` so prefill uses the split path): 1.9990 vs 1.9989.

## Kernel time, M3 Max, `test-backend-ops perf` (fork, same kernel code), best of 2-4 interleaved runs, us

| src0 (M x K) | n | before | after |
|---|---|---|---|
| f32 368 x 4096 | 32 | 122 | 30 |
| q8_0 1024 x 4096 | 32 | 118 | 44 |
| nvfp4 4096 x 8192 | 32 | 342 | 217 |
| iq2_xxs 4096 x 5376 | 32 | 231 | 157 |
| nvfp4 4096 x 8192 | 64 | 557 | 415 |
| nvfp4 18560 x 4096 | 24 | 520 | 478 |
| n = 512 shapes | | | unchanged |

Measured before the move to 2e0d86beb (kernel times on the fork, end-to-end on 8ca4060b7); not re-measured.

## End to end, M3 Max, `llama-batched-bench -pps`, depth 1024 (master vs master + patch), aggregate tg tok/s, 2 interleaved runs each, GPU otherwise idle

| model | 16 | 24 | 32 | 64 |
|---|---|---|---|---|
| Qwen3-8B Q4_K_M base | 375-384 | 305-306 | 381-382 | 433 |
| Qwen3-8B Q4_K_M split-K | 364-382 | 335-345 | 414-417 | 445-452 |
| Qwen3-30B-A3B Q4_K_M base | 257-260 | 239-241 | 360-363 | 496-507 |
| Qwen3-30B-A3B Q4_K_M split-K | 257-260 | 253-254 | 385 | 516 |

Qwen3-8B: +11% at 24, +9% at 32, +4% at 64, 16 within noise. Qwen3-30B-A3B: +6% at 24 and 32, +3% at 64, 16 unchanged.

Fork build, Bolt-74B-A9B (NVFP4 + IQ2_XXS), depth 4096: 24 users 125.4 -> 141.4, 32 users 139.3 -> 154.3; 1-16 users and prefill unchanged.

## Overlap with open PRs (checked 2026-10-28)
- **#27441** (draft, tugot17, last update 2026-09-01, base 9855ad69d): `kernel_mul_mm_s` with 32x16 tiles for
  ne11 <= 32 (env `GGML_METAL_MM_SKINNY_MAX`, default 32), plus a split-K variant with a fixed 4 slices when
  `ne01 <= 5120 && ne00 >= 1024`; partials in the dst scratch tail, fixed-order reduce; tensor-API devices
  excluded; env knobs for the mv_ext window (`GGML_METAL_MV_EXT_MAX/_R1`) and the mm threshold
  (`GGML_METAL_MM_MIN`). Measured on M4 Max for 8 < ne11 <= 32 (speculative-decoding verify batches).
  - Same idea and same scratch-tail pattern as this branch. Differences: this branch keeps the 64x32 tile and
    picks the split count per op (enough slices to reach ~512 threadgroups, 1..16, slices of >= 512 K in
    multiples of 256), for any ne11 > 8 while the tile count is < 512 (so also 33..64+ rows, the 24-64 user
    range); no new tile kernel and no env knobs; selection does not depend on ne01/ne00 thresholds.
  - Conflict: both rewrite the mm dispatch block in `ggml_metal_op_mul_mat` and add `GGML_OP_MUL_MAT` to
    `ggml_backend_metal_buffer_type_get_alloc_size`; they cannot both land as is. #27441 is written against the
    pre-split `ggml-metal.metal` and does not apply to 2e0d86beb (`git apply --check` fails in
    ggml-metal-device.h and ggml-metal-ops.cpp), so no A/B was run.
- **#29743** (draft, forforever73, 2026-10-03): per-device tuned mm tiles (NR1 8/16/32 template, NR0 function
  constant, generated tables) and a lower mv_ext -> mm switch inside ne11 2..8. Attacks the same
  under-filled-GPU problem with smaller tiles instead of K splits. Touches `kernels/mul_mm.metal` and the same
  dispatch code; does not apply to 2e0d86beb. If it lands, the split rule here would need the tile size it
  picks (tile count uses 64x32 today).
- **#30982** (open): lets K in 32..63 use mul_mm (`ggml_metal_op_mul_mat_use_mm`). Applies cleanly to 2e0d86beb
  and touches a different file; no conflict. Such small K never splits here (needs ne00 >= 1024).
- **#25377** (64x8 tile for q4_0): closed.

## Walkthrough
1. `ggml-metal-impl.h`: `ggml_metal_kargs_mul_mm` gets `nsk` (number of K splits, 1 = none) and `nkc`
   (K elements per split); new `ggml_metal_kargs_mul_mm_sk_reduce` (`n` floats per slice, `nsk`).
2. `ggml-metal-ops.cpp`, `ggml_metal_op_mul_mat_nkc`: counts 64x32 output tiles; returns ne00 (no split) when
   there are >= 512 tiles, K < 1024, or ne11 <= 8 (that case never reaches mul_mm). Otherwise
   nsk = min(ceil(512/tiles), K/512, 16) and the slice length is rounded up to a multiple of 256, so every
   slice starts on a block boundary for all quant types (largest block is 256).
   `ggml_metal_op_mul_mat_nsk` derives the slice count from that length (the last slice may be partial).
   `ggml_metal_op_mul_mat_extra_sk`: scratch = nsk * nbytes(dst) when splitting.
3. `ggml-metal.cpp`: the buffer-type alloc size adds that scratch for `GGML_OP_MUL_MAT`, the same mechanism
   MUL_MAT_ID uses for its tokens-per-expert map. The scratch sits right after dst.
4. `ggml_metal_op_mul_mat` (mm branch): nsk = 1 on tensor-API devices (that kernel is untouched). With a split,
   the mm kernel writes to the scratch (`bid_sk` = dst + nbytes(dst)), the grid z dimension becomes
   nsk * ne12 * ne13, then `ggml_metal_op_concurrency_reset` and one reduce dispatch (256 threads per
   threadgroup) sums the slices into dst.
5. `kernels/mul_mm.metal`, `kernel_mul_mm`: z is decoded as split * (ne12*ne13) + batch; the src0 block pointer
   and src1 pointer start at k0; the K loop runs k0..k1 (k1 clamped to ne00, so a short last slice is handled
   by the existing loop and bounds logic); the output offset uses z instead of the batch index, so slice s
   lands at s * ne0*ne1*ne12*ne13 in the scratch. With nsk = 1, z == batch and the kernel is unchanged.
6. `kernel_mul_mm_sk_reduce`: one thread per output float, sums nsk slices in order (deterministic).
7. `ggml-metal-device.cpp/h`: pipeline getter for the reduce kernel (no function constants).

## Open points a reviewer may ask about
- Not measured on other Apple GPUs; the 512 target was tuned on M3 Max only (128 / 256 / 512 / 1024 tried; 512 ~ 1024).
- Extra memory is reserved even when the op later takes the MMA/mv path (cheap: only small-tile shapes reserve, e.g. 32 x 4096 x 4 B x 4 slices = 2 MiB).
- Results are not bit-identical to the unsplit kernel (different summation order).
- Not applied to MUL_MAT_ID or to the tensor-API kernel.
