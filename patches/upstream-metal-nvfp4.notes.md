# metal: NVFP4 support (mat-vec, mat-mat, few-row MMA, MoE, get_rows)

Branch: `metal-nvfp4` in worktree `pr/metal-nvfp4`, on origin/master 2e0d86beb, uncommitted.
Patch: `patches/upstream-metal-nvfp4.patch` (see UPSTREAM-INDEX.md). Metal only; no CPU, test or model changes.

## What changes
- `ggml-metal-device.m`: NVFP4 is no longer excluded from MUL_MAT / MUL_MAT_ID / GET_ROWS in `supports_op`.
- New kernels / template instances for `block_nvfp4` (64 weights = 4 sub-blocks of 16, one UE4M3 scale per sub-block):
  - `kernel_mul_mv_nvfp4_f32` and `kernel_mul_mv_id_nvfp4_f32` (dedicated mat-vec, no threadgroup memory),
  - `kernel_mul_mv_ext_nvfp4_f32_r1_{2..5}` (2-5 src1 columns),
  - `kernel_mul_mm_nvfp4_{f32,f16}`, `kernel_mul_mm_id_nvfp4_{f32,f16}` (bf16 tiles when `GGML_METAL_HAS_BF16`, same as MXFP4),
  - few-row MMA (`MUL_MV_MMA_GEN("nvfp4", ...)`) plus the type lists in `ggml-metal-common.cpp` (rows_min 3, same as MXFP4),
  - `kernel_get_rows_nvfp4`.
- Helpers: `ue4m3_to_fp32` (common.h), `dequantize_nvfp4` / `dequantize_nvfp4_t4` (dequantize.h), `fp4_e2m1_dq1/lo4/hi4` (mul_mv.metal).
- Dispatch: N_R0_NVFP4 = 2, N_SG_NVFP4 = 2 in `get_pipeline_mul_mv` and `get_pipeline_mul_mv_id`; NVFP4 added to the mv_ext type list in `ggml_metal_op_mul_mat` (ne11 2..8).

## Why
On master any NVFP4 weight falls back to the CPU backend on Apple Silicon. NVIDIA ships Nemotron-3.5-Bolt-74B-A9B
(`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`) with its Mamba in/out projections in NVFP4 (2.47 GB, 80 tensors), so every
decode step round-trips activations to the CPU for those matmuls.

## Tests (M3 Max 40-core, macOS 27.0, master 2e0d86beb + patch)
- `test-backend-ops test -b MTL0 -o GET_ROWS`: 232/232 (8 NVFP4 cases ran).
- `-o MUL_MAT_ID`: 1104/1104 (73 NVFP4 cases ran).
- `-o MUL_MAT`: 2538/2538 (48 NVFP4 cases ran; 36 NVFP4 x f16-src1 cases are "not supported" because the CPU reference does not support them, same on master).
- NVFP4 is already in `all_types` / `base_types` on master, so no test changes were needed; these cases were skipped on Metal before.
- Perplexity, Bolt-74B NVIDIA GGUF, 8 x 2048 tokens of Python stdlib source (`-fa on -ngl 99`):
  master (NVFP4 on CPU) 2.0061 +/- 0.0389, branch (Metal NVFP4) 2.0061 +/- 0.0389; per-chunk values agree to within 0.0014.

## End to end (M3 Max, `llama-bench -fa 1 -ngl 99 -p 512 -n 64 -r 2`, NVIDIA Bolt GGUF, 2 interleaved runs each)

| build | pp512 t/s | tg64 t/s |
|---|---|---|
| master | 167.3, 167.9 | 9.66, 9.64 |
| master + patch | 445.1, 444.7 | 31.40, 31.33 |

3.25x decode, 2.66x prefill.

## Kernel time (`test-backend-ops perf -o MUL_MAT`, Bolt shapes, two runs, us; temporary perf cases, not in the patch)

| src0 M x K | n | nvfp4 | q8_0 | mxfp4 (master kernel) |
|---|---|---|---|---|
| 18560 x 4096 (ssm_in) | 1 | 103 | 209-211 | 110-147 |
| 4096 x 8192 (ssm_out) | 1 | 34 | 72-75 | 52-58 |
| 18560 x 4096 | 2 | 201-263 | 220-245 | 197-266 |
| 18560 x 4096 | 4 | 154-178 | 219-232 | 171-173 |
| 18560 x 4096 | 8 | 168-174 | 226 | 178-194 |
| 18560 x 4096 | 16 / 32 / 64 | 306-337 / 522-562 / 926-990 | 312-337 / 525-577 / 920-1020 | 303-338 / 533-549 / 939-990 |
| 18560 x 4096 | 512 | 6727-7306 | 6686-7802 | 6851-7401 |

- n = 1 runs at half the time of Q8_0, matching the byte ratio (0.5625 vs 1.0625 bytes per weight). NVFP4 is not faster
  than master's MXFP4 at n >= 4: those paths are the shared mv_ext / MMA / mat-mat templates with the table decode.
- Absolute GB/s from these runs is inflated (e.g. 4096 x 8192 at n = 1: 18.9 MB in 34 us = 555 GB/s, above the
  ~370 GB/s DRAM read rate): test-backend-ops reuses the same weights, so part of src0 stays in the system level cache.
  The ratio to Q8_0 is the meaningful number.
- n = 2 (mv_ext, r1 = 2) is ~2x the n = 1 time for 18560 x 4096, same as MXFP4; the dedicated mat-vec only handles one column.

## Overlap: open PR #20456 ("metal : add NVFP4 quantization support", opened 2026-03-12, last update 2026-06-23, no reviews)
- Targets the old single-file `ggml-metal.metal`; master has since split kernels into `kernels/*.metal`, so it does not apply.
- Decodes the UE4M3 scale through a 128-entry LUT copied into threadgroup memory per threadgroup (plus the 16-entry E2M1 LUT),
  in mat-vec; this branch decodes both with the f16 bit trick and uses no threadgroup memory in the mat-vec.
- Has mv, mv_id, mv_ext, mm, mm_id (half tiles only), get_rows. It has no few-row MMA path (`mul_mv_mma` did not exist then)
  and no `ggml-metal-common.cpp` type-list entries, and no bf16 mm_id variant.
- Reported 25.3 t/s tg128 on an unnamed model; not benchmarked here because it does not apply to current master.
- The user decides whether to build on #20456 (e.g. ask the author) or open a new PR that references it.

## Relation to other open PRs
- #31075 (MXFP4 mat-vec via the fp16 bit trick, opened 2026-10-24) uses the same E2M1 decode idea for MXFP4. This branch adds
  its own `fp4_e2m1_*` helpers (names chosen so they don't collide); if #31075 lands first, the NVFP4 mat-vec should reuse
  whatever helper it adds.
- #30047 (MMA kernels for MUL_MAT_ID): if it lands, NVFP4 needs an entry wherever it adds per-type kernels.

## Open points a reviewer may ask about
- Only measured on M3 Max. N_R0 / N_SG (2 / 2) copied from MXFP4, not swept.
- mv_ext, mat-mat and MMA paths use the table decode (`kvalues_mxfp4_f`), not the bit trick.
- `ue4m3_to_fp32` maps 0x7F (E4M3 NaN) to 0 and ignores the sign bit, same as the CPU `ggml_ue4m3_to_fp32`.
- The mat-vec reads each sub-block's 8 quant bytes as two uint32 loads; this relies on `sizeof(block_nvfp4)` = 36
  (4-byte multiple) and `qs` at offset 4, both fixed by the static_assert / struct in ggml-common.h.
- No tensor-API (M5) specific path; the generic templates are used there. Not tested on M5.
- CONTRIBUTING.md requires an AI-use disclosure in the PR.

## Walkthrough (hunk by hunk)
- `kernels/common.h`, `ue4m3_to_fp32`: UE4M3 has 4 exponent bits (bias 7) and 3 mantissa bits. Shifting the 7 value bits left
  by 7 drops them into an f16's exponent (bits 10-13) and top mantissa bits (7-9). The f16 then equals the E4M3 value times
  2^(7-15) = 2^-8, so `* 256` restores it. Subnormals (exponent 0) come out right too: f16 subnormal m*2^-17, times 256 = m*2^-9,
  which is the E4M3 subnormal. 0x7F (NaN) returns 0 to match the CPU reference.
- `kernels/dequantize.h`, `dequantize_nvfp4(xb, il, reg)`: the generic templates call this with `il` = 0..3, one 16-weight
  sub-block per call. Sub-block `il` is `qs[8*il .. 8*il+7]`; low nibbles are weights 0..7 and high nibbles weights 8..15
  (same layout as `quantize_row_nvfp4_ref`). Rows 0,1 of the 4x4 output take the low nibbles, rows 2,3 the high nibbles.
  `kvalues_mxfp4_f` holds the true E2M1 values (0, 0.5, ... 6), and `ue4m3_to_fp32` the true scale. (The CPU uses the doubled
  int8 table and halves the scale; the product is the same.)
- `dequantize_nvfp4_t4(xb, il, reg)`: 4 weights per call for mv_ext, `il` = 0..15. `s = il/4` is the sub-block, `c = il%4`
  picks bytes 0-3 or 4-7 (`c%2`) and low or high nibble (`c >= 2`). The template is instantiated with 64 weights per block.
- `ggml-metal-impl.h`: N_R0_NVFP4 / N_SG_NVFP4 = 2 rows per simdgroup, 2 simdgroups per threadgroup (MXFP4's values).
- `mul_mv.metal`, mv_ext instances: same as the MXFP4 ones with block size 64 and the `_t4` dequantizer.
- `mul_mv.metal`, `fp4_e2m1_dq1`: an E2M1 nibble is sign + 2 exponent bits + 1 mantissa bit. Placing the 3 magnitude bits at
  f16 bits 9-11 gives exponent e and mantissa bit m, i.e. 2^(e-15)*(1+m/2) = value * 2^-14 for e > 0, and for e = 0, m = 1 the
  f16 subnormal 2^-15 = 0.5 * 2^-14. So every nibble decodes to value * 2^-14 with one shift and a sign select, no table.
  `lo4` / `hi4` decode the low / high nibbles of the 4 bytes of a uint32.
- `kernel_mul_mv_nvfp4_f32_impl`: 32 lanes = 8 blocks x 4 sub-blocks (`ix = tiisg/4` block, `is = tiisg%4` sub-block). Each lane
  loads the 16 matching activations (`yl[0..3]`, reordered so `yl[0]`/`yl[2]` are weights 0-3 / 4-7 and `yl[1]`/`yl[3]` are 8-11 /
  12-15, matching the nibble order), reads its 8 quant bytes as two aligned uint32 (blocks are 36 bytes, `qs` starts at byte 4),
  and accumulates `scale * 16384 * dot` (16384 = 2^14 undoes the decode scaling). The simdgroup steps 8 blocks at a time and
  reduces with `simd_sum`. NR0 rows share each activation load. `ib < ns01` guards the row-stride view case the same way the MXFP4
  kernel does. The `kernel_mul_mv_nvfp4_f32` wrapper and the `mul_mv_id` instance follow the MXFP4 pattern exactly.
- `mul_mm.metal`: `kernel_mul_mm_nvfp4_*` and `kernel_mul_mm_id_nvfp4_*` are the existing templates with `nl = 4` (four 16-weight
  chunks per 64-weight block) and `dequantize_nvfp4`; the mm_id pair copies MXFP4's bf16/half `#if`.
- `mul_mv_mma.metal` + `ggml-metal-common.cpp`: one generator line for the few-row MMA kernels and the two type lists that enable
  them; `rows_min` 3 like MXFP4 (below 3 rows the mat-vec is used).
- `quantize.metal`: `kernel_get_rows_nvfp4` from the generic template (`nl = 4`).
- `ggml-metal-device.cpp`: NVFP4 cases in `get_pipeline_mul_mv` and `get_pipeline_mul_mv_id` set nsg / nr0; no `smem` because the
  kernel uses no threadgroup memory (MXFP4 sets 32 floats for its LUT).
- `ggml-metal-ops.cpp`: NVFP4 joins the types that take mv_ext for 2..8 src1 columns.
- `ggml-metal-device.m`: removes the two NVFP4 exclusions in `supports_op`.
