# metal: sigmoid + bias gating in the TOPK_MOE fusion

Branch: `metal-topk-moe-sigmoid` in worktree `pr/metal-topk-moe-sigmoid`, on origin/master 2e0d86beb, uncommitted.
Patch: `patches/upstream-metal-topk-moe-sigmoid.patch` (see UPSTREAM-INDEX.md).
Build note: cmake needs `-DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk` (the CommandLineTools SDK 27.0 .tbd files fail to link).

## What changes
- The Metal TOPK_MOE fusion matched only softmax routing. It now also matches the DeepSeek-V3 style chain that `build_moe_ffn` emits for sigmoid gating with `exp_probs_b`:
  - the chain: `SIGMOID -> ADD(bias) -> ARGSORT(top-k) -> GET_ROWS(unbiased probs) -> SUM_ROWS -> CLAMP -> DIV [-> SCALE]`;
  - the effect: one kernel per MoE layer instead of 7-8 dispatches.
- **Kernel** (`kernel_topk_moe_f32`): new function constant `FC_topk_moe_sigmoid` and a `bias` buffer (index 4).
  - **Sigmoid path:** experts are ranked by `sigmoid(x) + bias`. The weight that is gathered (and normalized/scaled) is the unbiased `sigmoid(x)`, as in the graph.
  - **Softmax path:** unchanged apart from the selection score (`sel`) being a copy of the softmax output.
- Only the norm variants are matched (with and without scale). Sigmoid without bias or without norm still runs unfused.

## Why
Nemotron-H MoE (Nemotron-3-Nano-30B-A3B, Nemotron-3.5-Bolt), GLM and DeepSeek-V3 style models use sigmoid + bias routing.
- **Metal before this change:** the routing ran as separate small kernels for every MoE layer and token step.
- **Other backends:** Vulkan (#18295, "extend topk_moe to handle sigmoid w/exp_probs_b for nemotron") and CUDA already fuse this.

## Tests (M3 Max, branch)
- `test-backend-ops test -o TOPK_MOE -b MTL0`: 736/736 passed.
  - Master's existing loop already covers `GATING_FUNC_SIGMOID` with bias, norm and scale. Two shapes were added to it, `{128, 22}` top-6 (Nemotron-3-Nano) and `{496, 22}` top-9 (Nemotron-3.5-Bolt), for all gating functions.
  - The `sigmoid=1` pipelines get compiled during the run, so the fused path is exercised.
- Fusion fires in the model (`GGML_METAL_FUSION_DEBUG=1`, Nemotron-3-Nano-30B-A3B Q4_K_M, `llama-bench -n 16`):
  - **branch:** `UNARY+ADD+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV+SCALE: 391` (23 MoE layers x 17 graphs);
  - **master:** no TOPK_MOE entry.
- Perplexity (Nemotron-3-Nano-30B-A3B Q4_K_M, `-c 2048 --chunks 16`, Python-source text): master 2.5123 +/- 0.0402, branch 2.5132 +/- 0.0402.

## End to end, M3 Max 40-core, Nemotron-3-Nano-30B-A3B Q4_K_M, master vs branch, interleaved runs
`llama-bench -fa 1 -p 512 -n 128 -r 1`, 3 rounds:

| | round 1 | round 2 | round 3 |
|---|---|---|---|
| tg128 master | 78.39 | 78.28 | 78.47 |
| tg128 branch | 82.73 | 80.49 | 80.43 |
| pp512 master | 1237.8 | 1237.8 | 1233.2 |
| pp512 branch | 1240.2 | 1234.4 | 1237.9 |

tg128: +2.6% median (+5.4% best vs best). pp512 unchanged.

`llama-batched-bench -npp 512 -ntg 64`, aggregate S_TG tok/s, 3 interleaved rounds (min-max):

| sequences | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| master | 89.9-93.9 | 116.5-119.5 | 138.5-140.7 | 147.2-149.7 |
| branch | 88.8-93.0 | 116.5-120.9 | 134.6-141.8 | 147.5-151.4 |

Multi-sequence decode is within run-to-run noise. The kernel runs one 32-thread threadgroup per token, so the saving is a fixed cost per layer per step that matters most at batch 1.

## Overlap check (gh search, 2026-10-28)
- No open Metal PR for sigmoid routing in TOPK_MOE.
- Related, merged:
  - #28948 "metal : add MoE and SSM_CONV fusion optimizations" (the softmax TOPK_MOE fusion this extends);
  - #18295 (Vulkan sigmoid + exp_probs_b);
  - #25217 (SYCL fused top-k MoE).
- Open and unrelated: #30680 (metal SCALE -> SIGMOID -> SCALE chain for qwen4exp).

## Walkthrough
1. `kernels/argsort.metal`
   - **Function constant:** `FC_topk_moe_sigmoid` added at `FC_TOPK_MOE + 3`.
   - **Kernel signature:** gains `device const float * bias` at buffer index 4.
   - **Two arrays per lane:** `wt` is the routing weight, `sel` is the ranking score.
     - Sigmoid path: `wt = 1/(1+exp(-x))` and `sel = wt + bias[expert]`. Lanes past `n_expert` get `sel = -inf`.
     - Softmax path: the existing softmax block, now indented inside `else`, followed by `sel = wt`.
   - **Top-k loop:** it now compares and masks `sel` but carries `best_wt` along through the simd shuffle. The output weight and the norm sum use `best_wt`. For softmax `sel == wt`, so results there do not change.
2. `ggml-metal-device.cpp/.h`
   - `ggml_metal_library_get_pipeline_topk_moe` takes `bool sigmoid`.
   - It adds `_sigmoid=%d` to the pipeline name and sets the function constant.
3. `ggml-metal-ops.cpp`
   - **`ggml_metal_op_unary`:** for a SIGMOID node with fusion enabled, it asks `can_fuse` for TOPK_MOE and hands off to `ggml_metal_op_topk_moe`. This mirrors how `ggml_metal_op_soft_max` does it. The `op->op == GGML_OP_UNARY` guard is needed because this function also encodes non-UNARY ops, and `ggml_get_unary_op` asserts on them.
   - **`ggml_metal_op_topk_moe`:** detects the sigmoid chain from the first node's op.
     - Node offsets shift by one because ADD comes before ARGSORT.
     - The bias is the ADD operand that is not the sigmoid.
     - With norm is always true. With scale means 8 fused nodes.
     - The bias is bound at buffer 4. For softmax the logits buffer is bound there as an unused placeholder.
     - The debug log names the chain.
4. `ggml-metal-fusion.cpp`
   - **Two raw op sequences:** `ops_topk_moe_sigmoid_norm` and `_norm_scale`, including the RESHAPE/VIEW nodes the kernel elides.
   - **`ggml_metal_fusion_check_topk_moe_sigmoid`:** follows the softmax check's structure.
     - It walks the raw nodes and handles the trailing RESHAPE of the no-scale variant.
     - Types: f32 logits, bias and output; i32 ids.
     - Limits: the same expert limits as softmax.
     - Shapes: the bias must be contiguous with `n_expert` elements.
     - Links: every src link of the chain is checked, so only this exact graph matches.
     - Outputs: the ids (ARGSORT view) and the final weights are the only outputs, enforced by `ggml_can_fuse_subgraph_ext`.
   - **Table entries:** two, with fused output index 2 (the ARGSORT in the non-empty sequence).
5. `tests/test-backend-ops.cpp`: two TOPK_MOE shapes added to the existing loop (Nemotron-3-Nano 128/6, Nemotron-3.5-Bolt 496/9).

## Open points a reviewer may ask about
- **What matches:** only sigmoid + bias + norm (+ scale), the Nemotron-H / DeepSeek-V3 form.
  - Not matched: sigmoid without bias, or without norm (Llama 4 style sigmoid after top-k is a different graph).
  - Not matched: grouped routing (`n_expert_groups > 1`); the group masking runs before the top-k.
- **NaN logits:** mapped to `-FLT_MAX` before the sigmoid (as in the softmax path), so they get weight 0. The unfused graph would propagate NaN.
- **Numerics:** the sigmoid is computed in the kernel as `1/(1+exp(-x))` in fp32. Perplexity agrees within 0.04%.
- **Measured hardware:** only the M3 Max.
- **Not included:** `GGML_METAL_MOE_REDUCE_MAX_EXPERTS` 8 -> 12 (the MUL+ADD reduce fusion for top-9/top-10 models such as Bolt) is in the fork but is a separate change.
- CONTRIBUTING.md requires an AI-use disclosure in the PR.
