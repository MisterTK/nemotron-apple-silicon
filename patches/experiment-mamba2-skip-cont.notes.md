# llama: skip ggml_cont of already-contiguous views in mamba-base

Branch: `mamba2-skip-cont` in worktree `pr/mamba2-skip-cont`, on origin/master 2e0d86beb, uncommitted.
Patch: `patches/experiment-mamba2-skip-cont.patch` (1 file, src/models/mamba-base.cpp, +4/-3).

## What changes
- `build_mamba_layer` and `build_mamba2_layer`: `ggml_cont(z)` -> `ggml_is_contiguous(z) ? z : ggml_cont(z)` before
  `ggml_swiglu_split`, and the same for `dt` before the `ssm_dt_b` add. At decode (one token per sequence) these views
  are already contiguous, so the copy node is dropped from the graph.

## Measured (M3 Max, best of 2)
- No measurable end-to-end effect on its own: Nano-30B tg128 77.7 -> 78.0, Nano-9B 40.2 -> 40.2 (master vs branch);
  Bolt (on master + metal-nvfp4) tg128 31.7 -> 31.2, i.e. noise.
- In combination with `metal-ssm-fusions` on Bolt it also adds nothing: scan+conv 38.1 tg128 /
  144.0 at 16 seqs, scan+conv+this 37.5 / 143.5. In the fork it was kept for the graph cleanup, not for speed.
  Probably not worth a PR on measured grounds.
- Perplexity, Nemotron-Nano-9B-v2 Q4_K_M, 16 x 512, `-ub 8`: 3.7133 master, 3.7133 branch.
- Backend-independent (graph change), so it touches every backend; test-backend-ops does not cover src/.

## Open points a reviewer may ask about
- Whether any backend's swiglu_split / add requires its own contiguous copy (the CPU and Metal ops accept the view;
  other backends not checked).
- Small value on its own; may fit better folded into a larger Mamba change.

## Walkthrough
1. Line ~138 (Mamba-1 path) and ~286 (Mamba-2 path): z is a view into the in-projection output; contiguous when the
   ubatch has one token per sequence. Skip the cont in that case.
2. Line ~248: same for dt, with a comment.
