# patches

`apply.sh fork|upstream|master [dir]` checks out the pinned llama.cpp commit, applies the set and builds it. The four `upstream-*` patches apply together on 418f905a4 (verified 2026-10-29).

Prefixes: `fork-` the Bolt fork diff, `upstream-` branches prepared for llama.cpp, `tool-` benchmark and profiling aids, `experiment-` measured changes not kept.

## Fork

`fork-llama.cpp-bolt-metal-opt.patch` applies on llama.cpp 620d2225c and is the whole `llama.cpp-bolt` worktree: 704366d76 (top-9 MoE kernel), PR #31075, Metal NVFP4, SSM scan/conv kernels and fusions, sigmoid routing fusion, split-K mat-mat, IQ2_XXS_U3, the batched-bench fixes. Regenerated 2026-10-29. See REPORT.md.

## Upstream branches

Worktrees under `pr/` (split-K: `llama.cpp-upstream-splitk`). The batched-bench fix (#31389) and the TOPK_MOE fusion (#31390, draft until #31389 merges; new contributors get one open PR) are on master 418f905a4. nvfp4 and ssm-fusions were rebased onto 418f905a4 on 2026-10-29 and are staged, uncommitted. Split-K is still on 2e0d86beb. Each has `upstream-<branch>.patch` and `upstream-<branch>.notes.md` (facts, numbers, tests, overlap, reviewer questions, walkthrough).
Configure fresh builds with `-DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk` when the Command Line Tools SDK fails to link.

| # | branch | size | headline (M3 Max) | overlap |
|---|---|---|---|---|
| 1 | batched-bench-pps-recurrent | 1 file, +10/-5 | master `-pps` fails on recurrent models ("inconsistent sequence positions"); fixed | none |
| 2 | metal-topk-moe-sigmoid | 6 files, +180/-29 | Nano-30B tg +3% (78 -> 80.5); TOPK_MOE 736/736 | none on Metal (CUDA, Vulkan have it) |
| 3 | metal-nvfp4 | 11 files, +153/-3 | NVIDIA Bolt GGUF: tg 9.65 -> 31.4, pp 167 -> 445; PPL 2.0061 both | #20456 (stale, old layout) |
| 4 | metal-ssm-fusions | 8 files, +753/-22 | Bolt (+nvfp4) 1 user +20%, 16 users 61 -> 144; Nano-30B tg +13% | #25533 (decode only, no cache fusion); builds on merged #28948 |
| 5 | metal-mul-mm-split-k | 7 files, +95/-8 | Qwen3-8B +9-13% at 24-32 seqs | conflicts with draft #27441; overlaps #29743 |

Suggested order: 1 and 2 first (small, self-contained). Open an issue before 3 and 4, as AGENTS.md asks for new features. Comment on #27441 before 5.

Not proposed: MUL_MAT_ID tokens-per-expert selection (#30047); full-threadgroup unary/GLU/binary and the MXFP4 mat-vec change (#31075); IQ2_XXS_U3 and the f16 SSM state (fork only). Candidate not yet prepared: MoE-reduce fusion limit 8 -> 12 experts for top-9 models.

llama.cpp policy: the contributor writes the PR descriptions, commit messages, issues and reviewer replies, and discloses AI use in the code.

## Tools

- `tool-batched-bench-pps-reuse.patch`: branch 1 plus `BB_REUSE_PP=1`, which prefills the shared prompt once and copies it to every sequence (used by `bench/scenarios.py`; matched fresh prefill within 0.3% on Bolt). Applied in `llama.cpp-upstream-base`.
- `tool-metal-op-skip-profiling.snippet.cpp`: `GGML_METAL_SKIP=<op>` drops every dispatch of an op type; the tok/s delta is that op's real in-graph cost. Paste into `ggml-metal-ops.cpp`.

## Experiments (measured, not kept)

- `experiment-ssm-f16-state.patch` (on the fork): f16 recurrent state with stochastic rounding and a vectorized decode scan. No end-to-end gain; REPORT.md sixth pass.
- `experiment-mamba2-skip-cont.patch` + notes: skip `ggml_cont` of contiguous views in mamba-base. No measurable gain.
