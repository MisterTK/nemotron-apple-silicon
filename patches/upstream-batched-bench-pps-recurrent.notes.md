# batched-bench: -pps with recurrent / hybrid models

Branch: `batched-bench-pps-recurrent` in worktree `pr/batched-bench-pps-recurrent`, on origin/master 2e0d86beb, uncommitted.
Patch: `patches/upstream-batched-bench-pps-recurrent.patch` (tools only).
`tools/batched-bench` is unchanged between 620d2225c and 2e0d86beb; the fork patch applied as is.

## What is broken on master
With `-pps` and the default non-unified KV (no `-kvu`), after the shared prompt is copied to the other
sequences, batched-bench decodes one dummy token on seq 0 at position `pp` to apply the copy, then calls
`llama_memory_seq_rm(mem, 0, pp, -1)` to drop it and starts text generation at position `pp` again.
Recurrent memory cannot drop the last token: `llama_memory_recurrent::seq_rm` only rolls back through
per-token snapshots (`n_rs_seq`), which batched-bench does not enable, so it returns false. The return value
is ignored, the state stays at position `pp`, and the first generation batch is rejected.

Master, Nemotron-Nano-9B-v2 Q4_K_M (hybrid Mamba-2), `-c 4096 -fa on -ngl 99 -pps -npp 512 -ntg 32 -npl 1,2,4`,
exit code 1, before any row is printed (fails at npl = 1 too):

```
E init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 512
 - the tokens for sequence 0 in the input batch have a starting position of Y = 512
 it is required that the sequence positions remain consecutive: Y = X + 1
E decode: failed to initialize batch
E llama_batched_bench: llama_decode() failed
```

With `-kvu` (unified KV) master works, because that path has no dummy token.

## What changes
- The dummy token is decoded on every sequence (positions `pp`), and kept instead of removed.
- Text generation starts at `tg_pos0 = pp + 1` when the dummy token was decoded, `pp` otherwise.
- `n_ctx_req` counts the extra token per sequence: `pl*(tg + 1)` instead of `pl*tg` in the shared-prompt case
  (so a point is not skipped or overflowed by one cell per sequence).

## Results (M3 Max, Metal, 2e0d86beb vs branch)

Nemotron-Nano-9B-v2 Q4_K_M, `-pps -npp 512 -ntg 32`, S_TG t/s:

| build | npl 1 | npl 2 | npl 4 |
|---|---|---|---|
| master, default KV | fails | fails | fails |
| master, `-kvu` | 40.0 | 43.5 | 59.3 |
| branch, default KV (2 runs) | 39.3 / 39.8 | 35.4 / 43.4 | 58.0 / 59.8 |

The branch now matches master's `-kvu` numbers. (The npl 2 run-to-run spread is the model, not the tool: master
`-kvu` shows the same 2-sequence point.)

Qwen3-8B Q4_K_M (non-recurrent), same flags, `-c 8192`, two interleaved runs each, S_TG t/s:

| build | npl 1 | npl 2 | npl 4 |
|---|---|---|---|
| master | 63.9 / 63.5 | 96.0 / 96.5 | 169.5 / 168.9 |
| branch | 63.4 / 63.0 | 96.5 / 96.3 | 170.0 / 173.7 |

Same within noise. The N_KV column grows by one per sequence (544 -> 545, 1088 -> 1090, 2176 -> 2180),
and each sequence generates its tokens one position later.

## Walkthrough
1. `n_ctx_req`: in the shared-prompt case each sequence now holds `pp` (copied) + 1 (dummy) + `tg` tokens
   when KV is not unified, so the check adds `pl` cells. With `-kvu` the formula over-counts by `pl` cells,
   which only matters if `-c` is set exactly to the requirement.
2. `int tg_pos0 = pp;` the first generation position; stays `pp` for `-kvu` and for non-shared prompts.
3. Dummy-token loop: one token per sequence `j` at position `pp`, instead of only seq 0. Every sequence's
   copy is applied by the same decode, and every sequence advances by one token, so positions stay
   consistent across sequences.
4. `llama_memory_seq_rm(mem, 0, pp, -1)` removed, `tg_pos0 = pp + 1` set instead. This is the line that
   silently failed for recurrent memory.
5. Both generation loops (`is_tg_separate` and interleaved) use `tg_pos0 + i` instead of `pp + i`.

## Open points a reviewer may ask about
- Alternative: keep master's behaviour for non-recurrent models and only skip `seq_rm` when it fails
  (check its return value). That keeps N_KV unchanged for attention-only models but has two code paths.
- Timing: the dummy decode is outside the timed TG region in both versions; the branch decodes `pl` tokens
  there instead of 1.
- Alternative: enable snapshots (`n_rs_seq`) so `seq_rm` can roll back; heavier and changes memory use.
