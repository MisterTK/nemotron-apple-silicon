#!/usr/bin/env python3
"""MLX decode benchmark: context length x concurrent users x KV precision.

Every user gets a prompt of exactly `--ctx` tokens, so per-user caches can be
concatenated along the batch axis with no padding or masking. That keeps batching
independent of KV precision (mlx-lm's BatchGenerator cannot batch QuantizedKVCache).

Prompt modes
  distinct : each user prefills its own full prompt (B=1), caches are then merged.
  shared   : one (ctx - suffix)-token prefix is prefilled once and physically copied
             per user (so KV bytes read per step are still per-user), then each user
             gets a distinct `--suffix`-token tail prefilled as one batch. States and
             MoE routing diverge per user; prefill time is not representative.

Decode is greedy, fixed-length (EOS ignored), pipelined with async_eval like
mlx_lm.generate_step. Each step's wall time is recorded for inter-token gaps.
"""
import argparse, glob, json, os, platform, statistics, sys, time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache, QuantizedKVCache, make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ctx", type=int, nargs="+", default=[4096])
ap.add_argument("--users", type=int, nargs="+", default=[1])
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--trials", type=int, default=1)
ap.add_argument("--kv-bits", type=int, default=0, help="0 = native (bf16/f16) KV")
ap.add_argument("--kv-group", type=int, default=64)
ap.add_argument("--mode", choices=["distinct", "shared"], default="distinct")
ap.add_argument("--suffix", type=int, default=1024)
ap.add_argument("--prefill-step", type=int, default=2048)
ap.add_argument("--gqa-kernel", action="store_true", help="use bench/gqa_decode_attn.py for decode attention")
ap.add_argument("--variants", default="", help="shared mode: comma list of kernel:kvbits, e.g. stock:0,gqa:0,gqa:8")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
ap.add_argument("--show", action="store_true", help="print user 0's generated text")
a = ap.parse_args()

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy, gqa_decode_attn
gqa_decode_attn.install()
gqa_decode_attn.ENABLED = a.gqa_kernel
model, tok = load(a.model)

# ---- fixture: real text (Python stdlib source), deterministic order --------------
LIB = os.path.dirname(os.__file__)
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture_tokens.npy")
_corpus = None


def corpus():
    global _corpus
    if _corpus is None:
        import numpy as np
        fix = FIX.replace(".npy", f".{os.path.basename(a.model.rstrip('/'))}.npy")
        if os.path.exists(fix):
            _corpus = np.load(fix).tolist()
        else:
            files = sorted(glob.glob(os.path.join(LIB, "**", "*.py"), recursive=True))
            ids = []
            for f in files:
                try:
                    txt = open(f, encoding="utf-8").read()
                except Exception:
                    continue
                ids += tok.encode(f"\n# ==== {os.path.relpath(f, LIB)} ====\n{txt}", add_special_tokens=False)
                if len(ids) > 9_000_000:
                    break
            np.save(fix, np.array(ids, dtype=np.int32))
            _corpus = ids
    return _corpus


def head_tail():
    msgs = [{"role": "user", "content": "<<DOC>>"}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre, post = s.split("<<DOC>>")
    head = tok.encode(pre, add_special_tokens=False)
    tail = tok.encode("\n\nExplain in detail what the code above does, file by file." + post, add_special_tokens=False)
    return head, tail


def user_prompt(u, L):
    head, tail = head_tail()
    n = L - len(head) - len(tail)
    c = corpus(); off = (u * 1_000_003) % (len(c) - n)
    return head + c[off:off + n] + tail


# ---- cache helpers ------------------------------------------------------------------
def prefill(cache, ids, B=1):
    """ids: (B, T) array. Prefill all but the last token; return the last token column.
    Runs the backbone only (no LM head: full-vocab logits for every position would be
    wasted memory) in chunks of ~prefill_step tokens total across the batch."""
    body = getattr(model, "backbone", None) or getattr(model, "model", None) or model.language_model.model
    T = ids.shape[1]
    step = max(64, a.prefill_step // B)
    i = 0
    while T - i > 1:
        n = min(step, T - 1 - i)
        body(ids[:, i:i + n], cache=cache)
        mx.eval([c.state for c in cache])
        i += n
        mx.clear_cache()
    return ids[:, -1:]


def quantize_bits(cache, bits):
    return [c.to_quantized(group_size=a.kv_group, bits=bits) if isinstance(c, KVCache) else c for c in cache]


def quantize(cache):
    if not a.kv_bits:
        return cache
    return [c.to_quantized(group_size=a.kv_group, bits=a.kv_bits) if isinstance(c, KVCache) else c for c in cache]


def merge_into(batch, cache):
    """Append a B=1 cache (list of layer caches) to the batch cache, layer by layer."""
    if batch is None:
        return cache
    for bc, c in zip(batch, cache):
        if isinstance(bc, ArraysCache):
            bc.state = [mx.concatenate([x, y]) for x, y in zip(bc.state, c.state)]
        elif isinstance(bc, QuantizedKVCache):
            bc.keys = tuple(mx.concatenate([x, y]) for x, y in zip(bc.keys, c.keys))
            bc.values = tuple(mx.concatenate([x, y]) for x, y in zip(bc.values, c.values))
        else:
            bc.keys = mx.concatenate([bc.keys, c.keys]); bc.values = mx.concatenate([bc.values, c.values])
        mx.eval(bc.state)
    del cache
    mx.clear_cache()
    return batch


def replicate(cache, B):
    for c in cache:
        if isinstance(c, ArraysCache):
            c.state = [mx.concatenate([x] * B) for x in c.state]
        elif isinstance(c, QuantizedKVCache):
            c.keys = tuple(mx.concatenate([x] * B) for x in c.keys)
            c.values = tuple(mx.concatenate([x] * B) for x in c.values)
        else:
            c.keys = mx.concatenate([c.keys] * B); c.values = mx.concatenate([c.values] * B)
        mx.eval(c.state)
    mx.clear_cache()
    return cache


def clone(cache, B, bits):
    """Fresh cache objects holding B physical copies of a B=1 cache, optionally KV-quantized."""
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            n = copy.copy(c); n.cache = [mx.concatenate([x] * B) for x in c.cache]
        else:
            qb = bits if isinstance(c, KVCache) else 0   # sliding-window caches stay native
            q = c.to_quantized(group_size=a.kv_group, bits=qb) if qb else c
            n = copy.copy(q)
            if qb:
                n.keys = tuple(mx.concatenate([x] * B) for x in q.keys)
                n.values = tuple(mx.concatenate([x] * B) for x in q.values)
            else:
                n.keys = mx.concatenate([q.keys] * B); n.values = mx.concatenate([q.values] * B)
        mx.eval(n.state)
        out.append(n)
    mx.clear_cache()
    return out


def trim_kv_buffers(cache):
    # KVCache over-allocates in 256-token steps; slice to offset so B copies are exact
    for c in cache:
        if isinstance(c, KVCache) and c.keys is not None:
            c.keys = c.keys[..., :c.offset, :]; c.values = c.values[..., :c.offset, :]
    return cache


# ---- one run --------------------------------------------------------------------------
def decode(batch, y, B, L, trial, t0, t_prefill, kernel, bits):
    mem_after_prefill = mx.get_active_memory() / 1e9
    gqa_decode_attn.ENABLED = kernel

    def step(y):
        logits = model(y, cache=batch)[:, -1, :]
        return mx.argmax(logits, axis=-1)[:, None]

    # pipelined decode
    gaps = []; out = [[] for _ in range(B)]
    nxt = step(y); mx.async_eval(nxt)
    first = None; last_t = None
    for i in range(a.gen):
        cur = nxt
        if i + 1 < a.gen:
            nxt = step(cur); mx.async_eval(nxt)
        toks = cur.squeeze(1).tolist()
        now = time.perf_counter()
        if first is None:
            first = now
        else:
            gaps.append(now - last_t)
        last_t = now
        for u in range(B):
            out[u].append(toks[u])
    dec_s = time.perf_counter() - first
    steps_ps = (a.gen - 1) / dec_s
    gaps.sort()
    rec = dict(tag=a.tag, gqa_kernel=kernel, model=os.path.basename(a.model.rstrip("/")), mode=a.mode, ctx=L, users=B,
               kv_bits=bits or "native", gen=a.gen, trial=trial,
               prefill_s=round(t_prefill, 2), ttft_s=round(first - t0, 2),
               per_user_tps=round(steps_ps, 2), agg_tps=round(steps_ps * B, 1),
               gap_p50_ms=round(1e3 * gaps[len(gaps) // 2], 2), gap_p95_ms=round(1e3 * gaps[int(len(gaps) * .95)], 2),
               gap_max_ms=round(1e3 * gaps[-1], 2),
               mem_prefill_gb=round(mem_after_prefill, 2), peak_gb=round(mx.get_peak_memory() / 1e9, 2),
               distinct_outputs=len({tuple(o) for o in out}))
    if a.show:
        print("--- user0:", repr(tok.decode(out[0])[:600]), file=sys.stderr)
    return rec


def run(L, B, trial):
    """distinct mode (or B == 1): every user prefills its own full prompt."""
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    batch = None; last = []
    for u in range(B):
        c = make_prompt_cache(model)
        ids = mx.array([user_prompt(u + 97 * trial, L)])
        last.append(prefill(c, ids))
        batch = merge_into(batch, quantize(trim_kv_buffers(c)))
    y = mx.concatenate(last)
    mx.eval([c.state for c in batch])
    rec = decode(batch, y, B, L, trial, t0, time.perf_counter() - t0, a.gqa_kernel, a.kv_bits)
    del batch
    mx.clear_cache()
    return rec


def kv_bytes_per_token(bits):
    if hasattr(model, "language_model"):  # gemma4: only full-attention layers grow with context
        per = sum(2 * l.self_attn.n_kv_heads * l.self_attn.head_dim for l in model.layers
                  if l.self_attn.layer_type == "full_attention")
        return per * 2 if not bits else per * (bits / 8 + 4 / a.kv_group)
    n = 0
    for c in make_prompt_cache(model):
        if isinstance(c, KVCache):
            n += 1
    attn = next(l for l in model.layers if getattr(l, "block_type", "*") == "*" or hasattr(l, "self_attn"))
    mix = getattr(attn, "mixer", None) or attn.self_attn
    nkv = getattr(mix, "num_key_value_heads", None) or mix.n_kv_heads
    hd = getattr(model.args, "head_dim", None) or getattr(mix, "head_dim", None) or model.args.hidden_size // model.args.num_attention_heads
    per = 2 * nkv * hd * n
    return per * 2 if not bits else per * (bits / 8 + 4 / a.kv_group)


def fixed_bytes_per_user():
    # gemma4 sliding-window layers: a RotatingKVCache of `sliding_window` tokens per user, always native
    if not hasattr(model, "language_model"):
        return 0
    # RotatingKVCache keeps up to window + (prefill chunk - 1) tokens after a chunked prefill
    w = model.language_model.args.sliding_window + a.prefill_step
    return sum(2 * l.self_attn.n_kv_heads * l.self_attn.head_dim * 2 * w for l in model.layers
               if l.self_attn.layer_type == "sliding_attention")


def shared_matrix(L, variants, trial=0):
    """Prefill one (L - suffix) prefix once; for each variant and user count, clone it per user,
    batch-prefill a distinct suffix, and decode."""
    S = a.suffix
    budget = mx.device_info()["max_recommended_working_set_size"] / 1e9 * 0.94
    t0 = time.perf_counter()
    base = make_prompt_cache(model)
    shared = user_prompt(97 * trial, L)[:L - S]
    prefill(base, mx.array([shared + [0]]))   # +[0]: prefill() leaves the last token out
    trim_kv_buffers(base)
    mx.eval([c.state for c in base])
    t_prefix = time.perf_counter() - t0
    resident = mx.get_active_memory() / 1e9
    for kernel, bits in variants:
        for B in a.users:
            need = resident + B * (L * kv_bytes_per_token(bits) + fixed_bytes_per_user()) / 1e9 * 1.05 + 1.0
            if need > budget:
                yield dict(tag=a.tag, gqa_kernel=kernel, ctx=L, users=B, kv_bits=bits or "native",
                           status="skipped_mem", est_gb=round(need, 1), budget_gb=round(budget, 1))
                continue
            try:
                mx.reset_peak_memory()
                t1 = time.perf_counter()
                gqa_decode_attn.ENABLED = kernel
                tails = [user_prompt(u + 1 + 97 * trial, L)[L - S:] for u in range(B)]
                if bits:
                    # mlx-lm quantized prefill materializes (q_len x L) scores; prefill each user's
                    # suffix in bf16, then quantize that user's cache and merge it into the batch
                    batch = None; ys = []
                    for u in range(B):
                        c = clone(base, 1, 0)
                        ys.append(prefill(c, mx.array([tails[u]])))
                        batch = merge_into(batch, quantize_bits(trim_kv_buffers(c), bits))
                    y = mx.concatenate(ys)
                else:
                    batch = clone(base, B, 0)
                    y = prefill(batch, mx.array(tails), B)
                mx.eval([c.state for c in batch])
                rec = decode(batch, y, B, L, trial, t1, time.perf_counter() - t1, kernel, bits)
                rec["prefix_prefill_s"] = round(t_prefix, 1)
                del batch, y
                mx.clear_cache()
            except RuntimeError as e:  # Metal OOM: record and move on
                batch = y = None; mx.clear_cache()
                yield dict(tag=a.tag, gqa_kernel=kernel, ctx=L, users=B, kv_bits=bits or "native",
                           status="fail", err=str(e)[:200], est_gb=round(need, 1))
                continue
            yield rec


meta = dict(machine=platform.machine(), mlx=mx.__version__, device=mx.device_info().get("device_name"))
def emit(r):
    r.update(meta)
    print(json.dumps(r), flush=True)
    if a.out:
        open(a.out, "a").write(json.dumps(r) + "\n")


variants = [(v.split(":")[0] == "gqa", int(v.split(":")[1])) for v in a.variants.split(",")] if a.variants \
    else [(a.gqa_kernel, a.kv_bits)]
for L in a.ctx:
    if a.mode == "shared":
        for t in range(a.trials):
            for r in shared_matrix(L, variants, t):
                emit(r)
        continue
    for B in a.users:
        for t in range(a.trials):
            try:
                r = run(L, B, t)
            except Exception as e:  # OOM etc.
                r = dict(tag=a.tag, ctx=L, users=B, kv_bits=a.kv_bits or "native", status="fail", err=str(e)[:200])
                mx.clear_cache()
            emit(r)
