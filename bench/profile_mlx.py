#!/usr/bin/env python3
"""Per-component decode profile for MLX Nemotron-H / Qwen3-MoE at B users, context L.

Each component is chained on its own output (x = f(x)) N times and evaluated once,
so ops are strictly serial and per-dispatch commit overhead is amortized. Bytes are
counted from the actual (quantized) arrays each component must read; GB/s is that
estimate over measured time, i.e. an *estimated* DRAM rate, not a counter reading.
"""
import argparse, json, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache, ArraysCache

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--users", type=int, default=1)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--kv-bits", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()

model, tok = load(a.model)
B, L, N = a.users, a.ctx, a.n
layers = model.layers
H = model.args.hidden_size


def nbytes(m):
    return sum(v.nbytes for _, v in nn.utils.tree_flatten(m.parameters()))


def bench(f, x):
    for _ in range(3):
        x0 = f(x)
    mx.eval(x0)
    t0 = time.perf_counter()
    y = x
    for _ in range(N):
        y = f(y)
    mx.eval(y)
    return (time.perf_counter() - t0) / N


x = (mx.random.normal((B, 1, H)) * 0.1).astype(mx.bfloat16)
mx.eval(x)
res = {}


def record(name, count, t, byts):
    res[name] = dict(count=count, us=round(t * 1e6, 1), total_ms=round(count * t * 1e3, 3),
                     est_gb=round(byts / 1e9, 4), est_gbps=round(byts / t / 1e9, 1) if byts else None)


# full decode step with real caches at context L (random KV / zero SSM state)
cache = make_prompt_cache(model)
cache_layers = [i for i, l in enumerate(layers) if getattr(l, "block_type", "*") in "M*"]
cidx = {li: ci for ci, li in enumerate(cache_layers)}
for c, l in zip(cache, [layers[i] for i in cache_layers]):
    if isinstance(c, KVCache):
        nkv, hd = l.mixer.num_key_value_heads, l.mixer.head_dim
        c.keys = (mx.random.normal((B, nkv, L, hd)) * 0.5).astype(mx.bfloat16)
        c.values = (mx.random.normal((B, nkv, L, hd)) * 0.5).astype(mx.bfloat16)
        c.offset = L
if a.kv_bits:
    cache = [c.to_quantized(group_size=64, bits=a.kv_bits) if isinstance(c, KVCache) else c for c in cache]
mx.eval([c.state for c in cache if not isinstance(c, ArraysCache)])
ids = mx.zeros((B, 1), dtype=mx.int32)


def full_step(_):
    global ids
    logits = model(ids, cache=cache)
    ids = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    return ids

# warm the SSM states so they exist, then time steps (each step appends 1 KV token)
for _ in range(3):
    mx.eval(full_step(None))
t0 = time.perf_counter()
for _ in range(30):
    mx.async_eval(full_step(None))
mx.eval(ids)
step = (time.perf_counter() - t0) / 30
res["_full_step"] = dict(ms=round(step * 1e3, 3), tok_s_per_user=round(1 / step, 1), agg_tok_s=round(B / step, 1))

kinds = {}
for i, l in enumerate(layers):
    kinds.setdefault(l.block_type, []).append(i)
print("layer pattern:", {k: len(v) for k, v in kinds.items()})

# Mamba blocks (with their own cache so state read/write is real)
if "M" in kinds:
    l = layers[kinds["M"][0]]
    c = cache[cidx[kinds["M"][0]]]
    mix = l.mixer
    st_bytes = sum(s.nbytes for s in c.state if s is not None)
    t = bench(lambda y: l(y, cache=c), x)
    record("mamba_block", len(kinds["M"]), t, nbytes(l) + 2 * st_bytes)
    t = bench(lambda y: y + mix.out_proj(mix.in_proj(y)[..., :mix.intermediate_size]), x)
    record("  mamba_in+out_proj", len(kinds["M"]), t, nbytes(mix.in_proj) + nbytes(mix.out_proj))
    record("  (ssm state bytes/layer, r+w)", 0, 1, 2 * st_bytes)

# MoE blocks
if "E" in kinds:
    l = layers[kinds["E"][0]]; moe = l.mixer
    E, K = moe.config.n_routed_experts, moe.num_experts_per_tok
    expert_b = nbytes(moe.switch_mlp)
    # experts touched by B users (upper bound B*K, capped at E); estimate with uniform routing
    touched = E * (1 - (1 - K / E) ** B)
    t = bench(lambda y: l(y), x)
    record("moe_block", len(kinds["E"]), t, nbytes(l) - expert_b + expert_b * touched / E)
    t = bench(lambda y: y + moe.shared_experts(y), x)
    record("  moe_shared_expert", len(kinds["E"]), t, nbytes(moe.shared_experts))

    def routed(y):
        inds, scores = moe.gate(y)
        z = moe.switch_mlp(y, inds)
        return y + (z * scores[..., None]).sum(axis=-2).astype(y.dtype)
    t = bench(routed, x)
    record("  moe_gate+routed", len(kinds["E"]), t, nbytes(moe.gate) + expert_b * touched / E)
    inds, _ = moe.gate(x); mx.eval(inds)
    t = bench(lambda y: y + moe.switch_mlp(y, inds).sum(axis=-2).astype(y.dtype), x)
    record("  moe_routed_only(fixed idx)", len(kinds["E"]), t, expert_b * len(set(inds.flatten().tolist())) / E)

# Qwen-style blocks (attention + MoE in one layer)
if hasattr(layers[0], "self_attn"):
    l = layers[0]
    print("qwen-style layers:", len(layers))

# Attention blocks at context L
if "*" in kinds:
    idx = kinds["*"][0]; l = layers[idx]; c = cache[cidx[idx]]
    kv_b = sum(v.nbytes for _, v in nn.utils.tree_flatten(c.state))
    off0 = c.offset

    def attn(y):
        c.offset = off0  # keep context fixed at L (overwrite the same slot)
        return l(y, cache=c)
    t = bench(attn, x)
    record("attn_block", len(kinds["*"]), t, nbytes(l) + kv_b)

# LM head
head = model.lm_head
t = bench(lambda y: y + head(model.backbone.norm_f(y))[..., :H], x)
record("lm_head", 1, t, nbytes(head))

tot = sum(v["total_ms"] for k, v in res.items() if not k.startswith("_") and not k.startswith("  "))
res["_sum_of_blocks_ms"] = round(tot, 3)
res["_unaccounted_ms"] = round(res["_full_step"]["ms"] - tot, 3)
meta = dict(model=a.model.rstrip("/").split("/")[-1], ctx=L, users=B, kv_bits=a.kv_bits or "native")
for k, v in res.items():
    print(f"{k:34s} {v}")
if a.out:
    open(a.out, "a").write(json.dumps(dict(meta, **res)) + "\n")
