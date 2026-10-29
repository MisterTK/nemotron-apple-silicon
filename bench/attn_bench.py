#!/usr/bin/env python3
"""Decode-attention microbenchmark at Nemotron-3-Nano shapes (32 q heads, 2 kv heads, D=128).

Variants: stock mx.fast sdpa; GQA-folded matmul (K/V streamed once for all 16 q heads
of a kv head); mlx-lm quantized-KV attention (8/4-bit, also GQA-folded matmuls).
Reports time per layer call and effective GB/s over the KV bytes that must be read.
"""
import argparse, time
import mlx.core as mx
from mlx_lm.models.base import quantized_scaled_dot_product_attention
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gqa_decode_attn import gqa_decode_attention

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 32768, 131072])
ap.add_argument("--users", type=int, nargs="+", default=[1])
ap.add_argument("--hq", type=int, default=32)
ap.add_argument("--hkv", type=int, default=2)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--variants", default="sdpa,fold,q8,q4")
a = ap.parse_args()
scale = a.d ** -0.5


def bench(f, n=a.n):
    mx.eval(f()); mx.eval(f())
    t0 = time.perf_counter()
    outs = [f() for _ in range(n)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / n


def fold(q, k, v):
    B, Hq, _, D = q.shape; Hk = k.shape[1]
    qf = q.reshape(B, Hk, Hq // Hk, D)
    s = (qf * scale) @ k.swapaxes(-1, -2)                    # (B, Hk, G, L)
    p = mx.softmax(s.astype(mx.float32), axis=-1, precise=True).astype(v.dtype)
    return (p @ v).reshape(B, Hq, 1, D)


for B in a.users:
    for L in a.ctx:
        q = mx.random.normal((B, a.hq, 1, a.d)).astype(mx.bfloat16)
        k = (mx.random.normal((B, a.hkv, L, a.d)) * 0.5).astype(mx.bfloat16)
        v = mx.random.normal((B, a.hkv, L, a.d)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        kv_bytes = k.nbytes + v.nbytes
        row = [f"B={B:<3} L={L:<7}"]
        for var in a.variants.split(","):
            if var == "sdpa":
                f = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); nb = kv_bytes
            elif var == "fold":
                f = lambda: fold(q, k, v); nb = kv_bytes
            elif var in ("q8", "q4"):
                bits = int(var[1:])
                kq = mx.quantize(k, group_size=64, bits=bits); vq = mx.quantize(v, group_size=64, bits=bits)
                mx.eval(kq, vq)
                nb = sum(x.nbytes for x in kq + vq)
                f = (lambda kq=kq, vq=vq, bits=bits: quantized_scaled_dot_product_attention(
                    q, kq, vq, scale=scale, mask=None, group_size=64, bits=bits))
            elif var == "gqa":
                f = lambda: gqa_decode_attention(q, k, v, scale); nb = kv_bytes
            elif var in ("gqa8", "gqa4"):
                bits = int(var[3:])
                kq = mx.quantize(k, group_size=64, bits=bits); vq = mx.quantize(v, group_size=64, bits=bits)
                mx.eval(kq, vq); nb = sum(x.nbytes for x in kq + vq)
                f = (lambda kq=kq, vq=vq, bits=bits: gqa_decode_attention(q, kq, vq, scale, quant=(64, bits)))
            out = f(); err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
            t = bench(f)
            row.append(f"{var}: {t*1e6:8.1f}us {nb/t/1e9:6.1f}GB/s err={err:.3f}")
        print(" | ".join(row), flush=True)
