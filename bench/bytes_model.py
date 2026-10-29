#!/usr/bin/env python3
"""Bytes moved per decode step: weights from the GGUF tensor table, KV/state from the architecture.

Routed experts touched by N users assume uniform routing: E * (1 - (1 - k/E)^N) per layer (an
upper bound; real routing is skewed, so fewer distinct experts are read).
"""
import os, sys, re, functools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama.cpp", "gguf-py"))
from gguf import GGUFReader

MiB = 2**20
KV_PER_TOKEN_F16 = {  # bytes per token per user (K + V, f16)
    "bolt": 8 * 2 * 128 * 2 * 2,          # 8 attention layers, 2 KV heads, d 128
    "qwen": 48 * 4 * 128 * 2 * 2,         # 48 layers, 4 KV heads, d 128
    "gemma": 5 * 2 * 512 * 2 * 2,         # 5 global layers, 2 KV heads, d 512
}
SWA_PER_TOKEN_F16 = {"gemma": 25 * 8 * 256 * 2 * 2}  # 25 sliding layers, window 1024
SWA_WINDOW = 1024
STATE_BYTES = {"bolt": (160 + 4.69) * MiB}  # SSM f32 state + conv state per user (measured RS buffer)
KV_SCALE = {"f16": 1.0, "q8_0": 34 / 64}


@functools.lru_cache(None)
def weights(path):
    r = GGUFReader(path)
    f = {k: v for k, v in r.fields.items()}
    def arr(key):
        fld = next(v for k, v in f.items() if k.endswith(key))
        return [fld.parts[i].tolist()[0] for i in fld.data]
    used = arr("expert_used_count")
    names = {t.name for t in r.tensors}
    dense, experts = 0, []  # experts: (layer bytes, n_expert, top_k)
    per_layer = {}
    for t in r.tensors:
        if "_exps" in t.name:
            layer = int(re.match(r"blk\.(\d+)\.", t.name).group(1))
            b, e = per_layer.get(layer, (0, int(t.shape[-1])))
            per_layer[layer] = (b + int(t.n_bytes), e)
        elif t.name == "token_embd.weight":
            if "output.weight" not in names:  # tied LM head reads the whole table
                dense += int(t.n_bytes)
        else:
            dense += int(t.n_bytes)
    for layer, (b, e) in per_layer.items():
        k = used[layer] if len(used) > 1 else used[0]
        experts.append((b, e, k))
    return dense, experts


def family(path):
    p = path.lower()
    return "bolt" if "bolt" in p else "qwen" if "qwen" in p else "gemma"


def step_bytes(path, users, depth, kv="f16"):
    fam = family(path)
    dense, experts = weights(path)
    w = dense + sum(b * (1 - (1 - k / e) ** users) for b, e, k in experts)
    kvb = users * depth * KV_PER_TOKEN_F16[fam] * KV_SCALE[kv]
    kvb += users * min(depth, SWA_WINDOW) * SWA_PER_TOKEN_F16.get(fam, 0) * KV_SCALE[kv]
    st = users * 2 * STATE_BYTES.get(fam, 0)  # read + write
    return dict(weights=w, kv=kvb, state=st, total=w + kvb + st)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d, ex = weights(path)
        print(path, f"dense {d / MiB:.0f} MiB, experts {sum(b for b, _, _ in ex) / MiB:.0f} MiB over {len(ex)} layers,"
              f" E {sorted({e for _, e, _ in ex})}, k {sorted({k for _, _, k in ex})}")
        for n in (1, 8, 32):
            for depth in (8192, 131072):
                s = step_bytes(path, n, depth)
                print(f"  users {n:3d} depth {depth:6d}: " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in s.items()))
