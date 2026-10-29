#!/usr/bin/env python3
"""Copy a GGUF, re-encoding IQ2_XXS tensors as IQ2_XXS_U3 (all else byte-identical).

IQ2_XXS stores 8 weights per byte-sized index into a 256-entry E8 grid; Metal decodes it with threadgroup-memory
lookups, which caps its mat-vec at ~650 G weights/s (compute-bound, ~150 GB/s in Bolt's graph). IQ2_XXS_U3 stores
each weight's level as a 3-bit field: the 6 IQ2_XXS levels +-{8,25,43} map to 17*(n - 2.5), n = 0..5 (within +-0.5
of the original), so the kernel decodes with one mask + one FMA per weight. 3.25 bpw (104 bytes / 256) vs 2.06.
Requires the llama.cpp-bolt build that defines GGML_TYPE_IQ2_XXS_U3 (= 43).

usage: python3 -I bench/gguf_iq2u3.py IN.gguf OUT.gguf [--match REGEX]
"""
import argparse, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("src_path"); ap.add_argument("dst_path")
ap.add_argument("--match", default="", help="only re-encode IQ2_XXS tensors whose name matches this regex")
ap.add_argument("--gguf-py", default=os.path.join(os.path.dirname(__file__), "..", "llama.cpp-bolt", "gguf-py"))
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
import numpy as np
import gguf
from gguf import quants

SRC, DST = gguf.GGMLQuantizationType.IQ2_XXS, gguf.GGMLQuantizationType.IQ2_XXS_U3
assert gguf.GGML_QUANT_SIZES[DST] == (256, 104)

quants.IQ2_XXS.init_grid()
GRID_CODE = np.searchsorted(np.array([8, 25, 43]), quants.IQ2_XXS.grid.reshape(256, 8)).astype(np.uint32)  # 0, 1, 2
KSIGNS = np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8)
CHUNK = 1 << 20  # blocks per numpy pass (~1 GB of temporaries)


def to_u3(blocks):
    """blocks: (nb, 66) uint8 IQ2_XXS -> (nb, 104) uint8 IQ2_XXS_U3"""
    nb = blocks.shape[0]
    d = blocks[:, :2]
    qs = np.ascontiguousarray(blocks[:, 2:]).view(np.uint32).reshape(nb, 8, 2)
    idx = qs[..., 0:1].view(np.uint8).reshape(nb, 8, 4)                       # grid index per 8 weights
    m = GRID_CODE[idx].reshape(nb, 8, 32)                                      # magnitude code per weight
    s7 = (qs[..., 1:2] >> np.array([0, 7, 14, 21], dtype=np.uint32)) & 127     # (nb, 8, 4)
    sg = (KSIGNS[s7][..., None] >> np.arange(8, dtype=np.uint8)) & 1           # (nb, 8, 4, 8)
    n = np.where(sg.reshape(nb, 8, 32) != 0, 2 - m, 3 + m).astype(np.uint32)

    sh = (3 * np.arange(10, dtype=np.uint32))
    q = np.stack([(n[..., 10*k:10*k + 10] << sh).sum(-1, dtype=np.uint32) for k in range(3)], -1)  # (nb, 8, 3)
    n30, n31 = n[..., 30], n[..., 31]
    q[..., 0] |= (n30 & 3) << 30
    q[..., 1] |= ((n30 >> 2) << 30) | ((n31 & 1) << 31)
    q[..., 2] |= (n31 >> 1) << 30

    scales = ((qs[..., 1] >> 28) << (4 * np.arange(8, dtype=np.uint32))).sum(-1, dtype=np.uint32)

    out = np.zeros((nb, 104), dtype=np.uint8)
    out[:, 0:2] = d
    out[:, 4:8] = scales.view(np.uint8).reshape(nb, 4)
    out[:, 8:] = q.reshape(nb, 24).view(np.uint8)
    return out


def u3_dequant(out):
    nb = out.shape[0]
    d = out[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1, 1)
    scales = out[:, 4:8].copy().view(np.uint32).reshape(nb, 1)
    s = ((scales >> (4 * np.arange(8, dtype=np.uint32))) & 15).astype(np.float32).reshape(nb, 8, 1)
    q = out[:, 8:].copy().view(np.uint32).reshape(nb, 8, 3)
    n = np.zeros((nb, 8, 32), dtype=np.uint32)
    for j in range(30):
        n[..., j] = (q[..., j // 10] >> (3 * (j % 10))) & 7
    n[..., 30] = (q[..., 0] >> 30) | (((q[..., 1] >> 30) & 1) << 2)
    n[..., 31] = (q[..., 1] >> 31) | ((q[..., 2] >> 30) << 1)
    return (d * (2*s + 1) * np.float32(17/16) * (2*n.astype(np.float32) - 5)).reshape(nb, 256)


r = gguf.GGUFReader(a.src_path)
w = gguf.GGUFWriter(a.dst_path, arch=r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)


def selected(t):
    return t.tensor_type == SRC and re.search(a.match, t.name) is not None


for t in r.tensors:
    if selected(t):
        shape = quants.quant_shape_to_byte_shape(tuple(int(x) for x in reversed(t.shape)), DST)
        w.add_tensor_info(t.name, shape, np.dtype(np.uint8), int(np.prod(shape)), DST)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()

n_conv, err = 0, 0.0
for t in r.tensors:
    if not selected(t):
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        continue
    src = np.asarray(t.data).reshape(-1, 66)
    out = np.empty((src.shape[0], 104), dtype=np.uint8)
    for i in range(0, src.shape[0], CHUNK):
        out[i:i + CHUNK] = to_u3(src[i:i + CHUNK])
    if n_conv == 0:  # spot-check the first tensor against the IQ2_XXS reference: |u3 - iq2| <= 0.5/8 of the sub-block scale
        k = min(4096, src.shape[0])
        ref = quants.dequantize(src[:k], SRC).reshape(k, 256)
        got = u3_dequant(out[:k])
        d = src[:k, :2].copy().view(np.float16).astype(np.float32).reshape(k, 1, 1)
        s = (src[:k, 2:].copy().view(np.uint32).reshape(k, 8, 2)[..., 1:2] >> 28).astype(np.float32)
        unit = d * (0.5 + s) * 0.25  # one IQ2_XXS level unit (levels are 8, 25, 43 units)
        err = float((np.abs(got - ref).reshape(k, 8, 32) / np.maximum(unit, 1e-30)).max())
        assert err <= 0.5 + 1e-3, err
    out_shape = quants.quant_shape_to_byte_shape(tuple(int(x) for x in reversed(t.shape)), DST)
    w.write_tensor_data(out.reshape(out_shape), tensor_endianess=r.endianess)
    n_conv += 1
w.close()
print(f"re-encoded {n_conv} IQ2_XXS tensors -> IQ2_XXS_U3; max |u3 - iq2| on the first tensor = {err:.3f} units (<= 0.5)")
