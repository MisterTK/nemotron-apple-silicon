#!/usr/bin/env python3
"""Copy a GGUF, re-encoding every tensor of one quant type into another (all else byte-identical).

Used for Nemotron-3.5-Bolt-74B-A9B-24GB: llama.cpp's Metal backend has no NVFP4 matmul, so its
80 NVFP4 Mamba projections (ssm_in/ssm_out) fall back to the CPU every token. Re-encoding them as
Q8_0 (FP4 values x E4M3 block scale -> int8 x f16 scale per 32) keeps them on the GPU; the per-tensor
`.scale` tensors are separate graph inputs and are left unchanged.

usage: python3 -I bench/gguf_retype.py IN.gguf OUT.gguf --src NVFP4 --dst Q8_0
       python3 -I bench/gguf_retype.py IN.gguf OUT.gguf --src F32 --dst F16 --match 'ffn_gate_inp\.weight$'
"""
import argparse, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("src_path"); ap.add_argument("dst_path")
ap.add_argument("--src", default="NVFP4"); ap.add_argument("--dst", default="Q8_0")
ap.add_argument("--match", default="", help="only re-encode tensors whose name matches this regex")
ap.add_argument("--gguf-py", default=os.path.join(os.path.dirname(__file__), "..", "llama.cpp-bolt", "gguf-py"))
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
import numpy as np
import gguf
from gguf import quants

src_t, dst_t = gguf.GGMLQuantizationType[a.src], gguf.GGMLQuantizationType[a.dst]
r = gguf.GGUFReader(a.src_path)
w = gguf.GGUFWriter(a.dst_path, arch=r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)

for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)


def selected(t):
    return t.tensor_type == src_t and re.search(a.match, t.name) is not None


def convert(t):
    x = quants.dequantize(t.data, src_t)                     # float32, logical shape
    if dst_t == gguf.GGMLQuantizationType.F16:
        return x.astype(np.float16)
    return quants.quantize(x.astype(np.float32), dst_t)      # uint8 raw blocks


bs, ts = gguf.GGML_QUANT_SIZES[dst_t]
n = 0
for t in r.tensors:
    if selected(t) and dst_t == gguf.GGMLQuantizationType.F16:
        shape = tuple(int(x) for x in reversed(t.shape))
        w.add_tensor_info(t.name, shape, np.dtype(np.float16), int(np.prod(shape))*2, dst_t)
        n += 1
    elif selected(t):
        shape = quants.quant_shape_to_byte_shape(tuple(int(x) for x in reversed(t.shape)), dst_t)
        w.add_tensor_info(t.name, shape, np.dtype(np.uint8), int(np.prod(shape)), dst_t)
        n += 1
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()

err = []
for t in r.tensors:
    if selected(t):
        q = convert(t)
        ref = quants.dequantize(t.data, src_t)
        back = q.astype(np.float32) if dst_t == gguf.GGMLQuantizationType.F16 else quants.dequantize(q, dst_t)
        err.append(float(np.abs(back - ref).max() / (np.abs(ref).max() + 1e-12)))
        w.write_tensor_data(q, tensor_endianess=r.endianess)
    else:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print(f"re-encoded {n} tensors {a.src} -> {a.dst}; max relative error (vs tensor max) {max(err):.2e}")
