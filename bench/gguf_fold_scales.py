#!/usr/bin/env python3
"""Fold Nemotron-H NVFP4 per-tensor scales into the RMSNorm weights that feed them, and drop the scales.

llama.cpp applies `blk.N.ssm_in.scale` / `blk.N.ssm_out.scale` as a separate MUL after each matmul
(80 extra dispatches per decode step on Bolt-74B). Both matmul inputs come straight out of an RMSNorm
with a per-channel weight (`attn_norm` for ssm_in, the grouped `ssm_norm` for ssm_out), so
W @ (n * g) * s == W @ (n * (g * s)) exactly up to f32 rounding. The scale tensors are optional
(TENSOR_NOT_REQUIRED), so the model loads unchanged without them.

usage: python3 -I bench/gguf_fold_scales.py IN.gguf OUT.gguf
"""
import argparse, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("src_path"); ap.add_argument("dst_path")
ap.add_argument("--gguf-py", default=os.path.join(os.path.dirname(__file__), "..", "llama.cpp-bolt", "gguf-py"))
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
import numpy as np
import gguf

r = gguf.GGUFReader(a.src_path)
w = gguf.GGUFWriter(a.dst_path, arch=r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)

for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

tensors = {t.name: t for t in r.tensors}
fold = {}  # norm tensor name -> scale
for name, t in tensors.items():
    m = re.fullmatch(r"blk\.(\d+)\.ssm_(in|out)\.scale", name)
    if not m:
        continue
    norm = f"blk.{m.group(1)}.{'attn_norm' if m.group(2) == 'in' else 'ssm_norm'}.weight"
    assert tensors[norm].tensor_type == gguf.GGMLQuantizationType.F32, norm
    fold[norm] = float(np.asarray(t.data, dtype=np.float32).reshape(-1)[0])

keep = [t for t in r.tensors if not t.name.endswith((".ssm_in.scale", ".ssm_out.scale"))]
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    data = t.data
    if t.name in fold:
        data = (np.asarray(data, dtype=np.float32) * np.float32(fold[t.name])).astype(np.float32).reshape(data.shape)
    w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
s = np.array(list(fold.values()))
print(f"folded {len(fold)} scales into norm weights (range {s.min():.4g} .. {s.max():.4g}); dropped {len(r.tensors) - len(keep)} tensors")
