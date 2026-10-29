"""Split-K GQA decode attention for MLX (one query token per sequence).

Pass 1 (Metal): one threadgroup per (batch, kv head, key split). A tile of T keys/values
is loaded into threadgroup memory once and reused by all G query heads of that kv head
(one simdgroup per q head). Online softmax runs per tile, so there is one simd reduction
per T keys instead of one per key. K/V are read in place through their strides (a
KVCache slice is not contiguous), so no copy of the cache is made.
Pass 2 (MLX ops): merge the per-split (max, sum, unnormalized out) partials.

Supports bf16/f16 K/V and mlx affine-quantized K/V (8 or 4 bit, any group size dividing D).
"""
import math, os
import mlx.core as mx

_T = 32  # keys per tile

_HEADER = """
template <typename X> inline float tof(X x) { return static_cast<float>(x); }
"""

_SRC = """
    // NSG = 8 simdgroups, G = 8*RB q heads (RB row blocks of 8), T = 32 keys (4 col blocks)
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint tid  = sg * 32 + lane;
    const uint tgz  = threadgroup_position_in_grid.z;
    const uint s    = tgz % S;
    const uint hk   = (tgz / S) % HK;
    const uint b    = tgz / (S * HK);
    const int  N    = n_keys[0];
    const int  C    = (((N + S - 1) / S) + T - 1) / T * T;
    const int  k0   = s * C;
    const int  k1   = min(N, k0 + C);
    constexpr int NT = 32 * NSG;

    threadgroup half  qs[G * D];
    threadgroup half  kt[T * D];
    threadgroup half  vt[T * D];
    threadgroup float ss[G * T];
    threadgroup half  ps[G * T];
    constexpr int RB  = G / 8;
    constexpr int HPS = G / 8;      // softmax heads per simdgroup
    constexpr int TPS = 2 * RB;     // output tiles per simdgroup
    threadgroup float dg[RB * 64];  // diag(corr) per 8-row block

    for (int i = tid; i < G * D; i += NT) {
        const int h = i / D, d = i % D;
        qs[i] = half(tof(q[b * q_strides[0] + (hk * G + h) * q_strides[1] + d]) * scale[0]);
    }
    for (int i = tid; i < RB * 64; i += NT) dg[i] = 0.0f;

    // softmax state: simdgroup sg owns heads HPS*sg .. HPS*sg+HPS-1
    float m0 = -INFINITY, l0 = 0.0f, m1 = -INFINITY, l1 = 0.0f;

    // score tile (r, c); output tiles sg*TPS .. sg*TPS+TPS-1 over RB x (D/8)
    const int r  = sg / 4;
    const int c  = sg % 4;
    const int ro = (sg * TPS) / (D / 8);
    simdgroup_float8x8 oacc[TPS];
    for (int i = 0; i < TPS; ++i) oacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int t0 = k0; t0 < k1; t0 += T) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        LOAD_TILE
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[r*8.., c*8..] = Q[r] . K[c]^T
        if (r < RB) {
            simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kd = 0; kd < D / 8; ++kd) {
                simdgroup_half8x8 qa, kb;
                simdgroup_load(qa, qs + r * 8 * D + kd * 8, D);
                simdgroup_load(kb, kt + c * 8 * D + kd * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qa, kb, sacc);
            }
            simdgroup_store(sacc, ss + r * 8 * T + c * 8, T);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax for this simdgroup's heads, lane = key
        {
            const bool valid = t0 + (int)lane < k1;
            for (int hh = 0; hh < HPS; ++hh) {
                const int h = HPS * sg + hh;
                float sc = valid ? ss[h * T + lane] : -INFINITY;
                float &m = hh ? m1 : m0;
                float &l = hh ? l1 : l0;
                const float m_new = max(m, simd_max(sc));
                const float corr  = fast::exp(m - m_new);
                const float p     = fast::exp(sc - m_new);
                l = l * corr + simd_sum(p);
                m = m_new;
                ps[h * T + lane] = half(p);
                if (lane == 0) dg[(h / 8) * 64 + (h % 8) * 9] = corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O[ro, oc] = diag(corr) * O + P[ro] . V[:, oc]
        simdgroup_float8x8 dm;
        simdgroup_load(dm, dg + ro * 64, 8);
        for (int i = 0; i < TPS; ++i) {
            const int oc = (sg * TPS + i) % (D / 8);
            simdgroup_multiply(oacc[i], dm, oacc[i]);
            for (int kj = 0; kj < T / 8; ++kj) {
                simdgroup_half8x8 pa, vb;
                simdgroup_load(pa, ps + ro * 8 * T + kj * 8, T);
                simdgroup_load(vb, vt + kj * 8 * D + oc * 8, D);
                simdgroup_multiply_accumulate(oacc[i], pa, vb, oacc[i]);
            }
        }
    }

    for (int i = 0; i < TPS; ++i) {
        device float * op = o_part + ((b * (HK * G) + hk * G + ro * 8) * S + s) * D + ((sg * TPS + i) % (D / 8)) * 8;
        simdgroup_store(oacc[i], op, S * D);
    }
    if (lane == 0) {
        const uint p0 = (b * (HK * G) + hk * G + HPS * sg) * S + s;
        m_part[p0] = m0; l_part[p0] = l0;
        if (HPS == 2) { m_part[p0 + S] = m1; l_part[p0 + S] = l1; }
    }
"""

# coalesced: lane i loads elements [4i, 4i+4) of a contiguous run of the tile
_LOAD_DENSE = """
        typedef vec<DT, 4> dt4;
        for (int w = tid; w < T * (D / 4); w += NT) {
            const int row = w / (D / 4), col = (w % (D / 4)) * 4;
            const int key = t0 + row;
            half4 kh = half4(0.0h), vh = half4(0.0h);
            if (key < k1) {
                kh = half4(*(const device dt4 *)(k + b * k_strides[0] + hk * k_strides[1] + key * k_strides[2] + col));
                vh = half4(*(const device dt4 *)(v + b * v_strides[0] + hk * v_strides[1] + key * v_strides[2] + col));
            }
            *(threadgroup half4 *)(kt + row * D + col) = kh;
            *(threadgroup half4 *)(vt + row * D + col) = vh;
        }
"""

# mlx affine quantization: one packed uint32 word (32/BITS values) per lane per step
_LOAD_QUANT = """
        constexpr int VALS = 32 / BITS;
        constexpr int WPR  = D / VALS;
        for (int w = tid; w < T * WPR; w += NT) {
            const int row = w / WPR, wc = w % WPR, col = wc * VALS, gi = col / GS;
            const int key = t0 + row;
            threadgroup half * kd = kt + row * D + col;
            threadgroup half * vd = vt + row * D + col;
            if (key < k1) {
                const uint kw = kq[b * kq_strides[0] + hk * kq_strides[1] + key * kq_strides[2] + wc];
                const uint vw = vq[b * vq_strides[0] + hk * vq_strides[1] + key * vq_strides[2] + wc];
                const half ks = half(tof(kscale[b * kscale_strides[0] + hk * kscale_strides[1] + key * kscale_strides[2] + gi]));
                const half kb = half(tof(kbias [b * kbias_strides[0]  + hk * kbias_strides[1]  + key * kbias_strides[2]  + gi]));
                const half vs = half(tof(vscale[b * vscale_strides[0] + hk * vscale_strides[1] + key * vscale_strides[2] + gi]));
                const half vb = half(tof(vbias [b * vbias_strides[0]  + hk * vbias_strides[1]  + key * vbias_strides[2]  + gi]));
                if (BITS == 8) {
                    *(threadgroup half4 *)kd = fma(half4(as_type<uchar4>(kw)), half4(ks), half4(kb));
                    *(threadgroup half4 *)vd = fma(half4(as_type<uchar4>(vw)), half4(vs), half4(vb));
                } else {  // BITS == 4: nibble e is value e; even/odd nibbles via two masks
                    const half4 klo = fma(half4(as_type<uchar4>(kw & 0x0F0F0F0Fu)), half4(ks), half4(kb));
                    const half4 khi = fma(half4(as_type<uchar4>((kw >> 4) & 0x0F0F0F0Fu)), half4(ks), half4(kb));
                    const half4 vlo = fma(half4(as_type<uchar4>(vw & 0x0F0F0F0Fu)), half4(vs), half4(vb));
                    const half4 vhi = fma(half4(as_type<uchar4>((vw >> 4) & 0x0F0F0F0Fu)), half4(vs), half4(vb));
                    *(threadgroup half4 *)kd       = half4(klo[0], khi[0], klo[1], khi[1]);
                    *(threadgroup half4 *)(kd + 4) = half4(klo[2], khi[2], klo[3], khi[3]);
                    *(threadgroup half4 *)vd       = half4(vlo[0], vhi[0], vlo[1], vhi[1]);
                    *(threadgroup half4 *)(vd + 4) = half4(vlo[2], vhi[2], vlo[3], vhi[3]);
                }
            } else {
                for (int e = 0; e < VALS; ++e) { kd[e] = 0.0h; vd[e] = 0.0h; }
            }
        }
"""

# ---- wide head_dim (D = 256/512, G = 8; Gemma 4 global layers) --------------------------------
# A T=32 K+V tile at D=512 needs 64 KB of threadgroup memory (limit 32 KB), so this variant uses
# T=16 and one buffer that holds the K tile for the scores and is then refilled with the V tile.
# Each simdgroup owns one q head for the softmax; the Q.K^T reduction over D is split across
# simdgroups (NC column blocks x NDP d-partitions) and summed from threadgroup memory.
_TW = 16

_SRC_WIDE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint tid  = sg * 32 + lane;
    const uint tgz  = threadgroup_position_in_grid.z;
    const uint s    = tgz % S;
    const uint hk   = (tgz / S) % HK;
    const uint b    = tgz / (S * HK);
    const int  N    = n_keys[0];
    const int  C    = (((N + S - 1) / S) + T - 1) / T * T;
    const int  k0   = s * C;
    const int  k1   = min(N, k0 + C);
    constexpr int NT  = 32 * NSG;
    constexpr int NC  = T / 8;          // score column blocks
    constexpr int NDP = NSG / NC;       // d-partitions of the score reduction
    constexpr int DPS = D / 8 / NDP;    // 8-wide d steps per partition
    constexpr int TPS = D / 8 / NSG;    // output tiles per simdgroup

    threadgroup half  qs[G * D];
    threadgroup half  kv[T * D];        // K tile, then V tile
    threadgroup float sp[NDP * G * T];  // partial scores per d-partition
    threadgroup half  ps[G * T];
    threadgroup float dg[64];           // diag(corr)

    for (int i = tid; i < G * D; i += NT) {
        const int h = i / D, d = i % D;
        qs[i] = half(tof(q[b * q_strides[0] + (hk * G + h) * q_strides[1] + d]) * scale[0]);
    }
    for (int i = tid; i < 64; i += NT) dg[i] = 0.0f;

    float m = -INFINITY, l = 0.0f;      // softmax state of head sg
    const int c = sg % NC, dp = sg / NC;
    simdgroup_float8x8 oacc[TPS];
    for (int i = 0; i < TPS; ++i) oacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int t0 = k0; t0 < k1; t0 += T) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        LOAD_K
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {   // partial S[:, c*8..] over this simdgroup's d-partition
            simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kd = 0; kd < DPS; ++kd) {
                const int d = (dp * DPS + kd) * 8;
                simdgroup_half8x8 qa, kb;
                simdgroup_load(qa, qs + d, D);
                simdgroup_load(kb, kv + c * 8 * D + d, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qa, kb, sacc);
            }
            simdgroup_store(sacc, sp + dp * G * T + c * 8, T);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {   // online softmax for head sg, lane = key; the V tile loads meanwhile (K is no longer read)
            const bool valid = (int)lane < T && t0 + (int)lane < k1;
            float sc = -INFINITY;
            if (valid) {
                sc = 0.0f;
                for (int p = 0; p < NDP; ++p) sc += sp[p * G * T + sg * T + lane];
            }
            const float m_new = max(m, simd_max(sc));
            const float corr  = fast::exp(m - m_new);
            const float pr    = valid ? fast::exp(sc - m_new) : 0.0f;
            l = l * corr + simd_sum(pr);
            m = m_new;
            if ((int)lane < T) ps[sg * T + lane] = half(pr);
            if (lane == 0) dg[sg * 9] = corr;
        }
        LOAD_V
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(corr) * O + P . V
        simdgroup_float8x8 dm;
        simdgroup_load(dm, dg, 8);
        for (int i = 0; i < TPS; ++i) {
            const int oc = sg * TPS + i;
            simdgroup_multiply(oacc[i], dm, oacc[i]);
            for (int kj = 0; kj < T / 8; ++kj) {
                simdgroup_half8x8 pa, vb;
                simdgroup_load(pa, ps + kj * 8, T);
                simdgroup_load(vb, kv + kj * 8 * D + oc * 8, D);
                simdgroup_multiply_accumulate(oacc[i], pa, vb, oacc[i]);
            }
        }
    }

    for (int i = 0; i < TPS; ++i) {
        device float * op = o_part + ((b * (HK * G) + hk * G) * S + s) * D + (sg * TPS + i) * 8;
        simdgroup_store(oacc[i], op, S * D);
    }
    if (lane == 0) {
        const uint p0 = (b * (HK * G) + hk * G + sg) * S + s;
        m_part[p0] = m; l_part[p0] = l;
    }
"""


def _load_dense_one(x):
    return f"""
        for (int w = tid; w < T * (D / 4); w += NT) {{
            const int row = w / (D / 4), col = (w % (D / 4)) * 4;
            const int key = t0 + row;
            half4 xh = half4(0.0h);
            if (key < k1)
                xh = half4(*(const device vec<DT, 4> *)({x} + b * {x}_strides[0] + hk * {x}_strides[1] + key * {x}_strides[2] + col));
            *(threadgroup half4 *)(kv + row * D + col) = xh;
        }}
"""


def _load_quant_one(x):
    return f"""
        for (int w = tid; w < T * (D / (32 / BITS)); w += NT) {{
            constexpr int VALS = 32 / BITS;
            const int row = w / (D / VALS), wc = w % (D / VALS), col = wc * VALS, gi = col / GS;
            const int key = t0 + row;
            threadgroup half * xd = kv + row * D + col;
            if (key < k1) {{
                const uint xw = {x}q[b * {x}q_strides[0] + hk * {x}q_strides[1] + key * {x}q_strides[2] + wc];
                const half xs = half(tof({x}scale[b * {x}scale_strides[0] + hk * {x}scale_strides[1] + key * {x}scale_strides[2] + gi]));
                const half xb = half(tof({x}bias [b * {x}bias_strides[0]  + hk * {x}bias_strides[1]  + key * {x}bias_strides[2]  + gi]));
                if (BITS == 8) {{
                    *(threadgroup half4 *)xd = fma(half4(as_type<uchar4>(xw)), half4(xs), half4(xb));
                }} else {{
                    const half4 lo = fma(half4(as_type<uchar4>(xw & 0x0F0F0F0Fu)), half4(xs), half4(xb));
                    const half4 hi = fma(half4(as_type<uchar4>((xw >> 4) & 0x0F0F0F0Fu)), half4(xs), half4(xb));
                    *(threadgroup half4 *)xd       = half4(lo[0], hi[0], lo[1], hi[1]);
                    *(threadgroup half4 *)(xd + 4) = half4(lo[2], hi[2], lo[3], hi[3]);
                }}
            }} else {{
                for (int e = 0; e < VALS; ++e) xd[e] = 0.0h;
            }}
        }}
"""


_kernels_wide = {}


def _kernel_wide(quant):
    if quant not in _kernels_wide:
        if quant:
            names = ["q", "kq", "kscale", "kbias", "vq", "vscale", "vbias", "n_keys", "scale"]
            src = _SRC_WIDE.replace("LOAD_K", _load_quant_one("k")).replace("LOAD_V", _load_quant_one("v"))
        else:
            names = ["q", "k", "v", "n_keys", "scale"]
            src = _SRC_WIDE.replace("LOAD_K", _load_dense_one("k")).replace("LOAD_V", _load_dense_one("v"))
        _kernels_wide[quant] = mx.fast.metal_kernel(
            name=f"gqa_decode_attn_wide_{'q' if quant else 'd'}",
            input_names=names, output_names=["o_part", "m_part", "l_part"],
            source=src, header=_HEADER, ensure_row_contiguous=False)
    return _kernels_wide[quant]


_kernels = {}


def _kernel(quant):
    if quant not in _kernels:
        if quant:
            names = ["q", "kq", "kscale", "kbias", "vq", "vscale", "vbias", "n_keys", "scale"]
            src = _SRC.replace("LOAD_TILE", _LOAD_QUANT)
        else:
            names = ["q", "k", "v", "n_keys", "scale"]
            src = _SRC.replace("LOAD_TILE", _LOAD_DENSE)
        _kernels[quant] = mx.fast.metal_kernel(
            name=f"gqa_decode_attn_{'q' if quant else 'd'}",
            input_names=names, output_names=["o_part", "m_part", "l_part"],
            source=src, header=_HEADER, ensure_row_contiguous=False)
    return _kernels[quant]


def n_splits(B, Hk, N, target_tgs=int(os.environ.get("GQA_TGS", 320)), T=_T):
    s = max(1, math.ceil(target_tgs / (B * Hk)))
    return max(1, min(s, math.ceil(N / (T * 4))))  # keep >= 4 tiles per split


def supported(G, D):
    return (D == 128 and G in (8, 16)) or (D in (256, 512) and G == 8)


def gqa_decode_attention(q, k, v, scale, n_keys=None, quant=None, splits=None):
    """q: (B, Hq, 1, D). Dense: k, v (B, Hk, >=N, D). Quantized: k, v are
    (packed, scales, biases) tuples, quant=(group_size, bits). Returns (B, Hq, 1, D)."""
    B, Hq, _, D = q.shape
    if quant:
        Hk, N = k[0].shape[1], (n_keys or k[0].shape[2])
    else:
        Hk, N = k.shape[1], (n_keys or k.shape[2])
    G = Hq // Hk
    assert supported(G, D), "kernel supports G 8/16 at head_dim 128, or G 8 at head_dim 256/512"
    wide = D != 128
    T = _TW if wide else _T
    S = splits or n_splits(B, Hk, N, T=T)
    tmpl = [("D", D), ("G", G), ("HK", Hk), ("S", S), ("T", T), ("NSG", 8)]
    nk = mx.array([N], dtype=mx.int32); sc = mx.array([scale], dtype=mx.float32)
    if quant:
        gs, bits = quant
        assert bits in (4, 8)
        tmpl += [("GS", gs), ("BITS", bits)]
        inputs = [q, k[0], k[1], k[2], v[0], v[1], v[2], nk, sc]
    else:
        tmpl += [("DT", k.dtype)]
        inputs = [q, k, v, nk, sc]
    o, m, l = (_kernel_wide if wide else _kernel)(bool(quant))(
        inputs=inputs, template=tmpl,
        grid=(32, 8, S * Hk * B), threadgroup=(32, 8, 1),
        output_shapes=[(B, Hq, S, D), (B, Hq, S), (B, Hq, S)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    out = o[:, :, 0, :] / l[..., :1] if S == 1 else _merge(o, m, l)
    return out[:, :, None, :].astype(q.dtype)


@mx.compile
def _merge(o, m, l):
    mm = m.max(axis=-1, keepdims=True)
    w = mx.exp(m - mm)
    return (o * w[..., None]).sum(axis=-2) / (l * w).sum(axis=-1, keepdims=True)


ENABLED = True
REF32 = False  # validation only: exact fp32 attention for single-token decode
# below this many keys the stock vector kernel wins at 1-2 users (extra dispatch + merge); 0 = always use the kernel
MIN_KEYS = int(os.environ.get("GQA_MIN_KEYS", 0))


def install(module_names=("mlx_lm.models.nemotron_h", "mlx_lm.models.qwen3_moe", "mlx_lm.models.gemma4_text")):
    """Route single-token decode attention in the given mlx-lm model modules to the kernel."""
    import importlib
    for name in module_names:
        mod = importlib.import_module(name)
        orig = mod.scaled_dot_product_attention

        def sdpa(queries, keys, values, cache, scale, mask, sinks=None, _orig=orig):
            quant = (cache.group_size, cache.bits) if hasattr(cache, "bits") else None
            Hk = (keys[0] if quant else keys).shape[1]
            N = (keys[0] if quant else keys).shape[2]
            if REF32 and not quant and queries.shape[2] == 1 and mask is None:
                B, Hq, _, D = queries.shape
                q32 = queries.astype(mx.float32).reshape(B, Hk, Hq // Hk, D) * scale
                s = q32 @ keys.astype(mx.float32).swapaxes(-1, -2)
                return (mx.softmax(s, axis=-1, precise=True) @ values.astype(mx.float32)).reshape(B, Hq, 1, D).astype(queries.dtype)
            if ENABLED and N >= MIN_KEYS and queries.shape[2] == 1 and mask is None and sinks is None and supported(queries.shape[1] // Hk, queries.shape[-1]):
                return gqa_decode_attention(queries, keys, values, scale, quant=quant)
            return _orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

        mod.scaled_dot_product_attention = sdpa
