#!/usr/bin/env python3
"""Quality checks for decode-path changes (kernel, KV precision) against stock bf16 MLX.

agreement: teacher-forced. A reference continuation is generated greedily with stock
           bf16 attention; every variant is then fed the same tokens from a clone of the
           same prefilled cache. Reports top-1 agreement, mean KL(ref || variant) and the
           max |delta logprob| of the reference token.
needle:    exact-answer retrieval at long context. A passphrase is planted at a given
           depth of a fixture document; each variant must reproduce it exactly.
"""
import argparse, copy, json, os, random, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gqa_decode_attn
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache, make_prompt_cache

def log_softmax(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--test", choices=["agreement", "needle"], default="agreement")
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
ap.add_argument("--variants", default="gqa:0,gqa:8,gqa:4,stock:8,stock:4")
ap.add_argument("--reference", default="stock", choices=["stock", "ref32"])
ap.add_argument("--out", default="")
a = ap.parse_args()

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
gqa_decode_attn.install()
model, tok = load(a.model)
fix = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"fixture_tokens.{os.path.basename(a.model.rstrip('/'))}.npy")
import numpy as np
corpus = np.load(fix).tolist()
variants = [(v.split(":")[0] == "gqa", int(v.split(":")[1])) for v in a.variants.split(",")]
name = lambda k, b: f"{'gqa' if k else 'stock'}:{b or 'bf16'}"


def chat_ids(content_ids, question):
    msgs = [{"role": "user", "content": "<<DOC>>"}]
    kw = {"enable_thinking": False} if any(m in a.model.lower() for m in ("nemotron", "gemma")) else {}
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    pre, post = s.split("<<DOC>>")
    return (tok.encode(pre, add_special_tokens=False) + content_ids +
            tok.encode("\n\n" + question + post, add_special_tokens=False))


def prefill(ids):
    cache = make_prompt_cache(model)
    body = getattr(model, "backbone", None) or getattr(model, "model", None) or model.language_model.model
    step = 512 if hasattr(model, "language_model") else 2048   # gemma4 global layers: D=512 scores
    for i in range(0, len(ids) - 1, step):
        body(mx.array([ids[i:min(i + step, len(ids) - 1)]]), cache=cache)
        mx.eval([c.state for c in cache]); mx.clear_cache()
    for c in cache:
        if isinstance(c, KVCache):
            c.keys = c.keys[..., :c.offset, :]; c.values = c.values[..., :c.offset, :]
    return cache, ids[-1]


def clone(cache, bits):
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            n = copy.copy(c); n.cache = [mx.array(x) for x in c.cache]
        elif bits and isinstance(c, KVCache):   # sliding-window (Rotating) caches stay native
            n = c.to_quantized(group_size=64, bits=bits)
        else:
            n = copy.copy(c); n.keys = mx.array(c.keys); n.values = mx.array(c.values)
        out.append(n)
    mx.eval([c.state for c in out])
    return out


def emit(r):
    print(json.dumps(r), flush=True)
    if a.out:
        open(a.out, "a").write(json.dumps(r) + "\n")


if a.test == "agreement":
    doc = corpus[123_457:123_457 + a.ctx - 64]
    ids = chat_ids(doc, "Explain in detail what the code above does.")
    base, last = prefill(ids)
    # reference: stock bf16 (or exact fp32 attention) greedy
    gqa_decode_attn.ENABLED = False
    gqa_decode_attn.REF32 = a.reference == "ref32"
    cache = clone(base, 0); y = mx.array([[last]]); ref_tok, ref_lp = [], []
    for _ in range(a.gen):
        lp = log_softmax(model(y, cache=cache)[:, -1, :].astype(mx.float32))
        t = mx.argmax(lp, axis=-1); mx.eval(lp, t)
        ref_tok.append(t.item()); ref_lp.append(lp[0]); y = t[None]
    del cache
    gqa_decode_attn.REF32 = False
    for kernel, bits in variants:
        gqa_decode_attn.ENABLED = kernel
        cache = clone(base, bits); y = mx.array([[last]]); agree = 0; kls = []; dmax = 0.0
        for i in range(a.gen):
            lp = log_softmax(model(y, cache=cache)[:, -1, :].astype(mx.float32))[0]
            p_ref = mx.exp(ref_lp[i])
            kl = (p_ref * (ref_lp[i] - lp)).sum()
            mx.eval(lp, kl)
            agree += int(mx.argmax(lp).item() == ref_tok[i]); kls.append(kl.item())
            dmax = max(dmax, abs(lp[ref_tok[i]].item() - ref_lp[i][ref_tok[i]].item()))
            y = mx.array([[ref_tok[i]]])  # teacher forcing
        emit(dict(test="agreement", reference=a.reference, model=os.path.basename(a.model.rstrip("/")), ctx=a.ctx, variant=name(kernel, bits),
                  top1_agree=round(agree / a.gen, 4), kl_mean=round(sum(kls) / len(kls), 6), kl_max=round(max(kls), 5),
                  ref_logprob_absdiff_max=round(dmax, 4), n=a.gen))
        del cache; mx.clear_cache()
    print("reference continuation:", repr(tok.decode(ref_tok)[:300]), file=sys.stderr)

else:
    rng = random.Random(7)
    words = "amber basalt cobalt dahlia ember fjord garnet harbor indigo juniper kestrel lantern meadow nickel orchid".split()
    for depth in a.depths:
        vault = rng.randint(100, 999)
        secret = f"{rng.choice(words)}-{rng.choice(words)}-{rng.randint(1000, 9999)}"
        needle = tok.encode(f"\n# NOTE: the secret passphrase for vault {vault} is \"{secret}\".\n", add_special_tokens=False)
        n = a.ctx - len(needle) - 96
        doc = corpus[555_555:555_555 + n]
        cut = int(n * depth)
        ids = chat_ids(doc[:cut] + needle + doc[cut:],
                       f"What is the secret passphrase for vault {vault}? Reply with the passphrase only.")
        t0 = time.time(); base, last = prefill(ids); tp = time.time() - t0
        for kernel, bits in [(False, 0)] + variants:
            gqa_decode_attn.ENABLED = kernel
            cache = clone(base, bits); y = mx.array([[last]]); out = []
            for _ in range(24):
                t = mx.argmax(model(y, cache=cache)[:, -1, :], axis=-1); mx.eval(t)
                if t.item() in tok.eos_token_ids:
                    break
                out.append(t.item()); y = t[None]
            ans = tok.decode(out).strip()
            emit(dict(test="needle", model=os.path.basename(a.model.rstrip("/")), ctx=len(ids), depth=depth,
                      variant=name(kernel, bits), expected=secret, answer=ans[:80], exact=secret in ans,
                      prefill_s=round(tp, 1)))
            del cache; mx.clear_cache()
        del base; mx.clear_cache()
