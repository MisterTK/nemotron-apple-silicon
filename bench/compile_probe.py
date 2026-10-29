import sys, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, _ = load(sys.argv[1]); layers = model.layers
x = (mx.random.normal((1, 1, model.args.hidden_size)) * .1).astype(mx.bfloat16); mx.eval(x)
def bench(f, n=200):
    y = x
    for _ in range(3): y = f(y)
    mx.eval(y); t0 = time.perf_counter(); y = x
    for _ in range(n): y = f(y)
    mx.eval(y); return (time.perf_counter() - t0) / n * 1e6
E = next(l for l in layers if l.block_type == "E")
print(f"moe block eager    {bench(lambda y: E(y)):7.1f} us")
cE = mx.compile(lambda y: E(y))
print(f"moe block compiled {bench(cE):7.1f} us")
M = next(l for l in layers if l.block_type == "M")
cache = make_prompt_cache(model)[0]
print(f"mamba block eager  {bench(lambda y: M(y, cache=cache)):7.1f} us")
