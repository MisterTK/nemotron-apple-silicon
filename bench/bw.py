import mlx.core as mx, time
x = mx.random.uniform(shape=(1024, 1024*1024)).astype(mx.float32)  # 4 GiB
mx.eval(x)
def t(f, n=10):
    mx.eval(f()); t0=time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter()-t0)/n
r = t(lambda: x.sum(axis=1)); print(f"read (reduce): {x.nbytes/r/1e9:.0f} GB/s")
c = t(lambda: x + 1.0);        print(f"copy (r+w):    {2*x.nbytes/c/1e9:.0f} GB/s")
