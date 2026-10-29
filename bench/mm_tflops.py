"""Matmul throughput (MLX): dense f16/bf16 mat-mat at a few sizes, and a bandwidth-bound f16 mat-vec.
Run on each machine to compare compute (matrix accelerators) against memory bandwidth."""
import time
import mlx.core as mx


def timeit(f, n=20):
    mx.eval(f())
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t0) / n


print(mx.device_info()["device_name"])
for dt in (mx.float16, mx.bfloat16):
    for (m, n, k) in ((4096, 4096, 4096), (8192, 8192, 8192), (32, 18560, 4096), (512, 18560, 4096)):
        a = mx.random.normal((m, k)).astype(dt)
        b = mx.random.normal((k, n)).astype(dt)
        mx.eval(a, b)
        t = timeit(lambda: a @ b)
        print(f"{str(dt):14s} mm  {m:5d}x{n:5d}x{k:5d}: {2*m*n*k/t/1e12:6.2f} TFLOPS")

# mat-vec: 1 GiB f16 weight, bandwidth-bound
w = mx.random.normal((32768, 16384)).astype(mx.float16)
x = mx.random.normal((16384, 1)).astype(mx.float16)
mx.eval(w, x)
t = timeit(lambda: w @ x)
print(f"f16 mat-vec 32768x16384 (1 GiB): {w.nbytes/t/1e9:6.0f} GB/s")
