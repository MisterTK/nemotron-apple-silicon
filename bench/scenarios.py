#!/usr/bin/env python3
"""Context x users matrix for one (model, build, KV type), via llama-batched-bench.

1. probe: largest number of sequences of a given context that the build can allocate
   (KV + recurrent state + compute buffers are all reserved at context creation), using a
   64-token prompt so each probe takes seconds.
2. run: one process per context depth, prompt prefilled once (BB_REUSE_PP=1, needs
   patches/tool-batched-bench-pps-reuse.patch) and copied into every
   sequence, so each sequence holds its own full-length KV/state copy.
"""
import argparse, json, os, re, subprocess, time, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--bin", required=True, help="llama-batched-bench")
ap.add_argument("--tag", required=True)
ap.add_argument("--kv", default="f16")
ap.add_argument("--grid", required=True, help="depth:u1,u2,...;depth:...")
ap.add_argument("--probe-max", type=int, default=128)
ap.add_argument("--no-probe", action="store_true")
ap.add_argument("--ntg", type=int, default=64)
ap.add_argument("--out", default="results/results_scenarios.jsonl")
a = ap.parse_args()

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
KV = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB")
RS = re.compile(r"llama_memory_recurrent: size =\s*([\d.]+) MiB")
CB = re.compile(r"MTL0 compute buffer size =\s*([\d.]+) MiB")
TOT = re.compile(r"allocated buffer, size =.*\(\s*([\d.]+) / [\d.]+\)")
base = dict(model=pathlib.Path(a.model).name, tag=a.tag, kv=a.kv)


def emit(rec):
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def bench(n_seq, depth, npp, users, reuse, timeout):
    ctx = n_seq * (((depth + a.ntg + 16 + 255) // 256) * 256)
    cmd = [a.bin, "-m", a.model, "-c", str(ctx), "-b", "2048", "-ub", "512", "-fa", "on", "-ngl", "99",
           "-pps", "-npp", str(npp), "-ntg", str(a.ntg if npp > 64 else 4), "-npl", ",".join(map(str, users)),
           "-ctk", a.kv, "-ctv", a.kv, "-v"]
    env = dict(os.environ)
    if reuse:
        env["BB_REUSE_PP"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
        log, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        log, rc = "timeout", -1
    rows = [m.groups() for m in map(ROW.match, log.splitlines()) if m]
    mem = dict(kv_mib=sum(map(float, KV.findall(log))), rec_mib=sum(map(float, RS.findall(log))),
               compute_mib=max(map(float, CB.findall(log)), default=None),
               metal_total_mib=max(map(float, TOT.findall(log)), default=None))
    ok = rc == 0 and len(rows) == len(users)
    err = [] if ok else [l for l in log.splitlines() if re.search(r"error|fail|memory", l, re.I)][-3:]
    return ok, rows, mem, round(time.time() - t0, 1), err


grid = []
for part in a.grid.split(";"):
    d, us = part.split(":")
    grid.append((int(d), [int(u) for u in us.split(",")]))

for depth, users in grid:
    cap = None
    if not a.no_probe:
        fits = {}
        def fit(n):
            if n not in fits:
                ok, _, mem, wall, err = bench(n, depth, 64, [n], False, 900)
                fits[n] = ok
                emit(dict(base, kind="probe", depth=depth, n=n, ok=ok, wall_s=wall, **mem, err=err))
            return fits[n]
        lo, hi = 0, None
        n = 1
        while n <= a.probe_max:
            if fit(n):
                lo = n; n *= 2
            else:
                hi = n; break
        if hi is not None:
            while hi - lo > 1 and (hi - lo) > max(1, lo // 8):
                mid = (lo + hi) // 2
                if fit(mid): lo = mid
                else: hi = mid
        cap = lo
        emit(dict(base, kind="capacity", depth=depth, max_seqs=lo, bound="probe_max" if hi is None else "fail"))
        if lo == 0:
            continue
        # the template sequence takes one slot; also measure the largest count that fits
        users = sorted({u for u in users if u <= lo - 1} | ({lo - 1} if lo > 1 else set()))
    if not users:
        continue
    while users:
        ok, rows, mem, wall, err = bench(max(users) + 1, depth, depth, users, True, 4 * 3600)
        if ok:
            for g in rows:
                emit(dict(base, kind="run", depth=depth, users=int(g[2]), pp_s=float(g[4]), pp_tps=float(g[5]),
                          tg_s=float(g[6]), tg_tps=float(g[7]), per_user_tps=float(g[7]) / int(g[2]),
                          wall_s=wall, cap=cap, **mem))
            break
        emit(dict(base, kind="runfail", depth=depth, users=users, wall_s=wall, err=err))
        users = users[:-1]
