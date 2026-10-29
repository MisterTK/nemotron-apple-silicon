#!/usr/bin/env python3
"""Decode-throughput sweep: context depth x concurrent users, via llama-batched-bench.

One process per (depth, users) point so each run allocates exactly the KV/recurrent
memory that point needs; OOM / allocation failures are recorded rather than fatal.
Prompt is shared (-pps) so prefill runs once and is copied to every sequence --
memory is still per-sequence (no unified KV), but we don't burn minutes re-prefilling.
"""
import os, argparse, json, re, subprocess, sys, time, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--depths", default="1024,8192,32768")
ap.add_argument("--users", default="1,2,4,8,16,32")
ap.add_argument("--ntg", type=int, default=64)
ap.add_argument("--extra", default="", help="extra llama-batched-bench args, e.g. '-ctk q8_0 -ctv q8_0'")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="results/results.jsonl")
ap.add_argument("--max-tokens", type=int, default=1_100_000, help="skip points whose total ctx exceeds this")
a = ap.parse_args()

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
MEM = re.compile(r"(llama_kv_cache|llama_memory_recurrent|graph_reserve|llama_context).*?size\s*=\s*([\d.]+)\s*MiB", re.I)
KVLINE = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB")
RSLINE = re.compile(r"llama_memory_recurrent: size =\s*([\d.]+) MiB")
COMPUTE = re.compile(r"MTL0 compute buffer size =\s*([\d.]+) MiB")

for d in map(int, a.depths.split(",")):
    for n in map(int, a.users.split(",")):
        ctx = n * (d + a.ntg) + 256
        rec = dict(model=pathlib.Path(a.model).name, tag=a.tag, depth=d, users=n, ntg=a.ntg, ctx=ctx, extra=a.extra)
        if ctx > a.max_tokens:
            rec["status"] = "skipped"; print(json.dumps(rec)); continue
        cmd = [os.environ.get("BENCH", "llama-batched-bench"), "-m", a.model, "-c", str(ctx), "-b", "2048", "-ub", "512",
               "-fa", "on", "-ngl", "99", "-pps", "-npp", str(d), "-ntg", str(a.ntg), "-npl", str(n)] + a.extra.split()
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        log = p.stdout + p.stderr
        rec["wall_s"] = round(time.time() - t0, 1)
        m = [ROW.match(l) for l in log.splitlines()]
        m = [x for x in m if x]
        if p.returncode != 0 or not m:
            rec["status"] = "fail"
            rec["err"] = [l for l in log.splitlines() if re.search(r"error|failed|out of memory|OOM", l, re.I)][-3:]
        else:
            g = m[-1].groups()
            rec.update(status="ok", pp_s=float(g[4]), pp_tps=float(g[5]), tg_s=float(g[6]),
                       tg_tps=float(g[7]), per_user_tps=float(g[7]) / n)
            kv = KVLINE.findall(log); rs = RSLINE.findall(log); cb = COMPUTE.findall(log)
            rec["kv_mib"] = sum(map(float, kv)); rec["recurrent_mib"] = sum(map(float, rs))
            rec["compute_mib"] = max(map(float, cb)) if cb else None
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
