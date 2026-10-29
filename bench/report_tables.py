#!/usr/bin/env python3
"""Print markdown tables from the results_*.jsonl files (paths relative to the repo root)."""
import json, sys
from collections import defaultdict


def load(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError:
        return []


def vname(r):
    return f"{'kernel' if r.get('gqa_kernel') else 'stock'} {('bf16' if r.get('kv_bits') in (0, 'native') else str(r.get('kv_bits')) + '-bit')}"


def matrix(rows, title, field="agg_tps"):
    by = defaultdict(dict)
    users = sorted({r["users"] for r in rows})
    for r in rows:
        key = (r["ctx"], vname(r))
        if r.get("status", "ok") == "ok":
            by[key][r["users"]] = f"{r[field]:.0f}" if field == "agg_tps" else f"{r[field]:.1f}"
        else:
            by[key][r["users"]] = "mem" if "mem" in r.get("status", "") else "fail"
    print(f"\n#### {title}\n")
    print("| context | variant | " + " | ".join(f"{u} user{'s' if u > 1 else ''}" for u in users) + " |")
    print("|---|---|" + "---|" * len(users))
    for (ctx, v), d in sorted(by.items()):
        print(f"| {ctx // 1024}K | {v} | " + " | ".join(d.get(u, "") for u in users) + " |")


for path in sys.argv[1:] or ["results_mlx_matrix.jsonl"]:
    rows = [r for r in load(path) if "users" in r]
    for model in sorted({r.get("model", "?") for r in rows if r.get("model")}):
        mr = [r for r in rows if r.get("model") == model or (r.get("status", "ok") != "ok" and r.get("tag", "").endswith(model.split("-")[1]))]
        matrix(mr, f"{model}: aggregate decode tok/s", "agg_tps")
        matrix(mr, f"{model}: per-user decode tok/s", "per_user_tps")
        matrix(mr, f"{model}: p95 inter-token gap (ms)", "gap_p95_ms")
