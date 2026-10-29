import sys, json
for l in sys.stdin:
    try:
        r = json.loads(l)
    except Exception:
        if "Warning" not in l:
            print(l.rstrip()[:300], flush=True)
        continue
    print(r.get("model", r.get("tag", ""))[:16], r["ctx"], r["users"], "gqa" if r.get("gqa_kernel") else "stock", r.get("kv_bits"),
          r.get("status", "ok"), "agg", r.get("agg_tps"), "per", r.get("per_user_tps"), "p95", r.get("gap_p95_ms"), "peak", r.get("peak_gb"), flush=True)
