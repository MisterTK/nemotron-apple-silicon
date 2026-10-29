#!/usr/bin/env python3
"""Concurrent-user load test against an OpenAI-compatible llama-server.

Each user sends a distinct long prompt (so no prefix-cache sharing) and streams
`--gen` tokens. Reports per-user TTFT and decode speed plus aggregate throughput.
"""
import argparse, json, statistics, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--users", type=int, default=4)
ap.add_argument("--prompt-words", type=int, default=2000)
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()

WORDS = ("memory bandwidth bounds decode on unified memory systems because every generated "
         "token must stream all active weights and the whole attention cache through the GPU ").split()


def prompt(i):
    body = " ".join(WORDS[(j * 7 + i) % len(WORDS)] for j in range(a.prompt_words))
    return f"User {i} document #{i * 7919}:\n{body}\n\nSummarize the document above in detail."


res = [None] * a.users


def run(i):
    req = urllib.request.Request(a.url + "/v1/completions", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(dict(prompt=prompt(i), max_tokens=a.gen, temperature=0, stream=True,
                             ignore_eos=True, cache_prompt=False)).encode())
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            if json.loads(line[6:])["choices"][0].get("text"):
                n += 1
                if first is None:
                    first = time.time()
    t1 = time.time()
    res[i] = dict(ttft=first - t0, decode_tps=(n - 1) / (t1 - first) if n > 1 else 0, n=n, end=t1, start=t0)


t0 = time.time()
th = [threading.Thread(target=run, args=(i,)) for i in range(a.users)]
[t.start() for t in th]; [t.join() for t in th]
wall = time.time() - t0
toks = sum(r["n"] for r in res)
out = dict(tag=a.tag, users=a.users, prompt_words=a.prompt_words, gen=a.gen, wall_s=round(wall, 2),
           agg_gen_tps=round(toks / wall, 1),
           ttft_p50=round(statistics.median(r["ttft"] for r in res), 2),
           ttft_max=round(max(r["ttft"] for r in res), 2),
           per_user_decode_tps_p50=round(statistics.median(r["decode_tps"] for r in res), 1),
           per_user_decode_tps_min=round(min(r["decode_tps"] for r in res), 1))
print(json.dumps(out))
if a.out:
    open(a.out, "a").write(json.dumps(out) + "\n")
