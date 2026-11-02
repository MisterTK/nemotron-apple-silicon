import json, urllib.request, time, sys, glob, os, uuid
# A long agent session: every turn appends ~2000 words of natural text (a tool result) and gets a 128-token reply.
# usage (from the repo root, server up): python3 webai/session.py TURNS TAG OUT.jsonl
#   WEBAI_URL server base (default http://127.0.0.1:11436); WEBAI_MODEL model id (oMLX needs it)
URL=os.environ.get("WEBAI_URL","http://127.0.0.1:11436"); MODEL=os.environ.get("WEBAI_MODEL")
docs=" ".join(open(f).read() for f in sorted(glob.glob('*.md')+glob.glob('patches/*.md')))
words=docs.split(); turns=int(sys.argv[1]); tag=sys.argv[2]; out=sys.argv[3]
msgs=[{"role":"system","content":f"You are an agent operating tools. Be brief. Session {uuid.uuid4()}"}]
pos=0
for t in range(turns):
    chunk=" ".join(words[pos:pos+2000]); pos=(pos+2000)%max(1,len(words)-2000)
    msgs.append({"role":"user","content":f"Tool result #{t}:\n{chunk}\n\nState the next step in one sentence."})
    body=dict(messages=msgs,max_tokens=128,temperature=0,stream=False)
    if MODEL: body["model"]=MODEL
    r=urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.load(urllib.request.urlopen(r,timeout=7200)); wall=time.time()-t0
    m=d["choices"][0]["message"]; msgs.append({"role":"assistant","content":m.get("content") or m.get("reasoning_content") or ""})
    if "timings" in d:
        tm=d["timings"]; rec=dict(tag=tag,turn=t+1,history_tokens=d["usage"]["prompt_tokens"],processed=tm["prompt_n"],ttft_s=round(tm["prompt_ms"]/1000,2),tg_tps=round(tm["predicted_per_second"],1),wall_s=round(wall,1))
    else:
        u=d["usage"]; rec=dict(tag=tag,turn=t+1,history_tokens=u["prompt_tokens"],processed=u["prompt_tokens"]-u.get("prompt_tokens_details",{}).get("cached_tokens",0),ttft_s=round(u["time_to_first_token"],2),tg_tps=round(u["generation_tokens_per_second"],1),wall_s=round(wall,1))
    print(json.dumps(rec),flush=True); open(out,'a').write(json.dumps(rec)+"\n")
