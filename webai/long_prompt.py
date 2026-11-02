import json, urllib.request, time, sys, glob, os, uuid
# One natural-text prompt (the repo's markdown, repeated to TOKENS) with a 256-token answer.
# usage (from the repo root, server up): python3 webai/long_prompt.py TOKENS TAG OUT.jsonl
#   WEBAI_URL   server base (default http://127.0.0.1:11436)
#   WEBAI_MODEL model id, required by oMLX, ignored by llama-server
# Works against llama-server (timings block) and oMLX (usage block). A random prefix defeats prompt caching.
URL=os.environ.get("WEBAI_URL","http://127.0.0.1:11436"); MODEL=os.environ.get("WEBAI_MODEL")
docs=" ".join(open(f).read() for f in sorted(glob.glob('*.md')+glob.glob('patches/*.md')))
target=int(sys.argv[1]); tag=sys.argv[2]
def post(path,body):
    r=urllib.request.Request(URL+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=7200))
def tok(t):
    body={"content":t}
    if MODEL: body["model"]=MODEL; body["prompt"]=t
    d=post("/tokenize",body); return len(d["tokens"])
per=tok(docs); text=(docs+"\n\n")*(target//per+1)
text=text[:int(len(text)*target/tok(text))]; n=tok(text)
msg=[{"role":"user","content":f"Session {uuid.uuid4()}\n\n"+text+"\n\nWrite a 300-word summary of the document above."}]
body=dict(messages=msg,max_tokens=256,temperature=0,stream=False,cache_prompt=False)
if MODEL: body["model"]=MODEL
t=time.time(); d=post("/v1/chat/completions",body); wall=time.time()-t
if "timings" in d:
    tm=d["timings"]
    out=dict(tag=tag,prompt_tokens=tm["prompt_n"],gen=tm["predicted_n"],pp_tps=round(tm["prompt_per_second"],1),tg_tps=round(tm["predicted_per_second"],1),ttft_s=round(tm["prompt_ms"]/1000,1),wall_s=round(wall,1),draft_n=tm.get("draft_n"),draft_accepted=tm.get("draft_n_accepted"))
else:
    u=d["usage"]
    out=dict(tag=tag,prompt_tokens=u["prompt_tokens"],cached=u.get("prompt_tokens_details",{}).get("cached_tokens"),gen=u["completion_tokens"],pp_tps=round(u["prompt_tokens_per_second"],1),tg_tps=round(u["generation_tokens_per_second"],1),ttft_s=round(u["time_to_first_token"],1),wall_s=round(wall,1))
print(json.dumps(out)); open(sys.argv[3],'a').write(json.dumps(out)+"\n")
