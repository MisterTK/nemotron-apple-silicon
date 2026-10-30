import json, urllib.request, time, sys, glob
# usage (from the repo root, server up): python3 webai/session.py TURNS TAG OUT.jsonl   [WEBAI_URL=http://127.0.0.1:11436]
import os; URL=os.environ.get("WEBAI_URL","http://127.0.0.1:11436")
# a long agent session: every turn appends ~2000 words of natural text (a tool result) and gets a 128-token reply
docs=" ".join(open(f).read() for f in sorted(glob.glob('*.md')+glob.glob('patches/*.md')))
words=docs.split(); turns=int(sys.argv[1]); tag=sys.argv[2]; out=sys.argv[3]
msgs=[{"role":"system","content":"You are an agent operating tools. Be brief."}]
pos=0
for t in range(turns):
    chunk=" ".join(words[pos:pos+2000]); pos=(pos+2000)%max(1,len(words)-2000)
    msgs.append({"role":"user","content":f"Tool result #{t}:\n{chunk}\n\nState the next step in one sentence."})
    r=urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(dict(messages=msgs,max_tokens=128,temperature=0,stream=False)).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.load(urllib.request.urlopen(r,timeout=3600)); tm=d['timings']
    msgs.append({"role":"assistant","content":d['choices'][0]['message']['content']})
    rec=dict(tag=tag,turn=t+1,history_tokens=d['usage']['prompt_tokens'],processed=tm['prompt_n'],ttft_s=round(tm['prompt_ms']/1000,2),tg_tps=round(tm['predicted_per_second'],1),wall_s=round(time.time()-t0,1))
    print(json.dumps(rec),flush=True); open(out,'a').write(json.dumps(rec)+"\n")
