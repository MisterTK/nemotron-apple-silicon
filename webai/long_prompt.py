import json, urllib.request, time, sys, glob
# usage (from the repo root, server up): python3 webai/long_prompt.py TOKENS TAG OUT.jsonl   [WEBAI_URL=http://127.0.0.1:11436]
import os; URL=os.environ.get("WEBAI_URL","http://127.0.0.1:11436")
# natural text: the repo's markdown and patch notes, repeated to the requested token count
docs=" ".join(open(f).read() for f in sorted(glob.glob('*.md')+glob.glob('patches/*.md'))+glob.glob('patches/*.notes.md'))
target=int(sys.argv[1]); tag=sys.argv[2]
def tok(t):
    r=urllib.request.Request(URL+"/tokenize",data=json.dumps({"content":t}).encode(),headers={"Content-Type":"application/json"})
    return len(json.load(urllib.request.urlopen(r))['tokens'])
per=tok(docs); text=(docs+"\n\n")*(target//per+1)
# trim by characters proportionally, then verify
text=text[:int(len(text)*target/tok(text))]; n=tok(text)
msg=[{"role":"user","content":text+"\n\nWrite a 300-word summary of the document above."}]
r=urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(dict(messages=msg,max_tokens=256,temperature=0,stream=False,cache_prompt=False)).encode(),headers={"Content-Type":"application/json"})
t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=3600)); tm=d['timings']
out=dict(tag=tag,prompt_tokens=tm['prompt_n'],gen=tm['predicted_n'],pp_tps=round(tm['prompt_per_second'],1),tg_tps=round(tm['predicted_per_second'],1),ttft_s=round(tm['prompt_ms']/1000,1),wall_s=round(time.time()-t,1),draft_n=tm.get('draft_n'),draft_accepted=tm.get('draft_n_accepted'))
print(json.dumps(out)); open(sys.argv[3],'a').write(json.dumps(out)+"\n")
