import glob, os, sys, numpy as np
from mlx_lm.utils import load_tokenizer
from pathlib import Path
for m in sys.argv[1:]:
    tok = load_tokenizer(Path(m))
    LIB = os.path.dirname(os.__file__); ids = []
    for f in sorted(glob.glob(os.path.join(LIB, "**", "*.py"), recursive=True)):
        try: txt = open(f, encoding="utf-8").read()
        except Exception: continue
        ids += tok.encode(f"\n# ==== {os.path.relpath(f, LIB)} ====\n{txt}", add_special_tokens=False)
        if len(ids) > 9_000_000: break
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fixture_tokens.{os.path.basename(m.rstrip('/'))}.npy")
    np.save(out, np.array(ids, dtype=np.int32)); print(m, len(ids), out)
