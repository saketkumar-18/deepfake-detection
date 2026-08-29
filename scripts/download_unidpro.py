"""Download the full unidpro dataset (81 files) — original vs synthetic per speaker."""
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "audio_unidpro"
DS = "unidpro/real-vs-fake-human-voice-deepfake-audio"
creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()

names = [l.strip() for l in open(ROOT.parent / "unidpro_filelist.txt") if l.strip()]
print(f"{len(names)} files to fetch")

ok, fail = 0, 0
for i, n in enumerate(names):
    dest = ROOT / n.replace("/", "_")
    if dest.exists() and dest.stat().st_size > 1000:
        ok += 1
        continue
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DS}/{n.replace('/', '%2F')}"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
        d = urllib.request.urlopen(req, timeout=90).read()
        if len(d) > 1000:
            dest.write_bytes(d)
            ok += 1
        else:
            fail += 1
    except Exception as e:
        fail += 1
        print(f"  FAIL {n}: {str(e)[:60]}")
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(names)} ok={ok} fail={fail}", flush=True)
    time.sleep(2.2)

print(f"DONE ok={ok} fail={fail}")
