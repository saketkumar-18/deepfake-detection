"""Download synthetic clips from jayjoshi dataset using the EXACT known ids
from data/audio_filelist_full.txt (no probing, no 404s)."""
import base64
import json
import os
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "audio_raw"
DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
PFX = "deepfake_audio_dataset_jay15k"
NEED = 150
PAUSE = 2.5

creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()

ids = []
for line in open(ROOT.parent / "audio_filelist_full.txt"):
    line = line.strip()
    if "/fake/" in line:
        ids.append(line.split("/")[-1].replace(".wav", ""))
random.shuffle(ids)
print(f"{len(ids)} known fake ids")

out = ROOT / "fake"
out.mkdir(parents=True, exist_ok=True)
have = {p.stem for p in out.glob("*.wav")}
got, fails = len(have), 0
t0 = time.time()
for sid in ids:
    if got >= len(have) + NEED:
        break
    if sid in have:
        continue
    time.sleep(PAUSE)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DS}/{PFX}%2Ffake%2F{sid}.wav"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
        d = urllib.request.urlopen(req, timeout=60).read()
        if d[:4] == b"RIFF" and len(d) > 5000:
            (out / f"{sid}.wav").write_bytes(d)
            got += 1
            if got % 20 == 0:
                print(f"  {got} files ({time.time()-t0:.0f}s)", flush=True)
        else:
            fails += 1
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("  429 rate-limited, backing off 60s", flush=True)
            time.sleep(60)
            fails += 1
        else:
            fails += 1
    except Exception:
        fails += 1
    if fails > 30:
        print("too many failures, stopping", flush=True)
        break
print(f"DONE extra={got - len(have)} total={got} fails={fails} {time.time()-t0:.0f}s")
