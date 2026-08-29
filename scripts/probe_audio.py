"""Quick probe of a few candidate file ids (rate-limit friendly)."""
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "audio_raw"
creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
PFX = "deepfake_audio_dataset_jay15k"

names = sys.argv[1:] or ["fake/6.wav", "real/0.wav", "real/3.wav", "real/5.wav"]
for n in names:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DS}/{PFX}%2F{n.replace('/', '%2F')}"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
        d = urllib.request.urlopen(req, timeout=60).read()
        print(n, "->", len(d), d[:4], flush=True)
    except Exception as e:
        print(n, "->", str(e)[:80], flush=True)
    time.sleep(2.5)
