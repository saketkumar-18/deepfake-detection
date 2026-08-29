"""Politely download a balanced audio subset (rate-limit friendly: 1 req / ~2.5s)."""
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "audio_raw"
DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
PREFIX = "deepfake_audio_dataset_jay15k"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 150
PAUSE = 2.5

creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()


def fetch(name: str) -> bytes | None:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DS}/{PREFIX}%2F{name.replace('/', '%2F')}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
            d = urllib.request.urlopen(req, timeout=90).read()
            if d[:4] == b"RIFF" and len(d) > 5000:
                return d
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(3)
    return None


def main(cls: str, need: int):
    out = ROOT / cls
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out.glob("*.wav")}
    got, i, misses = len(have), 0, 0
    t0 = time.time()
    while got < need and i < 4000 and misses < 80:
        sid = str(i)
        i += 1
        if sid in have:
            continue
        time.sleep(PAUSE)
        d = fetch(f"{cls}/{sid}.wav")
        if d:
            (out / f"{sid}.wav").write_bytes(d)
            have.add(sid)
            got += 1
            misses = 0
            if got % 10 == 0:
                el = time.time() - t0
                print(f"  {cls}: {got}/{need} ({el:.0f}s, ~{(el/got):.1f}s/file)", flush=True)
        else:
            misses += 1
    print(f"{cls}: DONE {got} files, {misses} misses, {time.time()-t0:.0f}s", flush=True)


main("fake", TARGET)
main("real", TARGET)
print("AUDIO_SUBSET_POLITE_DONE")
