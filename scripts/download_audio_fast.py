"""Parallel download of a balanced audio subset via the Kaggle v1 download API."""
import base64
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "audio_raw"
DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
PREFIX = "deepfake_audio_dataset_jay15k"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 200
WORKERS = 6

creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()


def fetch(name: str) -> tuple[int, str, bytes | None]:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DS}/{PREFIX}%2F{name.replace('/', '%2F')}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
            d = urllib.request.urlopen(req, timeout=90).read()
            if d[:4] == b"RIFF" and len(d) > 5000:
                return 0, name, d
            return 2, name, None  # not a wav
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return 1, name, None


def main(cls: str, need: int):
    out = ROOT / cls
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out.glob("*.wav")}
    ids = [i for i in range(0, 4000) if str(i) not in have]
    got, fails, i = len(have), 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = []
        while got + len(futures) < need and i < 4000 and fails < 120:
            for _ in range(WORKERS * 2):
                if i >= 4000 or got + len(futures) >= need:
                    break
                futures.append(ex.submit(fetch, f"{cls}/{i}.wav"))
                i += 1
            for f in futures:
                rc, name, data = f.result()
                if rc == 0:
                    (out / name.split("/")[-1]).write_bytes(data)
                    got += 1
                    if got % 20 == 0:
                        print(f"  {cls}: {got}/{need} ({time.time()-t0:.0f}s)", flush=True)
                elif rc == 1:
                    fails += 1
            futures = []
    print(f"{cls}: have {got} files, {fails} misses, {time.time()-t0:.0f}s", flush=True)


main("fake", TARGET)
main("real", TARGET)
print("AUDIO_SUBSET_FAST_DONE")
