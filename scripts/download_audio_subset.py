"""Download a balanced audio subset from Kaggle by probing numeric file ids."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "audio_raw"
DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
PREFIX = "deepfake_audio_dataset_jay15k"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300

def download(cls: str, need: int):
    out = ROOT / cls
    out.mkdir(parents=True, exist_ok=True)
    got = len(list(out.glob("*.wav")))
    i = 0
    misses = 0
    while got < need and misses < 40 and i < 4000:
        name = f"{cls}/{i}.wav"
        dest = out / f"{i}.wav"
        i += 1
        if dest.exists():
            continue
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", DS,
             "--file", f"{PREFIX}/{name}", "--path", out, "--unzip", "--force"],
            capture_output=True, text=True,
        )
        moved = sorted(out.rglob(f"{i}.wav"))
        if r.returncode == 0 and moved:
            moved[0].replace(dest)
            # clean empty dirs from unzip layout
            for d in sorted((out).rglob("*")):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            got += 1
            misses = 0
            if got % 25 == 0:
                print(f"  {cls}: {got}/{need}", flush=True)
        else:
            misses += 1
    print(f"{cls}: done, {got} files", flush=True)

download("fake", TARGET)
download("real", TARGET)
print("AUDIO_SUBSET_DONE")
