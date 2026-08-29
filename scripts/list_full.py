"""List the real/ subdir via the official KaggleApi (paginated, subdir filter)."""
import os
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "deepfake_audio_dataset_jay15k"

api = KaggleApi()
api.authenticate()

all_files = []
tok = None
for page in range(120):
    if tok:
        fl = api.dataset_list_files(DS, page_token=tok)
    else:
        fl = api.dataset_list_files(DS)
    files = fl.files
    all_files.extend(files)
    tok = getattr(fl, "nextPageToken", None)
    print(f"page {page+1}: {len(all_files)} files", flush=True)
    if not tok:
        break
    time.sleep(0.6)
print()
real = [f.name for f in all_files if "/real/" in f.name]
fake = [f.name for f in all_files if "/fake/" in f.name]
print(f"real: {len(real)}  fake: {len(fake)}")
if real:
    print("real examples:", real[:8])
if fake:
    print("fake examples:", fake[:3])
with open("data/audio_filelist_full.txt", "w") as fh:
    fh.write("\n".join(f.name for f in all_files))
print("full list saved:", len(all_files))
