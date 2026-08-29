"""List all files of the unidpro real-vs-fake audio dataset via KaggleApi."""
import time
from kaggle.api.kaggle_api_extended import KaggleApi

DS = "unidpro/real-vs-fake-human-voice-deepfake-audio"

api = KaggleApi()
api.authenticate()

all_files = []
tok = None
for page in range(200):
    fl = api.dataset_list_files(DS, page_token=tok) if tok else api.dataset_list_files(DS)
    all_files.extend(fl.files)
    tok = getattr(fl, "nextPageToken", None)
    print(f"page {page+1}: {len(all_files)} files", flush=True)
    if not tok:
        break
    time.sleep(0.5)

import collections
c = collections.Counter()
for f in all_files:
    parts = f.name.split("/")
    c[parts[0] + ("/" + parts[1] if len(parts) > 3 else "")] += 1
print(c.most_common(12))
with open("data/unidpro_filelist.txt", "w") as fh:
    fh.write("\n".join(f.name for f in all_files))
print("saved", len(all_files))
