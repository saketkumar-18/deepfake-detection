"""List actual file paths under the dataset's real/ folder via v1 API list."""
import base64
import json
import os
import urllib.request
import urllib.parse

creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
DS = "jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"

for subdir in ["deepfake_audio_dataset_jay15k/real", "deepfake_audio_dataset_jay15k", "real"]:
    url = (f"https://www.kaggle.com/api/v1/datasets/list/{DS}"
           f"?dir={urllib.parse.quote(subdir)}&pageSize=50")
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        files = d.get("files", [])
        print(f"=== {subdir}: {len(files)} files (nextToken: {bool(d.get('nextPageToken'))})")
        for f in files[:12]:
            print("   ", f.get("name"), f.get("totalBytes"))
        if not files:
            print("   raw keys:", list(d.keys()))
    except Exception as e:
        print(f"=== {subdir}: ERR {str(e)[:100]}")
