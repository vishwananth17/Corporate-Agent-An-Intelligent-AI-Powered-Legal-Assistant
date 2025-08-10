import os
import json
import time
import re
from typing import List, Dict

import requests


def _safe_filename_from_url(url: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if not name:
        name = re.sub(r"\W+", "_", url)[:60]
    return name


def download_official_references(json_path: str, dest_folder: str) -> List[Dict[str, str]]:
    os.makedirs(dest_folder, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    saved: List[Dict[str, str]] = []
    headers = {"User-Agent": "Corporate-Agent/1.0"}
    for ent in entries:
        url = ent.get("url")
        if not url:
            continue
        fname = _safe_filename_from_url(url)
        fpath = os.path.join(dest_folder, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            saved.append({"name": ent.get("name", fname), "file": fpath})
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200 and resp.content:
                with open(fpath, "wb") as out:
                    out.write(resp.content)
                saved.append({"name": ent.get("name", fname), "file": fpath})
            else:
                # fallback create a .txt pointer with URL
                pointer = fpath + ".url.txt"
                with open(pointer, "w", encoding="utf-8") as out:
                    out.write(url)
                saved.append({"name": ent.get("name", fname), "file": pointer})
            time.sleep(0.2)
        except Exception:
            pointer = fpath + ".url.txt"
            with open(pointer, "w", encoding="utf-8") as out:
                out.write(url)
            saved.append({"name": ent.get("name", fname), "file": pointer})
    return saved


