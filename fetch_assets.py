"""Fetch the versioned CoatingVision archive or the official YOLO11n starting weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ZIP = DATA / "CoatingVision-v1.zip"
DATA_URL = "https://ndownloader.figshare.com/files/55182476"
DATA_SIZE = 94_296_797
DATA_MD5 = "7cbbf14725f9762a64a81a12b858a2b4"
WEIGHTS_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
WEIGHTS_REVISION = "8b8ac7d1fae7468f85dbf89670dd66f41f485aab"


def digest(path, algo):
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset():
    DATA.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and digest(ZIP, "md5") == DATA_MD5:
        print("Verified", ZIP.name)
        return
    cache = DATA / ".download"
    cache.mkdir(exist_ok=True)
    block_size = 2 * 1024 * 1024
    def get_block(left):
        right = min(left + block_size, DATA_SIZE) - 1
        path = cache / str(left)
        if path.is_file() and path.stat().st_size == right-left+1:
            return path
        for attempt in range(5):
            try:
                with requests.get(DATA_URL, params={"offset": left, "end": right, "attempt": attempt},
                                  headers={"Range": f"bytes={left}-{right}", "Accept-Encoding": "identity"},
                                  timeout=(12, 45), stream=True) as response:
                    response.raise_for_status()
                    if response.headers.get("Content-Range") != f"bytes {left}-{right}/{DATA_SIZE}":
                        raise OSError("Unexpected byte range")
                    content = response.content
                    if len(content) != right-left+1:
                        raise OSError("Incomplete block")
                temp = path.with_suffix(".part")
                temp.write_bytes(content)
                temp.replace(path)
                return path
            except (requests.RequestException, OSError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=5) as pool:
        blocks = list(pool.map(get_block, range(0, DATA_SIZE, block_size)))
    combined = ZIP.with_suffix(".part")
    with combined.open("wb") as out:
        for path in blocks:
            out.write(path.read_bytes())
    if digest(combined, "md5") != DATA_MD5:
        for path in blocks:
            path.unlink(missing_ok=True)
        raise ValueError("Figshare checksum mismatch. Retry the download.")
    combined.replace(ZIP)
    for path in blocks:
        path.unlink(missing_ok=True)
    print("Verified CoatingVision v1", DATA_SIZE, DATA_MD5, flush=True)


def weights():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    sha = WEIGHTS_SHA256
    path = folder / "yolo11n.pt"
    if not path.exists() or digest(path, "sha256") != sha:
        url = f"https://huggingface.co/Ultralytics/YOLO11/resolve/{WEIGHTS_REVISION}/yolo11n.pt"
        with requests.get(url, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            temp = path.with_suffix(".part")
            with temp.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    out.write(chunk)
        if digest(temp, "sha256") != sha:
            raise ValueError("Official weights checksum mismatch")
        temp.replace(path)
    print(path.name, "SHA-256", sha, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=["data", "weights"])
    option = parser.parse_args().asset
    dataset() if option == "data" else weights()
