"""Paths, Unicode-safe image I/O and half-open box coordinates in source pixels."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
READY = ROOT / "data" / "prepared-v1"
RUNS = ROOT / "runs"


def read_image(path):
    buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR) if len(buf) else None
    if image is None:
        raise ValueError(f"Cannot decode {Path(path).name}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Image encoding failed for {path.name}")
    path.write_bytes(buf.tobytes())


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def yolo_box_pixels(cx, cy, bw, bh, width, height):
    x1 = max(0.0, (cx - bw / 2) * width)
    y1 = max(0.0, (cy - bh / 2) * height)
    x2 = min(float(width), (cx + bw / 2) * width)
    y2 = min(float(height), (cy + bh / 2) * height)
    return [x1, y1, x2, y2]


def items(split):
    manifest = json.loads((READY / "manifest.json").read_text(encoding="utf-8"))
    return [r for r in manifest if r["split"] == split]


def image_path(row):
    return READY / "images" / row["split"] / (row["id"] + ".jpg")


def annotate(image, boxes, color=(45, 225, 255)):
    painted = image.copy()
    h, w = painted.shape[:2]
    for box in boxes:
        x1, y1 = np.floor(np.round(box[:2], 5)).astype(int)
        x2, y2 = np.ceil(np.round(box[2:4], 5)).astype(int) - 1
        cv2.rectangle(painted, (max(0, x1), max(0, y1)), (min(w-1, x2), min(h-1, y2)), color, 2)
    return painted
