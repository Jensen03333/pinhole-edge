"""Export grayscale file inputs and reference Blob measurements for VisionMaster and VisionPro."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from baseline import residual
from vision_io import READY, RUNS, read_image, write_image, write_json

OUT = RUNS / "platform"
FIELDS = ["id", "count", "total_area_px"]


def measure(gray, cutoff):
    mask = (gray >= cutoff).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if 4 <= area <= 2500:
            boxes.append(dict(xyxy=[int(x),int(y),int(x+w),int(y+h)], area_px=int(area)))
    return mask * 255, boxes


def controls(threshold):
    boundary = np.zeros((480,640), np.uint8)
    diagonal = np.zeros_like(boundary)
    area = np.zeros_like(boundary)
    for x, level in zip((20, 40, 60), (threshold-1, threshold, threshold+1)):
        boundary[20:24, x:x+4] = level
    diagonal[20:22, 20:22] = diagonal[22:24, 22:24] = 255
    area[10:12,:2] = 255
    area[40, 50] = 255
    area[100:151,100:151] = 255
    return {"control_threshold":boundary, "control_diagonal":diagonal,
            "control_border_area":area, "control_empty":np.zeros_like(boundary)}


def build():
    params = json.loads((RUNS/"baseline.json").read_text(encoding="utf-8"))
    cutoff = params["cutoff"]
    frames = controls(cutoff)
    for ident in ("image_19","image_21","image_215","image_994"):
        im = read_image(READY/"images/val"/(ident+".jpg"))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        write_image(OUT / "original_gray" / (ident+".bmp"), gray)
        frames[ident] = residual(gray, params["polarity"], params["diameter"])
    results = []
    details = {}
    for ident, im in frames.items():
        mask, boxes = measure(im, cutoff)
        write_image(OUT / "inputs" / (ident+".bmp"), im)
        write_image(OUT / "masks" / (ident+".bmp"), mask)
        results.append(dict(id=ident, count=len(boxes), total_area_px=sum(b["area_px"] for b in boxes)))
        details[ident] = boxes
    with (OUT/"reference.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(results)
    with (OUT/"blank-results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows({"id": row["id"]} for row in results)
    write_json(OUT/"boxes.json", details)
    write_json(OUT/"recipe.json", dict(polarity="light", threshold_inclusive=cutoff, connectivity=8,
               area_px_inclusive=[4,2500], include_border=True, fill_holes=False, inputs="8-bit grayscale BMP"))
    print("Prepared 8 inputs; reference rows:", results)


def read_results(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("Expected id,count,total_area_px")
        found = {}
        for row in reader:
            ident = row["id"]
            if not ident or ident in found:
                raise ValueError("Duplicate or empty input ID")
            found[ident] = (int(row["count"]), int(row["total_area_px"]))
    return found


def compare(path):
    expected, actual = read_results(OUT/"reference.csv"), read_results(path)
    delta = {k: {"ref":expected.get(k), "measured":actual.get(k)}
             for k in sorted(expected.keys() | actual.keys()) if expected.get(k) != actual.get(k)}
    return delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.compare:
        differences = compare(args.compare)
        print(json.dumps(differences) if differences else "All counts and foreground pixel totals agree.")
        raise SystemExit(1 if differences else 0)
    build()
