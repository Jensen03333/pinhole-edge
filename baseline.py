"""Classical pinhole detector: local morphology, connected regions, shape and contrast."""
from itertools import product

import cv2
import numpy as np

from metrics import matched_rows, pick_threshold
from vision_io import RUNS, image_path, items, read_image, write_json


def residual(gray, polarity, diameter):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    operation = cv2.MORPH_TOPHAT if polarity == "bright" else cv2.MORPH_BLACKHAT
    return cv2.morphologyEx(gray, operation, kernel)


def regions(diff, cutoff):
    mask = (diff >= cutoff).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return []
    # Average residual measures how distinct a region is from its neighborhood.
    weights = np.bincount(labels.ravel(), weights=diff.ravel(), minlength=count)
    x, y, w, h, area = stats[1:].T
    keep = ((area >= 4) & (area <= 2500) & (np.minimum(w,h) >= 2) &
            (np.maximum(w,h) <= 3.5*np.minimum(w,h)) & (area >= 0.30*w*h))
    detections = []
    for idx in np.flatnonzero(keep):
        score = float(weights[idx+1] / (area[idx]*255))
        detections.append([int(x[idx]),int(y[idx]),int(x[idx]+w[idx]),int(y[idx]+h[idx]),score])
    return sorted(detections, key=lambda det:-det[4])[:200]


def predict(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = residual(gray, params["polarity"], params["diameter"])
    found = [p for p in regions(filtered, params["cutoff"]) if p[4] >= params["confidence"]]
    return expand(found, params["margin_px"], gray.shape)


def expand(found, margin, shape):
    # The bright core and the published box can differ by a few pixels on each edge.
    height, width = shape
    return [[max(0, x1-margin), max(0, y1-margin), min(width, x2+margin),
             min(height, y2+margin), score] for x1, y1, x2, y2, score in found]


def calibrate():
    cv2.setNumThreads(1)
    rows = items("val")
    images = [(r, cv2.cvtColor(read_image(image_path(r)), cv2.COLOR_BGR2GRAY)) for r in rows]
    grid = [v / 255 for v in (0, 8, 16, 24, 32, 48, 64, 80, 112)]
    trials = []
    for polarity, diameter in product(("bright","dark"), (15, 31)):
        detections = {t:{} for t in (10, 22, 38)}
        for row, gray in images:
            diff = residual(gray, polarity, diameter)
            for cutoff, found in detections.items():
                found[row["id"]] = regions(diff, cutoff)
        for cutoff, found in detections.items():
            for margin in (0, 2, 4):
                boxes = {r["id"]: expand(found[r["id"]], margin, (r["height"], r["width"])) for r in rows}
                metrics, score = pick_threshold(matched_rows(rows, boxes), grid)
                trials.append(dict(polarity=polarity, diameter=diameter, cutoff=cutoff, margin_px=margin,
                                   confidence=score, val_f1=metrics["f1"], val_fp=metrics["fp"], val_tp=metrics["tp"]))
    best = max(trials, key=lambda r:(r["val_f1"], -r["val_fp"]))
    write_json(RUNS / "baseline_trials.json", trials)
    write_json(RUNS / "baseline.json", best)
    print(best, flush=True)


if __name__ == "__main__":
    calibrate()
