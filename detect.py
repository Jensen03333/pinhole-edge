"""CPU ONNX inference and postprocessing for one-class YOLO11n pinhole detection."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from fetch_assets import digest
from vision_io import RUNS, annotate, read_image, write_image


def letterbox(image, size):
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = max(1, round(w * gain)), max(1, round(h * gain))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return x, (nw / w, nh / h, left, top)


def ious(box, others):
    boxes = np.asarray(others, dtype=np.float64).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = (box[2]-box[0])*(box[3]-box[1])
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return inter / np.maximum(1e-12, area + areas - inter)


def decode(output, transform, shape, threshold=0.01, nms_iou=0.5, limit=200):
    if output.ndim != 3 or output.shape[:2] != (1, 5):
        raise ValueError(f"Expected [1, 5, anchors], got {output.shape}")
    candidates = output[0].T
    candidates = candidates[candidates[:, 4] >= threshold]
    if not len(candidates):
        return []
    sx, sy, left, top = transform
    centers, halves = candidates[:, :2], candidates[:, 2:4] / 2
    boxes = np.concatenate((centers - halves, centers + halves), axis=1).astype(np.float64)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / sx, 0, shape[1])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / sy, 0, shape[0])
    scores = candidates[:, 4]
    valid = np.flatnonzero(np.all(boxes[:, 2:] > boxes[:, :2], axis=1))
    order = valid[np.argsort(-scores[valid], kind="stable")]
    detections = []
    while len(order) and len(detections) < limit:
        first = order[0]
        detections.append([*boxes[first].tolist(), float(scores[first])])
        rest = order[1:]
        order = rest[ious(boxes[first], boxes[rest]) <= nms_iou]
    return detections


class Detector:
    def __init__(self, model):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        info = self.session.get_inputs()[0]
        self.name, self.size = info.name, info.shape[2]
        if info.shape != [1, 3, self.size, self.size]:
            raise ValueError("This runner requires static square inputs and batch 1")

    def predict(self, image, threshold=0.01):
        x, transform = letterbox(image, self.size)
        result = self.session.run(None, {self.name: x})[0]
        return decode(result, transform, image.shape, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=RUNS / "edge/pinhole.onnx")
    parser.add_argument("--settings", type=Path, default=RUNS / "edge/operating_point.json")
    parser.add_argument("--out", type=Path, default=RUNS / "edge/prediction.png")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    if digest(args.model, "sha256") != settings["model_sha256"]:
        parser.error("The operating point belongs to a different model.")
    img = read_image(args.image)
    detections = Detector(args.model).predict(img, settings["confidence"])
    write_image(args.out, annotate(img, detections, (60, 220, 80)))
    print(json.dumps({"count": len(detections), "xyxy_score": detections}))
