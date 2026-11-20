"""Choose both operating points on validation, then score both methods on untouched sequences."""
import argparse
from collections import defaultdict
from html import escape
import json
from time import perf_counter_ns

import cv2
import numpy as np
from PIL import Image, ImageDraw

from baseline import predict as classical
from detect import Detector
from fetch_assets import digest
from metrics import ap50, at_threshold, matched_rows, match_image, pick_threshold
from vision_io import READY, ROOT, RUNS, annotate, image_path, items, read_image, write_image, write_json

MODEL = RUNS / "edge/pinhole.onnx"
SETTINGS = RUNS / "edge/operating_point.json"
OUT = RUNS / "comparison"
GRID = [.01, .03, .05, .075, .10, .15, .20, .25, .30, .35, .40, .45,
        .50, .55, .60, .65, .70, .75, .80, .85, .90]


def scores(rows, predictions, threshold):
    matched = matched_rows(rows, predictions)
    return {**at_threshold(matched, threshold), "ap50_101": ap50(matched)}


def audit_gallery(rows, final):
    """Select error types deterministically; every retained test result is in predictions.json."""
    cases = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        rid, truth = row["id"], row["boxes"]
        for method in ("opencv", "yolo"):
            pred = final[rid][method]
            matches = match_image(pred, truth)
            tp = sum(i >= 0 for _, i in matches)
            for tag, present in (("漏检", tp < len(truth)), ("误报框", len(pred) > tp),
                                 ("检出针孔", tp > 0), ("无针孔且无框", not truth and not pred)):
                if present and len(cases[method,tag]) < 2:
                    cases[method,tag].append(row)
    folder = OUT / "review"
    tiles = []
    for (method, tag), group in cases.items():
        for row in group:
            rid = row["id"]
            path = folder / f"{method}_{rid}.jpg"
            truth, pred = row["boxes"], final[rid][method]
            canvas = annotate(read_image(image_path(row)), truth, (40, 212, 255))
            # Thin predictions leave the wider reference outlines visible when they coincide.
            for x1, y1, x2, y2, _ in pred:
                cv2.rectangle(canvas, (int(x1), int(y1)), (max(int(x1), int(np.ceil(x2))-1),
                              max(int(y1), int(np.ceil(y2))-1)), (50, 218, 82), 1)
            write_image(path, canvas)
            tiles.append(f'<figure><a href="{path.name}"><img loading="lazy" src="{path.name}"></a>'
                         f'<figcaption>{method} · {tag} · {rid}<br>{escape(row["sequence"])} · '
                         f'标注 {len(truth)}，预测 {len(pred)}</figcaption></figure>')
    folder.mkdir(parents=True, exist_ok=True)
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>针孔定位错例</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:0 1em;color:#142c33}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}'
            'figure{margin:0}img{width:100%}figcaption{padding:.4em 0 1em}</style>'
            '<h1>针孔定位错例与对照</h1><p>黄框为公开人工标注，绿框为模型输出。点击原尺寸查看。'
            '按图像编号为每种结果选至多两例，一张图可能同时有误报和漏检。</p><main>'
            + ''.join(tiles) + '</main><p>图像经叠框和 JPEG 编码，来自 Vignesh Sampath 等人的 '
            '<a href="https://doi.org/10.6084/m9.figshare.29260121.v1">CoatingVision v1</a>，'
            '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。</p></html>')
    (folder / "index.html").write_text(html, encoding="utf-8")


def teaching_gallery(rows, final):
    """Show the same original-pixel crop in all columns, for two explicit error conditions."""
    failure, alarm = None, None
    for r in sorted(rows,key=lambda row:int(row["id"].split("_")[1])):
        preds = final[r["id"]]
        if (failure is None and len(r["boxes"]) == 1 and not preds["yolo"]
                and any(i >= 0 for _,i in match_image(preds["opencv"],r["boxes"]))):
            failure = (r,r["boxes"][0],"OpenCV located the pinhole, YOLO returned no box")
        if (alarm is None and r["sequence"].startswith("R7") and not r["boxes"]
                and 1 <= len(preds["opencv"]) <= 6 and not preds["yolo"]):
            alarm = (r,preds["opencv"][0],"OpenCV flagged a region on an image with no labeled pinhole")
        if failure and alarm:
            break
    cases = [c for c in (failure, alarm) if c]
    if not cases:
        return
    sheet = Image.new("RGB", (924, 48+len(cases)*242+30), "#f4f7f5")
    draw = ImageDraw.Draw(sheet)
    for i, label in enumerate(("Human label (yellow)", "OpenCV (green)", "YOLO11n (green)")):
        draw.text((12+306*i,12), label, fill="#0d3035")
    notes = []
    for num,(r,focus,label) in enumerate(cases):
        im = read_image(image_path(r))
        x = int(np.clip((focus[0]+focus[2])/2-75,0,im.shape[1]-150))
        y = int(np.clip((focus[1]+focus[3])/2-50,0,im.shape[0]-100))
        rid, pred = r["id"], final[r["id"]]
        ypage = 38+num*242
        for i, (boxes, color) in enumerate(((r["boxes"],(45,225,255)),(pred["opencv"],(45,220,75)),
                                            (pred["yolo"],(45,220,75)))):
            frame = annotate(im,boxes,color)[y:y+100,x:x+150]
            tile = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((300,200),Image.Resampling.NEAREST)
            sheet.paste(tile,(12+i*306,ypage))
        draw.text((12,ypage+205), f'{rid}   |   {label}   |   source crop x={x}, y={y}, 150 x 100 px', fill="#193b3f")
        notes.append(dict(id=rid,sequence=r["sequence"],crop=[x,y,x+150,y+100],condition=label,
                          published=len(r["boxes"]),opencv=len(pred["opencv"]),yolo=len(pred["yolo"])))
    draw.text((12,sheet.height-20), "CoatingVision v1 - Vignesh Sampath - CC BY 4.0 - cropped, enlarged and overlaid", fill="#365158")
    folder = ROOT / "docs/examples"
    folder.mkdir(parents=True,exist_ok=True)
    sheet.save(folder / "comparison.jpg",quality=94)
    write_json(folder / "comparison.json",notes)


def benchmark(detector, params, confidence):
    rows = items("val")
    indices = np.linspace(0, len(rows)-1, 64, dtype=int)
    images = [read_image(image_path(rows[i])) for i in indices]
    runners = {"opencv": lambda im: classical(im, params), "yolo": lambda im: detector.predict(im, confidence)}
    trials = {name: [] for name in runners}
    for run in runners.values():
        for img in images[:8]:
            run(img)
    for trial in range(3):
        order = ["opencv", "yolo"] if trial % 2 == 0 else ["yolo", "opencv"]
        for name in order:
            times = []
            for img in images:
                start = perf_counter_ns()
                runners[name](img)
                times.append((perf_counter_ns() - start) / 1e6)
            trials[name].append(times)
    clocks = {name: dict(unique_frames=len(images), repeats=len(rounds), warmup=8,
                         p50_ms=float(np.median(rounds)), p95_ms=float(np.percentile(rounds, 95)),
                         round_p50_ms=[float(np.median(r)) for r in rounds],
                         round_p95_ms=[float(np.percentile(r,95)) for r in rounds]) for name,rounds in trials.items()}
    write_json(OUT / "timing.json", dict(input="decoded BGR 640x480 in RAM", hardware="local CPU",
               opencv_threads=1, onnx_threads=2, excludes=["file I/O", "camera", "PLC", "render"], methods=clocks))
    print("Timing (ms)", clocks, flush=True)


def evaluate(detector, params):
    val = items("val")
    val_raw = {r["id"]: detector.predict(read_image(image_path(r)), min(GRID)) for r in val}
    best, confidence = pick_threshold(matched_rows(val, val_raw), GRID)
    config = dict(confidence=confidence, nms_iou=.5, max_detections=200, io="RGB FP32 / 255, NCHW",
                  model_sha256=digest(MODEL, "sha256"), manifest_sha256=digest(READY/"manifest.json","sha256"),
                  baseline=params, selection="validation only, best F1 then fewest FP then highest threshold",
                  validation_gt=best["gt"], yolo_validation=best, confidence_grid=GRID)
    write_json(SETTINGS, config)
    test = items("test")
    raw = {"opencv": {}, "yolo": {}}
    final = {}
    uncut = {**params, "confidence": 0.0}
    thresholds = {"opencv": params["confidence"], "yolo": confidence}
    for r in test:
        rid = r["id"]
        im = read_image(image_path(r))
        raw["opencv"][rid] = classical(im, uncut)
        raw["yolo"][rid] = detector.predict(im, min(GRID))
        final[rid] = {name: [p for p in found[rid] if p[4] >= thresholds[name]] for name, found in raw.items()}
    groups = {"all": test}
    for seq in sorted({r["sequence"] for r in test}):
        groups[seq] = [r for r in test if r["sequence"] == seq]
    report = {group: {name: scores(rows, raw[name], thresholds[name]) for name in raw}
              for group, rows in groups.items()}
    write_json(OUT / "test.json", report)
    write_json(OUT / "predictions.json", final)
    audit_gallery(test, final)
    teaching_gallery(test, final)
    print("Test", json.dumps(report["all"], ensure_ascii=False), flush=True)
    return confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    engine = Detector(MODEL)
    params = json.loads((RUNS / "baseline.json").read_text(encoding="utf-8"))
    if "margin_px" not in params:
        raise ValueError("Run baseline.py to finish validation calibration first.")
    if args.timing_only:
        frozen = json.loads(SETTINGS.read_text(encoding="utf-8"))
        if digest(MODEL, "sha256") != frozen["model_sha256"] or params != frozen["baseline"]:
            raise ValueError("Operating point is for a different model or baseline.")
        cutoff = frozen["confidence"]
    else:
        cutoff = evaluate(engine, params)
    benchmark(engine, params, cutoff)
