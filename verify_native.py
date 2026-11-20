"""Compare C++ and Python ONNX outputs on identical pixels, then build a local Windows demo."""
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from detect import Detector, decode, letterbox
from fetch_assets import digest
from vision_io import READY, ROOT, RUNS, image_path, items, read_image, write_image, write_json

EXE = ROOT / "build/native/Release/pinhole-inspect.exe"
MODEL = RUNS / "edge/pinhole.onnx"


def run_cpp(exe, model, threshold, files):
    args = [str(exe), str(model), str(threshold), *(str(f) for f in files)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=120, check=True)
    decoded = [json.loads(line) for line in result.stdout.splitlines()]
    if [row["index"] for row in decoded] != list(range(len(files))):
        raise RuntimeError("Native output indices and input frames disagree")
    return [row["boxes"] for row in decoded]


def compare(ref, actual):
    max_xy = max_score = 0
    same = matched_boxes = 0
    for truth, found in zip(ref, actual):
        if len(truth) != len(found):
            continue
        same += 1
        if found:
            diff = np.abs(np.asarray(truth) - np.asarray(found))
            max_xy = max(max_xy, float(diff[:, :4].max()))
            max_score = max(max_score, float(diff[:, 4].max()))
            matched_boxes += len(found)
    return dict(frames=len(ref), equal_counts=same, compared_boxes=matched_boxes,
                max_abs_edge_error_px=max_xy, max_abs_score_error=max_score)


def pytorch_boxes(rows, cutoff):
    # Use identical preprocessing and NMS so this check measures the export itself.
    from train_yolo import YOLO, torch
    torch.set_num_threads(2)
    model = YOLO(str(RUNS/"fit/weights/best.pt")).model.float().cpu().eval()
    model.fuse(verbose=False)
    boxes = []
    with torch.inference_mode():
        for r in rows:
            im = read_image(image_path(r))
            x, transform = letterbox(im,512)
            raw = model(torch.from_numpy(x))
            if isinstance(raw, (tuple,list)):
                raw = raw[0]
            boxes.append(decode(raw.numpy(),transform,im.shape,cutoff))
    return boxes


def package(threshold, sample_files):
    archive_path = RUNS / "pinhole-edge-windows-x64.zip"
    names = " ".join(f'"samples\\{p.name}"' for p in sample_files)
    command = ('@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n'
               +'echo Sample order: '+', '.join(p.stem for p in sample_files)+'\r\n'
               +f'pinhole-inspect.exe pinhole.onnx {threshold} {names}\r\nset "code=%errorlevel%"\r\n'
               +'if /i not "%~1"=="--no-pause" pause\r\nexit /b %code%\r\n')
    attribution = ('本地 Windows x64 CPU 演示。解压后双击 run-demo.cmd，索引从 0 开始。每个框为原图的 '
                   '[左, 上, 右, 下, 置信度]，右和下为半开边界。输入为 640×480 图像。\n'
                   '样例图来自 Vignesh Sampath 等，CoatingVision v1，已从 JPG 解码另存 BMP。\n'
                   'https://doi.org/10.6084/m9.figshare.29260121.v1 · https://creativecommons.org/licenses/by/4.0/\n'
                   '模型从 Ultralytics YOLO11n 权重微调。许可说明 https://www.ultralytics.com/license '
                   '与 https://huggingface.co/Ultralytics/YOLO11 。ONNX Runtime 1.20.1 的许可见 notices。\n'
                   '此软件在 Windows 笔记本测试；样本和运行时间均不代表产线效果。\n')
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as z:
        for path in [EXE, EXE.with_name("onnxruntime.dll"), MODEL, RUNS/"edge/operating_point.json"]:
            z.write(path, path.name)
        for path in (ROOT / "build/ort-sdk/notices").glob("*"):
            if path.is_file():
                z.write(path, "notices/" + path.name)
        for path in sample_files:
            z.write(path, "samples/" + path.name)
        z.writestr("run-demo.cmd", command.encode("ascii"))
        z.writestr("README.txt", attribution.encode("utf-8-sig"))
    # Smoke-test the unpacked user entry point, with no Python in the launched program.
    with TemporaryDirectory() as td, ZipFile(archive_path) as z:
        z.extractall(td)
        out = subprocess.run(["cmd.exe", "/d", "/c", "run-demo.cmd", "--no-pause"], cwd=td,
                             capture_output=True, check=True, timeout=60)
        rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith(b"{")]
        if len(rows) != len(sample_files):
            raise RuntimeError("Packaged command did not complete every sample")
    return {"name": archive_path.name, "bytes": archive_path.stat().st_size,
            "sample_detections": [len(row["boxes"]) for row in rows]}


def main():
    cfg = json.loads((RUNS/"edge/operating_point.json").read_text(encoding="utf-8"))
    if digest(MODEL, "sha256") != cfg["model_sha256"]:
        raise ValueError("Calibration belongs to a different model.")
    engine = Detector(MODEL)
    t = cfg["confidence"]
    val = items("val")
    rows = [r for r in val if r["boxes"]][:16] + [r for r in val if not r["boxes"]][:16]
    report = {"model_sha256": cfg["model_sha256"], "threshold": t, "validation_ids": [r["id"] for r in rows]}
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        refs, bitmaps = [], []
        for r in rows:
            im = read_image(image_path(r))
            path = folder / (r["id"] + ".bmp")
            write_image(path, im)
            refs.append(engine.predict(im, t))
            bitmaps.append(path)
        exported = compare(refs, pytorch_boxes(rows,t))
        report["pytorch_vs_onnx"] = exported
        if (exported["equal_counts"] != len(rows) or exported["max_abs_edge_error_px"] > .1
                or exported["max_abs_score_error"] > .001):
            raise AssertionError(exported)
        lossless = compare(refs, run_cpp(EXE, MODEL, t, bitmaps))
        report["same_pixels_bmp"] = lossless
        if (lossless["equal_counts"] != len(rows) or lossless["compared_boxes"] < 1 or
                lossless["max_abs_edge_error_px"] > .01 or lossless["max_abs_score_error"] > .0001):
            raise AssertionError(lossless)
        report["raw_jpeg_wic_vs_opencv"] = compare(refs, run_cpp(EXE, MODEL, t, [image_path(r) for r in rows]))
        demos = []
        for ident in ("image_1011", "image_1622", "image_1008", "image_2"):
            path = folder / (ident+".bmp")
            write_image(path, read_image(READY/"images/train"/(ident+".jpg")))
            demos.append(path)
        report["archive"] = package(t, demos)
    write_json(RUNS / "edge/native_parity.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "validation_ids"}, indent=2))


if __name__ == "__main__":
    main()
