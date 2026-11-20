"""Fine-tune a one-class YOLO11n detector on the training split; select weights on validation."""
import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
from ultralytics import YOLO, settings
from vision_io import READY, RUNS, write_json


def save_for_deployment(source, dest):
    import onnx
    graph = onnx.load(source)
    # The stock exporter writes the local training run path in the model description.
    for prop in graph.metadata_props:
        if prop.key == "description":
            prop.value = "YOLO11n pinhole localization on CoatingVision v1, single class."
    onnx.checker.check_model(graph)
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def export(best, size):
    net = YOLO(str(best))
    path = Path(net.export(format="onnx", imgsz=size, batch=1, simplify=False, dynamic=False,
                           opset=17, half=False, nms=False, device="cpu"))
    dest = RUNS / "edge" / "pinhole.onnx"
    return save_for_deployment(path, dest)


def main(args):
    settings.update({"sync": False, "wandb": False, "mlflow": False, "clearml": False,
                     "comet": False, "dvc": False, "hub": False, "tensorboard": False})
    torch.set_num_threads(3)
    best = RUNS / args.name / "weights/best.pt"
    if not args.export_only:
        device = 0 if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        model = YOLO(str(ROOT / "models/yolo11n.pt"))
        model.train(data=str(READY / "dataset.yaml"), imgsz=args.size, batch=args.batch,
                    epochs=args.epochs, device=device, workers=0, seed=41, deterministic=True,
                    optimizer="AdamW", lr0=0.001, lrf=0.1, warmup_epochs=1, weight_decay=0.0005,
                    freeze=8, amp=False, mosaic=0.0, close_mosaic=0, mixup=0.0,
                    hsv_h=0.0, hsv_s=0.05, hsv_v=0.12, translate=0.04, scale=0.12,
                    plots=False, verbose=False, cache=False, project=str(RUNS), name=args.name,
                    exist_ok=True, patience=args.epochs)
        write_json(RUNS / args.name / "hardware.json",
                   {"training_device": torch.cuda.get_device_name(0) if device != "cpu" else "CPU",
                    "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2) if device != "cpu" else None,
                    "epochs": args.epochs, "batch": args.batch, "image_size": args.size, "seed": 41})
    digest = export(best, args.size)
    print("Exported ONNX", digest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default="fit")
    parser.add_argument("--export-only", action="store_true")
    main(parser.parse_args())
