"""Audit published classification flags and boxes, then split by acquisition sequence."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np

from fetch_assets import DATA_MD5, ZIP, digest
from vision_io import READY, write_json, yolo_box_pixels

VAL = "R1-800um-top-to-bottom-center"
TEST_R1 = "R1-900um-top-to-bottom-center"
TEST_R7 = "R7-700um-middle"
SEED = 41


def partition(sequence):
    if sequence == VAL:
        return "val"
    if sequence in (TEST_R1, TEST_R7):
        return "test"
    if not sequence.startswith("R1-"):
        raise ValueError(f"Unexpected acquisition sequence {sequence}")
    return "train"


def read_targets(text, width, height):
    targets, kinds = [], set()
    for line in text.splitlines():
        if not line.strip():
            continue
        kind, cx, cy, bw, bh = map(float, line.split())
        numbers = np.asarray([cx, cy, bw, bh])
        if kind not in (0, 1) or not np.isfinite(numbers).all() or np.any(numbers < 0) or np.any(numbers > 1):
            raise ValueError("Malformed published annotation")
        kinds.add(int(kind))
        if kind == 0:
            box = yolo_box_pixels(cx, cy, bw, bh, width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Degenerate pinhole annotation")
            targets.append(box)
    return targets, kinds


def lines_for(boxes, w, h):
    lines = []
    for x1, y1, x2, y2 in boxes:
        values = ((x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h)
        lines.append("0 " + " ".join(f"{v:.9f}" for v in values))
    return "\n".join(lines) + ("\n" if lines else "")


def prepare():
    if not ZIP.is_file() or digest(ZIP, "md5") != DATA_MD5:
        raise ValueError("First run python fetch_assets.py data to obtain the verified archive")
    READY.parent.mkdir(parents=True, exist_ok=True)
    if (READY / "audit.json").is_file():
        print((READY / "audit.json").read_text(encoding="utf-8"))
        return
    with TemporaryDirectory(dir=READY.parent, prefix="preparing-") as work, ZipFile(ZIP) as archive:
        stage = Path(work) / "dataset"
        filenames = set(archive.namelist())
        table = list(csv.DictReader(io.StringIO(archive.read("classification/labels.csv").decode("utf-8-sig"))))
        records, excluded = [], []
        checks, decoded = Counter(), defaultdict(list)
        for row in table:
            fname = row["file_name"]
            ident = Path(fname).stem
            original = row["original_file_name"]
            match = re.fullmatch(r"(.+)_frame_(\d+)_patch_(\d+)\.png", original, re.I)
            if not match:
                raise ValueError(f"Unrecognized acquisition key {original}")
            seq, frame, patch = match.groups()
            raw = archive.read("detection/images/" + fname)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Corrupt image {fname}")
            h, w = image.shape[:2]
            labelname = "detection/labels/" + ident + ".txt"
            text = archive.read(labelname).decode("utf-8-sig") if labelname in filenames else ""
            boxes, kinds = read_targets(text, w, h)
            stated = int(row["Pinhole"])
            checks[f"CSV{stated}_boxes{int(bool(boxes))}"] += 1
            reason = ""
            if stated != int(bool(boxes)):
                reason = "pinhole_flag_box_disagreement"
            elif int(row["unclassified"]) or 1 in kinds:
                reason = "unclassified_defect"
            if reason:
                excluded.append(dict(id=ident, reason=reason))
                continue
            digest_bytes = hashlib.sha256(image.tobytes()).hexdigest()
            decoded[digest_bytes].append(ident)
            records.append(dict(id=ident, original=original, sequence=seq, source_frame=int(frame),
                                patch=int(patch), split=partition(seq), width=w, height=h,
                                boxes=boxes, sha256_pixels=digest_bytes))
        duplicate_ids = {key for ids in decoded.values() if len(ids) > 1 for key in ids}
        excluded.extend(dict(id=k, reason="duplicate_pixels") for k in sorted(duplicate_ids))
        records = [r for r in records if r["id"] not in duplicate_ids]
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Duplicate dataset IDs")
        train_pos = [r for r in records if r["split"] == "train" and r["boxes"]]
        train_neg = [r for r in records if r["split"] == "train" and not r["boxes"]]
        rng = random.Random(SEED)
        neg_used = rng.sample(train_neg, min(len(train_neg), round(1.5 * len(train_pos))))
        train_used = {r["id"] for r in train_pos + neg_used}
        stats = defaultdict(Counter)
        sequences = defaultdict(set)
        for r in records:
            sp = r["split"]
            r["used_for_fit"] = sp == "train" and r["id"] in train_used
            img = stage / "images" / sp
            lbl = stage / "labels" / sp
            img.mkdir(parents=True, exist_ok=True)
            lbl.mkdir(parents=True, exist_ok=True)
            (img / (r["id"] + ".jpg")).write_bytes(archive.read("detection/images/" + r["id"] + ".jpg"))
            (lbl / (r["id"] + ".txt")).write_text(lines_for(r["boxes"], r["width"], r["height"]), encoding="utf-8")
            for name in [sp] + (["fit_used"] if r["used_for_fit"] else []):
                stats[name]["images"] += 1
                stats[name]["pinhole_images"] += int(bool(r["boxes"]))
                stats[name]["confirmed_no_pinhole"] += int(not r["boxes"])
                stats[name]["pinholes"] += len(r["boxes"])
            sequences[sp].add(r["sequence"])
        if any(sequences[a] & sequences[b] for a,b in [("train","val"),("train","test"),("val","test")]):
            raise RuntimeError("Sequence leakage")
        (stage / "train_used.txt").write_text("".join(f"./images/train/{i}.jpg\n" for i in sorted(train_used)), encoding="utf-8")
        write_json(stage / "dataset.yaml", dict(path=READY.resolve().as_posix(), train="train_used.txt",
                                               val="images/val", test="images/test", names=["pinhole"]))
        summary = dict(source_doi="10.6084/m9.figshare.29260121.v1", md5=DATA_MD5, raw_images=len(table),
                       text_files=sum(s.startswith("detection/labels/") and s.endswith(".txt") for s in filenames),
                       classification_box_check=checks, exclusions=Counter(r["reason"] for r in excluded),
                       split_counts=stats, sequences={s:sorted(seq) for s,seq in sequences.items()},
                       train_negative_sampling_seed=SEED)
        write_json(stage / "manifest.json", records)
        write_json(stage / "excluded.json", excluded)
        write_json(stage / "audit.json", summary)
        if READY.exists():
            raise FileExistsError("Prepared destination already exists without audit.json. Inspect it before retrying.")
        stage.replace(READY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    prepare()
