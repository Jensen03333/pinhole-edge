"""Lesson 1. Draw published pinhole boxes from the YOLO text file on a real frame."""
import argparse
from math import ceil, floor

import cv2
from PIL import Image, ImageDraw

from vision_io import READY, ROOT, RUNS, image_path, items, read_image, write_image, write_json, yolo_box_pixels


def draw_one(image_file, label_file, output_file):
    image = read_image(image_file)
    height, width = image.shape[:2]
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, bw, bh = map(float, line.split())
        if class_id != 0:
            continue
        box = yolo_box_pixels(cx, cy, bw, bh, width, height)
        x1, y1, x2, y2 = (round(edge, 5) for edge in box)
        # OpenCV takes inclusive integer endpoints. The YOLO box uses continuous edges.
        cv2.rectangle(image, (floor(x1), floor(y1)), (ceil(x2)-1, ceil(y2)-1), (45, 225, 255), 2)
        boxes.append(box)
    write_image(output_file, image)
    return boxes


def make_examples():
    rows = items("train")
    positive = sorted((r for r in rows if r["boxes"]), key=lambda r: (len(r["boxes"]), r["id"]))
    negative = next(r for r in rows if not r["boxes"])
    one = next((r for r in positive if len(r["boxes"]) == 1 and
                min(r["boxes"][0][2]-r["boxes"][0][0], r["boxes"][0][3]-r["boxes"][0][1]) >= 12), positive[0])
    multiple = next((r for r in positive if 3 <= len(r["boxes"]) <= 8), positive[-1])
    small = next((r for r in positive if r["id"] not in (one["id"], multiple["id"]) and
                  min(max(b[2]-b[0], b[3]-b[1]) for b in r["boxes"]) < 22), positive[len(positive)//2])
    out = RUNS / "lesson"
    out.mkdir(parents=True, exist_ok=True)
    notes = []
    sheet = Image.new("RGB", (900, 2 * 373 + 56), "#f4f7f6")
    draw = ImageDraw.Draw(sheet)
    draw.text((15,10), "CoatingVision | Published pinhole boxes | Training samples", fill="#112b34")
    for i, r in enumerate([one, multiple, small, negative]):
        image = image_path(r)
        labels = READY / "labels/train" / (r["id"] + ".txt")
        dst = out / (r["id"] + ".png")
        boxes = draw_one(image, labels, dst)
        assert len(boxes) == len(r["boxes"])
        notes.append(dict(id=r["id"], original_width=r["width"], original_height=r["height"],
                          pinholes=len(boxes), first_yolo_line=next(iter(labels.read_text().splitlines()), None),
                          first_xyxy_px=boxes[0] if boxes else None))
        with Image.open(dst) as im:
            im.thumbnail((430, 325))
            x, y = 14 + (i % 2)*450, 40 + (i//2)*373
            sheet.paste(im, (x,y))
            draw.text((x,y+329), f'{r["id"]}   pinholes: {len(boxes)}', fill="#132d34")
    draw.text((14, sheet.height-19), "Source: CoatingVision (Sampath et al.) | CC BY 4.0 | boxes overlaid, resized for display", fill="#31464b")
    sheet.save(out / "annotation_gallery.png")
    preview = ROOT / "docs/examples/annotations.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview, quality=92)
    write_json(out / "examples.json", notes)
    return notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="A prepared training image ID, e.g. image_123")
    args = parser.parse_args()
    if args.id:
        row = next((r for r in items("train") if r["id"] == args.id), None)
        if row is None:
            parser.error("Choose an ID in the training manifest")
        boxes = draw_one(image_path(row), READY / "labels/train" / (args.id + ".txt"),
                         RUNS / "lesson" / (args.id + ".png"))
        print(args.id, len(boxes), boxes[:1])
    else:
        for example in make_examples():
            print(example)
