"""Object-level metrics. A prediction may match one unused ground-truth box per image."""
import numpy as np
from detect import ious


def match_image(predictions, truth, iou_min=0.5):
    predictions = sorted(predictions, key=lambda p: -p[4])
    gt = np.asarray(truth, dtype=float).reshape(-1, 4)
    unused = np.ones(len(gt), dtype=bool)
    matched = []
    for p in predictions:
        chosen = -1
        if unused.any():
            overlaps = ious(np.asarray(p[:4]), gt)
            overlaps[~unused] = -1
            index = int(np.argmax(overlaps))
            if overlaps[index] >= iou_min:
                chosen = index
                unused[index] = False
        matched.append((float(p[4]), chosen))
    return matched


def matched_rows(rows, predictions):
    return [(r, match_image(predictions[r["id"]], r["boxes"])) for r in rows]


def at_threshold(matched, threshold):
    tp = fp = gt_total = small = small_tp = bg = bg_flagged = 0
    for row, candidates in matched:
        truth = row["boxes"]
        tiny = [(b[2]-b[0])*(b[3]-b[1]) < 16**2 for b in truth]
        gt_total += len(truth)
        small += sum(tiny)
        active = [(s, i) for s,i in candidates if s >= threshold]
        for _, index in active:
            if index >= 0:
                tp += 1
                small_tp += int(tiny[index])
            else:
                fp += 1
        if not truth:
            bg += 1
            bg_flagged += int(bool(active))
    fn = gt_total - tp
    return dict(images=len(matched), gt=gt_total, tp=tp, fp=fp, fn=fn, precision=tp/max(1,tp+fp),
                recall=tp/max(1,gt_total), f1=2*tp/max(1,2*tp+fp+fn), false_boxes_per_image=fp/max(1,len(matched)),
                small_gt=small, small_tp=small_tp, small_recall=small_tp/small if small else None,
                no_pinhole_images=bg, no_pinhole_flagged=bg_flagged)


def ap50(matched):
    flags = sorted([(score, int(index >= 0)) for _, entries in matched for score,index in entries],
                   key=lambda x: -x[0])
    positives = sum(len(row["boxes"]) for row,_ in matched)
    if not flags or positives == 0:
        return 0.0
    true = np.cumsum([yes for _,yes in flags])
    recall = true / positives
    precision = true / np.arange(1, len(flags) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    values = [precision[i] if i < len(precision) else 0.0
              for i in np.searchsorted(recall, np.linspace(0, 1, 101), side="left")]
    return float(np.mean(values))


def pick_threshold(matched, grid):
    # Highest F1, then fewest false boxes, then highest threshold. Validation split only.
    scored = [(at_threshold(matched, float(t)), float(t)) for t in grid]
    return max(scored, key=lambda pair:(pair[0]["f1"], -pair[0]["fp"], pair[1]))
