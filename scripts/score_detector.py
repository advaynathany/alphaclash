"""Score the detector against hand-checked labels from the labeling tool.

Usage: uv run scripts/score_detector.py runs/yolo_v1.pt
Matches each label to the best unmatched detection of the same class at IoU >= 0.5.
Towers are skipped: they never move and the HUD reads them separately.
Prints precision / recall overall and per class, and side accuracy on matches.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from alphaclash.perception.detector import TOWERS as SKIP, Detector

ROOT = Path(__file__).resolve().parents[1]
IOU = 0.5


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


det = Detector(sys.argv[1], side_weights=str(ROOT / "runs/side_cnn.pt"))
tp, fp, fn = Counter(), Counter(), Counter()
side_ok = side_n = 0
files = sorted((ROOT / "data/eval/labels").glob("*.json"))
for lp in files:
    truth = [b for b in json.loads(lp.read_text()) if b["name"] not in SKIP]
    units = [u for u in det.detect_arena(np.asarray(Image.open(ROOT / "data/eval/images" / f"{lp.stem}.jpg").convert("RGB")))
             if u.name not in SKIP]
    used = set()
    for t in truth:
        cands = [(iou(t["xyxy"], u.xyxy), i) for i, u in enumerate(units) if i not in used and u.name == t["name"]]
        best = max(cands, default=(0.0, -1))
        if best[0] >= IOU:
            used.add(best[1])
            tp[t["name"]] += 1
            if t["side"] is not None:
                side_n += 1
                side_ok += units[best[1]].side == t["side"]
        else:
            fn[t["name"]] += 1
    for i, u in enumerate(units):
        if i not in used:
            fp[u.name] += 1

T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
print(f"{len(files)} labeled frames, {T + N} units")
print(f"precision {T / max(1, T + F):.2f}  recall {T / max(1, T + N):.2f}  side acc {side_ok / max(1, side_n):.2f} ({side_n} sided matches)")
print(f"{'class':24} {'tp':>4} {'fp':>4} {'fn':>4}")
for name in sorted(set(tp) | set(fp) | set(fn), key=lambda n: -(tp[n] + fn[n])):
    print(f"{name:24} {tp[name]:4} {fp[name]:4} {fn[name]:4}")
