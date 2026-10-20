"""Pre-fill the labeling tool with the detector's guesses.

Usage: uv run scripts/prelabel_eval.py runs/yolo_v1.pt
Writes data/eval/prelabels/<stem>.json (never touches hand-saved data/eval/labels/).
Uses a lower confidence than the live detector, since deleting a wrong box is
quicker than drawing a missing one.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from alphaclash.perception.detector import Detector

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data/eval/prelabels"
out.mkdir(parents=True, exist_ok=True)
det = Detector(sys.argv[1], side_weights=str(ROOT / "runs/side_cnn.pt"), conf=0.25)
n = 0
for p in sorted((ROOT / "data/eval/images").glob("*.jpg")):
    units = det.detect_arena(np.asarray(Image.open(p).convert("RGB")))
    (out / f"{p.stem}.json").write_text(json.dumps(
        [{"name": u.name, "side": u.side, "xyxy": [round(v, 1) for v in u.xyxy]} for u in units], indent=0))
    n += len(units)
print(f"{n} boxes -> {out}")
