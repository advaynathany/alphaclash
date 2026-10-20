"""Draw the detector's units on arena crops, to check it by eye.

Usage: uv run scripts/eval_detector.py runs/best.pt [images_dir] [out_dir]
Defaults: data/eval/images -> runs/eval_vis/. Red = enemy, blue = mine, grey = side unsure.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from alphaclash.perception.detector import Detector

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"enemy": (255, 60, 60), "mine": (60, 140, 255), None: (200, 200, 200)}

weights = sys.argv[1]
src = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/eval/images"
out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "runs/eval_vis"
out.mkdir(parents=True, exist_ok=True)

det = Detector(weights, side_weights=str(ROOT / "runs/side_cnn.pt"))
names: Counter[str] = Counter()
sides: Counter[str | None] = Counter()
for p in sorted(src.glob("*.jpg")):
    img = Image.open(p).convert("RGB")
    units = det.detect_arena(np.asarray(img))
    d = ImageDraw.Draw(img)
    for u in units:
        d.rectangle(u.xyxy, outline=COLORS[u.side], width=2)
        d.text((u.xyxy[0] + 2, u.xyxy[1] - 11), f"{u.name} {u.conf:.2f}", fill=COLORS[u.side])
        names[u.name] += 1
        sides[u.side] += 1
    img.save(out / p.name)
    print(p.name, [(u.name, u.side, u.conf, tuple(round(t, 1) for t in u.tile)) for u in units])
print("classes:", names.most_common(15))
print("sides:", dict(sides))
