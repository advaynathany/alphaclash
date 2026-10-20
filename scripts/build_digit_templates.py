"""Build clock digit templates from a recording of a real match.

The clock ticks in real seconds, so one known reading (frame, seconds left) plus
file timestamps labels every frame. Frames within 0.25 s of a predicted tick are
skipped; each digit's template is the mean of its glyphs.

Usage: uv run scripts/build_digit_templates.py data/frames/<session> <frame_idx> <M:SS> <last_idx>
e.g.   uv run scripts/build_digit_templates.py data/frames/20261019-223127 760 2:54 944
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from alphaclash.perception.clock import DIGIT_DIR, glyphs

src, anchor, mss, last = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
m, s = mss.split(":")
t0 = (src / f"{anchor:06d}.png").stat().st_mtime
left0 = int(m) * 60 + int(s)

acc = defaultdict(list)
used = 0
for i in range(anchor, last + 1):
    p = src / f"{i:06d}.png"
    left = left0 - (p.stat().st_mtime - t0)  # anchor frame sits somewhere inside its second
    frac = left - np.floor(left)
    if not 0.25 < frac < 0.75 and i != anchor:
        continue
    secs = int(np.ceil(left)) if i != anchor else left0
    g = glyphs(np.asarray(Image.open(p).convert("RGB")))
    if g is None or secs < 0:
        continue
    for d, v in zip((secs // 60, secs % 60 // 10, secs % 10), g):
        acc[d].append(v)
    used += 1

DIGIT_DIR.mkdir(parents=True, exist_ok=True)
for d, vs in sorted(acc.items()):
    v = np.mean(vs, 0)
    np.save(DIGIT_DIR / f"{d}.npy", v / np.linalg.norm(v))
print(f"{used} frames; glyphs per digit:", {d: len(v) for d, v in sorted(acc.items())})
