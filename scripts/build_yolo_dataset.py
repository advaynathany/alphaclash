"""Convert KataCR's hand-labeled arena frames into an Ultralytics YOLO dataset.

KataCR label lines: `cls x y w h side 0 0 0 0 0 0` (normalized xywh, side 1 = enemy).
Output classes are unit types only. Side (ally/enemy) is dropped: a unit looks the
same on both sides apart from its red/blue bar, so pooling sides gives each unit
far more examples (KataCR has 0 ally minions but 239 enemy ones). Side is decided
after detection from the bar color.
Images are symlinked, not copied. Train/val split is by episode folder, so
near-duplicate neighbouring frames never land on both sides.

Usage: uv run scripts/build_yolo_dataset.py
"""

import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/katacr/dataset/images/part2"
OUT = ROOT / "data/yolo"
VAL_FRAC = 0.15

names = [n for _, n in re.findall(r"^\s+(\d+): (\S+)", (SRC / "ClashRoyale_detection.yaml").read_text(), re.M)]
names = [n for n in names if not n.startswith("pad")]

frames = sorted(p for p in SRC.glob("*/*/*.jpg") if p.with_suffix(".txt").exists())
episodes = sorted({p.parent for p in frames})
random.Random(0).shuffle(episodes)
val_eps = set(episodes[: max(1, int(len(episodes) * VAL_FRAC))])

classes: dict[str, int] = {}
counts: Counter[str] = Counter()
for split in ("train", "val"):
    for sub in ("images", "labels"):
        (OUT / sub / split).mkdir(parents=True, exist_ok=True)

n = {"train": 0, "val": 0}
for p in frames:
    split = "val" if p.parent in val_eps else "train"
    rows = []
    for line in p.with_suffix(".txt").read_text().split("\n"):
        f = line.split()
        if len(f) < 6:
            continue
        cls = int(f[0])
        if cls >= len(names):
            continue
        name = names[cls]
        idx = classes.setdefault(name, len(classes))
        counts[name] += 1
        rows.append(f"{idx} {' '.join(f[1:5])}")
    stem = f"{p.parent.parent.name}_{p.parent.name}_{p.stem}"
    img = OUT / "images" / split / f"{stem}.jpg"
    if not img.exists():
        img.symlink_to(p)
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + "\n")
    n[split] += 1

yaml = [f"path: {OUT}", "train: images/train", "val: images/val", "names:"]
yaml += [f"  {i}: {name}" for name, i in sorted(classes.items(), key=lambda kv: kv[1])]
(OUT / "dataset.yaml").write_text("\n".join(yaml) + "\n")

print(f"frames: {n}  episodes: {len(episodes)} ({len(val_eps)} val)  classes: {len(classes)}")
print("most common:", counts.most_common(8))
print("rarest:", counts.most_common()[-8:])
