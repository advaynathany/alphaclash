"""Train a small CNN that says whether a detected unit is mine or the enemy's.

Uses KataCR's ground-truth sides. Input is the unit box plus the band above it
(where the level badge / HP bar sits), resized to 64x64: the net can use badge
colour, team tint and facing direction (enemies walk down the screen).
Train/val split is by episode.

Usage: uv run scripts/train_side_classifier.py
Writes runs/side_cnn.pt (a plain state_dict).
"""

import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from alphaclash.perception.detector import NON_UNITS, side_crop
from alphaclash.perception.side_net import make_net

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/katacr/dataset/images/part2"
SKIP = NON_UNITS | {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
MAX_PER_SPLIT = {"train": 40000, "val": 6000}

names = [n for _, n in re.findall(r"^\s+(\d+): (\S+)", (SRC / "ClashRoyale_detection.yaml").read_text(), re.M)]
labels = sorted(p for p in SRC.glob("*/*/*.txt") if "background" not in str(p))
episodes = sorted({p.parent for p in labels})
random.Random(0).shuffle(episodes)
val_eps = set(episodes[:6])
random.Random(1).shuffle(labels)

data: dict[str, tuple[list, list]] = {"train": ([], []), "val": ([], [])}
for p in labels:
    split = "val" if p.parent in val_eps else "train"
    xs, ys = data[split]
    if len(xs) >= MAX_PER_SPLIT[split]:
        continue
    img = np.asarray(Image.open(p.with_suffix(".jpg")).convert("RGB"))
    H, W = img.shape[:2]
    for line in p.read_text().split("\n"):
        f = line.split()
        if len(f) < 6 or int(f[0]) >= len(names) or names[int(f[0])] in SKIP:
            continue
        x, y, w, h = map(float, f[1:5])
        xs.append(side_crop(img, ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)))
        ys.append(int(float(f[5])))
print({k: len(v[0]) for k, v in data.items()}, "enemy frac", np.mean(data["train"][1]).round(3))


def tensors(split):
    xs, ys = data[split]
    return torch.tensor(np.stack(xs)).permute(0, 3, 1, 2).float() / 255, torch.tensor(ys).float()


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xtr, ytr = tensors("train")
    Xva, yva = tensors("val")
    net = make_net().to(dev)
    EPOCHS = 14
    opt = torch.optim.AdamW(net.parameters(), 5e-4, weight_decay=1e-4)
    steps = EPOCHS * ((len(Xtr) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    best = 0.0
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            xb = Xtr[idx].to(dev)
            if random.random() < 0.5:
                xb = xb.flip(3)  # horizontal flip keeps side (vertical would not)
            # brightness/contrast jitter per sample (stream compression, arenas)
            xb = ((xb - 0.5) * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, 1, device=dev)) + 0.5
                  + 0.1 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)).clamp(0, 1)
            loss = lossf(net(xb).squeeze(1), ytr[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(net(Xva[i : i + 1024].to(dev))).squeeze(1).cpu() for i in range(0, len(Xva), 1024)])
        acc = ((p > 0.5).float() == yva).float().mean().item()
        print(f"epoch {epoch}  val acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(net.state_dict(), ROOT / "runs/side_cnn.pt")
    print(f"best val acc {best:.3f} -> runs/side_cnn.pt")
