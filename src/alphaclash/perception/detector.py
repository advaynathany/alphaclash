"""Perception v1: units on the arena (YOLO), with side and tile position.

The arena crop maps the iPhone Mirroring frame onto KataCR's 568x896 arena view
(checked by overlaying frames), so a detector trained on KataCR's labels works
on our frames. Side is not a YOLO class: a small CNN (scripts/train_side_classifier.py,
~94% on held-out KataCR episodes) reads it from the unit crop plus the band above
it, where the level badge / HP bar sits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from alphaclash.perception.screen import REF_H, REF_W

ARENA_CROP = (34, 191, 618, 1138)  # x0, y0, x1, y1 in the 652x1436 reference frame
ARENA_SIZE = (568, 896)
# KataCR's tile grid (18 x 32) on their 576-wide background, rescaled to 568.
_SX = ARENA_SIZE[0] / 576
GRID_XYXY = (4 * _SX, 82, 574 * _SX, 884)
GRID = (18, 32)

# UI-ish classes the detector also finds; not units.
NON_UNITS = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar",
             "skeleton-king-bar", "clock", "emote", "text", "elixir", "selected"}


@dataclass
class Unit:
    name: str
    side: str | None  # "mine" | "enemy" | None (spells, unreadable)
    conf: float
    xyxy: tuple[float, float, float, float]  # arena-crop pixels
    tile: tuple[float, float]  # (x 0-18, y 0-32), y=0 at the enemy end


def arena(frame: np.ndarray) -> np.ndarray:
    """Crop the arena out of a full frame and resize it to KataCR's view."""
    img = Image.fromarray(frame)
    if img.size != (REF_W, REF_H):
        img = img.resize((REF_W, REF_H))
    return np.asarray(img.crop(ARENA_CROP).resize(ARENA_SIZE))


def to_tile(x: float, y: float) -> tuple[float, float]:
    x0, y0, x1, y1 = GRID_XYXY
    return ((x - x0) / (x1 - x0) * GRID[0], (y - y0) / (y1 - y0) * GRID[1])


def side_crop(img: np.ndarray, xyxy: tuple[float, float, float, float], size: int = 64) -> np.ndarray:
    """Unit box plus the band above it (level badge / HP bar), as a size x size RGB array."""
    x0, y0, x1, y1 = xyxy
    h = y1 - y0
    H, W = img.shape[:2]
    box = (int(max(0, x0 - 4)), int(max(0, y0 - 0.5 * h - 6)), int(min(W, x1 + 4)), int(min(H, y1)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return np.zeros((size, size, 3), np.uint8)
    return np.asarray(Image.fromarray(img).crop(box).resize((size, size)))


SIDE_UNSURE = 0.2  # |p_enemy - 0.5| below this -> side None


class Detector:
    def __init__(self, weights: str, side_weights: str = "runs/side_cnn.pt", conf: float = 0.35, imgsz: int = 640) -> None:
        import torch
        from ultralytics import YOLO  # heavy imports; only when detection is used

        from alphaclash.perception.side_net import make_net

        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz
        self.side_net = make_net()
        self.side_net.load_state_dict(torch.load(side_weights, map_location="cpu", weights_only=True))
        self.side_net.eval()
        self._torch = torch

    def sides(self, img: np.ndarray, boxes: list) -> list[str | None]:
        if not boxes:
            return []
        torch = self._torch
        x = torch.tensor(np.stack([side_crop(img, b) for b in boxes])).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            p = torch.sigmoid(self.side_net(x)).squeeze(1).tolist()
        return [None if abs(q - 0.5) < SIDE_UNSURE else ("enemy" if q > 0.5 else "mine") for q in p]

    def detect(self, frame: np.ndarray) -> list[Unit]:
        img = arena(frame)
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets = [
            (res.names[int(k)], c, tuple(b))
            for b, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())
            if res.names[int(k)] not in NON_UNITS
        ]
        sides = self.sides(img, [b for _, _, b in dets])
        return [
            Unit(name, side, round(c, 3), b, to_tile((b[0] + b[2]) / 2, b[3]))
            for (name, c, b), side in zip(dets, sides)
        ]
