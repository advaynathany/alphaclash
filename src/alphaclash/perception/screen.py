"""Perception v0: read the fixed HUD (elixir, hand, next card) from a frame.

Layout is measured on the iPhone Mirroring frame (652x1436, includes the phone
bezel). Frames of another size are resized to it first, which holds as long as
the aspect ratio matches; a new device needs a re-measured layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REF_W, REF_H = 652, 1436

# Elixir bar: sample rows, and fill-end x at 0 and 10 elixir.
ELIXIR_ROWS = (1360, 1372)
ELIXIR_X0, ELIXIR_X10 = 185, 613

# Hand slots (x0, x1); card art spans HAND_Y. Raised (selected) cards sit higher.
HAND_SLOTS = [(158, 260), (274, 377), (391, 494), (507, 610)]
HAND_Y = (1205, 1340)
NEXT_BOX = (48, 1316, 100, 1380)
NEXT_LABEL = (38, 1280, 112, 1305)  # white "Next:" text; only drawn in battle
RAISE_SHIFTS = range(-24, 1, 4)

CARD_DIR = Path(__file__).resolve().parents[3] / "data" / "cards"
MATCH_MIN = 0.6  # below this a slot is reported as None (occluded / unknown)


def _feat(img: Image.Image, size=(24, 30), tile=6) -> np.ndarray:
    """Contrast-normalized per tile, so dimmed (unaffordable) cards and the grey
    charging wipe across them only disturb the few tiles the wipe edge crosses."""
    g = np.asarray(ImageOps.equalize(img.convert("L")).resize(size), dtype=float)
    h, w = g.shape
    t = g.reshape(h // tile, tile, w // tile, tile)
    t = (t - t.mean((1, 3), keepdims=True)) / (t.std((1, 3), keepdims=True) + 8.0)
    v = t.ravel()
    return v / (np.linalg.norm(v) + 1e-6)


def _is_empty_slot(img: Image.Image) -> bool:
    a = np.asarray(img, dtype=int)
    return ((a[..., 2] > a[..., 0] + 40) & (a[..., 2] > 80)).mean() > 0.85


def _art(img: Image.Image) -> Image.Image:
    """Top part of a card: the art, without the cost badge."""
    w, h = img.size
    return img.crop((8, 17, w - 8, int(h * 0.78)))


def _load_cards() -> dict[str, np.ndarray]:
    return {
        p.stem: _feat(_art(Image.open(p).convert("RGB")))
        for p in sorted(CARD_DIR.glob("*.png"))
        if not p.stem.startswith("_")
    }


@dataclass
class Hud:
    in_battle: bool
    elixir: float | None
    hand: list[str | None]
    next_card: str | None


class Screen:
    def __init__(self) -> None:
        self.cards = _load_cards()
        self._names = list(self.cards)
        self._mat = np.stack([self.cards[n] for n in self._names])

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] == (REF_H, REF_W):
            return frame
        return np.asarray(Image.fromarray(frame).resize((REF_W, REF_H)))

    @staticmethod
    def _bar(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row = frame[ELIXIR_ROWS[0] : ELIXIR_ROWS[1]].astype(int).mean(0)
        r, g, b = row[:, 0], row[:, 1], row[:, 2]
        pink = (r > 150) & (b > 150) & (g < 170)
        empty = (b > r + 30) & (r < 90)
        return pink, empty

    def in_battle(self, frame: np.ndarray) -> bool:
        pink, empty = self._bar(frame)
        x0, y0, x1, y1 = NEXT_LABEL
        label = (frame[y0:y1, x0:x1].min(2) > 200).mean()
        return (pink | empty)[200:610].mean() > 0.9 and label > 0.1

    def elixir(self, frame: np.ndarray) -> float:
        pink, _ = self._bar(frame)
        xs = np.where(pink[150:640])[0]
        if not len(xs):
            return 0.0
        end = xs.max() + 150
        return float(np.clip((end - ELIXIR_X0) / (ELIXIR_X10 - ELIXIR_X0) * 10, 0, 10))

    def _classify(self, crop: Image.Image) -> tuple[str | None, float]:
        sims = self._mat @ _feat(crop)
        i = int(sims.argmax())
        name, s = self._names[i], float(sims[i])
        if s < MATCH_MIN:
            return None, s
        return name, s

    def hand(self, frame: np.ndarray) -> list[str | None]:
        img = Image.fromarray(frame)
        out = []
        for x0, x1 in HAND_SLOTS:
            if _is_empty_slot(_art(img.crop((x0, HAND_Y[0], x1, HAND_Y[1])))):
                out.append(None)
                continue
            best = max(
                (self._classify(_art(img.crop((x0, HAND_Y[0] + dy, x1, HAND_Y[1] + dy)))) for dy in RAISE_SHIFTS),
                key=lambda t: t[1],
            )
            out.append(best[0])
        return out

    def next_card(self, frame: np.ndarray) -> str | None:
        crop = Image.fromarray(frame).crop(NEXT_BOX)
        return self._classify(_art(crop.resize((HAND_SLOTS[0][1] - HAND_SLOTS[0][0], HAND_Y[1] - HAND_Y[0]))))[0]

    def read(self, frame: np.ndarray) -> Hud:
        f = self.normalize(frame)
        if not self.in_battle(f):
            return Hud(False, None, [None] * 4, None)
        return Hud(True, round(self.elixir(f), 2), self.hand(f), self.next_card(f))
