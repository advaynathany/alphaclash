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

# Tower HP bars: (x0, x1, y_center, color). The fill starts at x0 and shrinks
# from the right as HP drops. Bars sit up to ~15px higher/lower depending on the
# arena, so each is searched for within +-BAR_SEARCH rows of y_center.
# King bars only appear once that tower is damaged (or the king is activated).
TOWER_BARS = {
    "enemy_left": (120, 194, 352, "pink"),
    "enemy_right": (481, 555, 352, "pink"),
    "enemy_king": (288, 390, 214, "pink"),
    "my_left": (121, 194, 926, "blue"),
    "my_right": (481, 555, 926, "blue"),
    "my_king": (293, 383, 1065, "blue"),
}
BAR_SEARCH = 18
BADGE_W = 30  # gold level badge just left of each princess bar; gone once the tower falls

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
    # HP fraction 0-1 per tower; None = not readable this frame (bar occluded).
    towers: dict[str, float | None]


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

    @staticmethod
    def _bar_mask(a: np.ndarray, color: str) -> np.ndarray:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        if color == "pink":
            return (r > 200) & (g < 130) & (b > 110)
        return (b > 220) & (r < 140) & (g > 150)

    def _bar_fill(self, frame: np.ndarray, x0: int, x1: int, yc: int, color: str) -> tuple[float | None, bool]:
        """(HP fraction or None, whether any bar was seen at all).

        Bars are only a few px tall, have a border, and my HP digits sit on top of
        them, so: bar rows = rows with a solid run of bar color; fill end = rightmost
        bar-colored column in those rows (past it is only dark empty bar or digits).
        Accurate to ~+-5%.
        """
        ya = yc - BAR_SEARCH
        m = self._bar_mask(frame[ya : yc + BAR_SEARCH, x0 - 5 : x1 + 5].astype(int), color)
        rows = np.where(m.sum(1) >= 8)[0]
        if not len(rows):
            return None, False
        runs = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        rr = max(runs, key=len)
        cols = np.where(m[rr[0] : rr[-1] + 1].any(0))[0]
        start, end = cols.min() + x0 - 5, cols.max() + x0 - 5
        if start > x0 + 12:
            return None, True  # bar color, but not anchored at the bar's left edge
        frac = float(np.clip((end - x0 + 1) / (x1 - x0 + 1), 0, 1))
        if frac < 0.95:
            # A short fill is real only if the bar's right end shows the dark empty
            # color; otherwise something (troop, spell, tutorial hand) covers it.
            tail = frame[ya + rr[0] : ya + rr[-1] + 1, x1 - 8 : x1 - 1].astype(int)
            if not (tail.sum(2) < 330).mean() > 0.6:
                return None, True
        return frac, True

    @staticmethod
    def _badge(frame: np.ndarray, x0: int, yc: int) -> bool:
        a = frame[yc - BAR_SEARCH - 10 : yc + BAR_SEARCH + 10, x0 - BADGE_W : x0 - 1].astype(int)
        gold = (a[..., 0] > 200) & (a[..., 1] > 150) & (a[..., 2] < 110)
        return gold.mean() > 0.05

    def towers(self, frame: np.ndarray) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, (x0, x1, yc, color) in TOWER_BARS.items():
            fill, seen = self._bar_fill(frame, x0, x1, yc, color)
            if fill is not None:
                out[name] = round(fill, 3)
            elif seen:
                out[name] = None  # bar visible but partly covered
            elif name.endswith("king"):
                out[name] = 1.0  # bar hidden until first damage
            else:
                out[name] = None if self._badge(frame, x0, yc) else 0.0
        return out

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
            return Hud(False, None, [None] * 4, None, {})
        return Hud(True, round(self.elixir(f), 2), self.hand(f), self.next_card(f), self.towers(f))
